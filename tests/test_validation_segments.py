"""Stage 6.7: performance, calibration and reject rate by segment, with the fairness review."""

import numpy as np
import pandas as pd
import pytest

from credit_risk.data.db import connect
from credit_risk.validation.package import load_package, load_scoring_frame, score_with_package
from credit_risk.validation.segments import (
    SegmentError,
    SegmentSpec,
    add_derived_columns,
    age_band,
    assign_score_band,
    ext_source_missing_pattern,
    recompute_segments,
    score_band_edges,
    segment_checks,
    segments_markdown,
)

CONFIG = {
    "segments": [
        {"name": "score_band", "column": "score_band"},
        {"name": "contract_type", "column": "NAME_CONTRACT_TYPE"},
        {"name": "ext_source_missing_pattern", "column": "ext_source_missing_pattern"},
        {"name": "gender", "column": "CODE_GENDER", "fairness_review": True},
        {"name": "age_band", "column": "DAYS_BIRTH", "fairness_review": True},
    ],
    "segment_options": {"samples": ["test"], "min_group_size": 50, "score_bands": 4,
                        "age_bands_years": [30, 45, 60], "reject_rule": {"pd_above": 0.10}},
}


def test_the_spec_maps_days_birth_to_the_age_band_and_lists_the_source_columns():
    spec = SegmentSpec.from_config(CONFIG)
    assert [s.column for s in spec.segments] == ["score_band", "NAME_CONTRACT_TYPE", "ext_source_missing_pattern",
                                                 "CODE_GENDER", "age_band"]
    assert spec.source_columns == ["NAME_CONTRACT_TYPE", "CODE_GENDER", "DAYS_BIRTH",
                                   "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    assert [s.fairness_review for s in spec.segments] == [False, False, False, True, True]
    with pytest.raises(SegmentError):
        SegmentSpec.from_config({"segments": []})
    with pytest.raises(SegmentError):
        SegmentSpec.from_config({**CONFIG, "segment_options": {"age_bands_years": [45, 30]}})
    with pytest.raises(SegmentError):
        SegmentSpec.from_config({**CONFIG, "segment_options": {"reject_rule": {"pd_above": 1.5}}})


def test_derived_columns_are_what_they_say():
    edges = score_band_edges(np.arange(100, 200), 4)
    bands = assign_score_band(pd.Series([100, 124, 125, 150, 199]), edges)
    assert bands.iloc[0] == bands.iloc[1] and bands.iloc[1] != bands.iloc[2]
    assert str(bands.iloc[0]).startswith("01: <") and str(bands.iloc[-1]).startswith("04: >=")
    frame = pd.DataFrame({"EXT_SOURCE_1": [np.nan, 0.5, np.nan], "EXT_SOURCE_2": [0.3, 0.3, np.nan],
                          "EXT_SOURCE_3": [np.nan, 0.1, 0.1]})
    assert list(ext_source_missing_pattern(frame)) == ["missing 1+3", "none missing", "missing 1+2"]
    years = np.array([25, 30, 44.9, 45, 70])
    assert list(age_band(-years * 365.25, (30, 45, 60))) == ["< 30", "30-45", "30-45", "45-60", ">= 60"]


def synthetic_scored(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    gender = rng.choice(["F", "M"], n, p=[0.65, 0.35])
    risk = rng.normal(size=n) + np.where(gender == "M", 0.5, 0.0)
    pd_true = 1 / (1 + np.exp(-(risk - 2.6)))
    y = (rng.uniform(size=n) < pd_true).astype(int)
    score = np.round(600 - 30 * risk)
    return pd.DataFrame({
        "TARGET": y, "split": np.where(np.arange(n) < n // 2, "train", "test"), "score": score,
        "pd_champion": pd_true, "pd_challenger": pd_true, "CODE_GENDER": gender,
        "NAME_CONTRACT_TYPE": rng.choice(["Cash loans", "Revolving loans"], n, p=[0.9, 0.1]),
        "DAYS_BIRTH": -rng.integers(20, 70, n) * 365.25,
        "EXT_SOURCE_1": np.where(rng.uniform(size=n) < 0.5, np.nan, 0.5),
        "EXT_SOURCE_2": 0.5, "EXT_SOURCE_3": np.where(rng.uniform(size=n) < 0.2, np.nan, 0.5),
    })


def test_groups_partition_the_sample_and_the_overall_row_matches_the_whole():
    spec = SegmentSpec.from_config(CONFIG)
    frame = synthetic_scored()
    scored = add_derived_columns(frame, spec, (frame["split"] == "train").to_numpy())
    table = recompute_segments(scored, spec)
    checks = segment_checks(table, scored, spec)
    assert checks["ok"].all(), checks[~checks["ok"]].to_string()
    assert set(table["segment"]) == {s.name for s in spec.segments}
    overall = table[(table["group"] == "(all)") & (table["model"] == "champion")]
    assert overall["n"].nunique() == 1 and overall["n"].iloc[0] == (scored["split"] == "test").sum()
    gender = table[(table["segment"] == "gender") & (table["model"] == "champion") & (table["group"] != "(all)")]
    assert gender["share"].sum() == pytest.approx(1.0)
    # Men were built riskier: higher bad rate, higher reject rate, AIR below 1 for men and 1 for women.
    men, women = gender.set_index("group").loc["M"], gender.set_index("group").loc["F"]
    assert men["bad_rate"] > women["bad_rate"] and men["reject_rate"] > women["reject_rate"]
    assert women["adverse_impact_ratio"] == pytest.approx(1.0) and men["adverse_impact_ratio"] < 1
    # The PDs are the truth here, so no group is flagged for calibration.
    assert (table.loc[table["group"] != "(all)", "calibration_flag"] == "").all()


def test_a_group_the_model_under_predicts_is_flagged_and_small_groups_get_no_auc():
    spec = SegmentSpec.from_config(CONFIG)
    frame = synthetic_scored(seed=1)
    men = (frame["CODE_GENDER"] == "M") & (frame["split"] == "test")
    frame.loc[men, "pd_champion"] = frame.loc[men, "pd_champion"] * 0.5      # champion says men are half as risky
    scored = add_derived_columns(frame, spec, (frame["split"] == "train").to_numpy())
    table = recompute_segments(scored, spec)
    row = table[(table["segment"] == "gender") & (table["group"] == "M") & (table["model"] == "champion")].iloc[0]
    assert row["calibration_flag"] == "PD thấp (thiếu)" and row["calibration_z"] > 2
    other = table[(table["segment"] == "gender") & (table["group"] == "M") & (table["model"] == "challenger")].iloc[0]
    assert other["calibration_flag"] == ""
    text = segments_markdown(table, segment_checks(table, scored, spec), spec)
    assert "gender = M, champion, test: hiệu chỉnh: PD thấp (thiếu)" in text
    # A tiny group: size and rates are reported, AUC is not.
    frame.loc[frame.index[-5:], "CODE_GENDER"] = "XNA"
    scored = add_derived_columns(frame, spec, (frame["split"] == "train").to_numpy())
    table = recompute_segments(scored, spec)
    tiny = table[(table["segment"] == "gender") & (table["group"] == "XNA")]
    assert tiny["note"].str.startswith("ít quan sát").all() and tiny["auc"].isna().all() and (tiny["n"] == 5).all()


def test_missing_segment_columns_and_samples_are_refused():
    spec = SegmentSpec.from_config(CONFIG)
    frame = synthetic_scored()
    train = (frame["split"] == "train").to_numpy()
    with pytest.raises(SegmentError, match="EXT_SOURCE"):
        add_derived_columns(frame.drop(columns=["EXT_SOURCE_2"]), spec, train)
    with pytest.raises(SegmentError, match="not in the scored frame"):
        add_derived_columns(frame.drop(columns=["CODE_GENDER"]), spec, train)
    scored = add_derived_columns(frame, spec, train)
    with pytest.raises(SegmentError, match="no rows"):
        recompute_segments(scored[scored["split"] == "train"], spec)


# ---------------------------------------------------------------------------
# The real chain on the synthetic build, with the project's own validation.yaml
# ---------------------------------------------------------------------------

def test_on_the_synthetic_package_every_configured_segment_is_reported(stage6_build):
    settings, config = stage6_build
    folder = settings.processed_dir / "home_credit"
    package = load_package(folder / "model_package", config.model_dev, config.definitions)
    configured = SegmentSpec.from_config(config.validation)
    con = connect(settings)
    columns = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{(folder / 'features.parquet').as_posix()}')").fetchall()]
    spec, missing = configured.split_by_availability(columns)
    # The synthetic raw data has no family status: that segment is reported as not computable, not dropped silently.
    assert [s.name for s in missing] == ["family_status"]
    frame = load_scoring_frame(con, folder / "features.parquet", folder / "splits.parquet",
                               package.features + spec.source_columns)
    scored = score_with_package(package, frame).join(frame.drop(columns=["SK_ID_CURR"]))
    scored = add_derived_columns(scored, spec, (scored["split"] == "train").to_numpy())
    table = recompute_segments(scored, spec)
    checks = segment_checks(table, scored, spec)
    # Coverage checks must hold; the monotone bad rate across 10 bands may fail on 3,000 weak-signal
    # synthetic rows, which is a finding the engine should report, not a defect of the engine.
    coverage = checks[checks["check"].str.startswith("groups cover")]
    assert coverage["ok"].all(), coverage[~coverage["ok"]].to_string()
    assert set(table["segment"]) == {s.name for s in spec.segments}
    assert table.loc[table["fairness_review"], "segment"].drop_duplicates().tolist() == ["gender", "age_band"]
    assert set(table["model"]) == {"champion", "challenger"}
    text = segments_markdown(table, checks, spec)
    assert "rà soát công bằng" in text and "Kiểm tra:" in text
