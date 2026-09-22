"""Stage 2.6: every feature leaves the shortlist with one step and one reason,
and the same inputs always give the same shortlist."""

import numpy as np
import pandas as pd
import pytest

from credit_risk.features.binning import BinningSpec, fit_all
from credit_risk.features.iv_report import ReportThresholds, iv_report
from credit_risk.features.selection import (
    SelectionError,
    SelectionSpec,
    _vif,
    select_features,
    shortlist_markdown,
    shortlist_payload,
)

MODEL_DEV = {
    "binning": {"initial_quantile_bins": 20, "max_bins": 8, "min_bin_share": 0.05,
                "prefer_monotonic_trend": True, "missing_as_own_bin": True, "rare_category_share": 0.01},
    "feature_selection": {
        "iv_min": 0.02, "max_abs_correlation": 0.70, "max_abs_rank_correlation_raw": 0.90, "max_vif": 5.0,
        "min_iv_retention_validation": 0.5, "min_bad_rate_rank_corr": 0.8,
        "drop_on_flags": ["leakage_suspect", "iv_drops_out_of_sample", "trend_not_confirmed",
                          "unstable_vs_validation", "shift_vs_current"],
    },
}
DEFINITIONS = {"metric_thresholds": {
    "psi": {"stable_below": 0.10, "significant_above": 0.25, "zero_bin_epsilon": 1e-4},
    "woe": {"zero_count_adjustment": 0.5},
    "iv": {"min_useful": 0.02, "leakage_suspect_above": 0.50},
}}


def frame(n, seed):
    rng = np.random.default_rng(seed)
    a, b, c = rng.normal(size=(3, n))
    y = (rng.random(n) < 1 / (1 + np.exp(-(-2.5 + 0.5 * a + 0.4 * b + 0.3 * c)))).astype(int)
    return pd.DataFrame({
        "a": a,
        "a_copy": a * 1000 + rng.normal(scale=1e-3, size=n),   # the same quantity, another unit
        "a_near": a + rng.normal(scale=0.3, size=n),            # rank correlation with a ~0.95
        "d": 0.6 * a + 0.8 * rng.normal(size=n),                # correlated with a at ~0.6
        "b": b,
        "c": c,
        "noise": rng.normal(size=n),
        "TARGET": y,
    })


def run(features=("a", "a_copy", "b", "c", "noise"), selection=None, validation_seed=1):
    spec_bins = BinningSpec.from_config(MODEL_DEV, DEFINITIONS)
    train = frame(30000, 0)
    binnings = fit_all(train, list(features), "TARGET", spec_bins)
    th = ReportThresholds.from_config(DEFINITIONS, MODEL_DEV)
    report, _ = iv_report(binnings, frame(10000, validation_seed), frame(10000, 2), "TARGET", th)
    config = {**MODEL_DEV, "feature_selection": {**MODEL_DEV["feature_selection"], **(selection or {})}}
    spec = SelectionSpec.from_config(config, DEFINITIONS)
    return select_features(report, binnings, train, spec), spec, report, binnings, train


def decision(result, feature):
    return result.set_index("feature").loc[feature]


def test_every_feature_gets_exactly_one_decision():
    result, *_ = run()
    assert sorted(result["feature"]) == ["a", "a_copy", "b", "c", "noise"]
    assert set(result["step"]) <= {"selected", "policy", "iv_min", "flags", "correlation", "vif"}
    assert (result["selected"] == (result["step"] == "selected")).all()
    assert (result.loc[~result["selected"], "reason"].str.len() > 0).all()


def test_weak_features_are_dropped_at_the_iv_screen():
    result, *_ = run()
    assert decision(result, "noise").step == "iv_min"
    assert {"a", "b", "c"} <= set(result.loc[result["selected"], "feature"])


def test_the_duplicate_with_the_lower_iv_goes_and_names_its_partner():
    result, *_ = run()
    kept, dropped = sorted(["a", "a_copy"], key=lambda f: -decision(result, f).iv_train)
    assert decision(result, kept).selected
    assert decision(result, dropped).step == "correlation"
    assert kept in decision(result, dropped).reason


def test_raw_rank_correlation_catches_a_duplicate_the_woe_test_misses():
    # With the WoE threshold effectively off, only the raw Spearman test can see the near-duplicate.
    result, *_ = run(features=("a", "a_near", "b", "c", "noise"), selection={"max_abs_correlation": 0.9999})
    dropped = [f for f in ("a", "a_near") if not decision(result, f).selected]
    assert len(dropped) == 1
    assert "Spearman" in decision(result, dropped[0]).reason


def test_a_policy_exclusion_wins_over_statistics():
    result, *_ = run(selection={"policy_exclude": {"a": "protected attribute"}})
    row = decision(result, "a")
    assert row.step == "policy" and row.reason == "protected attribute"
    assert decision(result, "a_copy").selected  # its duplicate is no longer shadowed


def test_a_policy_exclusion_needs_a_reason():
    config = {**MODEL_DEV, "feature_selection": {**MODEL_DEV["feature_selection"], "policy_exclude": {"a": ""}}}
    with pytest.raises(SelectionError, match="needs a reason"):
        SelectionSpec.from_config(config, DEFINITIONS)


def test_a_review_flag_listed_in_drop_on_flags_removes_the_feature():
    result, spec, report, binnings, train = run()
    report = report.copy()
    report.loc[report.feature == "b", "review_flags"] = "trend_not_confirmed"
    result = select_features(report, binnings, train, spec)
    row = decision(result, "b")
    assert row.step == "flags" and row.reason.startswith("trend_not_confirmed")


def test_vif_drops_the_worst_offender_until_all_are_under_the_limit():
    # a and d correlate at ~0.6: under the 0.70 correlation limit, but their VIF is ~1.6.
    result, *_ = run(features=("a", "b", "c", "d"), selection={"max_vif": 1.3})
    kept = result[result["selected"]]
    assert (kept["vif"] <= 1.3).all()
    dropped = result[result["step"] == "vif"]
    assert list(dropped["feature"]) in (["a"], ["d"])


def test_vif_of_independent_columns_is_one_and_of_collinear_ones_is_large():
    rng = np.random.default_rng(5)
    x, y = rng.normal(size=(2, 5000))
    assert _vif(np.column_stack([x, y])) == pytest.approx([1, 1], abs=0.01)
    assert _vif(np.column_stack([x, y, x + y + rng.normal(scale=0.01, size=5000)])).max() > 100


def test_the_same_inputs_give_the_same_shortlist_whatever_the_row_order():
    result, spec, report, binnings, train = run()
    shuffled = select_features(report.sample(frac=1, random_state=3), binnings,
                               train.sample(frac=1, random_state=4).reset_index(drop=True), spec)
    meta = {"rows_train": len(train), "binning_fingerprint": "x" * 64}
    assert shortlist_payload(result, spec, meta)["fingerprint"] == shortlist_payload(shuffled, spec, meta)["fingerprint"]


def test_a_report_feature_without_a_binning_is_rejected():
    result, spec, report, binnings, train = run()
    extra = pd.concat([report, report.head(1).assign(feature="ghost")])
    with pytest.raises(SelectionError, match="without a fitted binning"):
        select_features(extra, binnings, train, spec)


def test_markdown_lists_every_rule_and_every_non_iv_drop():
    result, spec, *_ = run()
    text = shortlist_markdown(result, spec, {"rows_train": 30000, "binning_fingerprint": "f" * 64})
    assert "Spearman" in text and "VIF <= 5.0" in text
    for row in result[~result["selected"] & (result["step"] != "iv_min")].itertuples():
        assert f"`{row.feature}`" in text
