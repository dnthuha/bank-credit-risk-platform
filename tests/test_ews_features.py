"""Stage 9.1: features at t read rows <= t only, follow the availability matrix,
and are computed as their definitions say on a hand-checked loan."""

import copy

import pandas as pd
import pytest

from credit_risk.data.db import connect
from credit_risk.ews import cohort as ch
from credit_risk.ews import features as ft
from credit_risk.features.availability import parse_availability
from credit_risk.monitoring.panel import PanelSpec, build_panel

from conftest import ews_markov_loans

SPLITS = {"train": (pd.Timestamp("2018-03-01"), pd.Timestamp("2018-12-01")),
          "validation": (pd.Timestamp("2019-04-01"), pd.Timestamp("2019-06-01")),
          "stress_test": (pd.Timestamp("2019-10-01"), pd.Timestamp("2020-02-01")),
          "out_of_time": (pd.Timestamp("2020-06-01"), None)}


def build(settings, config, perf, orig, name="a"):
    folder = settings.processed_dir / "freddie_mac" / name
    folder.mkdir(parents=True)
    perf.to_parquet(folder / "performance.parquet", index=False)
    orig.to_parquet(folder / "origination.parquet", index=False)
    con = connect(settings)
    panel_spec = PanelSpec.from_config(config.ews, config.definitions)
    build_panel(con, folder, panel_spec)
    cohort = ch.CohortSpec(**{**ch.CohortSpec.from_config(config.ews, config.definitions).__dict__, "splits": SPLITS})
    ch.build_labels(con, folder, cohort)
    spec = ft.FeatureSpec.from_config(config.ews, config.definitions)
    path, month = ft.build_dataset(con, folder, spec, panel_spec, ch.last_period(con, folder))
    data = pd.read_parquet(path)
    data["period"] = pd.to_datetime(data["period"])
    return con, folder, spec, month, data.sort_values(["loan_id", "period"]).reset_index(drop=True)


def test_features_at_t_do_not_move_when_later_months_change(settings, project_config):
    perf, orig = ews_markov_loans()
    _, _, spec, _, before = build(settings, project_config, perf, orig, "before")
    cutoff = pd.Timestamp("2019-06-01").date()
    later = perf["period"] > cutoff
    changed = perf.copy()
    changed.loc[later, "delinquency_status"] = "02"
    changed.loc[later, "current_upb"] = 1.0
    changed.loc[later, "current_interest_rate"] = 9.0
    changed.loc[later, "borrower_assistance_plan"] = "F"
    changed.loc[later, "modification_flag"] = "Y"
    _, _, _, _, after = build(settings, project_config, changed, orig, "after")
    key = ["loan_id", "period"]
    early = lambda d: d[d["period"] <= pd.Timestamp(cutoff)].drop(columns=["y", "event_month_offset", "split"])
    both = early(before).merge(early(after), on=key, suffixes=("_b", "_a"))
    assert len(both) > 1000
    for f in spec.names:
        b, a = both[f"{f}_b"], both[f"{f}_a"]
        assert ((b == a) | (b.isna() & a.isna())).all(), f


def test_a_hand_checked_loan(settings, project_config):
    months = ["00", "00", "01", "02", "00", "00", "00", "01", "00", "00", "00", "00"]
    perf = pd.DataFrame({
        "loan_id": "F17Q10000001", "period": [(pd.Period("2018-01", "M") + k).to_timestamp().date() for k in range(12)],
        "loan_age": range(3, 15), "current_upb": [100000.0 - 100 * k for k in range(12)], "zero_balance_removal_upb": None,
        "delinquency_status": months, "zero_balance_code": None, "current_interest_rate": [4.0] * 8 + [3.5] * 4,
        "current_non_interest_bearing_upb": [0.0] * 8 + [1000.0] * 4, "modification_flag": [None] * 8 + ["Y"] * 4,
        "payment_deferral_flag": None, "delinquency_due_to_disaster": None,
        "borrower_assistance_plan": [None, None, None, "F", "F", None, None, None, "T", None, None, None],
    })
    _, orig = ews_markov_loans(n=1)
    orig["loan_id"], orig["orig_upb"] = "F17Q10000001", 200000.0
    filler, filler_orig = ews_markov_loans(n=40, seed=5)                  # other loans, so the panel is not trivial
    _, _, _, _, data = build(settings, project_config, pd.concat([perf, filler]), pd.concat([orig, filler_orig]))
    row = data[(data["loan_id"] == "F17Q10000001") & (data["period"] == pd.Timestamp("2018-09-01"))]
    assert len(row) == 1                                                  # train, y = 0: three clean months follow
    r = row.iloc[0]
    assert r["split"] == "train" and r["y"] == 0
    assert r["mob"] == 11 and r["bucket_num"] == 0
    assert r["bucket_change_1m"] == -1                                     # 01 -> 00
    assert r["months_30plus_window"] == 2                                  # window 2018-04..2018-09: 04 (02) and 08 (01)
    assert r["max_bucket_window"] == 2
    assert r["months_since_delinquent"] == 1
    assert r["history_months_window"] == 6
    assert r["upb_ratio"] == pytest.approx((100000 - 800) / 200000)
    assert r["paydown_window"] == pytest.approx(500 / 200000)              # 99,700 at 2018-04 -> 99,200
    assert r["rate_change"] == pytest.approx(-0.5)
    assert r["non_interest_bearing_share"] == pytest.approx(1000 / 99200)
    assert r["assistance_months_window"] == 3                              # 2018-04, 2018-05 and 2018-09
    assert r["modified"] == 1 and r["assistance_plan_at_t"] == "T"


def test_every_check_passes_and_scoring_rows_have_no_label(settings, project_config):
    perf, orig = ews_markov_loans()
    con, folder, spec, month, data = build(settings, project_config, perf, orig)
    checks, summary = ft.dataset_checks(con, folder, spec, month)
    assert checks["ok"].all(), checks[~checks["ok"]]
    assert month == pd.Timestamp("2020-12-01") and summary["scoring_rows"] > 0
    scoring = data[data["split"] == ft.SCORING]
    assert scoring["y"].isna().all() and (scoring["period"] == month).all()
    assert set(data["split"]) == set(ch.SPLITS) | {ft.SCORING}


def test_a_feature_reading_a_blocked_column_stops_the_build(project_config):
    spec = ft.FeatureSpec.from_config(project_config.ews, project_config.definitions)
    raw = copy.deepcopy(project_config.feature_availability)
    raw["freddie_mac"]["performance"]["columns"]["current_interest_rate"] = {"availability": "post_decision"}
    with pytest.raises(ValueError, match="rate_change <- performance.current_interest_rate"):
        ft.check_availability(spec, parse_availability(raw, "freddie_mac"))
    frame = ft.check_availability(spec, parse_availability(project_config.feature_availability, "freddie_mac"))
    assert frame["use"].all() and set(frame["feature"]) == set(spec.names)


def test_unknown_or_repeated_features_are_refused(project_config):
    ews = copy.deepcopy(project_config.ews)
    ews["ews"]["features"]["numeric"].append("future_balance")
    with pytest.raises(ValueError, match="unknown features"):
        ft.FeatureSpec.from_config(ews, project_config.definitions)
    ews = copy.deepcopy(project_config.ews)
    ews["ews"]["features"]["categorical"].append("fico")
    with pytest.raises(ValueError, match="listed twice"):
        ft.FeatureSpec.from_config(ews, project_config.definitions)
