"""Stage 9.0: the SQL label on the whole panel equals the pandas reference, and the split.

The panel is a seeded Markov chain over buckets with payoffs, liquidations,
other exits, unknown (XX) months, reporting gaps and re-defaults after cure, so
every branch of the label is hit. The reference sample covers every loan, so the
check compares all rows.
"""

import copy

import numpy as np
import pandas as pd
import pytest

from credit_risk.data.db import connect
from credit_risk.ews import cohort as ch
from credit_risk.ews.labels import build_ews_labels
from credit_risk.monitoring.panel import PanelSpec, build_panel

START, END = pd.Period("2018-01", "M"), pd.Period("2020-12", "M")


def markov_loans(n=300, seed=3):
    rng = np.random.default_rng(seed)
    perf, orig = [], []
    for i in range(n):
        loan_id = f"F17Q{i % 4 + 1}{i:07d}"
        orig.append({"loan_id": loan_id, "orig_upb": 100000.0, "fico": 700, "orig_ltv": 80, "orig_dti": 35,
                     "property_state": "CA", "channel": "R", "harp_indicator": "N", "source_file": "sample_orig_2017.txt"})
        month, state, age = START + int(rng.integers(0, 6)), 0, int(rng.integers(0, 3))
        while month <= END:
            u, zb = rng.random(), None
            if state == 0:
                state = 1 if u < 0.06 else 0
                zb = "01" if u > 0.98 else None
            elif state == 1:
                state = 0 if u < 0.4 else 2 if u < 0.75 else 1
            elif state == 2:
                state = 0 if u < 0.2 else 3 if u < 0.7 else 2
            else:
                zb = "03" if u < 0.1 else "96" if u < 0.12 else None
                state = 0 if u > 0.85 else state + 1
            status = "XX" if rng.random() < 0.01 else "RA" if state >= 6 else f"{min(state, 99):02d}"
            if rng.random() > 0.01 or zb:                                  # ~1% of months not reported
                perf.append({"loan_id": loan_id, "period": month.to_timestamp().date(), "loan_age": age,
                             "current_upb": 0.0 if zb else 90000.0, "zero_balance_removal_upb": 90000.0 if zb else None,
                             "delinquency_status": status, "zero_balance_code": zb})
            if zb:
                break
            month, age = month + 1, age + 1
    return pd.DataFrame(perf), pd.DataFrame(orig)


@pytest.fixture
def spec(project_config):
    base = ch.CohortSpec.from_config(project_config.ews, project_config.definitions)
    splits = {"train": (pd.Timestamp("2018-03-01"), pd.Timestamp("2018-12-01")),
              "validation": (pd.Timestamp("2019-04-01"), pd.Timestamp("2019-06-01")),
              "stress_test": (pd.Timestamp("2019-10-01"), pd.Timestamp("2020-02-01")),
              "out_of_time": (pd.Timestamp("2020-06-01"), None)}
    return ch.CohortSpec(**{**base.__dict__, "splits": splits, "reference_sample_loans": 10_000})


@pytest.fixture
def built(settings, project_config, spec):
    folder = settings.processed_dir / "freddie_mac"
    folder.mkdir(parents=True)
    perf, orig = markov_loans()
    perf.to_parquet(folder / "performance.parquet", index=False)
    orig.to_parquet(folder / "origination.parquet", index=False)
    con = connect(settings)
    build_panel(con, folder, PanelSpec.from_config(project_config.ews, project_config.definitions))
    ch.build_labels(con, folder, spec)
    return con, folder, perf


def test_sql_labels_equal_the_pandas_reference_on_every_row(built, spec, definitions):
    con, folder, perf = built
    ref, sql = ch.reference_labels(con, folder, spec, definitions)
    assert len(ref) == len(sql) == len(perf)
    for col in ("bucket_at_t", "ever_90plus_before_t", "y", "event_month_offset", "exclusion_reason"):
        assert ch._same(ref[col], sql[col]) == 0, col
    assert set(sql["exclusion_reason"].dropna()) >= {"inactive_at_t", "already_90plus_at_t", "status_not_available_at_t",
                                                     "incomplete_performance_window"}
    assert sql["y"].sum() > 20 and (sql["y"] == 0).sum() > 1000


def test_every_check_passes(built, spec, definitions):
    con, folder, _ = built
    checks, info = ch.label_checks(con, folder, spec, definitions)
    assert checks["ok"].all(), checks[~checks["ok"]]
    assert info["summary"]["splits"]["out_of_time"] == ["2020-06", "2020-09"]   # last month 2020-12 minus 3
    assert info["by_split"]["split"].tolist() == list(ch.SPLITS)


def test_split_and_regime_follow_the_periods(built, spec):
    con, folder, _ = built
    labels = pd.read_parquet(folder / ch.LABELS_FILE)
    labels["period"] = pd.to_datetime(labels["period"])
    first_split = labels.groupby("split")["period"].agg(["min", "max"])
    assert first_split.loc["train", "min"] == pd.Timestamp("2018-03-01")
    assert first_split.loc["out_of_time", "max"] == pd.Timestamp("2020-09-01")
    gap = labels[(labels["period"] > pd.Timestamp("2018-12-01")) & (labels["period"] < pd.Timestamp("2019-04-01"))]
    assert gap["split"].isna().all()                                    # between segments: no split
    assert labels.loc[labels["period"] > pd.Timestamp("2020-09-01"), "split"].isna().all()
    assert set(labels.loc[labels["period"] == pd.Timestamp("2020-03-01"), "regime"]) == {"covid"}
    assert set(labels.loc[labels["period"] < pd.Timestamp("2020-03-01"), "regime"]) == {"pre_covid"}


def test_a_label_window_may_not_reach_the_next_segment(project_config):
    ews = copy.deepcopy(project_config.ews)
    ews["ews"]["time_split"]["train"]["end"] = "2018-11"                  # + 3 months = 2019-02 >= validation start
    with pytest.raises(ValueError, match="at least 3 months before validation"):
        ch.CohortSpec.from_config(ews, project_config.definitions)
    ews = copy.deepcopy(project_config.ews)
    ews["ews"]["time_split"] = dict(reversed(list(ews["ews"]["time_split"].items())))
    with pytest.raises(ValueError, match="in that order"):
        ch.CohortSpec.from_config(ews, project_config.definitions)


def test_prior_90plus_exclusion_follows_the_definition(settings, project_config, built, spec):
    con, folder, perf = built
    definitions = copy.deepcopy(project_config.definitions)
    definitions["mortgage"]["redefault_after_cure_counts_as_event"] = False
    strict = ch.CohortSpec(**{**spec.__dict__, "prior_90plus_excluded": True})
    ch.build_labels(con, folder, strict)
    sql = pd.read_parquet(folder / ch.LABELS_FILE).sort_values(["loan_id", "period"]).reset_index(drop=True)
    ref = build_ews_labels(perf, definitions, horizon=3).sort_values(["loan_id", "period"]).reset_index(drop=True)
    assert (sql["exclusion_reason"] == "prior_90plus_excluded").sum() > 0
    assert ch._same(ref["exclusion_reason"], sql["exclusion_reason"]) == 0
    assert ch._same(ref["y"], sql["y"]) == 0
