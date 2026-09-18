"""Three-month 90+ DPD label: one test per rule, horizon = 3 months."""

import copy

import pandas as pd
import pytest

from credit_risk.ews.labels import (
    ALREADY_90PLUS,
    INACTIVE_AT_T,
    INCOMPLETE_WINDOW,
    PRIOR_90PLUS,
    STATUS_NOT_AVAILABLE,
    build_ews_labels,
)
from credit_risk.monitoring.buckets import status_to_bucket


def make_panel(loans: dict[str, list]) -> pd.DataFrame:
    """loans: {loan_id: [status | (status, zero_balance_code) | None for a missing month]}, from 2019-01."""
    rows = []
    for loan_id, months in loans.items():
        for offset, entry in enumerate(months):
            if entry is None:
                continue  # month not reported
            status, zb = entry if isinstance(entry, tuple) else (entry, None)
            period = (pd.Period("2019-01", "M") + offset).to_timestamp()
            rows.append((loan_id, period, status, zb))
    return pd.DataFrame(rows, columns=["loan_id", "period", "delinquency_status", "zero_balance_code"])


def label_at(labels: pd.DataFrame, loan_id: str, month: str) -> pd.Series:
    row = labels[(labels["loan_id"] == loan_id) & (labels["period"] == pd.Timestamp(f"{month}-01"))]
    assert len(row) == 1, f"no observation row for {loan_id} at {month}"
    return row.iloc[0]


def assert_label(labels, loan_id, month, y=None, reason=None, offset=None):
    row = label_at(labels, loan_id, month)
    if reason is None:
        assert pd.isna(row["exclusion_reason"]), f"unexpected exclusion {row['exclusion_reason']}"
        assert row["y"] == y
    else:
        assert row["exclusion_reason"] == reason
        assert pd.isna(row["y"])
    if offset is not None:
        assert row["event_month_offset"] == offset


@pytest.fixture
def labels_for(definitions):
    def run(loans, **overrides):
        defs = copy.deepcopy(definitions)
        defs["mortgage"].update(overrides)
        return build_ews_labels(make_panel(loans), defs)
    return run


def test_status_codes_map_to_buckets(definitions):
    codes = definitions["mortgage"]["delinquency_status_codes"]
    status = pd.Series(["00", "01", "02", "03", "99", "RA", "XX", None, " 01 "])
    buckets = status_to_bucket(status, codes)
    expected = ["current", "dpd30", "dpd60", "dpd90plus", "dpd90plus", "dpd90plus", None, None, "dpd30"]
    assert [None if pd.isna(b) else b for b in buckets] == expected


def test_performing_loan_is_good_when_window_fully_observed(labels_for):
    labels = labels_for({"L": ["00"] * 6})
    for month in ("2019-01", "2019-02", "2019-03"):
        assert_label(labels, "L", month, y=0)


def test_end_of_data_before_t_plus_3_is_excluded(labels_for):
    labels = labels_for({"L": ["00"] * 6})  # Jan..Jun
    for month in ("2019-04", "2019-05", "2019-06"):
        assert_label(labels, "L", month, reason=INCOMPLETE_WINDOW)


def test_event_in_window_and_horizon_boundary(labels_for):
    labels = labels_for({"L": ["00", "00", "01", "02", "03", "RA"]})  # 90+ first in May
    assert_label(labels, "L", "2019-01", y=0)  # May is t+4: outside the window
    assert_label(labels, "L", "2019-02", y=1, offset=3)
    assert_label(labels, "L", "2019-03", y=1, offset=2)
    assert_label(labels, "L", "2019-04", y=1, offset=1)


def test_already_90plus_at_t_is_excluded(labels_for):
    labels = labels_for({"L": ["00", "03", "04", "RA", "RA"]})
    assert_label(labels, "L", "2019-02", reason=ALREADY_90PLUS)
    assert_label(labels, "L", "2019-04", reason=ALREADY_90PLUS)  # RA counts as 90+


def test_voluntary_payoff_in_window_is_good(labels_for):
    labels = labels_for({"L": ["00", "01", "00", ("00", "01")]})  # pays off in April
    assert_label(labels, "L", "2019-01", y=0)
    assert_label(labels, "L", "2019-03", y=0)
    assert_label(labels, "L", "2019-04", reason=INACTIVE_AT_T)


def test_liquidation_counts_as_event_even_without_90plus_status(labels_for):
    labels = labels_for({"L": ["00", "02", ("02", "03")]})  # short sale / charge off in March
    assert_label(labels, "L", "2019-01", y=1, offset=2)


def test_other_exit_hides_the_outcome(labels_for):
    labels = labels_for({"L": ["00", "00", ("00", "96")]})  # defect repurchase
    assert_label(labels, "L", "2019-01", reason=INCOMPLETE_WINDOW)


def test_reporting_gap_in_window_is_excluded(labels_for):
    labels = labels_for({"L": ["00", "00", None, "00", "00", "00"]})  # March missing
    assert_label(labels, "L", "2019-01", reason=INCOMPLETE_WINDOW)
    assert_label(labels, "L", "2019-02", reason=INCOMPLETE_WINDOW)
    assert_label(labels, "L", "2019-04", reason=INCOMPLETE_WINDOW)  # May, Jun, then data ends


def test_event_observed_before_a_gap_still_counts(labels_for):
    labels = labels_for({"L": ["00", "03", None, "RA"]})
    assert_label(labels, "L", "2019-01", y=1, offset=1)


@pytest.mark.parametrize("unknown", ["XX", (None, None)], ids=["raw_XX", "null_after_ingest"])
def test_unknown_status(labels_for, unknown):
    labels = labels_for({"L": ["00", unknown, "00", "00", "00"]})
    assert_label(labels, "L", "2019-01", reason=INCOMPLETE_WINDOW)
    assert_label(labels, "L", "2019-02", reason=STATUS_NOT_AVAILABLE)


def test_rows_after_termination_are_inactive(labels_for):
    labels = labels_for({"L": ["00", ("00", "01"), "00", "00", "00"]})  # trailing records after payoff
    assert_label(labels, "L", "2019-03", reason=INACTIVE_AT_T)


def test_redefault_after_cure(labels_for):
    loans = {"L": ["03", "00", "00", "03", "00", "00"]}
    counted = labels_for(loans)
    assert_label(counted, "L", "2019-02", y=1, offset=2)
    assert bool(label_at(counted, "L", "2019-02")["ever_90plus_before_t"])

    excluded = labels_for(loans, redefault_after_cure_counts_as_event=False)
    assert_label(excluded, "L", "2019-02", reason=PRIOR_90PLUS)


def test_future_beyond_window_does_not_change_label(labels_for):
    base = labels_for({"L": ["00"] * 4})
    extended = labels_for({"L": ["00"] * 4 + ["03", "RA"]})
    assert label_at(base, "L", "2019-01")["y"] == label_at(extended, "L", "2019-01")["y"] == 0


def test_loans_do_not_leak_into_each_other(labels_for):
    labels = labels_for({"A": ["00", "00", "00", "00"], "B": ["00", "03", "RA", "RA"]})
    assert_label(labels, "A", "2019-01", y=0)
    assert_label(labels, "B", "2019-01", y=1, offset=1)


def test_one_output_row_per_input_row(labels_for):
    loans = {"A": ["00"] * 5, "B": ["00", "01", ("00", "01")]}
    assert len(labels_for(loans)) == len(make_panel(loans))


def test_rejects_duplicated_loan_months(definitions):
    panel = make_panel({"L": ["00", "00"]})
    panel = pd.concat([panel, panel.iloc[[0]]])
    with pytest.raises(ValueError, match="duplicated"):
        build_ews_labels(panel, definitions)
