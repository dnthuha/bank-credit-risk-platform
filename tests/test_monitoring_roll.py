"""Stage 8.3: transitions, roll / cure rates and migration on a hand-built panel.

From 2019-01 to 2019-02 (UPB in brackets, weights are the January UPB):
    A current (100) -> dpd30        B current (300) -> current
    C dpd30   (200) -> current      D dpd30   (100) -> dpd60
    E current (400) -> payoff       F current (100) -> status XX (unknown)
"""

import pandas as pd
import pytest

from credit_risk.data.db import connect
from credit_risk.monitoring import roll as rl
from credit_risk.monitoring.panel import PanelSpec, build_panel

PAIRS = {  # loan: ((jan status, jan upb), (feb status, feb zero balance code))
    "F19Q10000001": (("00", 100.0), ("01", None)),
    "F19Q10000002": (("00", 300.0), ("00", None)),
    "F19Q10000003": (("01", 200.0), ("00", None)),
    "F19Q10000004": (("01", 100.0), ("02", None)),
    "F19Q10000005": (("00", 400.0), ("00", "01")),
    "F19Q10000006": (("00", 100.0), (None, None)),
}
FEB = pd.Timestamp("2019-02-01")


@pytest.fixture
def built(settings, project_config):
    folder = settings.processed_dir / "freddie_mac"
    folder.mkdir(parents=True)
    perf = []
    for loan_id, ((s1, u1), (s2, z2)) in PAIRS.items():
        perf.append({"loan_id": loan_id, "period": pd.Timestamp("2019-01-01").date(), "delinquency_status": s1,
                     "zero_balance_code": None, "current_upb": u1, "zero_balance_removal_upb": None, "loan_age": 0})
        perf.append({"loan_id": loan_id, "period": FEB.date(), "delinquency_status": s2, "zero_balance_code": z2,
                     "current_upb": 0.0 if z2 else u1, "zero_balance_removal_upb": u1 if z2 else None, "loan_age": 1})
    pd.DataFrame(perf).to_parquet(folder / "performance.parquet", index=False)
    pd.DataFrame([{"loan_id": l, "orig_upb": 500.0, "fico": 700, "orig_ltv": 80, "orig_dti": 30, "property_state": "CA",
                   "channel": "R", "harp_indicator": "N", "source_file": "sample_orig_2019.txt"} for l in PAIRS]
                 ).to_parquet(folder / "origination.parquet", index=False)
    con = connect(settings)
    build_panel(con, folder, PanelSpec.from_config(project_config.ews, project_config.definitions))
    return con, folder, rl.transition_counts(con, folder)


def test_destinations_include_exits_and_unknown(built):
    *_, counts = built
    got = {(r.from_state, r.to_state): r.loans for r in counts.itertuples()}
    assert got == {("current", "current"): 1, ("current", "dpd30"): 1, ("current", "payoff"): 1, ("current", "unknown"): 1,
                   ("dpd30", "current"): 1, ("dpd30", "dpd60"): 1}


def test_roll_and_cure_rates_by_count_and_upb(built):
    *_, counts = built
    r = rl.roll_rates(counts).iloc[0]
    assert r["current_to_dpd30_count"] == pytest.approx(1 / 4)
    assert r["current_to_dpd30_upb"] == pytest.approx(100 / 900)
    assert r["dpd30_to_dpd60_count"] == pytest.approx(1 / 2) and r["dpd30_to_dpd60_upb"] == pytest.approx(100 / 300)
    assert r["dpd30_to_current_count"] == pytest.approx(1 / 2) and r["dpd30_to_current_upb"] == pytest.approx(200 / 300)
    assert pd.isna(r["dpd60_to_dpd90plus_count"])                    # nobody was dpd60 in January


def test_migration_rows_sum_to_one_and_empty_rows_stay_empty(built):
    *_, counts = built
    m = rl.matrix(counts)
    assert list(m.columns) == rl.TO_STATES
    assert m.loc["current"].sum() == pytest.approx(1) and m.loc["current", "payoff"] == pytest.approx(0.25)
    assert m.loc["dpd60"].isna().all()


def test_checks_pass_including_the_pandas_reference(built):
    con, folder, counts = built
    checks = rl.roll_checks(con, folder, counts, FEB, sample_loans=100, seed=7)
    assert checks["ok"].all(), checks[~checks["ok"]]
    assert checks["check"].str.contains("transition_matrix on 6 sampled loans").any()


def test_a_lost_transition_is_caught(built):
    con, folder, counts = built
    checks = rl.roll_checks(con, folder, counts.iloc[1:], FEB, sample_loans=100, seed=7)
    assert not checks.loc[checks["check"].str.startswith("transitions counted"), "ok"].iloc[0]


def test_markdown_formats_tiny_rates_without_rounding_them_to_zero():
    assert rl._pct(0.0) == "0.00%" and rl._pct(0.00003) == "0.0030%" and rl._pct(1e-8) == "<0.0001%"
