"""Stage 8.1: monthly portfolio metrics on a three-loan panel computed by hand.

    loan A  2019-01 00 (100)   2019-02 01 (100)        2019-03 03 (100)  -> first default in March
    loan B  2019-01 00 (200)   2019-02 payoff (prev 200)
    loan C                     2019-02 00 (300)        2019-03 00 (300)  -> enters in February
"""

import copy

import pandas as pd
import pytest

from credit_risk.data.db import connect
from credit_risk.monitoring import portfolio as pf
from credit_risk.monitoring.panel import PanelSpec, build_panel

ROWS = [  # loan_id, period, status, zero balance code, current_upb
    ("F19Q10000001", "2019-01", "00", None, 100.0), ("F19Q10000001", "2019-02", "01", None, 100.0),
    ("F19Q10000001", "2019-03", "03", None, 100.0),
    ("F19Q10000002", "2019-01", "00", None, 200.0), ("F19Q10000002", "2019-02", "00", "01", 0.0),
    ("F19Q10000003", "2019-02", "00", None, 300.0), ("F19Q10000003", "2019-03", "00", None, 300.0),
]


@pytest.fixture
def built(settings, project_config):
    folder = settings.processed_dir / "freddie_mac"
    folder.mkdir(parents=True)
    perf = pd.DataFrame([{"loan_id": l, "period": pd.Timestamp(f"{p}-01").date(), "delinquency_status": s, "zero_balance_code": z,
                          "current_upb": u, "zero_balance_removal_upb": 200.0 if z else None, "loan_age": 0}
                         for l, p, s, z, u in ROWS])
    perf.to_parquet(folder / "performance.parquet", index=False)
    pd.DataFrame([{"loan_id": l, "fico": 700, "orig_ltv": 80, "orig_dti": 30, "property_state": "CA", "channel": "R",
                   "harp_indicator": "N", "source_file": "sample_orig_2019.txt"} for l in sorted({r[0] for r in ROWS})]
                 ).to_parquet(folder / "origination.parquet", index=False)
    con = connect(settings)
    build_panel(con, folder, PanelSpec.from_config(project_config.ews, project_config.definitions))
    spec = pf.PortfolioSpec.from_config(project_config.ews, project_config.definitions)
    return con, folder, spec, pf.portfolio_monthly(con, folder, spec)


def at(monthly, month):
    return monthly[monthly["period"] == pd.Timestamp(f"{month}-01")].iloc[0]


def test_first_month_has_no_start_of_month(built):
    *_, monthly = built
    jan = at(monthly, "2019-01")
    assert (jan["start_loans"], jan["end_loans"], jan["entries"]) == (0, 2, 2)
    assert jan["dpd30plus_rate_count"] == 0 and pd.isna(jan["payoff_rate_count"])     # no start, no flow rate


def test_february_delinquency_and_payoff_by_count_and_upb(built):
    *_, monthly = built
    feb = at(monthly, "2019-02")
    assert (feb["start_loans"], feb["start_upb"], feb["end_loans"], feb["end_upb"]) == (2, 300.0, 2, 400.0)
    assert feb["dpd30_rate_count"] == pytest.approx(1 / 2) and feb["dpd30_rate_upb"] == pytest.approx(100 / 400)
    assert feb["payoff_rate_count"] == pytest.approx(1 / 2) and feb["payoff_rate_upb"] == pytest.approx(200 / 300)
    assert feb["cpr_count"] == pytest.approx(1 - 0.5 ** 12)
    assert (feb["entries"], feb["entries_active"], feb["exits_from_start"]) == (1, 1, 1)


def test_march_first_default(built):
    *_, monthly = built
    mar = at(monthly, "2019-03")
    assert (mar["start_loans"], mar["start_upb"]) == (2, 400.0)
    assert mar["new_default_rate_count"] == pytest.approx(1 / 2) and mar["new_default_rate_upb"] == pytest.approx(100 / 400)
    assert mar["dpd90plus_rate_count"] == pytest.approx(1 / 2) and mar["dpd90plus_rate_upb"] == pytest.approx(100 / 400)
    assert mar["dpd30plus_rate_count"] == pytest.approx(1 / 2)


def test_checks_pass_and_catch_a_missing_exit(built):
    con, folder, spec, monthly = built
    month = pf.reporting_month(monthly, spec)
    assert month == pd.Timestamp("2019-03-01")
    checks = pf.portfolio_checks(con, folder, monthly, spec, month)
    assert checks["ok"].all(), checks[~checks["ok"]]
    broken = monthly.copy()
    broken.loc[broken["period"] == pd.Timestamp("2019-02-01"), "payoff_loans"] = 0
    checks = pf.portfolio_checks(con, folder, broken, spec, month)
    assert not checks.loc[checks["check"].str.startswith("payoff exits"), "ok"].iloc[0]


def test_regimes_follow_the_covid_period(project_config):
    spec = pf.PortfolioSpec.from_config(project_config.ews, project_config.definitions)
    months = pd.Series(pd.to_datetime(["2020-02-01", "2020-03-01", "2021-12-01", "2022-01-01"]))
    assert spec.regime(months).tolist() == ["pre_covid", "covid", "covid", "post_covid"]


def test_a_reporting_month_outside_the_panel_is_an_error(built, project_config):
    *_, monthly = built
    ews = copy.deepcopy(project_config.ews)
    ews["monitoring"]["reporting_month"] = "2030-01"
    spec = pf.PortfolioSpec.from_config(ews, project_config.definitions)
    with pytest.raises(ValueError, match="not in the panel"):
        pf.reporting_month(monthly, spec)
    ews["monitoring"]["reporting_month"] = "2019-02"
    assert pf.reporting_month(monthly, pf.PortfolioSpec.from_config(ews, project_config.definitions)) == pd.Timestamp("2019-02-01")
