"""Stage 8.2: vintage curves on hand-built loan outcomes, then on a small panel end to end."""

import numpy as np
import pandas as pd
import pytest

from credit_risk.data.db import connect
from credit_risk.monitoring.panel import PanelSpec, build_panel
from credit_risk.monitoring.vintage import (
    checkpoint_table,
    loan_outcomes,
    max_reported_mob,
    vintage_checks,
    vintage_curves,
)


def outcomes(rows):
    return pd.DataFrame(rows, columns=["loan_id", "vintage", "orig_upb", "left_truncated", "default_mob", "payoff_mob", "reachable_mob"]
                        ).astype({"default_mob": "Float64", "payoff_mob": "Float64"})


LOANS = outcomes([
    ("a", "2019Q1", 100.0, False, 2, None, 10),
    ("b", "2019Q1", 300.0, False, None, 1, 10),
    ("c", "2019Q1", 100.0, False, None, None, 10),
    ("d", "2019Q1", 500.0, False, None, None, 10),
    ("e", "2019Q1", 999.0, True, 0, None, 10),          # left truncated: never in the pool
])


def test_cumulative_rates_by_count_and_by_upb():
    c = vintage_curves(LOANS, "vintage", 1.0).set_index("mob")
    assert (c["pool_loans"].iloc[0], c["pool_upb"].iloc[0]) == (4, 1000.0)
    assert c.loc[[0, 1, 2, 3], "cum_default_rate_count"].tolist() == [0, 0, 0.25, 0.25]
    assert c.loc[2, "cum_default_rate_upb"] == pytest.approx(100 / 1000)
    assert c.loc[[0, 1], "cum_payoff_rate_count"].tolist() == [0, 0.25]
    assert c.loc[1, "cum_payoff_rate_upb"] == pytest.approx(300 / 1000)
    assert c.index.max() == 10


def test_the_curve_stops_where_enough_of_the_pool_could_have_reached():
    reach = pd.Series([3] + [40] * 199)                      # one loan of 200 is very young
    assert max_reported_mob(reach, 1.0) == 3
    assert max_reported_mob(reach, 0.99) == 40
    loans = LOANS.assign(reachable_mob=[10, 10, 10, 4, 10])
    assert vintage_curves(loans, "vintage", 1.0)["mob"].max() == 4
    with pytest.raises(ValueError, match="min_reach_share"):
        vintage_curves(LOANS, "vintage", 0.0)


def test_checkpoint_table_leaves_unreached_mobs_empty():
    wide = checkpoint_table(vintage_curves(LOANS, "vintage", 1.0), "vintage", checkpoints=(6, 12))
    assert wide.loc["2019Q1", 6] == 0.25 and np.isnan(wide.loc["2019Q1", 12])


@pytest.fixture
def small_panel(settings, project_config):
    """Two 2019Q1 loans from MOB 0 (one defaults at MOB 2), one left truncated, panel ends 2019-06."""
    folder = settings.processed_dir / "freddie_mac"
    folder.mkdir(parents=True)
    perf = []
    for loan_id, age0, statuses in (("F19Q10000001", 0, ["00", "01", "03", "04", "05", "06"]),
                                    ("F19Q10000002", 0, ["00"] * 6),
                                    ("F19Q10000003", 7, ["00"] * 6)):
        for k, s in enumerate(statuses):
            perf.append({"loan_id": loan_id, "period": (pd.Period("2019-01", "M") + k).to_timestamp().date(), "loan_age": age0 + k,
                         "delinquency_status": s, "zero_balance_code": None, "current_upb": 100.0, "zero_balance_removal_upb": None})
    pd.DataFrame(perf).to_parquet(folder / "performance.parquet", index=False)
    pd.DataFrame([{"loan_id": l, "orig_upb": u, "fico": 700, "orig_ltv": 80, "orig_dti": 30, "property_state": "CA",
                   "channel": "R", "harp_indicator": "N", "source_file": "sample_orig_2019.txt"}
                  for l, u in (("F19Q10000001", 100.0), ("F19Q10000002", 300.0), ("F19Q10000003", 50.0))]
                 ).to_parquet(folder / "origination.parquet", index=False)
    con = connect(settings)
    build_panel(con, folder, PanelSpec.from_config(project_config.ews, project_config.definitions))
    return con, folder


def test_end_to_end_on_a_panel(small_panel):
    con, folder = small_panel
    loans = loan_outcomes(con, folder).set_index("loan_id")
    assert loans.loc["F19Q10000001", "default_mob"] == 2 and pd.isna(loans.loc["F19Q10000002", "default_mob"])
    assert loans.loc["F19Q10000002", "reachable_mob"] == 5 and bool(loans.loc["F19Q10000003", "left_truncated"])
    curves = vintage_curves(loans.reset_index(), "vintage", 1.0).set_index("mob")
    assert curves["pool_loans"].iloc[0] == 2 and curves.index.max() == 5
    assert curves.loc[2, "cum_default_rate_count"] == 0.5 and curves.loc[2, "cum_default_rate_upb"] == pytest.approx(0.25)
    checks = vintage_checks(con, folder, loans.reset_index(), vintage_curves(loans.reset_index(), "vintage", 1.0), "vintage")
    assert checks["ok"].all(), checks[~checks["ok"]]
