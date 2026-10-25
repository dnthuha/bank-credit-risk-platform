"""Stage 8.4: segment shares, rates, HHI and mix PSI on a hand-built segment table."""

import numpy as np
import pandas as pd
import pytest

from credit_risk.monitoring.concentration import (
    concentration_checks,
    concentration_markdown,
    concentration_summary,
    group_order,
    reporting_view,
)
from credit_risk.monitoring.panel import PanelSpec

MONTH = pd.Timestamp("2020-01-01")
YEAR_AGO = pd.Timestamp("2019-01-01")


def seg(dimension, grp, period, loans, upb, d90=0, d90_upb=0.0, d30=0, known=None):
    known = loans if known is None else known
    return {"dimension": dimension, "grp": grp, "period": period, "loans": loans, "upb": upb, "known_loans": known,
            "known_upb": upb, "dpd30plus_loans": d30, "dpd30plus_upb": 0.0, "dpd90plus_loans": d90, "dpd90plus_upb": d90_upb}


TABLE = pd.DataFrame([
    seg("property_state", "CA", MONTH, 60, 600.0, d90=3, d90_upb=60.0, d30=6),
    seg("property_state", "TX", MONTH, 40, 400.0, d90=1, d90_upb=10.0, d30=2),
    seg("property_state", "CA", YEAR_AGO, 50, 500.0, d90=1),
    seg("property_state", "TX", YEAR_AGO, 50, 500.0, d90=1),
])


def test_shares_rates_and_trailing_rate():
    view = reporting_view(TABLE, MONTH).set_index("grp")
    assert view.loc["CA", "share_loans"] == pytest.approx(0.6) and view.loc["TX", "share_upb"] == pytest.approx(0.4)
    assert view.loc["CA", "dpd90plus_rate_count"] == pytest.approx(3 / 60)
    assert view.loc["CA", "dpd90plus_rate_upb"] == pytest.approx(60 / 600)
    assert view.loc["CA", "dpd30plus_rate_count"] == pytest.approx(6 / 60)
    assert view.loc["CA", "dpd90plus_rate_count_trailing"] == pytest.approx(1 / 50)


def test_hhi_top_share_and_mix_psi():
    s = concentration_summary(TABLE, MONTH, epsilon=1e-4).iloc[0]
    assert s["groups"] == 2 and s["hhi_upb"] == pytest.approx(0.6 ** 2 + 0.4 ** 2)
    assert s["top_group"] == "CA" and s["top_share_upb"] == pytest.approx(0.6)
    expected = (0.6 - 0.5) * np.log(0.6 / 0.5) + (0.4 - 0.5) * np.log(0.4 / 0.5)
    assert s["mix_psi_vs_year_ago"] == pytest.approx(expected)


def test_a_new_group_counts_in_the_psi_with_epsilon():
    table = pd.concat([TABLE, pd.DataFrame([seg("property_state", "NY", MONTH, 10, 100.0)])], ignore_index=True)
    s = concentration_summary(table, MONTH, epsilon=1e-4).iloc[0]
    assert s["groups"] == 3 and s["mix_psi_vs_year_ago"] > concentration_summary(TABLE, MONTH, 1e-4).iloc[0]["mix_psi_vs_year_ago"]


def test_groups_must_add_up_to_the_book():
    monthly = pd.DataFrame({"period": [YEAR_AGO, MONTH], "end_loans": [100, 100], "dpd90plus_loans": [2, 4]})
    checks = concentration_checks(TABLE, monthly, MONTH)
    assert checks.loc[checks["check"].str.contains("group loans|group 90"), "ok"].all()
    monthly.loc[1, "end_loans"] = 101
    checks = concentration_checks(TABLE, monthly, MONTH)
    assert not checks.loc[checks["check"].str.contains("group loans"), "ok"].iloc[0]


def test_bands_are_shown_in_configured_order(project_config):
    spec = PanelSpec.from_config(project_config.ews, project_config.definitions)
    order = group_order(spec.bands)
    assert order["dti_band"] == ["<=36", "37-43", "44-50", ">50", "missing"]
    table = pd.DataFrame([seg("dti_band", g, MONTH, 10, 100.0) for g in ("44-50", "missing", "<=36")])
    text = concentration_markdown(reporting_view(table, MONTH), concentration_summary(table, MONTH, 1e-4), MONTH,
                                  pd.DataFrame({"ok": [True]}), 0.1, 0.25, order)
    text = text[text.index("### dti_band"):]
    assert text.index("| <=36 |") < text.index("| 44-50 |") < text.index("| missing |")
