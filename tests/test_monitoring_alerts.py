"""Stage 8.5: each alert rule flips at its threshold, in the right direction, from the config."""

import copy

import numpy as np
import pandas as pd
import pytest

from credit_risk.monitoring import alerts as al

MONTH = pd.Timestamp("2020-01-01")


@pytest.fixture
def spec(project_config):
    return al.AlertSpec.from_config(project_config.ews, project_config.definitions)


def monthly_with(value, baseline=0.01, months=12, name="dpd90plus_rate_count"):
    periods = pd.date_range(end=MONTH, periods=months + 1, freq="MS")
    return pd.DataFrame({"period": periods, name: [baseline] * months + [value]})


def one(rows, item):
    return next(r for r in rows if r["item"] == item)


def test_light_thresholds_and_direction(spec):
    assert spec.light(0.0119, 0.01)[0] == "green"
    assert spec.light(0.0121, 0.01)[0] == "amber"
    assert spec.light(0.0151, 0.01)[0] == "red"
    assert spec.light(0.0079, 0.01, worse="lower")[0] == "amber"      # a 21% fall in a cure rate
    assert spec.light(0.0121, 0.01, worse="lower")[0] == "green"      # a rise in a cure rate is good news
    light, change = spec.light(0.02, 0.0)
    assert light == "green" and np.isnan(change)                       # no baseline, no alert


def test_metric_alert_uses_the_trailing_mean_only(spec):
    spec = al.AlertSpec(**{**spec.__dict__, "metrics": [{"name": "dpd90plus_rate_count", "worse": "higher"}]})
    monthly = monthly_with(0.0125)
    monthly.loc[0, "dpd90plus_rate_count"] = 1.0                        # 13 months back: outside the window
    rows = al.metric_alerts(monthly.iloc[1:].reset_index(drop=True), pd.DataFrame({"period": monthly["period"]}), MONTH, spec)
    r = one(rows, "dpd90plus_rate_count")
    assert r["baseline"] == pytest.approx(0.01) and r["light"] == "amber" and r["relative_change"] == pytest.approx(0.25)


def test_an_unknown_metric_is_an_error(spec):
    bad = al.AlertSpec(**{**spec.__dict__, "metrics": [{"name": "nope", "worse": "higher"}]})
    with pytest.raises(ValueError, match="unknown metric"):
        al.metric_alerts(monthly_with(0.01), pd.DataFrame({"period": monthly_with(0.01)["period"]}), MONTH, bad)


def test_segment_alerts_skip_small_groups(spec):
    view = pd.DataFrame([
        {"dimension": "fico_band", "grp": "780+", "loans": 50000, "dpd90plus_rate_count": 0.0035, "dpd90plus_rate_count_trailing": 0.002},
        {"dimension": "fico_band", "grp": "<620", "loans": 100, "dpd90plus_rate_count": 0.05, "dpd90plus_rate_count_trailing": 0.01},
        {"dimension": "vintage", "grp": "2024", "loans": 50000, "dpd90plus_rate_count": 0.05, "dpd90plus_rate_count_trailing": 0.01},
    ])
    rows = al.segment_alerts(view, spec)
    assert [r["item"] for r in rows] == ["fico_band = 780+"]              # small group and non-configured dimension left out
    assert rows[0]["light"] == "red"


def test_mix_psi_lights_and_exclusions(spec):
    summary = pd.DataFrame([
        {"dimension": "fico_band", "mix_psi_vs_year_ago": 0.05, "top_group": "780+", "top_share_upb": 0.3},
        {"dimension": "ltv_band", "mix_psi_vs_year_ago": 0.15, "top_group": "61-80", "top_share_upb": 0.4},
        {"dimension": "channel", "mix_psi_vs_year_ago": 0.30, "top_group": "R", "top_share_upb": 0.5},
        {"dimension": "vintage", "mix_psi_vs_year_ago": 0.90, "top_group": "2025", "top_share_upb": 0.2},
    ])
    lights = {r["item"]: r["light"] for r in al.mix_alerts(summary, spec)}
    assert lights == {"fico_band": "green", "ltv_band": "amber", "channel": "red", "vintage": "excluded"}


def test_new_vintage_compares_the_latest_vintage_with_the_benchmark(spec):
    rows = []
    for v, rate in ((2012, 0.002), (2013, 0.002), (2014, 0.002), (2015, 0.002), (2016, 0.002), (2023, 0.0023), (2024, 0.004)):
        rows.append({"vintage_year": v, "mob": 12, "cum_default_rate_count": rate})
    rows.append({"vintage_year": 2023, "mob": 24, "cum_default_rate_count": 0.0024})
    rows += [{"vintage_year": v, "mob": 24, "cum_default_rate_count": 0.002} for v in range(2012, 2017)]
    out = {r["item"]: r for r in al.vintage_alerts(pd.DataFrame(rows), spec)}
    assert out["default cộng dồn MOB 12, lứa 2024"]["light"] == "red"
    assert out["default cộng dồn MOB 24, lứa 2023"]["light"] == "green"   # +20% exactly is not above amber


def test_overall_status_ignores_excluded(spec):
    alerts = pd.DataFrame({"light": ["green", "excluded", "amber"]})
    assert al.overall_status(alerts) == "amber"
    assert al.overall_status(pd.DataFrame({"light": ["excluded"]})) == "green"


def test_config_errors_are_explicit(project_config):
    ews = copy.deepcopy(project_config.ews)
    ews["monitoring"]["alerts"]["amber_relative_increase"] = 0.6
    with pytest.raises(ValueError, match="amber_relative_increase"):
        al.AlertSpec.from_config(ews, project_config.definitions)
    ews = copy.deepcopy(project_config.ews)
    ews["monitoring"]["alerts"]["metrics"][0]["worse"] = "up"
    with pytest.raises(ValueError, match="higher or lower"):
        al.AlertSpec.from_config(ews, project_config.definitions)
