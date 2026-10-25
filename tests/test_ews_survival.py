"""Stage 9.4: Kaplan-Meier and Aalen-Johansen with delayed entry, by hand."""

import copy

import numpy as np
import pandas as pd
import pytest

from credit_risk.ews import survival as sv

D, P, C = sv.DEFAULT, sv.PAYOFF, sv.CENSORED


def test_without_censoring_or_truncation_cif_is_the_observed_share():
    rng = np.random.default_rng(0)
    n = 2000
    time = rng.integers(0, 40, n)
    cause = rng.choice([D, P], n, p=[0.1, 0.9])
    c = sv.curves(np.zeros(n, int), time, cause).set_index("mob")
    for m in (5, 20, 39):
        assert c.loc[m, "cif_default"] == pytest.approx(((time <= m) & (cause == D)).mean())
        assert c.loc[m, "cif_payoff"] == pytest.approx(((time <= m) & (cause == P)).mean())


def test_hand_example_with_delayed_entry_and_censoring():
    # loan: entry, time, cause
    entry = [0, 0, 0, 2, 0]
    time = [1, 2, 3, 3, 2]
    cause = [D, P, C, D, C]
    c = sv.curves(entry, time, cause).set_index("mob")
    assert c["at_risk"].tolist() == [4, 4, 4, 2]                      # the late loan joins at MOB 2
    # MOB 1: 1 default of 4; MOB 2: 1 payoff of 4 (+1 censored); MOB 3: 1 default of 2
    assert c.loc[1, "hazard_default"] == 0.25 and c.loc[2, "hazard_payoff"] == 0.25 and c.loc[3, "hazard_default"] == 0.5
    assert c.loc[3, "km_default"] == pytest.approx(1 - 0.75 * 0.5)
    s1, s2 = 0.75, 0.75 * 0.75
    assert c.loc[3, "cif_default"] == pytest.approx(0.25 + s2 * 0.5)
    assert c.loc[3, "cif_payoff"] == pytest.approx(s1 * 0.25)
    assert (c["survival"] + c["cif_default"] + c["cif_payoff"]).round(12).eq(1).all()
    assert (c["km_default"] >= c["cif_default"]).all()


def test_curve_stops_when_the_risk_set_is_small():
    n = 100
    time = np.r_[np.arange(n)]
    c = sv.curves(np.zeros(n, int), time, np.full(n, C), min_at_risk=30)
    assert c["mob"].max() == 70 and c["at_risk"].min() >= 30


def test_bad_input_is_refused():
    with pytest.raises(ValueError, match="entry <= time"):
        sv.curves([3], [2], [D])


def loans_frame():
    return pd.DataFrame({
        "loan_id": list("abcdef"), "entry": [0, 0, 6, 6, 0, 1], "time": [10, 20, 6, 12, 30, 15],
        "cause": [D, P, D, C, C, P], "left_truncated": [False, False, True, True, False, False],
        "bad_at_entry": [False, False, True, False, False, False], "vintage_year": [2015, 2015, 2016, 2016, 2016, 2015],
    })


def test_left_truncated_loans_bad_at_entry_are_left_out():
    loans = loans_frame()
    assert sv.usable(loans).tolist() == [True, True, False, True, True, True]
    spec = sv.SurvivalSpec(groups=["vintage_year"], checkpoints=[6, 12], min_at_risk=1)
    tables = sv.survival_tables(loans, spec)
    assert set(tables["dimension"]) == {"all", "vintage_year"}
    assert tables.loc[tables["dimension"] == "all", "defaults"].sum() == 1
    checks, info = sv.survival_checks(loans, tables, None, spec.checkpoints)
    assert checks["ok"].all(), checks[~checks["ok"]]
    assert info["summary"]["left_truncated_bad_at_entry"] == 1 and info["summary"]["usable_loans"] == 5


def test_static_pool_comparison_flags_a_large_gap():
    loans = loans_frame()
    spec = sv.SurvivalSpec(groups=[], checkpoints=[12], min_at_risk=1)
    tables = sv.survival_tables(loans, spec)
    pool = loans[sv.usable(loans) & ~loans["left_truncated"]]
    good = []
    for year, part in pool.groupby("vintage_year"):
        c = sv.curves(part["entry"], part["time"], part["cause"]).set_index("mob")
        good.append({"vintage_year": year, "mob": 12, "cum_default_rate_count": c.loc[12, "cif_default"] if 12 in c.index else 0.0})
    checks, _ = sv.survival_checks(loans, tables, pd.DataFrame(good), [12])
    assert checks["ok"].all()
    bad = pd.DataFrame(good).assign(cum_default_rate_count=lambda d: d["cum_default_rate_count"] + 0.5)
    checks, _ = sv.survival_checks(loans, tables, bad, [12])
    assert not checks.loc[checks["check"].str.startswith("static pool"), "ok"].iloc[0]


def test_config_is_validated(project_config):
    ews = copy.deepcopy(project_config.ews)
    ews["ews"]["survival"]["groups"] = ["zip_code"]
    with pytest.raises(ValueError, match="unknown"):
        sv.SurvivalSpec.from_config(ews)
    spec = sv.SurvivalSpec.from_config(project_config.ews)
    assert spec.checkpoints == sorted(spec.checkpoints)
