"""Stage 8.0: the monitoring panel built in DuckDB from tiny processed parquet files.

Each loan below exercises one rule: buckets (RA and 90+ codes), exit types, the
first bad month (a re-default after cure is bad again but not first_bad), the
previous-month columns across a reporting gap, left truncation, vintage from
loan_id and the bands. The build's own checks must all pass on it.
"""

import copy

import pandas as pd
import pytest

from credit_risk.data.db import connect
from credit_risk.monitoring.panel import MISSING, PanelSpec, assign_band, build_panel, panel_checks


def month(offset, start="2019-01"):
    return (pd.Period(start, "M") + offset).to_timestamp()


LOANS = {
    # loan_id: (first loan_age, [status | (status, zero_balance_code) | None for a gap])
    "F19Q10000001": (0, ["00", "01", "02", "03", "RA"]),
    "F19Q20000002": (0, ["00", "00", ("00", "01")]),                 # pays off in month 3
    "F18Q40000003": (1, ["00", "03", "00", "05"]),                    # bad, cures, bad again
    "F19Q30000004": (5, ["00", None, "01", ("01", "03")]),            # left truncated, gap, liquidated
    "F19Q10000005": (0, ["00", None, ("00", "96")]),                  # unknown status, then defect exit
}


@pytest.fixture
def folder(settings, project_config):
    perf, orig = [], []
    for loan_id, (age0, months) in LOANS.items():
        year = 2000 + int(loan_id[1:3])
        orig.append({"loan_id": loan_id, "fico": {1: 600, 2: 700, 3: 790}.get(int(loan_id[-1]), None),
                     "orig_ltv": 85, "orig_dti": None if loan_id.endswith("5") else 40, "property_state": "CA",
                     "channel": "R", "harp_indicator": "N", "source_file": f"sample_orig_{year}.txt"})
        for k, entry in enumerate(months):
            if entry is None and loan_id.endswith("4"):
                continue                                              # month not reported
            status, zb = entry if isinstance(entry, tuple) else (entry, None)
            perf.append({"loan_id": loan_id, "period": month(k).date(), "loan_age": age0 + k, "current_upb": 0.0 if zb else 100000.0 - k,
                         "zero_balance_removal_upb": 99000.0 if zb else None, "delinquency_status": status, "zero_balance_code": zb})
    out = settings.processed_dir / "freddie_mac"
    out.mkdir(parents=True)
    pd.DataFrame(perf).to_parquet(out / "performance.parquet", index=False)
    pd.DataFrame(orig).to_parquet(out / "origination.parquet", index=False)
    return out


@pytest.fixture
def spec(project_config):
    return PanelSpec.from_config(project_config.ews, project_config.definitions)


@pytest.fixture
def panel(settings, folder, spec):
    path = build_panel(connect(settings), folder, spec)
    return pd.read_parquet(path).sort_values(["loan_id", "period"]).reset_index(drop=True)


def rows(panel, loan_id):
    return panel[panel["loan_id"] == loan_id].reset_index(drop=True)


def test_buckets_exits_and_bad_flags(panel):
    a = rows(panel, "F19Q10000001")
    assert a["bucket"].tolist() == ["current", "dpd30", "dpd60", "dpd90plus", "dpd90plus"]
    assert a["is_bad"].tolist() == [False, False, False, True, True]
    assert a["first_bad"].tolist() == [False, False, False, True, False]
    b = rows(panel, "F19Q20000002")
    assert b["exit_type"].tolist()[-1] == "payoff" and not b["is_bad"].any()
    c = rows(panel, "F18Q40000003")
    assert c["is_bad"].tolist() == [False, True, False, True]
    assert c["first_bad"].tolist() == [False, True, False, False]       # re-default is bad, not first
    d = rows(panel, "F19Q30000004")
    assert d["exit_type"].tolist()[-1] == "liquidation" and d["first_bad"].tolist()[-1]
    e = rows(panel, "F19Q10000005")
    assert pd.isna(e["bucket"][1]) and e["exit_type"][2] == "other_exit" and not e["is_bad"].any()


def test_previous_month_columns_break_at_a_gap(panel):
    d = rows(panel, "F19Q30000004")                                  # months 0, 2, 3 reported
    assert d["has_prev"].tolist() == [False, False, True]
    assert panel["has_prev"].notna().all()
    assert pd.isna(d["prev_bucket"][1]) and d["prev_bucket"][2] == "dpd30"
    a = rows(panel, "F19Q10000001")
    assert a["prev_bucket"].tolist()[1:] == ["current", "dpd30", "dpd60", "dpd90plus"]
    assert a["prev_upb"][1] == 100000.0


def test_vintage_left_truncation_and_bands(panel):
    first = panel.groupby("loan_id").first()
    assert first.loc["F18Q40000003", "vintage"] == "2018Q4" and first.loc["F18Q40000003", "vintage_year"] == 2018
    assert first["left_truncated"].to_dict() == {"F18Q40000003": False, "F19Q10000001": False, "F19Q10000005": False,
                                                 "F19Q20000002": False, "F19Q30000004": True}   # first loan_age 1 is kept
    assert first.loc["F19Q10000001", "fico_band"] == "<620"
    assert first.loc["F19Q20000002", "fico_band"] == "680-739"
    assert first.loc["F18Q40000003", "fico_band"] == "780+"
    assert first.loc["F19Q30000004", "fico_band"] == MISSING
    assert first.loc["F19Q10000005", "dti_band"] == MISSING and first.loc["F19Q10000001", "dti_band"] == "37-43"
    assert set(first["ltv_band"]) == {"81-90"}


def test_the_build_checks_pass_and_summarise(settings, folder, spec, panel):
    checks, summary = panel_checks(connect(settings), folder, spec)
    assert checks["ok"].all(), checks[~checks["ok"]]
    assert summary["loans"] == 5 and summary["left_truncated_loans"] == 1
    assert summary["loans_ever_bad"] == 3 and summary["unknown_status_rows"] == 1


def test_a_drifted_bucket_mapping_is_caught(settings, folder, spec, panel, monkeypatch):
    import credit_risk.monitoring.panel as mod
    codes = copy.deepcopy(spec.codes)
    codes["dpd90plus_min_numeric_code"] = 4                          # SQL now disagrees with buckets.py on "03"
    drifted = PanelSpec(spec.bands, spec.max_first_loan_age, spec.vintage_granularity, codes, spec.zero_balance)
    reference = mod.status_to_bucket
    monkeypatch.setattr(mod, "status_to_bucket", lambda s, _codes: reference(s, spec.codes))
    checks, _ = panel_checks(connect(settings), folder, drifted)
    assert not checks.loc[checks["check"].str.startswith("SQL bucket"), "ok"].iloc[0]


def test_bands_are_inclusive_at_both_ends(spec):
    assert [assign_band(v, spec.bands["fico"]) for v in (619, 620, 679, 680, 779, 780)] == \
           ["<620", "620-679", "620-679", "680-739", "740-779", "780+"]
    assert [assign_band(v, spec.bands["ltv"]) for v in (60, 61, 100, 101)] == ["<=60", "61-80", "96-100", ">100"]
    assert assign_band(None, spec.bands["dti"]) == MISSING


def test_config_errors_are_explicit(project_config):
    ews = copy.deepcopy(project_config.ews)
    ews["monitoring"]["bands"]["fico"].append({"label": "<620", "min": 900})
    with pytest.raises(ValueError, match="unique"):
        PanelSpec.from_config(ews, project_config.definitions)
    ews = copy.deepcopy(project_config.ews)
    ews["monitoring"]["vintage_granularity"] = "month"
    with pytest.raises(ValueError, match="year or quarter"):
        PanelSpec.from_config(ews, project_config.definitions)
