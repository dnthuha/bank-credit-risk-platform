"""Shared fixtures: isolated settings and tiny synthetic raw files in the real formats."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from credit_risk.config import load_config
from credit_risk.settings import REPO_ROOT, Settings

CONFIG_DIR = REPO_ROOT / "configs"


@pytest.fixture(scope="session")
def project_config():
    return load_config(CONFIG_DIR)


@pytest.fixture(scope="session")
def definitions(project_config):
    return project_config.definitions


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        repo_root=REPO_ROOT,
        config_dir=CONFIG_DIR,
        data_dir=tmp_path / "data",
        artifacts_dir=tmp_path / "artifacts",
        duckdb_memory_limit="512MB",
        duckdb_threads=2,
    )


# ---------------------------------------------------------------------------
# Home Credit: CSV with header, only the columns the contract declares (+1 extra)
# ---------------------------------------------------------------------------

def write_home_credit_raw(settings: Settings) -> Path:
    rng = np.random.default_rng(0)
    raw = settings.raw_dir / "home_credit"
    raw.mkdir(parents=True, exist_ok=True)
    n = 40
    ids = np.arange(100001, 100001 + n)
    test_ids = np.arange(200001, 200011)

    pd.DataFrame(
        {
            "SK_ID_CURR": ids,
            "TARGET": (rng.random(n) < 0.2).astype(int),
            "NAME_CONTRACT_TYPE": rng.choice(["Cash loans", "Revolving loans"], n),
            "CODE_GENDER": rng.choice(["M", "F"], n),
            "AMT_INCOME_TOTAL": rng.uniform(5e4, 3e5, n).round(1),
            "AMT_CREDIT": rng.uniform(1e5, 1e6, n).round(1),
            "AMT_ANNUITY": rng.uniform(5e3, 5e4, n).round(1),
            "DAYS_BIRTH": -rng.integers(7000, 25000, n),
            "DAYS_EMPLOYED": np.where(rng.random(n) < 0.1, 365243, -rng.integers(0, 10000, n)),
            "EXT_SOURCE_1": rng.random(n),
            "EXT_SOURCE_2": rng.random(n),
            "EXT_SOURCE_3": rng.random(n),
            "FLAG_OWN_CAR": rng.choice(["Y", "N"], n),
            "DAYS_REGISTRATION": -10.0 * np.arange(n),
            "DAYS_ID_PUBLISH": -7 * np.arange(n),
            "DAYS_LAST_PHONE_CHANGE": -3.0 * np.arange(n),
        }
    ).to_csv(raw / "application_train.csv", index=False)

    pd.DataFrame(
        {
            "SK_ID_CURR": test_ids,
            "AMT_INCOME_TOTAL": rng.uniform(5e4, 3e5, 10).round(1),
            "AMT_CREDIT": rng.uniform(1e5, 1e6, 10).round(1),
            "DAYS_BIRTH": -rng.integers(7000, 25000, 10),
            "DAYS_EMPLOYED": np.r_[365243, -100 * np.arange(1, 10)],
            "DAYS_REGISTRATION": -10.0 * np.arange(10),
            "DAYS_ID_PUBLISH": -7 * np.arange(10),
            "DAYS_LAST_PHONE_CHANGE": -3.0 * np.arange(10),
        }
    ).to_csv(raw / "application_test.csv", index=False)

    bureau_ids = np.arange(5000001, 5000061)
    pd.DataFrame(
        {
            "SK_ID_BUREAU": bureau_ids,
            "SK_ID_CURR": rng.choice(np.r_[ids, test_ids], 60),
            "CREDIT_ACTIVE": rng.choice(["Closed", "Active"], 60),
            "DAYS_CREDIT": -rng.integers(1, 3000, 60),
            "DAYS_ENDDATE_FACT": np.where(np.arange(60) % 2 == 0, np.nan, -5.0 * np.arange(60)),
            "DAYS_CREDIT_UPDATE": -2 * np.arange(60),
        }
    ).to_csv(raw / "bureau.csv", index=False)

    pd.DataFrame(
        {
            "SK_ID_BUREAU": np.repeat(bureau_ids[:10], 3),
            "MONTHS_BALANCE": np.tile([0, -1, -2], 10),
            "STATUS": rng.choice(["C", "X", "0", "1"], 30),
        }
    ).to_csv(raw / "bureau_balance.csv", index=False)

    prev_ids = np.arange(1000001, 1000031)
    prev_curr = rng.choice(ids, 30)
    pd.DataFrame(
        {
            "SK_ID_PREV": prev_ids,
            "SK_ID_CURR": prev_curr,
            "NAME_CONTRACT_STATUS": rng.choice(["Approved", "Refused"], 30),
            "DAYS_DECISION": -rng.integers(1, 2000, 30),
            "DAYS_FIRST_DRAWING": np.where(np.arange(30) % 3 == 0, 365243.0, -50.0 - np.arange(30)),
            "DAYS_FIRST_DUE": -40.0 - np.arange(30),
            "DAYS_LAST_DUE": np.where(np.arange(30) % 4 == 0, 365243.0, -10.0 - np.arange(30)),
            "DAYS_TERMINATION": np.where(np.arange(30) % 4 == 0, 365243.0, -5.0 - np.arange(30)),
        }
    ).to_csv(raw / "previous_application.csv", index=False)

    monthly = pd.DataFrame(
        {
            "SK_ID_PREV": np.repeat(prev_ids[:10], 2),
            "SK_ID_CURR": np.repeat(prev_curr[:10], 2),
            "MONTHS_BALANCE": np.tile([-1, -2], 10),
        }
    )
    monthly.to_csv(raw / "POS_CASH_balance.csv", index=False)
    monthly.to_csv(raw / "credit_card_balance.csv", index=False)

    pd.DataFrame(
        {
            "SK_ID_PREV": np.repeat(prev_ids[:10], 2),
            "SK_ID_CURR": np.repeat(prev_curr[:10], 2),
            "NUM_INSTALMENT_NUMBER": np.tile([1, 2], 10),
            "DAYS_INSTALMENT": np.tile([-60.0, -30.0], 10),
            "DAYS_ENTRY_PAYMENT": np.tile([-61.0, -29.0], 10),
            "AMT_INSTALMENT": np.full(20, 1000.0),
            "AMT_PAYMENT": np.full(20, 1000.0),
        }
    ).to_csv(raw / "installments_payments.csv", index=False)
    return raw


# ---------------------------------------------------------------------------
# Freddie Mac: pipe-delimited, no header, field order taken from the contract
# ---------------------------------------------------------------------------

def freddie_columns(table: str) -> list[str]:
    contract = yaml.safe_load((CONFIG_DIR / "data_contracts" / "freddie_mac.yaml").read_text(encoding="utf-8"))
    return [c["name"] for c in contract["tables"][table]["columns"]]


def _pipe_line(columns: list[str], values: dict[str, str]) -> str:
    unknown = set(values) - set(columns)
    assert not unknown, f"fixture uses unknown fields {unknown}"
    return "|".join(values.get(c, "") for c in columns)


def write_freddie_raw(settings: Settings, year: int = 2015) -> Path:
    raw = settings.raw_dir / "freddie_mac"
    raw.mkdir(parents=True, exist_ok=True)
    orig_cols, perf_cols = freddie_columns("origination"), freddie_columns("performance")

    loans = {
        "F15Q10000001": ["00", "00", "01", "02", "03", "RA"],
        "F15Q10000002": ["00", "00", "00", "00", "00", "00"],
        "F15Q10000003": ["00", "01", "00", "00"],  # pays off in month 4
    }
    orig_lines, perf_lines = [], []
    for i, (loan_id, statuses) in enumerate(loans.items()):
        orig_lines.append(_pipe_line(orig_cols, {
            "fico": "9999" if i == 1 else "720", "first_payment_date": f"{year}03",
            "first_time_homebuyer": "N", "maturity_date": f"{year + 30}02", "msa": "",
            "mi_pct": "0", "num_units": "1", "occupancy_status": "P", "orig_cltv": "80",
            "orig_dti": "999" if i == 2 else "36", "orig_upb": "200000.00", "orig_ltv": "80",
            "orig_interest_rate": "4.125", "channel": "R", "prepayment_penalty": "N",
            "amortization_type": "FRM", "property_state": "CA", "property_type": "SF",
            "postal_code": "945", "loan_id": loan_id, "loan_purpose": "P",
            "orig_loan_term": "360", "num_borrowers": "2", "seller_name": "OTHER",
            "super_conforming_flag": "N", "harp_indicator": "N",
            "property_valuation_method": "2", "interest_only_indicator": "N",
            "vantagescore4": "9999",
        }))
        for m, status in enumerate(statuses):
            month = pd.Period(f"{year}-03", "M") + m
            payoff = loan_id == "F15Q10000003" and m == len(statuses) - 1
            perf_lines.append(_pipe_line(perf_cols, {
                "loan_id": loan_id, "period": month.strftime("%Y%m"),
                "current_upb": "0.00" if payoff else "199000.00", "delinquency_status": status,
                "loan_age": str(m), "remaining_months_to_maturity": str(360 - m),
                "zero_balance_code": "01" if payoff else "",
                "zero_balance_effective_date": month.strftime("%Y%m") if payoff else "",
                "current_interest_rate": "4.125", "current_non_interest_bearing_upb": "0.00",
                "eltv": "999", "current_interest_bearing_upb": "199000.00",
                "mi_cancellation_indicator": "7", "servicer_name": "OTHER",
            }))
    (raw / f"sample_orig_{year}.txt").write_text("\n".join(orig_lines) + "\n", encoding="utf-8")
    (raw / f"sample_perf_{year}.txt").write_text("\n".join(perf_lines) + "\n", encoding="utf-8")
    return raw


@pytest.fixture
def raw_data(settings: Settings) -> Settings:
    write_home_credit_raw(settings)
    write_freddie_raw(settings)
    return settings


