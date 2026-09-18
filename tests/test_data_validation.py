"""Contracts, ingest and validate-data on synthetic raw files, including an invalid fixture."""

import duckdb
import pandas as pd
import pytest

from credit_risk.config import ConfigError
from credit_risk.data.contracts import parse_contract
from credit_risk.data.db import connect
from credit_risk.data.ingest import RawDataError, ingest_source, processed_path
from credit_risk.data.validate import validate_source
from conftest import write_freddie_raw, write_home_credit_raw


@pytest.fixture(scope="module")
def contracts(project_config):
    return {name: parse_contract(raw) for name, raw in project_config.contracts.items()}


def failed_checks(report):
    return {(r.table, r.check, r.column) for r in report.results if not r.passed}


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------

def test_both_contracts_parse(contracts):
    assert set(contracts) == {"home_credit", "freddie_mac"}
    assert len(contracts["home_credit"].tables) == 8


def test_freddie_layout_matches_july_2026_file_layout(contracts):
    # Field positions from Freddie Mac file_layout_july_2026.xlsx (1-based).
    orig = [c.name for c in contracts["freddie_mac"].tables["origination"].columns]
    perf = [c.name for c in contracts["freddie_mac"].tables["performance"].columns]
    assert len(orig) == 31 and len(perf) == 35
    assert orig[0] == "fico" and orig[19] == "loan_id" and orig[30] == "vantagescore4"
    assert perf[:4] == ["loan_id", "period", "current_upb", "delinquency_status"]
    assert perf[8] == "zero_balance_code" and perf[28] == "delinquency_due_to_disaster"
    assert perf[33] == "servicer_name" and perf[34] == "bankruptcy_cramdown_costs"


def test_contract_typo_is_rejected(project_config):
    broken ={**project_config.contracts["home_credit"]}
    broken["tables"] = {
        "application_train": {
            **broken["tables"]["application_train"],
            "columns": {"SK_ID_CURR": {"dtype": "integer", "nulable": False}},
        }
    }
    with pytest.raises(ConfigError, match="nulable"):
        parse_contract(broken)


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------

def test_missing_raw_files_fail_with_instructions(settings, contracts):
    with pytest.raises(RawDataError, match="data/README.md"):
        ingest_source(contracts["home_credit"], settings, connect(settings), "test")


def test_freddie_ingest_types_and_na_codes(settings, contracts):
    write_freddie_raw(settings)
    contract = contracts["freddie_mac"]
    con = connect(settings)
    manifest = ingest_source(contract, settings, con, "test")
    assert manifest["tables"]["origination"]["rows"] == 3
    assert manifest["tables"]["performance"]["rows"] == 16

    orig = pd.read_parquet(processed_path(contract, contract.tables["origination"], settings))
    assert pd.isna(orig.loc[orig["loan_id"] == "F15Q10000002", "fico"]).all()  # 9999 -> NULL
    assert pd.isna(orig.loc[orig["loan_id"] == "F15Q10000003", "orig_dti"]).all()  # 999 -> NULL
    assert orig["first_payment_date"].iloc[0] == pd.Timestamp("2015-03-01").date()

    perf = pd.read_parquet(processed_path(contract, contract.tables["performance"], settings))
    assert perf["delinquency_status"].tolist().count("RA") == 1  # text code kept
    assert perf["zero_balance_code"].dropna().tolist() == ["01"]  # leading zero kept
    assert perf["source_file"].unique().tolist() == ["sample_perf_2015.txt"]


def test_freddie_field_count_mismatch_is_caught(settings, contracts):
    raw = write_freddie_raw(settings)
    path = raw / "sample_orig_2015.txt"
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(line + "|EXTRA" for line in lines) + "\n", encoding="utf-8")
    with pytest.raises(RawDataError, match="32 fields"):
        ingest_source(contracts["freddie_mac"], settings, connect(settings), "test")


def test_value_that_breaks_the_contract_type_fails_loudly(settings, contracts):
    raw = write_freddie_raw(settings)
    path = raw / "sample_perf_2015.txt"
    text = path.read_text(encoding="utf-8").replace("|4.125|", "|four|", 1)
    path.write_text(text, encoding="utf-8")
    with pytest.raises(duckdb.Error):
        ingest_source(contracts["freddie_mac"], settings, connect(settings), "test")


# ---------------------------------------------------------------------------
# validate-data
# ---------------------------------------------------------------------------

def test_valid_fixtures_pass(settings, contracts):
    write_home_credit_raw(settings)
    write_freddie_raw(settings)
    con = connect(settings)
    for name in ("home_credit", "freddie_mac"):
        ingest_source(contracts[name], settings, con, "test")
        report = validate_source(contracts[name], settings, con)
        assert report.passed, report.to_markdown()
        assert report.warnings == [], report.to_markdown()


def test_invalid_fixture_fails_on_the_broken_rules(settings, contracts):
    raw = write_home_credit_raw(settings)
    train = pd.read_csv(raw / "application_train.csv")
    train.loc[1, "SK_ID_CURR"] = train.loc[0, "SK_ID_CURR"]  # duplicate key -> error
    train.loc[2, "TARGET"] = 2                                # outside {0, 1} -> error
    train.loc[3, "CODE_GENDER"] = "Unknown"                    # unexpected category -> warn
    train.loc[4, "DAYS_BIRTH"] = 500                           # positive relative date -> warn
    train.to_csv(raw / "application_train.csv", index=False)

    contract = contracts["home_credit"]
    con = connect(settings)
    ingest_source(contract, settings, con, "test")
    report = validate_source(contract, settings, con)

    assert not report.passed
    assert {(r.table, r.check, r.column) for r in report.errors} == {
        ("application_train", "unique_key", "SK_ID_CURR"),
        ("application_train", "allowed_values", "TARGET"),
    }
    assert {(r.table, r.check, r.column) for r in report.warnings} == {
        ("application_train", "allowed_values", "CODE_GENDER"),
        ("application_train", "range", "DAYS_BIRTH"),
    }
    duplicate = next(r for r in report.errors if r.check == "unique_key")
    assert duplicate.n_failed == 1


def test_sentinel_values_are_not_range_violations(settings, contracts):
    write_home_credit_raw(settings)  # fixture contains DAYS_EMPLOYED = 365243
    contract = contracts["home_credit"]
    con = connect(settings)
    ingest_source(contract, settings, con, "test")
    report = validate_source(contract, settings, con)
    assert ("application_train", "range", "DAYS_EMPLOYED") not in failed_checks(report)


def test_missing_column_and_missing_table(settings, contracts):
    raw = write_home_credit_raw(settings)
    pd.read_csv(raw / "bureau.csv").drop(columns=["CREDIT_ACTIVE"]).to_csv(raw / "bureau.csv", index=False)
    contract = contracts["home_credit"]
    con = connect(settings)
    ingest_source(contract, settings, con, "test")
    processed_path(contract, contract.tables["credit_card_balance"], settings).unlink()

    failed = failed_checks(validate_source(contract, settings, con))
    assert ("bureau", "required_column", "CREDIT_ACTIVE") in failed
    assert ("credit_card_balance", "table_exists", None) in failed


def test_orphan_foreign_keys_are_reported(settings, contracts):
    raw = write_home_credit_raw(settings)
    bureau = pd.read_csv(raw / "bureau.csv")
    bureau.loc[0, "SK_ID_CURR"] = 999999
    bureau.to_csv(raw / "bureau.csv", index=False)
    contract = contracts["home_credit"]
    con = connect(settings)
    ingest_source(contract, settings, con, "test")
    report = validate_source(contract, settings, con)
    orphan = next(r for r in report.results if r.check == "foreign_key" and r.table == "bureau")
    assert orphan.n_failed == 1 and orphan.severity == "warn"
    assert report.passed  # warn does not block


def test_orphans_are_still_found_when_referenced_keys_contain_null(settings, contracts):
    raw = write_home_credit_raw(settings)
    test_apps = pd.read_csv(raw / "application_test.csv")
    test_apps["SK_ID_CURR"] = test_apps["SK_ID_CURR"].astype("Int64")
    test_apps.loc[0, "SK_ID_CURR"] = pd.NA
    test_apps.to_csv(raw / "application_test.csv", index=False)
    bureau = pd.read_csv(raw / "bureau.csv")
    bureau.loc[0, "SK_ID_CURR"] = 999999
    bureau.to_csv(raw / "bureau.csv", index=False)

    contract = contracts["home_credit"]
    con = connect(settings)
    ingest_source(contract, settings, con, "test")
    report = validate_source(contract, settings, con)
    orphan = next(r for r in report.results if r.check == "foreign_key" and r.table == "bureau")
    assert orphan.n_failed >= 1
