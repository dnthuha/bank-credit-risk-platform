"""Stage 2.3: the modelling table is one row per applicant, built only from
information that existed at application time."""

import pandas as pd
import pytest

from credit_risk.data.contracts import parse_contract
from credit_risk.data.db import connect
from credit_risk.data.ingest import ingest_source, processed_path
from credit_risk.features.availability import parse_availability
from credit_risk.features.build import SENTINEL, build_features
from conftest import write_home_credit_raw


@pytest.fixture
def built(settings, project_config):
    """Ingest the fixture, then build features; returns (frame, summary, raw dir)."""
    raw = write_home_credit_raw(settings)
    contract = parse_contract(project_config.contracts["home_credit"])
    matrix = parse_availability(project_config.feature_availability, "home_credit")
    con = connect(settings)
    ingest_source(contract, settings, con, "test")
    summary = build_features(contract, matrix, settings, con)
    frame = pd.read_parquet(settings.processed_dir / "home_credit" / "features.parquet")
    return frame, summary, raw


def rebuild(settings, project_config):
    contract = parse_contract(project_config.contracts["home_credit"])
    matrix = parse_availability(project_config.feature_availability, "home_credit")
    con = connect(settings)
    ingest_source(contract, settings, con, "test")
    build_features(contract, matrix, settings, con)
    return pd.read_parquet(settings.processed_dir / "home_credit" / "features.parquet")


def test_one_row_per_applicant_from_both_populations(built, settings, project_config):
    frame, summary, _ = built
    contract = parse_contract(project_config.contracts["home_credit"])
    train = pd.read_parquet(processed_path(contract, contract.tables["application_train"], settings))
    test = pd.read_parquet(processed_path(contract, contract.tables["application_test"], settings))

    assert len(frame) == len(train) + len(test) == summary["rows"]
    assert not frame["SK_ID_CURR"].duplicated().any()
    assert set(frame["population"]) == {"train", "test"}
    assert frame.loc[frame["population"] == "test", "TARGET"].isna().all()


def test_keys_and_unusable_columns_never_become_features(built):
    frame, summary, _ = built
    assert "SK_ID_BUREAU" not in frame.columns and "SK_ID_PREV" not in frame.columns
    assert "FLAG_MOBIL" not in frame.columns  # use: false in the availability matrix
    assert "SK_ID_CURR" not in summary["feature_columns"]
    assert "TARGET" not in summary["feature_columns"]


def test_no_history_means_null_not_zero(built):
    frame, _, _ = built
    without = frame[frame["HAS_BUR_HISTORY"] == 0]
    assert len(without) > 0
    assert without["BUR_COUNT"].isna().all()
    assert without["BUR_DEBT_TO_CREDIT"].isna().all()
    with_history = frame[frame["HAS_BUR_HISTORY"] == 1]
    assert (with_history["BUR_COUNT"] > 0).all()


def test_the_bureau_row_filter_keeps_post_application_updates_out(settings, project_config):
    raw = write_home_credit_raw(settings)
    before = rebuild(settings, project_config).set_index("SK_ID_CURR")["BUR_COUNT"]

    bureau = pd.read_csv(raw / "bureau.csv")
    # An applicant with several bureau loans, so losing one still leaves a history.
    victim = bureau["SK_ID_CURR"].value_counts().idxmax()
    row = bureau.index[bureau["SK_ID_CURR"] == victim][0]
    bureau.loc[row, "DAYS_CREDIT_UPDATE"] = 10  # bureau updated after the application date
    bureau.to_csv(raw / "bureau.csv", index=False)
    after = rebuild(settings, project_config).set_index("SK_ID_CURR")["BUR_COUNT"]

    assert before[victim] >= 2
    assert after[victim] == before[victim] - 1


def test_the_days_employed_sentinel_becomes_a_flag(built):
    frame, _, _ = built
    assert (frame["DAYS_EMPLOYED"] == SENTINEL).sum() == 0
    flagged = frame[frame["APP_NOT_EMPLOYED_FLAG"] == 1]
    assert len(flagged) > 0
    assert flagged["DAYS_EMPLOYED"].isna().all()
    assert flagged["APP_EMPLOYED_TO_AGE"].isna().all()


def test_previous_application_sentinels_are_excluded_from_statistics(built, settings, project_config):
    frame, _, _ = built
    contract = parse_contract(project_config.contracts["home_credit"])
    prev = pd.read_parquet(processed_path(contract, contract.tables["previous_application"], settings))

    assert (frame["PREV_DAYS_TERMINATION_MAX"] == SENTINEL).sum() == 0
    expected = prev.groupby("SK_ID_CURR")["DAYS_TERMINATION"].apply(lambda s: (s == SENTINEL).mean())
    got = frame.set_index("SK_ID_CURR")["PREV_TERMINATION_UNKNOWN_RATE"].dropna()
    pd.testing.assert_series_equal(
        got.sort_index(), expected.sort_index(), check_names=False, check_dtype=False
    )


def test_credit_card_utilisation_is_undefined_when_the_limit_is_zero(built):
    frame, _, _ = built
    cards = frame[frame["HAS_CC_HISTORY"] == 1]
    assert len(cards) > 0
    assert cards["CC_UTILISATION_MAX"].notna().any()   # limits of 10,000 in the fixture
    assert (cards["CC_UTILISATION_MAX"] != float("inf")).all()  # limits of 0 are skipped, not infinite


def test_installment_behaviour_matches_an_independent_recomputation(built, settings, project_config):
    frame, _, _ = built
    contract = parse_contract(project_config.contracts["home_credit"])
    ins = pd.read_parquet(processed_path(contract, contract.tables["installments_payments"], settings))
    ins["late"] = (ins["DAYS_ENTRY_PAYMENT"] - ins["DAYS_INSTALMENT"] > 0).astype(int)
    ins["short"] = (ins["AMT_INSTALMENT"] - ins["AMT_PAYMENT"] > 0.01).astype(int)
    expected = ins.groupby("SK_ID_CURR").agg(late_rate=("late", "mean"), underpay=("short", "mean"))

    got = frame.set_index("SK_ID_CURR")[["INS_LATE_RATE", "INS_UNDERPAY_RATE"]].dropna()
    assert len(got) == len(expected)
    pd.testing.assert_series_equal(
        got["INS_LATE_RATE"].sort_index(), expected["late_rate"].sort_index(),
        check_names=False, check_dtype=False,
    )
    pd.testing.assert_series_equal(
        got["INS_UNDERPAY_RATE"].sort_index(), expected["underpay"].sort_index(),
        check_names=False, check_dtype=False,
    )


def test_summary_reports_coverage_of_every_history_group(built):
    _, summary, _ = built
    assert set(summary["history_coverage"]) == {"BUR", "BB", "PREV", "POS", "CC", "INS"}
    assert all(0.0 <= share <= 1.0 for share in summary["history_coverage"].values())
    assert summary["columns"] == len(summary["feature_columns"]) + 3  # id, population, target
