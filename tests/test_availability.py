"""Stage 2.2: the availability matrix classifies every column, and post-decision
information can never reach a feature."""

import pytest
import yaml

from credit_risk.data.contracts import parse_contract
from credit_risk.features.availability import (
    AvailabilityError,
    availability_frame,
    availability_markdown,
    parse_availability,
)
from conftest import CONFIG_DIR

RAW = {
    "home_credit": {
        "application_train": {
            "grain": "Một hồ sơ",
            "default": {"availability": "at_application"},
            "columns": {
                "SK_ID_CURR": {"availability": "identifier"},
                "TARGET": {"availability": "target"},
                "FLAG_MOBIL": {"availability": "at_application", "use": False, "caveat": "hằng số"},
            },
            "patterns": [{"match": "FLAG_DOCUMENT_*", "availability": "at_application", "caveat": "kiểm IV"}],
        },
        "bureau": {
            "grain": "Một khoản vay bureau",
            "default": {"availability": "historical"},
            "row_filter": "DAYS_CREDIT_UPDATE <= 0",
            "columns": {"SK_ID_CURR": {"availability": "identifier"}},
        },
    }
}


@pytest.fixture
def matrix():
    return parse_availability(RAW, "home_credit")


@pytest.fixture
def committed():
    raw = yaml.safe_load((CONFIG_DIR / "feature_availability.yaml").read_text(encoding="utf-8"))
    return parse_availability(raw, "home_credit")


@pytest.fixture
def home_credit_contract(project_config):
    return parse_contract(project_config.contracts["home_credit"])


def test_rules_resolve_in_order_column_pattern_default(matrix):
    resolved = {c.column: c for c in matrix.classify(
        "application_train", ["SK_ID_CURR", "FLAG_DOCUMENT_3", "AMT_CREDIT"])}
    assert resolved["SK_ID_CURR"].matched_by == "column"
    assert resolved["FLAG_DOCUMENT_3"].matched_by == "pattern:FLAG_DOCUMENT_*"
    assert resolved["AMT_CREDIT"].matched_by == "default"


def test_identifiers_and_targets_are_never_features(matrix):
    features = matrix.feature_columns("application_train", ["SK_ID_CURR", "TARGET", "AMT_CREDIT", "FLAG_MOBIL"])
    assert features == ["AMT_CREDIT"]


def test_a_table_without_rules_is_an_error(matrix):
    with pytest.raises(AvailabilityError, match="no availability rules"):
        matrix.classify("new_table", ["X"])


@pytest.mark.parametrize(
    "broken, message",
    [
        ({"default": {"availability": "someday"}}, "availability must be one of"),
        ({"default": {"availability": "target", "use": True}}, "never be used as a feature"),
        ({"default": {"availability": "historical", "typo": 1}}, "unknown keys"),
        ({"grain": "x"}, "default rule is required"),
        ({"default": {"availability": "historical"}, "patterns": [{"availability": "historical"}]}, "'match' glob"),
    ],
)
def test_broken_rules_are_rejected(broken, message):
    raw = {"home_credit": {"t": broken}}
    with pytest.raises(AvailabilityError, match=message):
        parse_availability(raw, "home_credit")


def test_a_table_marked_unusable_yields_no_features():
    raw = {"home_credit": {"application_test": {
        "grain": "PSI sample", "use_for_features": False,
        "default": {"availability": "at_application"},
    }}}
    matrix = parse_availability(raw, "home_credit")
    assert matrix.feature_columns("application_test", ["AMT_CREDIT"]) == []


def test_frame_and_markdown_describe_every_column(matrix):
    columns = {"application_train": ["SK_ID_CURR", "TARGET", "AMT_CREDIT"], "bureau": ["SK_ID_CURR", "DAYS_CREDIT"]}
    frame = availability_frame(matrix, columns)
    assert len(frame) == 5
    assert set(frame.columns) == {"table", "column", "availability", "use", "matched_by", "caveat"}

    text = availability_markdown(matrix, frame)
    assert "DAYS_CREDIT_UPDATE <= 0" in text  # the row filter is documented
    for column in ["SK_ID_CURR", "TARGET", "AMT_CREDIT", "DAYS_CREDIT"]:
        assert f"`{column}`" in text


# --- the committed matrix, against the real contract -----------------------

def test_committed_matrix_covers_every_contract_table(committed, home_credit_contract):
    assert set(committed.tables) == set(home_credit_contract.tables)


def test_committed_matrix_never_lets_the_target_or_keys_through(committed, home_credit_contract):
    for name, table in home_credit_contract.tables.items():
        declared = [c.name for c in table.columns]
        features = set(committed.feature_columns(name, declared))
        for column in table.columns:
            if column.role in {"key", "target"}:
                assert column.name not in features, f"{name}.{column.name} must not be a feature"


def test_application_test_is_only_a_psi_sample(committed):
    spec = committed.table("application_test")
    assert spec.role == "psi_current_sample" and spec.use_for_features is False
