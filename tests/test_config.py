"""Definitions are coherent, and any config change is visible in the config hash."""

import copy
import math
import shutil

import pytest
import yaml

from credit_risk.config import check_definitions, load_config
from conftest import CONFIG_DIR


def test_committed_definitions_are_coherent(definitions):
    assert check_definitions(definitions) == []


def test_config_hash_is_stable(project_config):
    assert load_config(CONFIG_DIR).hash == project_config.hash


def test_config_hash_changes_when_a_definition_changes(tmp_path, project_config):
    config_dir = tmp_path / "configs"
    shutil.copytree(CONFIG_DIR, config_dir)
    path = config_dir / "definitions.yaml"
    content = yaml.safe_load(path.read_text(encoding="utf-8"))
    content["windows"]["performance_months"] = 6
    path.write_text(yaml.safe_dump(content), encoding="utf-8")
    assert load_config(config_dir).hash != project_config.hash


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda d: d["delinquency_buckets"][1].update(dpd_min=31), "not contiguous"),
        (lambda d: d["delinquency_buckets"][-1].update(dpd_max=120), "open-ended"),
        (lambda d: d["windows"].update(performance_months=0), "performance_months"),
        (lambda d: d["metric_thresholds"]["psi"].update(stable_below=0.3), "stable_below"),
        (lambda d: d["covid_period"].update(treatment="ignore"), "treatment"),
        (lambda d: d["mortgage"]["zero_balance_codes"]["other_exit"].append("01"), "more than one group"),
    ],
)
def test_incoherent_definitions_are_reported(definitions, mutate, message):
    broken = copy.deepcopy(definitions)
    mutate(broken)
    assert any(message in problem for problem in check_definitions(broken))


def test_scorecard_scaling_matches_overview_example(definitions):
    # Overview: BaseScore 600, BaseOdds 50:1, PDO 20 -> Factor 28.85, Offset 487.1
    scaling = definitions["scorecard_scaling"]
    factor = scaling["pdo"] / math.log(2)
    offset = scaling["base_score"] - factor * math.log(scaling["base_odds"])
    assert factor == pytest.approx(28.85, abs=0.01)
    assert offset == pytest.approx(487.1, abs=0.05)


def test_split_fractions_sum_to_one(project_config):
    fractions = project_config.model_dev["split"]["fractions"]
    assert sum(fractions.values()) == pytest.approx(1.0)


def test_ews_time_split_leaves_room_for_the_label_window(project_config, definitions):
    horizon = definitions["windows"]["performance_months"]
    periods = [(name, span) for name, span in project_config.ews["ews"]["time_split"].items()]
    for (name, span), (next_name, next_span) in zip(periods, periods[1:]):
        end = int(span["end"][:4]) * 12 + int(span["end"][5:7])
        next_start = int(next_span["start"][:4]) * 12 + int(next_span["start"][5:7])
        assert end + horizon < next_start, f"labels of {name} overlap {next_name}"
