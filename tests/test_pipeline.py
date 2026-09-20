"""One command runs the whole chain and leaves a traceable manifest."""

import json

import pandas as pd

from credit_risk.cli import main
from credit_risk.pipeline import STEPS, run_pipeline


def read_manifest(settings):
    manifests = sorted(settings.artifacts_dir.glob("*/run_manifest.json"))
    assert len(manifests) == 1
    return json.loads(manifests[0].read_text(encoding="utf-8")), manifests[0].parent


def test_full_pipeline_runs_end_to_end(raw_data):
    settings = raw_data
    assert run_pipeline(settings=settings) == 0

    manifest, run_dir = read_manifest(settings)
    assert manifest["status"] == "ok"
    assert [s["step"] for s in manifest["steps"]] == list(STEPS)
    statuses = {s["step"]: s["status"] for s in manifest["steps"]}
    assert statuses["ingest"] == statuses["validate-data"] == statuses["features"] == "ok"
    assert all(statuses[s] == "stub" for s in list(STEPS)[3:])
    assert manifest["run_id"].endswith(manifest["config_hash"][:8])
    assert manifest["packages"]["duckdb"] is not None

    assert (run_dir / "data_validation" / "home_credit.md").exists()
    assert (run_dir / "features" / "splits_summary.json").exists()
    assert (settings.processed_dir / "freddie_mac" / "performance.parquet").exists()
    assert (settings.processed_dir / "home_credit" / "_ingest_manifest.json").exists()

    splits = pd.read_parquet(settings.processed_dir / "home_credit" / "splits.parquet")
    population = pd.read_parquet(settings.processed_dir / "home_credit" / "application_train.parquet")
    assert set(splits["SK_ID_CURR"]) == set(population["SK_ID_CURR"])


def test_pipeline_stops_when_raw_data_is_missing(settings):
    assert run_pipeline(steps=["ingest", "validate-data", "train"], settings=settings) == 1
    manifest, _ = read_manifest(settings)
    assert manifest["status"] == "failed"
    assert [(s["step"], s["status"]) for s in manifest["steps"]] == [("ingest", "failed")]


def test_pipeline_stops_on_data_validation_error(raw_data):
    settings = raw_data
    path = settings.raw_dir / "home_credit" / "application_train.csv"
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines + [lines[1]]) + "\n", encoding="utf-8")  # duplicate a row

    assert run_pipeline(settings=settings) == 1
    manifest, _ = read_manifest(settings)
    assert [(s["step"], s["status"]) for s in manifest["steps"]] == [
        ("ingest", "ok"),
        ("validate-data", "failed"),
    ]


def test_steps_always_run_in_canonical_order(raw_data):
    settings = raw_data
    assert run_pipeline(steps=["validate-data", "ingest"], sources=["freddie_mac"], settings=settings) == 0
    manifest, _ = read_manifest(settings)
    assert [s["step"] for s in manifest["steps"]] == ["ingest", "validate-data"]


def test_cli_check_config(capsys):
    assert main(["check-config"]) == 0
    assert "definitions: OK" in capsys.readouterr().out
