"""Stage 6 Definition of Done, end to end through the pipeline: one command on the
frozen package writes a machine-readable metrics file and a full validation report,
reproducibly; a package that was altered or a population that does not reconcile
stops the step; and the report ties itself to the package, the seed and the config."""

import json
import shutil

import pandas as pd
import pytest

from credit_risk.pipeline import run_pipeline
from credit_risk.validation.engine import MODULES


def validate(settings):
    assert run_pipeline(steps=["validate-model"], sources=["home_credit"], settings=settings) == 0
    run_dir = max(p for p in settings.artifacts_dir.iterdir() if (p / "run_manifest.json").exists())
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    out = run_dir / "validate-model"
    return manifest, json.loads((out / "metrics.json").read_text(encoding="utf-8")), (out / "validation_report.md").read_text(encoding="utf-8")


def comparable(metrics):
    """Everything but the run id, which changes by construction."""
    payload = json.loads(json.dumps(metrics))
    payload["provenance"].pop("run_id")
    return payload


def test_one_command_gives_metrics_and_a_report_that_cover_every_module(stage6_build):
    settings, config = stage6_build
    manifest, metrics, report = validate(settings)
    step = next(s for s in manifest["steps"] if s["step"] == "validate-model")
    assert step["status"] == "ok"
    assert sorted(step["outputs"]) == ["validate-model/metrics.json", "validate-model/validation_report.md"]
    assert set(metrics["sections"]) == set(MODULES)
    assert {c["module"] for c in metrics["checks"]} == set(MODULES)
    assert all(set(c) >= {"module", "check", "claimed", "recounted", "source", "ok"} for c in metrics["checks"])
    assert "# Báo cáo thẩm định độc lập" in report
    for heading in ("Đối chiếu dân số", "Sức phân biệt", "Hiệu chỉnh", "Độ ổn định", "Độ bất định", "Theo phân khúc", "Đối chiếu triển khai"):
        assert heading in report, heading


def test_the_report_ties_itself_to_the_package_the_seed_and_the_config(stage6_build):
    settings, config = stage6_build
    package = json.loads((settings.processed_dir / "home_credit" / "model_package" / "package.json").read_text(encoding="utf-8"))
    manifest, metrics, report = validate(settings)
    p = metrics["provenance"]
    assert p["package_fingerprint"] == package["fingerprint"]
    assert p["champion_fingerprint"] == package["champion_fingerprint"]
    assert p["challenger_fingerprint"] == package["challenger_fingerprint"]
    assert p["split_fingerprint"] == package["split_fingerprint"]
    assert p["config_hash"] == manifest["config_hash"] == config.hash
    assert p["seed"] == config.validation["seed"]
    assert p["bootstrap_resamples"] == config.validation["uncertainty"]["bootstrap_resamples"]
    assert p["implementation_sample"] == min(config.validation["implementation_check"]["sample_size"], sum(p["rows"].values()))
    assert set(p["rows"]) == {"train", "validation", "calibration", "test", "application_test"}
    assert package["fingerprint"][:12] in report and str(p["seed"]) in report


def test_running_twice_gives_the_same_metrics(stage6_build):
    settings, _ = stage6_build
    _, first, _ = validate(settings)
    _, second, _ = validate(settings)
    assert comparable(first) == comparable(second)


def test_an_altered_package_stops_the_step(stage6_build, tmp_path):
    settings, _ = stage6_build
    isolated = settings.__class__(**{**settings.__dict__, "data_dir": tmp_path / "data", "artifacts_dir": tmp_path / "artifacts"})
    shutil.copytree(settings.data_dir, isolated.data_dir)
    path = isolated.processed_dir / "home_credit" / "model_package" / "challenger_model.txt"
    path.write_bytes(path.read_bytes() + b"\n")
    assert run_pipeline(steps=["validate-model"], sources=["home_credit"], settings=isolated) != 0
    run_dir = max(isolated.artifacts_dir.iterdir())
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] != "ok"
    assert "sha256 challenger_model.txt" in json.dumps(manifest, ensure_ascii=False)


def test_a_population_that_does_not_reconcile_stops_the_step(stage6_build, tmp_path):
    settings, _ = stage6_build
    isolated = settings.__class__(**{**settings.__dict__, "data_dir": tmp_path / "data", "artifacts_dir": tmp_path / "artifacts"})
    shutil.copytree(settings.data_dir, isolated.data_dir)
    path = isolated.processed_dir / "home_credit" / "application_train.parquet"
    frame = pd.read_parquet(path)
    frame.iloc[1:].to_parquet(path, index=False)                 # one applicant of the stated population is gone
    assert run_pipeline(steps=["validate-model"], sources=["home_credit"], settings=isolated) != 0
    run_dir = max(isolated.artifacts_dir.iterdir())
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert "population does not reconcile" in json.dumps(manifest, ensure_ascii=False)


def test_findings_are_listed_with_module_and_severity(stage6_build):
    settings, _ = stage6_build
    _, metrics, report = validate(settings)
    for finding in metrics["findings"]:
        assert finding["module"] in MODULES
        assert finding["severity"] in {"note", "major", "fairness", "blocking"}
    failed = [c for c in metrics["checks"] if not c["ok"]]
    assert metrics["provenance"]["checks_failed"] == len(failed)
    assert metrics["provenance"]["findings"] == len(metrics["findings"])
    assert "## Phát hiện" in report
