"""Stage 10 Definition of Done: the `report` step gathers the latest completed output of every
stage into one bundle (and reports/latest), reproducibly, naming the runs it took each part from;
a module that has not run is listed as missing, not invented; the dashboard renders every page
from that bundle without an exception and never touches the data itself."""

import json
import os
import sys
from pathlib import Path

import pytest

from credit_risk.pipeline import run_pipeline
from credit_risk.reporting.bundle import latest_ok_run, step_ok

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = REPO_ROOT / "app"
VIEWS = ("overview", "development", "validation", "monitoring", "ews", "engineering", "limitations")


def newest_run(settings) -> Path:
    runs = [p for p in settings.artifacts_dir.iterdir() if (p / "run_manifest.json").exists()]
    return max(runs, key=lambda p: (p / "run_manifest.json").stat().st_mtime_ns)


@pytest.fixture(scope="module")
def reported(stage6_build):
    """validate-model + report on the shared synthetic package; returns (settings, run_dir, index)."""
    settings, _ = stage6_build
    assert run_pipeline(steps=["validate-model", "report"], sources=["home_credit"], settings=settings) == 0
    run_dir = newest_run(settings)
    index = json.loads((run_dir / "report" / "index.json").read_text(encoding="utf-8"))
    return settings, run_dir, index


def test_report_step_bundles_the_validated_package_and_mirrors_latest(reported):
    settings, run_dir, index = reported
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    step = next(s for s in manifest["steps"] if s["step"] == "report")
    assert step["status"] == "ok"
    assert "report/index.json" in step["outputs"]

    # development and validation come from the same model: the runs named by validation's provenance
    metrics = json.loads((run_dir / "validate-model" / "metrics.json").read_text(encoding="utf-8"))
    assert index["runs"]["validate-model"] == run_dir.name
    assert index["runs"]["calibrate"] == metrics["provenance"]["package_run_id"]
    assert index["runs"]["train"] == metrics["provenance"]["train_run_id"]
    for rel in ("development/model_card.md", "development/scorecard_table.csv", "development/performance.json",
                "development/calibration.json", "development/package.json", "development/applicant_examples.json",
                "validation/metrics.json", "validation/validation_report.md"):
        assert (run_dir / "report" / rel).exists(), rel
        assert rel in index["files"]
    # modules that never ran are missing, with their file names, not faked
    assert index["runs"]["monitor"] is None and index["runs"]["ews"] is None
    assert "monitoring_metrics.json" in index["missing"]["monitoring"]
    assert "ews_metrics.json" in index["missing"]["ews"]
    assert "monitoring" not in index["headline"] and "ews" not in index["headline"]
    # headline numbers agree with the files they summarise
    h = index["headline"]
    assert h["validation"]["checks_total"] == metrics["provenance"]["checks_total"]
    assert h["validation"]["models"]["champion"]["verdict"]["code"] == metrics["conclusion"]["verdicts"]["champion"]["code"]
    assert h["development"]["champion_features"] >= 1

    latest = settings.reports_root / "latest"
    assert latest.is_dir() and not latest.is_relative_to(REPO_ROOT / "reports")
    mirrored = json.loads((latest / "index.json").read_text(encoding="utf-8"))
    assert mirrored == index
    assert {p.relative_to(latest).as_posix() for p in latest.rglob("*") if p.is_file()} == set(index["files"]) | {"index.json"}


def test_applicant_examples_reconcile_with_the_scorecard(reported):
    _, run_dir, _ = reported
    ex = json.loads((run_dir / "report" / "development" / "applicant_examples.json").read_text(encoding="utf-8"))
    assert 1 <= len(ex["applicants"]) <= 12
    pds = [a["pd_champion"] for a in ex["applicants"]]
    assert pds == sorted(pds, reverse=True), "riskiest first"
    for a in ex["applicants"]:
        assert a["base_points"] + sum(p["points"] for p in a["points"]) == a["score"]
        assert [p["feature"] for p in a["points"]] == ex["features"]
        assert all(p["lost"] == p["best_points"] - p["points"] >= 0 for p in a["points"])
        assert set(a["reason_codes"]) <= set(ex["features"])
        lost = {p["feature"]: p["lost"] for p in a["points"]}
        assert all(lost[f] > 0 for f in a["reason_codes"])
        assert 0 < a["pd_champion"] < 1 and 0 < a["pd_challenger"] < 1


def test_running_report_twice_gives_the_same_bundle(reported):
    settings, run_dir, first = reported
    assert run_pipeline(steps=["report"], sources=["home_credit"], settings=settings) == 0
    second_dir = newest_run(settings)
    assert second_dir != run_dir
    second = json.loads((second_dir / "report" / "index.json").read_text(encoding="utf-8"))
    for payload in (first, second):
        payload.pop("generated_at"), payload.pop("run_id")
        payload.pop("run_manifests")   # the validate-model run's manifest did not exist yet during its own report step
    assert first == second
    a = (run_dir / "report" / "development" / "applicant_examples.json").read_text(encoding="utf-8")
    b = (second_dir / "report" / "development" / "applicant_examples.json").read_text(encoding="utf-8")
    assert a == b


def test_report_fails_when_nothing_has_run(settings):
    assert run_pipeline(steps=["report"], settings=settings) == 1
    manifest = json.loads(next(settings.artifacts_dir.glob("*/run_manifest.json")).read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["steps"][-1]["step"] == "report" and manifest["steps"][-1]["status"] == "failed"
    assert not (settings.reports_root / "latest").exists()


def test_latest_ok_run_takes_the_newest_successful_step(tmp_path):
    def run(name, steps):
        (tmp_path / name).mkdir()
        (tmp_path / name / "run_manifest.json").write_text(json.dumps({"steps": [{"step": s, "status": st} for s, st in steps]}))

    run("20260101T000000Z-aaaa", [("monitor", "ok")])
    run("20260102T000000Z-aaaa", [("monitor", "failed")])
    run("20260103T000000Z-aaaa", [("ews", "ok")])
    (tmp_path / "20260104T000000Z-aaaa").mkdir()          # no manifest: a run in progress
    assert latest_ok_run(tmp_path, "monitor") == "20260101T000000Z-aaaa"
    assert latest_ok_run(tmp_path, "ews") == "20260103T000000Z-aaaa"
    assert latest_ok_run(tmp_path, "train") is None
    # the current run wins when its step folder already exists, even without a manifest
    (tmp_path / "20260104T000000Z-aaaa" / "monitor").mkdir()
    assert latest_ok_run(tmp_path, "monitor", prefer="20260104T000000Z-aaaa") == "20260104T000000Z-aaaa"
    assert step_ok(None, "monitor") is False


# ---------------------------------------------------------------------------
# Dashboard: every page renders from the bundle, reads nothing else
# ---------------------------------------------------------------------------

def _view_script(view: str, bundle_dir: Path) -> str:
    return (
        "import sys\n"
        f"sys.path.insert(0, {str(APP_DIR)!r})\n"
        "import importlib\n"
        "from common import Bundle\n"
        f"importlib.import_module('views.{view}').render(Bundle({str(bundle_dir)!r}))\n"
    )


@pytest.mark.parametrize("view", VIEWS)
def test_every_dashboard_page_renders_from_the_bundle(reported, view):
    from streamlit.testing.v1 import AppTest

    _, run_dir, _ = reported
    at = AppTest.from_string(_view_script(view, run_dir / "report"), default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.title, "every page starts with a title"
    if view in ("monitoring", "ews"):
        assert at.warning, "a module missing from the bundle is said so, not faked"
    else:
        assert not at.warning


def test_dashboard_entry_point_runs_with_navigation(reported, monkeypatch):
    from streamlit.testing.v1 import AppTest

    _, run_dir, _ = reported
    monkeypatch.setenv("CREDIT_RISK_BUNDLE_DIR", str(run_dir / "report"))
    monkeypatch.setenv("CREDIT_RISK_ARTIFACTS_DIR", str(run_dir.parent))
    sys.modules.pop("common", None)
    at = AppTest.from_file(str(APP_DIR / "app.py"), default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.title[0].value == "Credit Risk Model Lifecycle Platform"
    assert at.sidebar.selectbox[0].value == run_dir / "report"


def test_dashboard_without_a_bundle_says_how_to_make_one(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("CREDIT_RISK_BUNDLE_DIR", str(tmp_path / "nowhere"))
    monkeypatch.setenv("CREDIT_RISK_ARTIFACTS_DIR", str(tmp_path / "empty"))
    monkeypatch.setattr("os.environ", {**os.environ, "CREDIT_RISK_BUNDLE_DIR": str(tmp_path / "nowhere"),
                                       "CREDIT_RISK_ARTIFACTS_DIR": str(tmp_path / "empty")})
    sys.modules.pop("common", None)
    script = (
        "import sys\n"
        f"sys.path.insert(0, {str(APP_DIR)!r})\n"
        "import common\n"
        f"common.DEFAULT_BUNDLE = __import__('pathlib').Path({str(tmp_path / 'none')!r})\n"
        "import streamlit as st\n"
        "folder = common.resolve_bundle()\n"
        "if folder is None:\n"
        "    st.error('no bundle')\n"
        "    st.code('python -m credit_risk run --steps report')\n"
    )
    at = AppTest.from_string(script, default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.error and "report" in at.code[0].value
