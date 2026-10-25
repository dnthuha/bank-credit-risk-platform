"""Stage 8 Definition of Done, end to end through the pipeline on a synthetic
multi-vintage panel: one command writes the monitoring metrics, the report and the
tables, reproducibly; every check passes; the report opens with the status and
the alerts; the lights follow the thresholds in ews.yaml; the reporting month can
be set; and a panel that does not reconcile with its source stops the step."""

import json
import shutil

import pandas as pd
import yaml

from credit_risk.monitoring.engine import MODULES
from credit_risk.pipeline import run_pipeline


def monitor(settings):
    assert run_pipeline(steps=["monitor"], sources=["freddie_mac"], settings=settings) == 0
    return last_run(settings)


def last_run(settings):
    # newest manifest, not the largest name: runs in the same second differ only by config hash
    run_dir = max((p for p in settings.artifacts_dir.iterdir() if (p / "run_manifest.json").exists()),
                  key=lambda p: (p / "run_manifest.json").stat().st_mtime_ns)
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    out = run_dir / "monitor"
    metrics = json.loads((out / "monitoring_metrics.json").read_text(encoding="utf-8")) if (out / "monitoring_metrics.json").exists() else None
    report = (out / "monitoring_report.md").read_text(encoding="utf-8") if (out / "monitoring_report.md").exists() else None
    return manifest, metrics, report, out


def with_ews(settings, tmp_path, edit):
    config_dir = tmp_path / "configs"
    shutil.copytree(settings.config_dir, config_dir)
    path = config_dir / "ews.yaml"
    ews = yaml.safe_load(path.read_text(encoding="utf-8"))
    edit(ews)
    path.write_text(yaml.safe_dump(ews, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return settings.__class__(**{**settings.__dict__, "config_dir": config_dir})


def test_one_command_writes_metrics_report_and_tables(stage8_build):
    settings, config = stage8_build
    manifest, metrics, report, out = monitor(settings)
    step = next(s for s in manifest["steps"] if s["step"] == "monitor")
    assert step["status"] == "ok"
    for name in ("monitoring_metrics.json", "monitoring_report.md", "portfolio_monthly.csv", "vintage_curves.csv", "roll_rates.csv",
                 "transition_counts.csv", "segments_monthly.csv", "concentration.csv", "alerts.csv", "checks.csv"):
        assert f"monitor/{name}" in step["outputs"] and (out / name).exists(), name
    assert {c["module"] for c in metrics["checks"]} == set(MODULES)
    assert all(c["ok"] for c in metrics["checks"]), [c for c in metrics["checks"] if not c["ok"]]
    p = metrics["provenance"]
    assert p["config_hash"] == manifest["config_hash"] == config.hash
    assert p["reporting_month"] == "2019-06" and p["panel"]["last_month"] == "2019-06"
    assert p["panel"]["left_truncated_loans"] == 60                     # one in twenty of 1,200 loans
    assert p["status"] == metrics["status"] in {"green", "amber", "red"}
    lights = pd.Series([a["light"] for a in metrics["alerts"]]).value_counts()
    assert p["alerts"] == {k: int(lights.get(k, 0)) for k in ("red", "amber", "green", "excluded")}
    assert f"status {metrics['status']}" in step["detail"]


def test_running_twice_gives_the_same_metrics(stage8_build):
    settings, _ = stage8_build
    _, first, _, _ = monitor(settings)
    _, second, _, _ = monitor(settings)
    for payload in (first, second):
        payload["provenance"].pop("run_id")
    assert first == second


def test_the_report_opens_with_status_and_alerts_then_limitations(stage8_build):
    settings, config = stage8_build
    _, metrics, report, _ = monitor(settings)
    headings = ("## Trạng thái", "### Cảnh báo", "### Mọi quy tắc", "## Phạm vi và giới hạn", "## Kiểm tra theo module",
                "## Danh mục tháng 2019-06", "## Vintage curve", "## Roll rate và migration", "## Tập trung và phân khúc")
    order = [report.index(h) for h in headings]
    assert order == sorted(order)
    for line in config.ews["monitoring"]["report"]["limitations"]:
        assert f"- {line}" in report
    for a in metrics["alerts"]:
        if a["light"] in ("red", "amber"):
            assert f"| {a['item']} |" in report.split("### Mọi quy tắc")[0]


def test_the_lights_follow_the_thresholds_in_ews_yaml(stage8_build, tmp_path):
    settings, _ = stage8_build

    def lenient(ews):
        a = ews["monitoring"]["alerts"]
        a["amber_relative_increase"], a["red_relative_increase"] = 100.0, 200.0
        a["mix_psi"]["exclude_dimensions"] = list(ews["monitoring"]["concentration_dimensions"])

    _, metrics, _, _ = monitor(with_ews(settings, tmp_path / "lenient", lenient))
    assert metrics["status"] == "green"
    assert {a["light"] for a in metrics["alerts"]} <= {"green", "excluded"}

    def strict(ews):
        a = ews["monitoring"]["alerts"]
        a["amber_relative_increase"], a["red_relative_increase"] = 1e-9, 2e-9
        a["metrics"] = [{"name": "dpd90plus_rate_count", "worse": "higher"}, {"name": "dpd90plus_rate_count", "worse": "lower"}]

    _, metrics, _, _ = monitor(with_ews(settings, tmp_path / "strict", strict))
    assert metrics["status"] == "red"                                  # one of the two directions must be worse


def test_the_reporting_month_can_be_set(stage8_build, tmp_path):
    settings, _ = stage8_build
    _, metrics, report, _ = monitor(with_ews(settings, tmp_path, lambda e: e["monitoring"].update(reporting_month="2018-12")))
    assert metrics["provenance"]["reporting_month"] == "2018-12"
    assert "## Danh mục tháng 2018-12" in report


def test_a_panel_that_does_not_reconcile_stops_the_step(stage8_build, tmp_path):
    settings, _ = stage8_build
    isolated = settings.__class__(**{**settings.__dict__, "data_dir": tmp_path / "data", "artifacts_dir": tmp_path / "artifacts"})
    shutil.copytree(settings.data_dir, isolated.data_dir)
    path = isolated.processed_dir / "freddie_mac" / "performance.parquet"
    perf = pd.read_parquet(path)
    pd.concat([perf, perf.iloc[[0]]]).to_parquet(path, index=False)    # one loan-month twice
    assert run_pipeline(steps=["monitor"], sources=["freddie_mac"], settings=isolated) != 0
    manifest, *_ = last_run(isolated)
    step = next(s for s in manifest["steps"] if s["step"] == "monitor")
    assert step["status"] == "failed" and "no duplicated (loan, month)" in step["detail"]
