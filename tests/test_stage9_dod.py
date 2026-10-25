"""Stage 9 Definition of Done, end to end through the pipeline on a synthetic panel
2013-2023: one command labels, builds features, fits and selects the models, scores
every segment, writes the watchlist and the survival curves, reproducibly; every
check passes; the report opens with the results and the watchlist; the capacity and
the scoring month follow ews.yaml; and a feature that reads a column the
availability matrix does not allow stops the step."""

import json
import math
import shutil

import pandas as pd
import yaml

from credit_risk.ews.engine import MODULES
from credit_risk.pipeline import run_pipeline


def ews(settings):
    assert run_pipeline(steps=["ews"], sources=["freddie_mac"], settings=settings) == 0
    return last_run(settings)


def last_run(settings):
    # newest manifest, not the largest name: runs in the same second differ only by config hash
    run_dir = max((p for p in settings.artifacts_dir.iterdir() if (p / "run_manifest.json").exists()),
                  key=lambda p: (p / "run_manifest.json").stat().st_mtime_ns)
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    out = run_dir / "ews"
    metrics = json.loads((out / "ews_metrics.json").read_text(encoding="utf-8")) if (out / "ews_metrics.json").exists() else None
    report = (out / "ews_report.md").read_text(encoding="utf-8") if (out / "ews_report.md").exists() else None
    return manifest, metrics, report, out


def with_config(settings, tmp_path, name, edit):
    config_dir = tmp_path / "configs"
    shutil.copytree(settings.config_dir, config_dir)
    path = config_dir / f"{name}.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    edit(raw)
    path.write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return settings.__class__(**{**settings.__dict__, "config_dir": config_dir})


def test_one_command_writes_metrics_report_watchlist_and_models(stage9_build):
    settings, config = stage9_build
    manifest, metrics, report, out = ews(settings)
    step = next(s for s in manifest["steps"] if s["step"] == "ews")
    assert step["status"] == "ok"
    for name in ("ews_metrics.json", "ews_report.md", "watchlist.csv", "by_split.csv", "monthly_lists.csv", "calibration.csv",
                 "segments.csv", "survival_curves.csv", "trials.csv", "checks.csv", "feature_columns.csv",
                 "models/challenger.txt", "models/baseline.json", "models/encoder.json"):
        assert f"ews/{name}" in step["outputs"] and (out / name).exists(), name
    assert {c["module"] for c in metrics["checks"]} == set(MODULES)
    assert all(c["ok"] for c in metrics["checks"]), [c for c in metrics["checks"] if not c["ok"]]
    p = metrics["provenance"]
    assert p["config_hash"] == manifest["config_hash"] == config.hash
    assert p["scoring_month"] == "2023-06" and p["splits"]["out_of_time"] == ["2022-01", "2023-03"]
    assert p["selected_model"] in {"baseline", "challenger"}
    assert p["selected_model"] == max(p["validation_selection"], key=lambda m: (p["validation_selection"][m], m == "baseline"))
    assert {s["split"] for s in metrics["sections"]["by_split"]} == {"train", "validation", "stress_test", "out_of_time"}
    assert f"selected {p['selected_model']}" in step["detail"]
    watch = pd.read_csv(out / "watchlist.csv")
    assert len(watch) == p["watchlist"]["listed"] == math.ceil(0.05 * p["watchlist"]["scoring_rows"])


def test_running_twice_gives_the_same_metrics(stage9_build):
    settings, _ = stage9_build
    _, first, _, out1 = ews(settings)
    _, second, _, out2 = ews(settings)
    for payload in (first, second):
        payload["provenance"].pop("run_id")
    assert first == second
    assert (out1 / "watchlist.csv").read_text(encoding="utf-8") == (out2 / "watchlist.csv").read_text(encoding="utf-8")


def test_the_report_opens_with_results_and_watchlist_then_limitations(stage9_build):
    settings, config = stage9_build
    _, metrics, report, _ = ews(settings)
    headings = ("## Kết quả chính", "## Watchlist 2023-06", "## Phạm vi và giới hạn", "## Kiểm tra theo module",
                "## Nhãn và cohort", "## Mô hình", "## Backtest watchlist", "## Calibration", "## Phân khúc", "## Survival")
    order = [report.index(h) for h in headings]
    assert order == sorted(order)
    for line in config.ews["ews"]["report"]["limitations"]:
        assert f"- {line}" in report
    assert f"**Mô hình cho watchlist:** {metrics['provenance']['selected_model']}" in report


def test_capacity_and_scoring_month_follow_ews_yaml(stage9_build, tmp_path):
    settings, _ = stage9_build

    def edit(raw):
        raw["ews"]["alert"]["capacity_share"] = 0.10
        raw["ews"]["scoring_month"] = "2022-12"

    _, metrics, report, out = ews(with_config(settings, tmp_path, "ews", edit))
    p = metrics["provenance"]
    assert p["scoring_month"] == "2022-12" and "## Watchlist 2022-12" in report
    watch = pd.read_csv(out / "watchlist.csv")
    assert len(watch) == math.ceil(0.10 * p["watchlist"]["scoring_rows"])
    assert set(watch["period"].str[:7]) == {"2022-12"}


def test_a_feature_reading_a_blocked_column_stops_the_step(stage9_build, tmp_path):
    settings, _ = stage9_build

    def edit(raw):
        raw["freddie_mac"]["performance"]["columns"]["current_interest_rate"] = {"availability": "post_decision"}

    blocked = with_config(settings, tmp_path, "feature_availability", edit)
    assert run_pipeline(steps=["ews"], sources=["freddie_mac"], settings=blocked) != 0
    manifest, *_ = last_run(blocked)
    step = next(s for s in manifest["steps"] if s["step"] == "ews")
    assert step["status"] == "failed" and "rate_change <- performance.current_interest_rate" in step["detail"]
