"""Stage 7 Definition of Done, end to end through the pipeline: the same one command
that validates the frozen package also concludes; the conclusion is in metrics.json
and recomputable from it alone; the report opens with it, with what each model needs
and with the scope and limitations; and the verdicts follow the thresholds in
validation.yaml, not anything written in code."""

import json
import shutil

import yaml

from credit_risk.config import load_config
from credit_risk.pipeline import run_pipeline
from credit_risk.validation.conclusion import RULES, conclude


def validate(settings):
    assert run_pipeline(steps=["validate-model"], sources=["home_credit"], settings=settings) == 0
    return read_last_run(settings)


def read_last_run(settings):
    # newest manifest, not the largest name: runs in the same second differ only by config hash
    run_dir = max((p for p in settings.artifacts_dir.iterdir() if (p / "run_manifest.json").exists()),
                  key=lambda p: (p / "run_manifest.json").stat().st_mtime_ns)
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    out = run_dir / "validate-model"
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8")) if (out / "metrics.json").exists() else None
    report = (out / "validation_report.md").read_text(encoding="utf-8") if (out / "validation_report.md").exists() else None
    return manifest, metrics, report


def with_validation(settings, tmp_path, edit):
    """Settings whose validation.yaml is the project's after `edit`; data and artifacts are shared, read only."""
    config_dir = tmp_path / "configs"
    shutil.copytree(settings.config_dir, config_dir)
    path = config_dir / "validation.yaml"
    validation = yaml.safe_load(path.read_text(encoding="utf-8"))
    edit(validation)
    path.write_text(yaml.safe_dump(validation, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return settings.__class__(**{**settings.__dict__, "config_dir": config_dir})


def test_one_command_concludes_and_metrics_json_reproduces_the_conclusion(stage6_build):
    settings, config = stage6_build
    manifest, metrics, report = validate(settings)
    c = metrics["conclusion"]
    assert set(c["verdicts"]) == {"champion", "challenger"}
    assert {r["rule"] for r in c["rules"]} == set(config.validation["conclusion_rules"]["rules"]) <= set(RULES)
    assert c["recommendation"]["text"]
    recomputed = conclude({k: v for k, v in metrics.items() if k != "conclusion"}, config.validation)
    assert recomputed.to_dict() == c
    assert metrics["provenance"]["verdicts"] == {m: v["code"] for m, v in c["verdicts"].items()}
    assert metrics["provenance"]["deploy"] == c["recommendation"]["deploy"]
    step = next(s for s in manifest["steps"] if s["step"] == "validate-model")
    assert f"champion {c['verdicts']['champion']['code']}, challenger {c['verdicts']['challenger']['code']}" in step["detail"]


def test_the_report_opens_with_the_conclusion_then_actions_then_limitations(stage6_build):
    settings, _ = stage6_build
    _, metrics, report = validate(settings)
    c = metrics["conclusion"]
    order = [report.index(h) for h in ("## Kết luận", "## Con số chính (test)", "## Việc cần làm",
                                       "## Phạm vi và giới hạn", "## Kiểm tra theo module", "## Phát hiện",
                                       "## So sánh hai mô hình")]
    assert order == sorted(order)
    assert c["recommendation"]["text"] in report
    for model in ("champion", "challenger"):
        assert f"{model} {c['verdicts'][model]['text']}" in report.split("\n\n")[2]      # the summary line
        assert f"**{model}** ({c['verdicts'][model]['text']})" in report
    for rule in c["rules"]:
        if rule["outcome"] in ("note", "remediate"):
            assert f"`{rule['rule']}`: {rule['detail']}" in report


def test_the_limitations_come_from_the_config_and_from_the_run(stage6_build):
    settings, config = stage6_build
    _, metrics, report = validate(settings)
    configured = config.validation["report"]["limitations"]
    assert metrics["limitations"][:len(configured)] == configured
    assert any("out-of-time" in x for x in configured)
    for line in metrics["limitations"]:
        assert f"- {line}" in report


def test_the_verdicts_follow_the_thresholds_in_validation_yaml(stage6_build, tmp_path):
    settings, _ = stage6_build

    def strict(v):        # no AUC is good enough
        v["conclusion_rules"]["rules"]["auc_ci_lower"] = {"remediate_below": 0.999, "note_below": 0.9995}

    _, metrics, report = validate(with_validation(settings, tmp_path / "strict", strict))
    c = metrics["conclusion"]
    assert {m: v["code"] for m, v in c["verdicts"].items()} == {"champion": "remediate", "challenger": "remediate"}
    assert c["recommendation"]["deploy"] is None and "Không mô hình nào" in report

    def lenient(v):       # only the integrity rule is left
        v["conclusion_rules"]["rules"] = {"blocking_checks": {}}

    _, metrics, _ = validate(with_validation(settings, tmp_path / "lenient", lenient))
    c = metrics["conclusion"]
    assert {m: v["code"] for m, v in c["verdicts"].items()} == {"champion": "pass", "challenger": "pass"}
    assert c["recommendation"]["deploy"] in {"champion", "challenger"}
    assert len(c["rules"]) == 2


def test_the_configured_limitations_are_printed_as_written(stage6_build, tmp_path):
    settings, _ = stage6_build

    def edit(v):
        v["report"]["limitations"] = ["Giới hạn thử nghiệm: chỉ một dòng."]

    _, metrics, report = validate(with_validation(settings, tmp_path, edit))
    assert metrics["limitations"][0] == "Giới hạn thử nghiệm: chỉ một dòng."
    assert "- Giới hạn thử nghiệm: chỉ một dòng." in report
    assert not any("out-of-time" in x for x in metrics["limitations"])


def test_missing_conclusion_rules_stop_the_step_with_a_clear_message(stage6_build, tmp_path):
    settings, _ = stage6_build
    broken = with_validation(settings, tmp_path, lambda v: v.pop("conclusion_rules"))
    assert run_pipeline(steps=["validate-model"], sources=["home_credit"], settings=broken) != 0
    manifest, _, _ = read_last_run(broken)
    step = next(s for s in manifest["steps"] if s["step"] == "validate-model")
    assert step["status"] != "ok"
    assert "conclusion_rules" in step["detail"]


def test_the_project_config_loads_with_the_report_block(stage6_build):
    settings, _ = stage6_build
    validation = load_config(settings.config_dir).validation
    assert validation["report"]["limitations"] and all(isinstance(x, str) and x for x in validation["report"]["limitations"])
