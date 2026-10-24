"""Stage 7.1: conclusion rules turn metrics.json into a per-model verdict and a recommendation.

A small metrics factory exercises every rule at both sides of its thresholds and
the four branches of the recommendation; then the real chain: the engine's
metrics on the synthetic build conclude without error, and the verdict follows the
thresholds in validation.yaml.
"""

import copy

import pytest

from credit_risk.validation.conclusion import (
    RULES,
    ConclusionError,
    conclude,
    conclusion_markdown,
)
from credit_risk.validation.engine import metrics_payload, run_validation

RULES_CFG = {
    "headline_sample": "test", "recommendation": True,
    "rules": {
        "blocking_checks": {},
        "auc_ci_lower": {"remediate_below": 0.70, "note_below": 0.72},
        "validation_test_gap": {"metric": "gini", "note_if_ci_excludes_zero": True},
        "slope": {"note_outside": [0.90, 1.10], "note_z_above": 3.0, "remediate_outside": [0.80, 1.20]},
        "bad_rate_to_pd_ratio": {"note_outside": [0.95, 1.05]},
        "flagged_bands": {"note_at_least": 3},
        "output_psi": {"remediate_light": "red", "note_light": "amber"},
        "feature_psi": {"note_light": "red"},
        "sensitive_calibration": {"min_group_size": 500},
        "adverse_impact_ratio": {"note_below": 0.80, "remediate_below": 0.60},
        "group_auc_gap": {"note_below_minus": 0.05},
    },
}


def metrics(**over):
    """A clean metrics dictionary where both models pass every rule; `over` patches leaves."""
    def per_model(build):
        return [build(m) for m in ("champion", "challenger")]

    base = {
        "checks": [{"module": "package", "check": "x", "ok": True}],
        "sections": {
            "uncertainty": {
                "intervals": per_model(lambda m: {"split": "test", "model": m, "metric": "auc", "estimate": 0.76, "ci_low": 0.75, "ci_high": 0.77}),
                "differences": [
                    *per_model(lambda m: {"comparison": "validation - test", "split": "validation, test", "model": m, "metric": "gini",
                                          "paired": False, "estimate": 0.01, "ci_low": -0.01, "ci_high": 0.03, "excludes_zero": False}),
                    {"comparison": "challenger - champion", "split": "test", "model": "both", "metric": "gini", "paired": True,
                     "estimate": 0.0, "ci_low": -0.01, "ci_high": 0.01, "excludes_zero": False},
                ],
            },
            "calibration": {"summary": per_model(lambda m: {"model": m, "split": "test", "slope": 1.0, "slope_se": 0.02, "slope_z_vs_1": 0.0,
                                                           "ratio_bad_rate_to_pd": 1.0, "bands_flagged": 0, "bands_flagged_list": ""})},
            "stability": {"table": [
                *per_model(lambda m: {"kind": "output", "model": m, "name": "score" if m == "champion" else "pd_challenger",
                                      "current": "application_test", "segment": "all", "psi": 0.01, "light": "green"}),
                *per_model(lambda m: {"kind": "feature", "model": m, "name": "F1", "current": "application_test", "segment": "all",
                                      "psi": 0.02, "light": "green"}),
            ]},
            "segments": {"table": per_model(lambda m: {"sample": "test", "segment": "gender", "fairness_review": True, "group": "M", "model": m,
                                                       "n": 1000, "calibration_flag": "", "calibration_z": 0.1,
                                                       "adverse_impact_ratio": 0.95, "auc_gap_vs_all": -0.01})},
        },
    }
    for path, value in over.items():
        node = base
        *parents, leaf = path.split("__")
        for key in parents:
            node = node[int(key)] if isinstance(node, list) else node[key]
        if isinstance(node, list):
            node[int(leaf)] = value
        else:
            node[leaf] = value
    return base


def verdicts(m, cfg=RULES_CFG):
    return conclude(m, {"conclusion_rules": cfg}).verdicts


def test_a_clean_metrics_file_passes_every_rule_and_recommends_the_simpler_model():
    c = conclude(metrics(), {"conclusion_rules": RULES_CFG})
    assert c.verdicts == {"champion": "pass", "challenger": "pass"}
    assert all(r.outcome == "pass" for r in c.results)
    assert len(c.results) == 2 * len(RULES)
    assert c.recommendation["deploy"] == "champion" and c.recommendation["benchmark"] == "challenger"
    assert "không khác nhau có ý nghĩa" in c.recommendation["text"]
    text = conclusion_markdown(c)
    assert "champion: Đạt" in text and "challenger: Đạt" in text


def test_blocking_checks_are_always_remediate():
    m = metrics(); m["checks"].append({"module": "population", "check": "rows", "ok": False})
    c = conclude(m, {"conclusion_rules": RULES_CFG})
    assert c.verdicts == {"champion": "remediate", "challenger": "remediate"}
    assert c.recommendation["deploy"] is None and "Không mô hình nào" in c.recommendation["text"]


@pytest.mark.parametrize("patch, model, expected", [
    ({"sections__uncertainty__intervals__0__ci_low": 0.715}, "champion", "note"),
    ({"sections__uncertainty__intervals__0__ci_low": 0.69}, "champion", "remediate"),
    ({"sections__uncertainty__differences__1__excludes_zero": True}, "challenger", "note"),
    ({"sections__calibration__summary__0__slope": 0.88}, "champion", "note"),
    ({"sections__calibration__summary__0__slope_z_vs_1": -3.5}, "champion", "note"),
    ({"sections__calibration__summary__1__slope": 0.75}, "challenger", "remediate"),
    ({"sections__calibration__summary__0__ratio_bad_rate_to_pd": 1.08}, "champion", "note"),
    ({"sections__calibration__summary__1__bands_flagged": 3}, "challenger", "note"),
    ({"sections__stability__table__0__light": "amber"}, "champion", "note"),
    ({"sections__stability__table__1__light": "red"}, "challenger", "remediate"),
    ({"sections__stability__table__2__light": "red"}, "champion", "note"),
    ({"sections__segments__table__0__calibration_flag": "PD thấp (thiếu)"}, "champion", "note"),
    ({"sections__segments__table__1__adverse_impact_ratio": 0.7}, "challenger", "note"),
    ({"sections__segments__table__1__adverse_impact_ratio": 0.5}, "challenger", "remediate"),
    ({"sections__segments__table__0__auc_gap_vs_all": -0.08}, "champion", "note"),
])
def test_each_rule_flips_at_its_threshold_for_the_right_model_only(patch, model, expected):
    v = verdicts(metrics(**patch))
    other = "challenger" if model == "champion" else "champion"
    assert v[model] == expected and v[other] == "pass"


def test_a_small_sensitive_group_does_not_trigger_the_calibration_rule():
    v = verdicts(metrics(sections__segments__table__0__calibration_flag="PD cao (thừa)", sections__segments__table__0__n=200))
    assert v["champion"] == "pass"


def test_thresholds_come_from_the_config():
    cfg = copy.deepcopy(RULES_CFG)
    cfg["rules"]["adverse_impact_ratio"]["remediate_below"] = 0.75
    assert verdicts(metrics(sections__segments__table__0__adverse_impact_ratio=0.7), cfg)["champion"] == "remediate"
    cfg["rules"] = {"blocking_checks": {}}
    c = conclude(metrics(sections__segments__table__0__adverse_impact_ratio=0.2), {"conclusion_rules": cfg})
    assert c.verdicts["champion"] == "pass" and len(c.results) == 2


def test_the_recommendation_branches():
    # one model remediate: deploy the other, keep the failed one as benchmark
    c = conclude(metrics(sections__segments__table__0__adverse_impact_ratio=0.5), {"conclusion_rules": RULES_CFG})
    assert (c.recommendation["deploy"], c.recommendation["benchmark"]) == ("challenger", "champion")
    # both pass, challenger significantly better: deploy challenger
    m = metrics(); m["sections"]["uncertainty"]["differences"][2].update({"estimate": 0.05, "ci_low": 0.04, "ci_high": 0.06, "excludes_zero": True})
    c = conclude(m, {"conclusion_rules": RULES_CFG})
    assert c.recommendation["deploy"] == "challenger" and "cao hơn có ý nghĩa" in c.recommendation["text"]
    # conditions of the deployed model are appended once each
    m = metrics(sections__stability__table__3__light="red"); m["sections"]["uncertainty"]["differences"][2].update({"estimate": 0.05, "ci_low": 0.04, "ci_high": 0.06, "excludes_zero": True})
    c = conclude(m, {"conclusion_rules": RULES_CFG})
    assert c.verdicts["challenger"] == "note" and "Điều kiện kèm theo" in c.recommendation["text"]
    assert c.recommendation["text"].count("Theo dõi biến này") == 1
    # recommendation switched off
    cfg = copy.deepcopy(RULES_CFG); cfg["recommendation"] = False
    assert conclude(metrics(), {"conclusion_rules": cfg}).recommendation == {}


def test_config_errors_are_explicit():
    with pytest.raises(ConclusionError, match="empty"):
        conclude(metrics(), {"conclusion_rules": {}})
    cfg = copy.deepcopy(RULES_CFG); cfg["rules"]["lift"] = {}
    with pytest.raises(ConclusionError, match="unknown"):
        conclude(metrics(), {"conclusion_rules": cfg})
    broken = metrics(); del broken["sections"]["calibration"]
    with pytest.raises(ConclusionError, match="sections.calibration.summary"):
        conclude(broken, {"conclusion_rules": RULES_CFG})


def test_on_the_synthetic_build_the_engine_metrics_conclude_under_the_project_rules(stage6_build):
    settings, config = stage6_build
    result = run_validation(settings, config, "test-run")
    payload = metrics_payload(result)
    c = conclude(payload, config.validation)
    assert set(c.verdicts) == {"champion", "challenger"}
    assert all(r.outcome in {"pass", "note", "remediate", "skipped"} for r in c.results)
    assert c.recommendation["text"]
    assert "## Kết luận" in conclusion_markdown(c)
