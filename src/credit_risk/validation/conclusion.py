"""Stage 7.1: from measurements to a conclusion.

The engine (Stage 6) measures and lists findings; it does not judge. This
module applies `validation.yaml -> conclusion_rules` to the engine's
metrics.json and gives, for each model, one of three verdicts:

    Đạt                 every rule passes;
    Đạt có lưu ý        at least one rule says "note", none says "remediate";
    Phải khắc phục      at least one rule says "remediate".

Every rule is a small function with its thresholds in the config, so the
report can show, rule by rule, the value it looked at, the threshold it
applied and the outcome. Each violated rule carries a recommendation
sentence; the findings table of the report is built from these.

An overall recommendation (which model to deploy, which to keep as the
benchmark) is derived from the two verdicts and the paired bootstrap
difference, never written by hand. The rules read the metrics dictionary
only, so they can run on a metrics.json file long after the engine did.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

MODELS = ("champion", "challenger")
VERDICTS = {"pass": "Đạt", "note": "Đạt có lưu ý", "remediate": "Phải khắc phục"}
RANK = {"pass": 0, "note": 1, "remediate": 2}
LIGHT_RANK = {"green": 0, "amber": 1, "red": 2}


class ConclusionError(ValueError):
    pass


@dataclass(frozen=True)
class RuleResult:
    rule: str
    model: str
    outcome: str                 # pass | note | remediate | skipped
    value: str                   # what was looked at, formatted
    threshold: str               # what it was compared with
    detail: str = ""             # one sentence, for the report
    recommendation: str = ""     # one sentence, when violated


@dataclass
class Conclusion:
    results: list[RuleResult] = field(default_factory=list)
    verdicts: dict[str, str] = field(default_factory=dict)          # model -> pass|note|remediate
    recommendation: dict[str, Any] = field(default_factory=dict)

    def table(self) -> pd.DataFrame:
        return pd.DataFrame([r.__dict__ for r in self.results])

    def verdict_text(self, model: str) -> str:
        return VERDICTS[self.verdicts[model]]

    def to_dict(self) -> dict[str, Any]:
        return {"verdicts": {m: {"code": v, "text": VERDICTS[v]} for m, v in self.verdicts.items()},
                "rules": [r.__dict__ for r in self.results], "recommendation": self.recommendation}


# ---------------------------------------------------------------------------
# Access to the engine's metrics.json
# ---------------------------------------------------------------------------

def _rows(metrics: dict[str, Any], section: str, table: str) -> pd.DataFrame:
    try:
        return pd.DataFrame(metrics["sections"][section][table])
    except KeyError as exc:
        raise ConclusionError(f"metrics.json lacks sections.{section}.{table}") from exc


def _worst(outcomes) -> str:
    outcomes = [o for o in outcomes if o in RANK]
    return max(outcomes, key=RANK.__getitem__) if outcomes else "pass"


def _fmt(v, nd=4) -> str:
    try:
        return format(float(v), f".{nd}f")
    except (TypeError, ValueError):
        return str(v)


# ---------------------------------------------------------------------------
# Rules: each returns one RuleResult per model
# ---------------------------------------------------------------------------

def rule_blocking_checks(metrics, cfg, sample) -> list[RuleResult]:
    checks = pd.DataFrame(metrics.get("checks", []))
    blocking = checks[(~checks["ok"]) & (checks["module"].isin(["package", "population", "implementation"]))] if len(checks) else checks
    out = []
    for model in MODELS:
        n = int(len(blocking))
        out.append(RuleResult("blocking_checks", model, "remediate" if n else "pass", f"{n} kiểm tra lệch", "0",
                              "Gói, dân số và chấm lại đều khớp." if not n else
                              f"{n} kiểm tra về gói / dân số / chấm lại không khớp: mọi con số khác nói về một mô hình hoặc mẫu khác.",
                              "" if not n else "Dừng phê duyệt; làm rõ nguồn lệch (gói bị sửa, dữ liệu khác, pipeline khác) rồi thẩm định lại."))
    return out


def rule_auc_ci_lower(metrics, cfg, sample) -> list[RuleResult]:
    iv = _rows(metrics, "uncertainty", "intervals")
    out = []
    for model in MODELS:
        row = iv[(iv["model"] == model) & (iv["split"] == sample) & (iv["metric"] == "auc")]
        if row.empty:
            out.append(RuleResult("auc_ci_lower", model, "skipped", "-", "-", f"không có khoảng tin cậy AUC trên {sample}"))
            continue
        low, est = float(row["ci_low"].iloc[0]), float(row["estimate"].iloc[0])
        outcome = "remediate" if low < cfg["remediate_below"] else "note" if low < cfg["note_below"] else "pass"
        out.append(RuleResult("auc_ci_lower", model, outcome, f"AUC {est:.4f}, biên dưới {low:.4f}",
                              f">= {cfg['note_below']} (lưu ý), >= {cfg['remediate_below']} (khắc phục)",
                              f"Sức phân biệt trên {sample} chắc chắn ít nhất AUC {low:.3f}.",
                              "" if outcome == "pass" else "Sức phân biệt dưới mức chấp nhận: xem lại danh sách biến và dữ liệu đầu vào."))
    return out


def rule_validation_test_gap(metrics, cfg, sample) -> list[RuleResult]:
    diffs = _rows(metrics, "uncertainty", "differences")
    metric = cfg.get("metric", "gini")
    out = []
    for model in MODELS:
        row = diffs[(diffs["model"] == model) & (diffs["metric"] == metric) & (diffs["comparison"].str.contains(sample))]
        if row.empty:
            out.append(RuleResult("validation_test_gap", model, "skipped", "-", "-", "không có so sánh validation - test"))
            continue
        r = row.iloc[0]
        violated = bool(r["excludes_zero"]) and float(r["estimate"]) > 0 and cfg.get("note_if_ci_excludes_zero", True)
        out.append(RuleResult("validation_test_gap", model, "note" if violated else "pass",
                              f"{metric} {r['comparison']} = {float(r['estimate']):+.4f} [{float(r['ci_low']):+.4f}, {float(r['ci_high']):+.4f}]",
                              "khoảng 95% chứa 0",
                              "Mức tụt từ mẫu phát triển sang mẫu test nằm trong nhiễu mẫu." if not violated else
                              "Mẫu phát triển lạc quan hơn test một cách có ý nghĩa.",
                              "" if not violated else "Báo cáo hiệu năng theo con số trên test, không theo validation; xem lại mức tinh chỉnh trên validation."))
    return out


def rule_slope(metrics, cfg, sample) -> list[RuleResult]:
    summary = _rows(metrics, "calibration", "summary")
    lo_n, hi_n = cfg["note_outside"]
    lo_r, hi_r = cfg["remediate_outside"]
    out = []
    for model in MODELS:
        row = summary[(summary["model"] == model) & (summary["split"] == sample)]
        if row.empty:
            out.append(RuleResult("slope", model, "skipped", "-", "-", f"không có hiệu chỉnh trên {sample}"))
            continue
        r = row.iloc[0]
        slope, z = float(r["slope"]), abs(float(r["slope_z_vs_1"]))
        outcome = "remediate" if not lo_r <= slope <= hi_r else "note" if (not lo_n <= slope <= hi_n or z > cfg["note_z_above"]) else "pass"
        out.append(RuleResult("slope", model, outcome, f"slope {slope:.3f} (SE {float(r['slope_se']):.3f}, |z| {z:.1f})",
                              f"[{lo_n}, {hi_n}] và |z| <= {cfg['note_z_above']} (lưu ý); [{lo_r}, {hi_r}] (khắc phục)",
                              "PD không quá cực đoan cũng không quá dè dặt." if outcome == "pass" else
                              ("PD hơi cực đoan ở hai đầu." if slope < 1 else "PD hơi dè dặt ở hai đầu."),
                              "" if outcome == "pass" else "Hiệu chỉnh lại trên mẫu mới hoặc xem lại Platt; theo dõi slope theo thời gian."))
    return out


def rule_bad_rate_to_pd_ratio(metrics, cfg, sample) -> list[RuleResult]:
    summary = _rows(metrics, "calibration", "summary")
    lo, hi = cfg["note_outside"]
    out = []
    for model in MODELS:
        row = summary[(summary["model"] == model) & (summary["split"] == sample)]
        if row.empty:
            out.append(RuleResult("bad_rate_to_pd_ratio", model, "skipped", "-", "-", "không có"))
            continue
        ratio = float(row["ratio_bad_rate_to_pd"].iloc[0])
        outcome = "pass" if lo <= ratio <= hi else "note"
        out.append(RuleResult("bad_rate_to_pd_ratio", model, outcome, f"bad rate / PD trung bình = {ratio:.3f}", f"[{lo}, {hi}]",
                              "PD đúng mức tổng thể." if outcome == "pass" else ("PD thấp hơn bad rate thật." if ratio > 1 else "PD cao hơn bad rate thật."),
                              "" if outcome == "pass" else "Điều chỉnh intercept (central tendency) trước khi dùng PD cho định giá hay tổn thất dự kiến."))
    return out


def rule_flagged_bands(metrics, cfg, sample) -> list[RuleResult]:
    summary = _rows(metrics, "calibration", "summary")
    out = []
    for model in MODELS:
        row = summary[(summary["model"] == model) & (summary["split"] == sample)]
        if row.empty:
            out.append(RuleResult("flagged_bands", model, "skipped", "-", "-", "không có"))
            continue
        n = int(row["bands_flagged"].iloc[0]); which = str(row["bands_flagged_list"].iloc[0] or "")
        outcome = "note" if n >= cfg["note_at_least"] else "pass"
        out.append(RuleResult("flagged_bands", model, outcome, f"{n} band lệch hệ thống" + (f" ({which})" if which else ""),
                              f"< {cfg['note_at_least']}",
                              "Không có vùng PD lệch kéo dài." if outcome == "pass" else "Nhiều vùng PD lệch hệ thống.",
                              "" if outcome == "pass" else "Xem lại dạng hiệu chỉnh (isotonic hoặc theo phân khúc) cho các band bị cờ."))
    return out


def rule_output_psi(metrics, cfg, sample) -> list[RuleResult]:
    table = _rows(metrics, "stability", "table")
    out = []
    for model in MODELS:
        rows = table[(table["kind"] == "output") & (table["model"] == model) & (table["current"] == "application_test") & (table["segment"] == "all")]
        if rows.empty:
            out.append(RuleResult("output_psi", model, "skipped", "-", "-", "không có application_test"))
            continue
        r = rows.iloc[0]
        light = str(r["light"])
        outcome = "remediate" if light == cfg["remediate_light"] else "note" if light == cfg["note_light"] else "pass"
        out.append(RuleResult("output_psi", model, outcome, f"PSI {r['name']} so với application_test = {float(r['psi']):.4f} ({light})",
                              f"{cfg['note_light']} = lưu ý, {cfg['remediate_light']} = khắc phục",
                              "Phân phối đầu ra trên dân số mới giống dân số học." if outcome == "pass" else "Đầu ra trôi trên dân số mới.",
                              "" if outcome == "pass" else "Điều tra nguồn trôi (mix sản phẩm, kênh) trước khi dùng ngưỡng điểm cũ."))
    return out


def rule_feature_psi(metrics, cfg, sample) -> list[RuleResult]:
    table = _rows(metrics, "stability", "table")
    out = []
    for model in MODELS:
        rows = table[(table["kind"] == "feature") & (table["model"] == model) & (table["current"] == "application_test")]
        if rows.empty:
            out.append(RuleResult("feature_psi", model, "skipped", "-", "-", "không có"))
            continue
        bad = rows[rows["light"] == cfg["note_light"]]
        worst = rows.sort_values("psi", ascending=False).iloc[0]
        outcome = "note" if not bad.empty else "pass"
        out.append(RuleResult("feature_psi", model, outcome,
                              f"{len(bad)} / {len(rows)} biến {cfg['note_light']}; lớn nhất {worst['name']} = {float(worst['psi']):.4f}",
                              f"không biến nào {cfg['note_light']}",
                              "Các biến ảnh hưởng nhất ổn định trên dân số mới." if outcome == "pass" else
                              "Biến ảnh hưởng lớn đã trôi: " + ", ".join(f"{n} ({p:.2f})" for n, p in zip(bad["name"], bad["psi"])) + ".",
                              "" if outcome == "pass" else "Theo dõi biến này trong monitoring; cân nhắc cắt bin hoặc thay biến nếu triển khai."))
    return out


def _sensitive(metrics, sample):
    table = _rows(metrics, "segments", "table")
    return table[(table["fairness_review"]) & (table["group"] != "(all)") & (table["sample"] == sample)]


def rule_sensitive_calibration(metrics, cfg, sample) -> list[RuleResult]:
    fair = _sensitive(metrics, sample)
    out = []
    for model in MODELS:
        rows = fair[(fair["model"] == model) & (fair["n"] >= cfg.get("min_group_size", 500))]
        flagged = rows[rows["calibration_flag"].fillna("") != ""]
        outcome = "note" if not flagged.empty else "pass"
        out.append(RuleResult("sensitive_calibration", model, outcome, f"{len(flagged)} / {len(rows)} nhóm bị cờ",
                              "0 nhóm",
                              "PD đúng mức trong mọi nhóm nhạy cảm." if outcome == "pass" else
                              "PD lệch hệ thống ở: " + "; ".join(f"{s} = {g} ({f}, z {z:+.1f})" for s, g, f, z in
                                                                zip(flagged["segment"], flagged["group"], flagged["calibration_flag"], flagged["calibration_z"])) + ".",
                              "" if outcome == "pass" else "Kiểm tra lại cách biến nhạy cảm vào mô hình; cân nhắc hiệu chỉnh theo nhóm hoặc bỏ biến (PROJECT_SCOPE #7)."))
    return out


def rule_adverse_impact_ratio(metrics, cfg, sample) -> list[RuleResult]:
    fair = _sensitive(metrics, sample)
    out = []
    for model in MODELS:
        rows = fair[(fair["model"] == model) & fair["adverse_impact_ratio"].notna()]
        if rows.empty:
            out.append(RuleResult("adverse_impact_ratio", model, "skipped", "-", "-", "không có AIR"))
            continue
        worst = rows.sort_values("adverse_impact_ratio").iloc[0]
        air = float(worst["adverse_impact_ratio"])
        outcome = "remediate" if air < cfg["remediate_below"] else "note" if air < cfg["note_below"] else "pass"
        below = rows[rows["adverse_impact_ratio"] < cfg["note_below"]]
        out.append(RuleResult("adverse_impact_ratio", model, outcome, f"AIR thấp nhất {air:.2f} ({worst['segment']} = {worst['group']})",
                              f">= {cfg['note_below']} (lưu ý), >= {cfg['remediate_below']} (khắc phục)",
                              "Tỷ lệ duyệt giữa các nhóm nhạy cảm trong mốc 4/5." if outcome == "pass" else
                              "Nhóm bị duyệt ít hơn mốc 4/5: " + ", ".join(f"{s} = {g} ({a:.2f})" for s, g, a in zip(below["segment"], below["group"], below["adverse_impact_ratio"])) + ".",
                              "" if outcome == "pass" else "Đối chiếu tỷ lệ từ chối với bad rate thật của nhóm; phần không giải thích được cần xử lý trước khi triển khai."))
    return out


def rule_group_auc_gap(metrics, cfg, sample) -> list[RuleResult]:
    fair = _sensitive(metrics, sample)
    out = []
    for model in MODELS:
        rows = fair[(fair["model"] == model) & fair["auc_gap_vs_all"].notna()]
        if rows.empty:
            out.append(RuleResult("group_auc_gap", model, "skipped", "-", "-", "không có AUC theo nhóm"))
            continue
        bad = rows[rows["auc_gap_vs_all"] < -cfg["note_below_minus"]]
        worst = rows.sort_values("auc_gap_vs_all").iloc[0]
        outcome = "note" if not bad.empty else "pass"
        out.append(RuleResult("group_auc_gap", model, outcome, f"lệch AUC lớn nhất {float(worst['auc_gap_vs_all']):+.3f} ({worst['segment']} = {worst['group']})",
                              f">= -{cfg['note_below_minus']}",
                              "Sức phân biệt đồng đều giữa các nhóm nhạy cảm." if outcome == "pass" else
                              "Mô hình phân biệt kém hơn rõ ở: " + ", ".join(f"{s} = {g} ({a:+.3f})" for s, g, a in zip(bad["segment"], bad["group"], bad["auc_gap_vs_all"])) + ".",
                              "" if outcome == "pass" else "Cân nhắc biến hoặc mô hình riêng cho nhóm yếu; ghi rõ giới hạn trong model card."))
    return out


RULES: dict[str, Callable[[dict, dict, str], list[RuleResult]]] = {
    "blocking_checks": rule_blocking_checks,
    "auc_ci_lower": rule_auc_ci_lower,
    "validation_test_gap": rule_validation_test_gap,
    "slope": rule_slope,
    "bad_rate_to_pd_ratio": rule_bad_rate_to_pd_ratio,
    "flagged_bands": rule_flagged_bands,
    "output_psi": rule_output_psi,
    "feature_psi": rule_feature_psi,
    "sensitive_calibration": rule_sensitive_calibration,
    "adverse_impact_ratio": rule_adverse_impact_ratio,
    "group_auc_gap": rule_group_auc_gap,
}


# ---------------------------------------------------------------------------
# Conclude
# ---------------------------------------------------------------------------

def conclude(metrics: dict[str, Any], validation: dict[str, Any]) -> Conclusion:
    cfg = validation.get("conclusion_rules") or {}
    rules_cfg = cfg.get("rules")
    if not rules_cfg:
        raise ConclusionError("validation.conclusion_rules.rules is empty: nothing to conclude with")
    unknown = set(rules_cfg) - set(RULES)
    if unknown:
        raise ConclusionError(f"unknown conclusion rules {sorted(unknown)}; known: {sorted(RULES)}")
    sample = str(cfg.get("headline_sample", "test"))
    conclusion = Conclusion()
    for name in RULES:                       # fixed order, so the report reads the same every time
        if name in rules_cfg:
            conclusion.results.extend(RULES[name](metrics, rules_cfg[name] or {}, sample))
    for model in MODELS:
        conclusion.verdicts[model] = _worst(r.outcome for r in conclusion.results if r.model == model)
    if cfg.get("recommendation", True):
        conclusion.recommendation = recommend(metrics, conclusion, sample)
    return conclusion


def recommend(metrics: dict[str, Any], conclusion: Conclusion, sample: str) -> dict[str, Any]:
    """Which model to deploy and which to keep as the benchmark, from the verdicts and the paired difference."""
    v = conclusion.verdicts
    diffs = _rows(metrics, "uncertainty", "differences")
    paired = diffs[(diffs["paired"] == True) & (diffs["split"] == sample) & (diffs["metric"] == "gini")]  # noqa: E712
    better, gap = None, None
    if not paired.empty:
        r = paired.iloc[0]
        gap = {"estimate": float(r["estimate"]), "ci_low": float(r["ci_low"]), "ci_high": float(r["ci_high"]), "excludes_zero": bool(r["excludes_zero"])}
        if gap["excludes_zero"]:
            better = "challenger" if gap["estimate"] > 0 else "champion"
    deployable = [m for m in MODELS if v[m] != "remediate"]
    if not deployable:
        deploy, benchmark = None, None
        text = "Không mô hình nào đủ điều kiện triển khai: cả hai đều Phải khắc phục. Xử lý các điểm khắc phục rồi thẩm định lại."
    elif len(deployable) == 1:
        deploy, benchmark = deployable[0], [m for m in MODELS if m != deployable[0]][0]
        text = (f"Triển khai {deploy} ({VERDICTS[v[deploy]]}); {benchmark} Phải khắc phục nên chỉ giữ làm đối chứng, "
                f"không dùng ra quyết định.")
    else:
        if better is not None:
            deploy, benchmark = better, [m for m in MODELS if m != better][0]
            text = (f"Cả hai đủ điều kiện ({VERDICTS[v['champion']]} / {VERDICTS[v['challenger']]}). Triển khai {deploy}: Gini trên {sample} "
                    f"cao hơn có ý nghĩa ({gap['estimate']:+.3f}, khoảng [{gap['ci_low']:+.3f}, {gap['ci_high']:+.3f}]); giữ {benchmark} làm đối chứng.")
        else:
            deploy, benchmark = "champion", "challenger"
            text = ("Cả hai đủ điều kiện và không khác nhau có ý nghĩa về Gini: triển khai champion (đơn giản, giải thích được), "
                    "giữ challenger làm đối chứng.")
    notes = [r for r in conclusion.results if r.model == deploy and r.outcome == "note"] if deploy else []
    if notes:
        conditions = list(dict.fromkeys(r.recommendation.rstrip(".") for r in notes if r.recommendation))
        text += " Điều kiện kèm theo: " + "; ".join(conditions) + "."
    return {"deploy": deploy, "benchmark": benchmark, "gini_gap_test": gap, "text": text}


def conclusion_markdown(conclusion: Conclusion) -> str:
    lines = ["## Kết luận", ""]
    for model in MODELS:
        lines.append(f"- **{model}: {conclusion.verdict_text(model)}**")
    if conclusion.recommendation:
        lines += ["", f"**Khuyến nghị:** {conclusion.recommendation['text']}"]
    lines += ["", "| Quy tắc | Mô hình | Giá trị | Ngưỡng | Kết quả | Diễn giải |", "|---|---|---|---|---|---|"]
    for r in conclusion.results:
        mark = {"pass": "đạt", "note": "**lưu ý**", "remediate": "**KHẮC PHỤC**", "skipped": "bỏ qua"}[r.outcome]
        lines.append(f"| {r.rule} | {r.model} | {r.value} | {r.threshold} | {mark} | {r.detail} |")
    return "\n".join(lines) + "\n"
