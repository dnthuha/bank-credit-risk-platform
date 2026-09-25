"""Stage 4 challenger report: how the LightGBM was tuned, how it compares with
the champion scorecard on the samples development may look at, what drives it,
and how it explains single applicants.

As for the model card, only train and validation are evaluated; calibration and
test are left for stages 5 and 6.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from credit_risk.models.challenger import Challenger
from credit_risk.reporting.model_card import SENSITIVE


def global_importance(contributions: pd.DataFrame, challenger: Challenger) -> pd.DataFrame:
    """Mean |SHAP| (log-odds) and share of split gain per feature, ranked by mean |SHAP|."""
    shap = contributions[challenger.features].abs().mean()
    gain = pd.Series(challenger.booster.feature_importance("gain", iteration=challenger.best_iteration),
                     index=challenger.booster.feature_name())
    frame = pd.DataFrame({"feature": challenger.features,
                          "mean_abs_shap": shap.reindex(challenger.features).to_numpy(),
                          "gain_share": (gain / gain.sum()).reindex(challenger.features).to_numpy()})
    frame = frame.sort_values(["mean_abs_shap", "feature"], ascending=[False, True]).reset_index(drop=True)
    frame.insert(0, "rank", np.arange(1, len(frame) + 1))
    return frame


def example_rows(pd_hat: pd.Series, n: int) -> list:
    """Index labels of `n` applicants spread over the PD ranking, riskiest first (deterministic)."""
    ordered = pd_hat.sort_values(ascending=False, kind="stable")
    positions = np.unique(np.round(np.linspace(0, len(ordered) - 1, n)).astype(int))
    return [ordered.index[p] for p in positions]


def agreement(pd_challenger, score_champion, y, top_share: float = 0.10) -> dict[str, float]:
    """How far the two models rank alike: Spearman rho of the risk ranks, and the overlap of their
    riskiest `top_share` of applicants (share of the challenger's riskiest also in the champion's)."""
    risk_challenger = np.asarray(pd_challenger, dtype=float)
    risk_champion = -np.asarray(score_champion, dtype=float)
    k = max(1, int(round(top_share * len(risk_challenger))))
    top_challenger = set(np.argsort(-risk_challenger, kind="stable")[:k])
    top_champion = set(np.argsort(-risk_champion, kind="stable")[:k])
    y = np.asarray(y)
    only_challenger = np.array(sorted(top_challenger - top_champion), dtype=int)
    only_champion = np.array(sorted(top_champion - top_challenger), dtype=int)
    return {
        "spearman": float(spearmanr(risk_challenger, risk_champion).statistic),
        "top_share": top_share,
        "top_overlap": len(top_challenger & top_champion) / k,
        "bad_rate_only_challenger_top": float(y[only_challenger].mean()) if only_challenger.size else float("nan"),
        "bad_rate_only_champion_top": float(y[only_champion].mean()) if only_champion.size else float("nan"),
    }


def _value(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "missing"
    if isinstance(v, (float, np.floating)):
        return f"{v:.4g}"
    return str(v).replace("|", r"\|")


def challenger_report_markdown(challenger: Challenger, payload: dict[str, Any], context: dict[str, Any]) -> str:
    """`context`: meta, performance, champion, agreement, psi, importance, examples - computed by the pipeline."""
    spec = challenger.spec
    meta = context["meta"]
    perf = context["performance"]
    champion = context["champion"]
    trials = challenger.trials
    best = trials[trials["selected"]].iloc[0]
    lines = [
        "# Challenger report - Home Credit PD, LightGBM",
        "",
        f"Run `{meta['run_id']}` | challenger fingerprint `{payload['fingerprint'][:12]}` | "
        f"champion `{meta['champion_fingerprint'][:12]}` | binning `{meta['binning_fingerprint'][:12]}` | "
        f"split `{meta['split_fingerprint'][:12]}`",
        "",
        "## 1. Mục đích",
        "",
        "- Challenger của scorecard: đo xem một mô hình phi tuyến trên cùng dữ liệu hơn scorecard bao nhiêu, và "
        "đổi lại mất gì về khả năng giải thích.",
        "- PD là đầu ra thô của LightGBM, **chưa calibrate** (chặng 5). Không dùng để ra quyết định tín dụng thật.",
        "",
        "## 2. Dữ liệu và đầu vào",
        "",
        f"- Fit trên **{meta['rows_train']:,}** hồ sơ train; early stopping và chọn cấu hình trên "
        f"{meta['rows_validation']:,} hồ sơ validation. Calibration và test **không được đọc**.",
        f"- **{len(challenger.features)} biến** của bảng feature Stage 2.3 (đã qua availability matrix), giá trị gốc, "
        f"không WoE: {len(challenger.categories)} biến categorical tách native, missing do LightGBM tự định tuyến. "
        "Category train chưa từng thấy được coi là missing.",
        "- Khác champion: challenger thấy mọi biến, kể cả các biến Stage 2.6 đã loại vì IV thấp, tương quan hay "
        "VIF. Chênh lệch hiệu năng vì vậy gồm cả phần do thêm biến, không chỉ do mô hình phi tuyến.",
        "",
        "## 3. Tuning",
        "",
        f"{len(trials)} cấu hình lấy ngẫu nhiên (seed {spec.seed}) từ lưới "
        + ", ".join(f"`{k}` {v}" for k, v in sorted(spec.search_space.items()))
        + f"; learning rate {spec.learning_rate:g}, early stopping {spec.early_stopping_rounds} vòng trên AUC "
        f"validation, {spec.num_threads} luồng, `deterministic`.",
        "",
        f"**Quy tắc chọn**: các cấu hình có AUC validation cách cấu hình tốt nhất không quá "
        f"{spec.selection_tolerance:g} coi như bằng nhau (chênh nhỏ hơn sai số lấy mẫu); trong nhóm đó chọn cấu hình "
        "**ít overfit nhất**, tức chênh AUC train - validation nhỏ nhất.",
        "",
        "| Trial | " + " | ".join(sorted(spec.search_space)) + " | Số cây | AUC train | AUC validation | Chênh | "
        "Trong ngưỡng |",
        "|---:|" + "---:|" * len(spec.search_space) + "---:|---:|---:|---:|---|",
    ]
    for r in trials.sort_values(["auc_validation", "trial"], ascending=[False, True]).itertuples(index=False):
        row = r._asdict()
        mark = " **(chọn)**" if row["selected"] else ""
        lines.append(f"| {row['trial']}{mark} | " + " | ".join(f"{row[k]:g}" for k in sorted(spec.search_space))
                     + f" | {row['best_iteration']} | {row['auc_train']:.4f} | {row['auc_validation']:.4f} | "
                     f"{row['gap']:.4f} | {'có' if row['within_tolerance'] else ''} |")
    spread = trials["auc_validation"].max() - trials["auc_validation"].min()
    top = trials.sort_values(["auc_validation", "trial"], ascending=[False, True]).iloc[0]
    lines += [
        "",
        f"Xếp theo AUC validation. {len(trials)} cấu hình cách nhau {spread:.4f} AUC: kết quả ít phụ thuộc cấu hình. "
        f"{int(trials['within_tolerance'].sum())} cấu hình nằm trong ngưỡng; trial {int(best['trial'])} được chọn "
        f"(AUC validation {best['auc_validation']:.4f}, {int(best['best_iteration'])} cây, chênh {best['gap']:.4f}) "
        f"thay vì trial {int(top['trial'])} có AUC cao nhất ({top['auc_validation']:.4f}, "
        f"{int(top['best_iteration'])} cây, chênh {top['gap']:.4f})."
        if int(top["trial"]) != int(best["trial"]) else
        f"Xếp theo AUC validation. {len(trials)} cấu hình cách nhau {spread:.4f} AUC. Trial {int(best['trial'])} vừa có "
        f"AUC cao nhất vừa ít overfit nhất trong ngưỡng; dùng {int(best['best_iteration'])} cây.",
        "",
        "## 4. So với champion",
        "",
        "| Mẫu | Mô hình | AUC | Gini | KS |",
        "|---|---|---:|---:|---:|",
    ]
    for sample in ("train", "validation"):
        c, m = champion[sample], perf[sample]
        lines.append(f"| {sample} | champion (điểm nguyên) | {c['auc']:.4f} | {c['gini']:.4f} | {c['ks']:.4f} |")
        lines.append(f"| {sample} | challenger | {m['auc']:.4f} | {m['gini']:.4f} | {m['ks']:.4f} |")
    gain = perf["validation"]["gini"] - champion["validation"]["gini"]
    agree = context["agreement"]
    lines += [
        "",
        f"- Challenger hơn champion **{gain:+.4f} Gini** trên validation. Con số này **lạc quan cho challenger**: "
        "validation đã dùng để dừng boosting và chọn cấu hình. So sánh công bằng là trên test, ở chặng 6.",
        f"- Chênh AUC train - validation: challenger {perf['train']['auc'] - perf['validation']['auc']:.4f}, "
        f"champion {champion['train']['auc'] - champion['validation']['auc']:.4f}.",
        f"- Hai mô hình xếp hạng giống nhau ở mức Spearman {agree['spearman']:.3f}. Trong {agree['top_share']:.0%} "
        f"hồ sơ rủi ro nhất theo challenger, {agree['top_overlap']:.0%} cũng nằm trong {agree['top_share']:.0%} rủi "
        "ro nhất theo champion. Bad rate của phần không trùng: chỉ challenger xếp vào nhóm này "
        f"{agree['bad_rate_only_challenger_top']:.2%}, chỉ champion xếp vào {agree['bad_rate_only_champion_top']:.2%}.",
        "",
        "## 5. Biến quan trọng nhất",
        "",
        "Trung bình |SHAP| trên validation (log-odds của bad) và tỷ trọng gain trong các lần tách.",
        "",
        "| Hạng | Biến | TB \\|SHAP\\| | Tỷ trọng gain | Trong scorecard |",
        "|---:|---|---:|---:|---|",
    ]
    importance = context["importance"]
    in_card = set(meta["champion_features"])
    for r in importance.head(20).itertuples(index=False):
        lines.append(f"| {r.rank} | `{r.feature}` | {r.mean_abs_shap:.4f} | {r.gain_share:.1%} | "
                     f"{'có' if r.feature in in_card else ''} |")
    unused = int((importance["gain_share"] == 0).sum())
    lines += [
        "",
        f"{unused} trong {len(importance)} biến không được dùng trong cây nào. Bảng đầy đủ: `challenger_importance.csv`.",
        "",
        "## 6. Thuộc tính nhạy cảm (PROJECT_SCOPE #7)",
        "",
        "Challenger không loại biến theo chính sách, nên cả ba thuộc tính đều là đầu vào:",
        "",
        "| Thuộc tính | Hạng theo TB \\|SHAP\\| | TB \\|SHAP\\| | Trong scorecard |",
        "|---|---:|---:|---|",
    ]
    by_feature = importance.set_index("feature")
    for f in SENSITIVE:
        if f in by_feature.index:
            r = by_feature.loc[f]
            lines.append(f"| `{f}` | {int(r['rank'])} / {len(importance)} | {r['mean_abs_shap']:.4f} | "
                         f"{'có' if f in in_card else 'không'} |")
    lines += [
        "",
        "`DAYS_BIRTH` đã bị loại khỏi scorecard vì dấu hệ số; challenger vẫn dùng nó với hình dạng tự do. Module 2 "
        "phải báo cáo hiệu năng theo giới tính, tình trạng hôn nhân và nhóm tuổi cho **cả hai** mô hình.",
        "",
        "## 7. Giải thích từng hồ sơ",
        "",
        f"Mã lý do: {spec.reason_codes} biến có đóng góp SHAP dương lớn nhất (đẩy rủi ro lên). Ví dụ "
        f"{len(context['examples'])} hồ sơ validation trải đều theo thứ hạng PD, rủi ro nhất trước:",
    ]
    for ex in context["examples"]:
        lines += [
            "",
            f"**Hồ sơ {ex['id']}** - PD {ex['pd']:.2%}, TARGET {ex['target']}. Log-odds = base "
            f"{ex['base']:+.3f} + tổng đóng góp {ex['total'] - ex['base']:+.3f} = {ex['total']:+.3f}.",
            "",
            "| Biến | Giá trị | Đóng góp (log-odds) |",
            "|---|---|---:|",
        ]
        for name, value, contribution in ex["top"]:
            lines.append(f"| `{name}` | {_value(value)} | {contribution:+.3f} |")
        lines.append(f"| {ex['n_rest']} biến còn lại | | {ex['rest']:+.3f} |")

    psi = context["psi"]
    lines += [
        "",
        "## 8. Giới hạn",
        "",
        "- **AUC validation lạc quan** (mục 4); **không có out-of-time thật**, như champion.",
        "- **Không ràng buộc đơn điệu**: một biến có thể làm rủi ro tăng rồi giảm ở những khoảng khác nhau, khó "
        "giải thích với khách hàng hơn bảng điểm.",
        "- **Mã lý do từ SHAP** giải thích đóng góp so với trung bình mẫu, không phải so với \"bin tốt nhất\" như "
        "scorecard: hai mô hình có thể nêu lý do khác nhau cho cùng một hồ sơ.",
        "- PSI của PD so với application_test: "
        + ", ".join(f"{'tất cả' if k == 'all' else k} {v:.4f}" for k, v in psi.items()) + ".",
        "",
    ]
    return "\n".join(lines)
