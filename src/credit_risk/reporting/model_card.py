"""Stage 3 model card for the champion scorecard: what it is, how well it ranks
on the samples development may look at, and what it must not be used for.

Only train and validation are evaluated here. Calibration is for stage 5 and
test for the independent validation of stage 6; neither is read.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from credit_risk.models.scorecard import Scorecard
from credit_risk.validation.metrics import auc, ks, psi

SENSITIVE = ("CODE_GENDER", "NAME_FAMILY_STATUS", "DAYS_BIRTH")


def discrimination(y, risk) -> dict[str, float]:
    """AUC, Gini and KS of a risk score (higher = riskier)."""
    a = auc(y, risk)
    return {"auc": a, "gini": 2 * a - 1, "ks": ks(y, risk).statistic}


def score_bands(score, y, pd_hat, n_bands: int = 10) -> pd.DataFrame:
    """Equal-count score bands (by rank, lowest score first) with the bad rate and mean PD of each."""
    s = pd.Series(np.asarray(score))
    band = pd.qcut(s.rank(method="first"), n_bands, labels=False) + 1
    frame = pd.DataFrame({"band": band, "score": s, "bad": np.asarray(y), "pd": np.asarray(pd_hat)})
    out = frame.groupby("band").agg(score_min=("score", "min"), score_max=("score", "max"),
                                    n=("bad", "size"), bad_rate=("bad", "mean"), pd_mean=("pd", "mean"))
    return out.reset_index()


def score_psi(reference, current, contract_ref, contract_cur, n_bins: int, epsilon: float) -> dict[str, float]:
    """Score PSI, overall and within each contract type (PROJECT_SCOPE #1: application_test has a different mix)."""
    out = {"all": psi(reference, current, n_bins, epsilon).value}
    for contract in sorted(set(contract_ref) & set(contract_cur)):
        ref = np.asarray(reference)[np.asarray(contract_ref) == contract]
        cur = np.asarray(current)[np.asarray(contract_cur) == contract]
        out[contract] = psi(ref, cur, n_bins, epsilon).value
    return out


def _cell(text: str) -> str:
    """Categorical bin labels join categories with '|', which would split a markdown cell."""
    return text.replace("|", r"\|")


def _fmt_share(x) -> str:
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.1%}"


def model_card_markdown(card: Scorecard, payload: dict[str, Any], context: dict[str, Any]) -> str:
    """`context`: performance, bands, psi, iv, contract_mix, splits, meta - all computed by the pipeline."""
    spec = card.spec
    meta = context["meta"]
    perf = context["performance"]
    lines = [
        "# Model card - Home Credit PD scorecard (champion)",
        "",
        f"Run `{meta['run_id']}` | scorecard fingerprint `{payload['fingerprint'][:12]}` | "
        f"binning `{meta['binning_fingerprint'][:12]}` | shortlist `{meta['shortlist_fingerprint'][:12]}` | "
        f"split `{meta['split_fingerprint'][:12]}`",
        "",
        "## 1. Mục đích",
        "",
        "- Xếp hạng rủi ro một hồ sơ vay tiêu dùng **tại ngày nộp đơn**; bad = `TARGET = 1` "
        "(khách gặp khó khăn trả nợ ở những kỳ đầu, theo định nghĩa của nguồn).",
        "- Đây là model của một portfolio project trên dữ liệu công khai, **không dùng để ra quyết định "
        "tín dụng thật**.",
        "- PD trong tài liệu này là PD **chưa calibrate**, suy ra từ thang điểm; calibration ở chặng 5.",
        "",
        "## 2. Dữ liệu",
        "",
        f"- Population: `application_train`, stratified random split 60 / 10 / 15 / 15 (seed {spec.seed}).",
        f"- Fit trên **{meta['rows_train']:,}** hồ sơ train (bad rate {meta['bad_rate_train']:.2%}); "
        f"đánh giá trên {meta['rows_validation']:,} hồ sơ validation.",
        "- Calibration (15%) và test (15%) **không được đọc** ở chặng này.",
        "",
        "## 3. Mô hình",
        "",
        f"- Hồi quy logistic trên WoE, L2 (C = {spec.C:g}), fit bằng scikit-learn.",
        f"- Đầu vào: {meta['shortlisted']} biến của shortlist Stage 2.6; **{len(card.features)} biến giữ lại** sau "
        "kiểm tra dấu hệ số (mục 5).",
        f"- Thang điểm: Score = Offset + Factor x ln(odds Good:Bad); BaseScore {spec.base_score:g} tại odds "
        f"{spec.base_odds:g}:1, PDO {spec.pdo:g} -> Factor {spec.factor:.4f}, Offset {spec.offset:.4f}.",
        f"- Điểm mỗi bin làm tròn thành số nguyên; điểm cơ sở (intercept) = **{card.base_points}**. "
        f"Score của một hồ sơ = điểm cơ sở + tổng điểm các biến.",
        f"- Mã lý do: {spec.reason_codes} biến làm hồ sơ mất nhiều điểm nhất so với bin tốt nhất của biến đó.",
        "",
        "## 4. Hiệu năng (development)",
        "",
        "| Mẫu | Mô hình | AUC | Gini | KS |",
        "|---|---|---:|---:|---:|",
    ]
    for sample in ("train", "validation"):
        for variant, label in (("exact", "log-odds chính xác"), ("points", "điểm nguyên")):
            m = perf[sample][variant]
            lines.append(f"| {sample} | {label} | {m['auc']:.4f} | {m['gini']:.4f} | {m['ks']:.4f} |")
    lines += [
        "",
        f"Làm tròn điểm làm Gini validation thay đổi "
        f"{perf['validation']['points']['gini'] - perf['validation']['exact']['gini']:+.4f}. "
        f"Trước khi loại biến theo dấu, {meta['shortlisted']} biến cho AUC validation "
        f"{meta['auc_validation_all_features']:.4f}.",
        "",
        "Score theo decile trên validation (band 1 = điểm thấp nhất = rủi ro cao nhất):",
        "",
        "| Band | Score | Hồ sơ | Bad rate | PD chưa calibrate (TB) |",
        "|---:|---|---:|---:|---:|",
    ]
    for r in context["bands"].itertuples(index=False):
        lines.append(f"| {r.band} | {r.score_min:.0f} - {r.score_max:.0f} | {r.n:,} | {r.bad_rate:.2%} | "
                     f"{r.pd_mean:.2%} |")

    trail = payload["dropped_for_sign"]
    rules = context["selection_rules"]
    lines += [
        "",
        "## 5. Chọn biến: tiêu chí và kết quả",
        "",
        "### 5.1 Tiêu chí giữ lại",
        "",
        "Một biến vào scorecard khi đạt **cả ba** tiêu chí:",
        "",
        "| # | Tiêu chí | Ngưỡng | Kiểm ở |",
        "|---:|---|---|---|",
        f"| 1 | Qua shortlist Stage 2.6 | IV train >= {rules['iv_min']:g}; không có cờ ngoài mẫu của 2.5; "
        f"\\|r\\| WoE <= {rules['max_abs_correlation']:g} với biến IV cao hơn; VIF <= {rules['max_vif']:g} | "
        "`features/shortlist.md` |",
        "| 2 | Hệ số trên WoE **âm** | < 0 (WoE = ln(%Good / %Bad): bin nhiều good hơn phải ít rủi ro hơn) | "
        "mô hình fit trên toàn bộ train |",
        f"| 3 | Dấu hệ số **ổn định** | âm trong >= {spec.min_sign_share:.0%} của {spec.bootstrap_resamples} lần "
        "fit lại trên bootstrap phân tầng của train | cùng vòng với tiêu chí 2 |",
        "",
        "Tiêu chí 2 và 3 được kiểm theo vòng: mỗi vòng fit lại với các biến còn lại, loại **một** biến kém "
        "ổn định nhất (tỷ lệ âm thấp nhất; hoà thì hệ số lớn hơn), rồi fit lại, đến khi mọi biến đều đạt. "
        "Loại từng biến một vì bỏ một biến có thể làm dấu của biến tương quan với nó đổi theo.",
        "",
        f"### 5.2 Kết quả: {len(card.features)} giữ, {len(trail)} loại trên {meta['shortlisted']} biến của shortlist",
        "",
        "| # | Biến | IV train | Hệ số (đủ biến) | Hệ số (cuối / lúc loại) | Âm trong bootstrap | Khoảng điểm | "
        "Kết quả | Lý do |",
        "|---:|---|---:|---:|---:|---:|---|---|---|",
    ]
    initial = context["initial_coefficients"]
    kept = sorted(card.features, key=lambda f: (-(max(card.points[f].values()) - min(card.points[f].values())), f))
    rows = []
    for f in kept:
        pts = card.points[f].values()
        rows.append((f, card.coefficients[f], payload["sign_share"][f], f"{min(pts)} .. {max(pts)}", "**Giữ**",
                     "đạt cả ba tiêu chí"))
    for r in trail:
        why = ("hệ số dương: ngược chiều với chính các bin của nó" if r["coefficient"] > 0
               else f"hệ số âm nhưng không ổn định (dưới {spec.min_sign_share:.0%})")
        rows.append((r["feature"], r["coefficient"], r["negative_share"], "", f"Loại (vòng {r['round']})", why))
    for i, (f, coef, share, pts, decision, why) in enumerate(rows, 1):
        lines.append(f"| {i} | `{f}` | {context['iv'].get(f, float('nan')):.4f} | {initial.get(f, float('nan')):+.4f} | "
                     f"{coef:+.4f} | {share:.0%} | {pts} | {decision} | {why} |")
    lines += [
        "",
        "- *Hệ số (đủ biến)*: vòng 1, khi cả shortlist cùng trong mô hình. *Hệ số (cuối / lúc loại)*: mô hình cuối "
        "với biến được giữ, vòng bị loại với biến bị loại. Tỷ lệ âm trong bootstrap đọc theo cùng mô hình đó.",
        "- Biến giữ xếp theo độ rộng khoảng điểm (biến ảnh hưởng score nhiều nhất lên trước); biến loại xếp "
        "theo vòng. Bảng điểm đầy đủ theo bin: `scorecard_table.csv`; lịch sử loại: `sign_trail.csv`.",
    ]

    lines += [
        "",
        "## 6. Thuộc tính nhạy cảm (PROJECT_SCOPE #7)",
        "",
        "Không loại theo chính sách: ba thuộc tính dưới đây vào mô hình theo cùng tiêu chí thống kê như mọi biến.",
        "",
        "| Thuộc tính | Trong scorecard | Điểm theo bin |",
        "|---|---|---|",
    ]
    dropped_names = {r["feature"]: r for r in trail}
    for f in SENSITIVE:
        if f in card.points:
            bins = "; ".join(f"{_cell(label)}: {p:+d}" for label, p in card.points[f].items()
                             if any(row["label"] == label and row["n"] > 0 for row in card.binnings[f].bins))
            lines.append(f"| `{f}` | có | {bins} |")
        elif f in dropped_names:
            lines.append(f"| `{f}` | không - bị loại ở vòng {dropped_names[f]['round']} vì dấu hệ số | |")
        else:
            lines.append(f"| `{f}` | không - không qua shortlist Stage 2 | |")
    lines += [
        "",
        "Hệ quả: scorecard cho điểm khác nhau theo các nhóm này. **Không phù hợp để dùng ở nơi luật cấm các "
        "thuộc tính này.** Module 2 bắt buộc báo cáo AUC, calibration và tỷ lệ bị từ chối theo giới tính, "
        "tình trạng hôn nhân và nhóm tuổi (`validation.yaml` -> `fairness_review`), kể cả khi thuộc tính "
        "không còn trong scorecard: biến khác vẫn có thể mang thông tin thay cho nó.",
        "",
        "## 7. Giới hạn",
        "",
        "- **Không có out-of-time thật**: Home Credit không có ngày nộp đơn; split là stratified random, nên "
        "hiệu năng ở đây lạc quan hơn khi dùng trên một giai đoạn sau.",
        "- **`application_test` có population khác train**: Revolving loans "
        f"{_fmt_share(context['contract_mix']['current'].get('Revolving loans'))} so với "
        f"{_fmt_share(context['contract_mix']['train'].get('Revolving loans'))} ở train. PSI của score so với "
        "application_test:",
        "",
        "| Phân khúc | PSI score |",
        "|---|---:|",
    ]
    for segment, value in context["psi"].items():
        lines.append(f"| {'tất cả' if segment == 'all' else segment} | {value:.4f} |")
    lines += [
        "",
        "- **PD chưa calibrate**: PD suy từ thang điểm chỉ đúng bằng odds mà logistic regression học trên train.",
        "- **Dữ liệu công khai, TARGET do nguồn định nghĩa**: không kiểm chứng được cửa sổ quan sát của nhãn.",
        "",
    ]
    return "\n".join(lines)
