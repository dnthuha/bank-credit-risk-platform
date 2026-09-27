"""Stage 5 calibration report: how far each model's PD was off, which method
fixed it and why, how the calibrated PD holds on validation, and what was frozen.
"""

from __future__ import annotations

from typing import Any

import numpy as np

LABEL = {"none": "không calibrate", "platt": "Platt", "isotonic": "isotonic"}


def _metrics_rows(rows: list[tuple[str, dict[str, float]]]) -> list[str]:
    out = ["| Mẫu / cách | PD TB | Bad rate | Brier | Log loss | Slope | Intercept |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for label, m in rows:
        out.append(f"| {label} | {m['mean_pd']:.2%} | {m['bad_rate']:.2%} | {m['brier']:.5f} | {m['log_loss']:.5f} | "
                   f"{m['slope']:.3f} | {m['intercept']:+.3f} |")
    return out


def calibration_report_markdown(context: dict[str, Any]) -> str:
    """`context`: meta, spec, models (per model: before, cv, chosen, params, after, reliability, by_contract,
    auc), anchors, package - all computed by the pipeline."""
    meta, spec = context["meta"], context["spec"]
    lines = [
        "# Calibration report - Home Credit PD",
        "",
        f"Run `{meta['run_id']}` | package fingerprint `{context['package']['fingerprint'][:12]}` | "
        f"champion `{meta['champion_fingerprint'][:12]}` | challenger `{meta['challenger_fingerprint'][:12]}`",
        "",
        "## 1. Mục đích",
        "",
        "AUC / Gini chỉ đo thứ hạng. Calibration kiểm tra và chỉnh để **PD đúng về mức**: trong nhóm hồ sơ được "
        "gán PD 5%, khoảng 5% phải thực sự trở thành bad. PD đúng mức là điều kiện để đặt ngưỡng duyệt, định giá "
        "theo rủi ro, hay tính tổn thất dự kiến.",
        "",
        "## 2. Dữ liệu",
        "",
        f"- Fit calibrator trên **{meta['rows_calibration']:,}** hồ sơ calibration (15%, bad rate "
        f"{meta['bad_rate_calibration']:.2%}), tập chưa mô hình nào đọc tới.",
        f"- Chọn phương pháp bằng {spec['cv_folds']}-fold cross-validation **bên trong** tập calibration.",
        f"- Kiểm tra độc lập trên {meta['rows_validation']:,} hồ sơ validation (không dùng để fit calibrator).",
        "- Test (15%) **không được đọc**: dành cho chặng 6.",
        "",
        "Chỉ số: **Brier** (sai số bình phương PD - kết quả, càng nhỏ càng tốt); **slope** (lý tưởng 1; dưới 1 là "
        "PD quá cực đoan, trên 1 là quá rụt rè); **intercept** với slope cố định 1 (lý tưởng 0; âm là PD trung bình "
        "quá cao, dương là quá thấp).",
    ]

    for number, (name, m) in enumerate(context["models"].items(), start=3):
        title = "Champion (scorecard)" if name == "champion" else "Challenger (LightGBM)"
        lines += ["", f"## {number}. {title}", "", "### Trước calibration", ""]
        lines += _metrics_rows([(sample, m["before"][sample]) for sample in ("train", "validation", "calibration")])
        lines += [
            "",
            "### Chọn phương pháp (cross-validation trong tập calibration)",
            "",
            "| Cách | Brier ngoài fold | Log loss ngoài fold | AUC ngoài fold | Trong ngưỡng | Chọn |",
            "|---|---:|---:|---:|---|---|",
        ]
        for r in m["cv"]:
            lines.append(f"| {LABEL[r['method']]} | {r['brier_oof']:.5f} | {r['log_loss_oof']:.5f} | {r['auc_oof']:.4f} | "
                         f"{'mốc so sánh' if r['method'] == 'none' else ('có' if r['within_tolerance'] else '')} | "
                         f"{'**chọn**' if r['chosen'] else ''} |")
        params = m["params"]
        if m["chosen"] == "platt":
            fitted = f"PD = sigmoid({params['a']:+.4f} + {params['b']:.4f} x log-odds gốc)"
        else:
            fitted = f"hàm bậc thang {len(params['x'])} mức"
        lines += [
            "",
            f"Chọn **{LABEL[m['chosen']]}**: {fitted}. Quy tắc: trong các phương pháp có Brier ngoài fold cách "
            f"phương pháp tốt nhất không quá {spec['selection_tolerance_brier']:g}, chọn phương pháp đơn giản hơn "
            "(Platt trước isotonic: 2 tham số, giữ nguyên thứ hạng).",
            "",
            "### Sau calibration, trên validation",
            "",
        ]
        lines += _metrics_rows([("trước", m["before"]["validation"]), ("sau", m["after"]["validation"])])
        lines += [
            "",
            f"AUC validation trước / sau: {m['auc']['before']:.4f} / {m['auc']['after']:.4f}.",
            "",
            "Reliability trên validation (10 nhóm bằng nhau theo PD đã calibrate, thấp nhất trước):",
            "",
            "| Nhóm | PD từ - đến | Hồ sơ | PD TB | Bad rate | Bad rate / PD |",
            "|---:|---|---:|---:|---:|---:|",
        ]
        for r in m["reliability"]:
            lines.append(f"| {r['band']} | {r['pd_min']:.2%} - {r['pd_max']:.2%} | {r['n']:,} | {r['mean_pd']:.2%} | "
                         f"{r['bad_rate']:.2%} | {r['ratio']:.2f} |")
        lines += ["", "Theo loại hợp đồng (validation):", "",
                  "| Loại hợp đồng | Hồ sơ | PD TB | Bad rate | Bad rate / PD |", "|---|---:|---:|---:|---:|"]
        for r in m["by_contract"]:
            lines.append(f"| {r['segment']} | {r['n']:,} | {r['mean_pd']:.2%} | {r['bad_rate']:.2%} | {r['ratio']:.2f} |")

    anchors = context["anchors"]
    number = 3 + len(context["models"])
    lines += [
        "",
        f"## {number}. Thang điểm scorecard sau calibration",
        "",
        "Bảng điểm không đổi; chỉ cách đọc điểm thành PD thay đổi.",
        "",
        "| Score | PD theo thang gốc | PD đã calibrate |",
        "|---:|---:|---:|",
    ]
    for r in anchors:
        lines.append(f"| {r['score']} | {r['pd_scaling']:.2%} | {r['pd_calibrated']:.2%} |")

    package = context["package"]
    lines += [
        "",
        f"## {number + 1}. Model package đã đóng băng",
        "",
        f"Thư mục `{package['folder']}`. Chặng 6 chỉ đọc gói này; mọi thay đổi sau đây làm đổi fingerprint gói "
        f"(`{package['fingerprint'][:12]}`).",
        "",
        "| File | SHA-256 |",
        "|---|---|",
    ]
    for name, digest in package["files"].items():
        lines.append(f"| `{name}` | `{digest[:16]}` |")
    lines += [
        "",
        f"## {number + 2}. Giới hạn",
        "",
        "- Calibration khớp với bad rate của **dữ liệu phát triển** (8.07%). Danh mục thật có thể có bad rate khác; "
        "khi đó PD cần hiệu chỉnh lại theo kỳ vọng dài hạn (central tendency), việc này ngoài phạm vi project.",
        "- **Không có out-of-time**: calibration đúng trên một mẫu ngẫu nhiên cùng giai đoạn với train.",
        "- Tập calibration có khoảng 3,700 bad: các nhóm PD cao có ít quan sát, bad rate / PD ở đó dao động nhiều hơn.",
        "",
    ]
    return "\n".join(lines)


def anchors_table(offset: float, factor: float, calibrator, scores=(500, 525, 550, 575, 600, 625, 650)) -> list[dict]:
    """PD at a few scores, before and after calibration. log-odds(bad) = -(score - Offset) / Factor."""
    x = -(np.asarray(scores, dtype=float) - offset) / factor
    return [{"score": int(s), "pd_scaling": float(1 / (1 + np.exp(-v))), "pd_calibrated": float(p)}
            for s, v, p in zip(scores, x, calibrator.apply(x))]
