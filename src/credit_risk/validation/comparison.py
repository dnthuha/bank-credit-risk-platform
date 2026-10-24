"""Stage 7.2: champion against challenger, side by side and head to head.

Side by side   one table, both models, every headline figure the engine
               produced on the test split: discrimination with its interval,
               calibration, stability, fairness. Built from the other
               modules' tables, so it cannot disagree with them.

Head to head   computed here from the rescored applicants:
    agreement      Spearman rank correlation of the two risk orderings and
                   the overlap of their riskiest 10% (with the bad rate of
                   the applicants only one model puts there), as Stage 4
                   reported on the validation split; recomputed on both
                   validation (to compare with the published figure) and test;
    swap set       at the common reject rule (calibrated PD above the
                   configured threshold), every applicant falls in one of
                   four cells: both reject, both accept, champion-only
                   reject, challenger-only reject. Counts and bad rates per
                   cell say what changes if the challenger replaces the
                   champion: who it would newly decline, who it would newly
                   accept, and how risky those people actually were;
    equal volume   at the champion's reject rate, the challenger's bad rate
                   among the applicants it would approve, and vice versa: a
                   comparison that does not depend on the two PD scales.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata

MODELS = ("champion", "challenger")


class ComparisonError(ValueError):
    pass


@dataclass(frozen=True)
class ComparisonSpec:
    sample: str = "test"
    agreement_samples: tuple[str, ...] = ("validation", "test")
    top_share: float = 0.10
    reject_pd_above: float = 0.10
    abs_tolerance_metric: float = 1e-6

    @classmethod
    def from_config(cls, validation: dict[str, Any]) -> ComparisonSpec:
        opt = validation.get("segment_options") or {}
        rule = opt.get("reject_rule") or {}
        cmp = validation.get("comparison") or {}
        spec = cls(sample=str((validation.get("conclusion_rules") or {}).get("headline_sample", "test")),
                   agreement_samples=tuple(str(s) for s in cmp.get("agreement_samples", ("validation", "test"))),
                   top_share=float(cmp.get("top_share", 0.10)),
                   reject_pd_above=float(rule.get("pd_above", 0.10)),
                   abs_tolerance_metric=float(cmp.get("abs_tolerance_metric", 1e-6)))
        if not 0 < spec.top_share < 1 or not 0 < spec.reject_pd_above < 1:
            raise ComparisonError("comparison: top_share and reject_rule.pd_above must be in (0, 1)")
        return spec


# ---------------------------------------------------------------------------
# Head to head
# ---------------------------------------------------------------------------

def spearman(a, b) -> float:
    """Spearman rho as Pearson correlation of average ranks (own implementation, handles ties)."""
    ra, rb = rankdata(np.asarray(a, dtype=float)), rankdata(np.asarray(b, dtype=float))
    ra, rb = ra - ra.mean(), rb - rb.mean()
    denominator = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    if denominator == 0:
        raise ComparisonError("a constant ranking has no rank correlation")
    return float((ra * rb).sum() / denominator)


def agreement(score_champion, pd_challenger, y, top_share: float) -> dict[str, float]:
    """Rank agreement and overlap of the riskiest `top_share`, as Stage 4 defined it
    (ties in the integer score broken by row order, stable)."""
    risk_champion = -np.asarray(score_champion, dtype=float)
    risk_challenger = np.asarray(pd_challenger, dtype=float)
    y = np.asarray(y, dtype=float)
    k = max(1, int(round(top_share * len(y))))
    top_champion = set(np.argsort(-risk_champion, kind="stable")[:k])
    top_challenger = set(np.argsort(-risk_challenger, kind="stable")[:k])
    only_champion = np.array(sorted(top_champion - top_challenger), dtype=int)
    only_challenger = np.array(sorted(top_challenger - top_champion), dtype=int)
    both = np.array(sorted(top_champion & top_challenger), dtype=int)
    return {
        "n": int(len(y)), "top_share": top_share, "top_k": k,
        "spearman": spearman(risk_champion, risk_challenger),
        "top_overlap": len(both) / k,
        "bad_rate_both_top": float(y[both].mean()) if both.size else float("nan"),
        "bad_rate_only_champion_top": float(y[only_champion].mean()) if only_champion.size else float("nan"),
        "bad_rate_only_challenger_top": float(y[only_challenger].mean()) if only_challenger.size else float("nan"),
        "n_only_each": int(len(only_champion)),
    }


def swap_set(y, pd_champion, pd_challenger, threshold: float) -> pd.DataFrame:
    """Four cells at the common reject rule, with count, share and bad rate."""
    y = np.asarray(y, dtype=float)
    reject_c = np.asarray(pd_champion, dtype=float) > threshold
    reject_g = np.asarray(pd_challenger, dtype=float) > threshold
    cells = [("both reject", reject_c & reject_g), ("champion only rejects (challenger accepts)", reject_c & ~reject_g),
             ("challenger only rejects (champion accepts)", ~reject_c & reject_g), ("both accept", ~reject_c & ~reject_g)]
    rows = []
    for name, mask in cells:
        n = int(mask.sum())
        rows.append({"cell": name, "n": n, "share": n / len(y), "bads": int(y[mask].sum()),
                     "bad_rate": float(y[mask].mean()) if n else float("nan")})
    return pd.DataFrame(rows)


def decision_outcomes(y, pd_hat, threshold: float) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    reject = np.asarray(pd_hat, dtype=float) > threshold
    return {"reject_rate": float(reject.mean()),
            "bad_rate_approved": float(y[~reject].mean()) if (~reject).any() else float("nan"),
            "bads_caught_share": float(y[reject].sum() / y.sum()) if y.sum() else float("nan")}


def at_equal_volume(y, pd_reference, pd_other, threshold: float) -> dict[str, float]:
    """Reject with `pd_other` exactly as many applicants as `pd_reference` rejects at `threshold`."""
    y = np.asarray(y, dtype=float)
    k = int((np.asarray(pd_reference, dtype=float) > threshold).sum())
    order = np.argsort(-np.asarray(pd_other, dtype=float), kind="stable")
    reject = np.zeros(len(y), dtype=bool)
    reject[order[:k]] = True
    return {"rejected": k, "reject_rate": k / len(y),
            "bad_rate_approved": float(y[~reject].mean()) if (~reject).any() else float("nan"),
            "bads_caught_share": float(y[reject].sum() / y.sum()) if y.sum() else float("nan")}


def head_to_head(scored: pd.DataFrame, spec: ComparisonSpec) -> dict[str, Any]:
    for column in ("TARGET", "split", "score", "pd_champion", "pd_challenger"):
        if column not in scored.columns:
            raise ComparisonError(f"scored frame lacks '{column}'")
    out: dict[str, Any] = {"agreement": {}, "sample": spec.sample, "reject_pd_above": spec.reject_pd_above}
    for sample in spec.agreement_samples:
        part = scored[scored["split"] == sample]
        if part.empty:
            continue
        out["agreement"][sample] = agreement(part["score"], part["pd_challenger"], part["TARGET"], spec.top_share)
    part = scored[scored["split"] == spec.sample]
    if part.empty:
        raise ComparisonError(f"split {spec.sample} has no rows")
    y = part["TARGET"].to_numpy(dtype=float)
    if np.isnan(y).any():
        raise ComparisonError(f"split {spec.sample} has applicants without TARGET")
    pc, pg = part["pd_champion"].to_numpy(dtype=float), part["pd_challenger"].to_numpy(dtype=float)
    out["swap_set"] = swap_set(y, pc, pg, spec.reject_pd_above)
    out["decisions"] = {"champion": decision_outcomes(y, pc, spec.reject_pd_above),
                        "challenger": decision_outcomes(y, pg, spec.reject_pd_above)}
    out["equal_volume"] = {"challenger_at_champion_volume": at_equal_volume(y, pc, pg, spec.reject_pd_above),
                           "champion_at_challenger_volume": at_equal_volume(y, pg, pc, spec.reject_pd_above)}
    return out


# ---------------------------------------------------------------------------
# Side by side, from the engine's tables
# ---------------------------------------------------------------------------

def side_by_side(discrimination: pd.DataFrame, intervals: pd.DataFrame, calibration: pd.DataFrame,
                 stability: pd.DataFrame, segments: pd.DataFrame, sample: str) -> pd.DataFrame:
    rows = []

    def ci(model, metric):
        r = intervals[(intervals["model"] == model) & (intervals["split"] == sample) & (intervals["metric"] == metric)]
        return (float(r["ci_low"].iloc[0]), float(r["ci_high"].iloc[0])) if not r.empty else (np.nan, np.nan)

    for model in MODELS:
        d = discrimination[(discrimination["model"] == model) & (discrimination["split"] == sample)].iloc[0]
        c = calibration[(calibration["model"] == model) & (calibration["split"] == sample)].iloc[0]
        s = stability[(stability["kind"] == "output") & (stability["model"] == model) & (stability["current"] == "application_test") & (stability["segment"] == "all")]
        f = stability[(stability["kind"] == "feature") & (stability["model"] == model) & (stability["current"] == "application_test")]
        fair = segments[(segments["fairness_review"]) & (segments["group"] != "(all)") & (segments["model"] == model) & (segments["sample"] == sample)]
        row = {"model": model, "n": int(d["n"]), "bads": int(d["bads"])}
        for metric in ("auc", "gini", "ks"):
            lo, hi = ci(model, metric)
            row[metric] = float(d[metric]); row[f"{metric}_ci"] = f"[{lo:.4f}, {hi:.4f}]"
        row.update({"ks_cut": d["ks_cut"], "brier": float(c["brier"]), "slope": float(c["slope"]), "slope_se": float(c["slope_se"]),
                    "ratio_bad_rate_to_pd": float(c["ratio_bad_rate_to_pd"]), "bands_flagged": int(c["bands_flagged"]),
                    "hl_p_value": float(c["hl_p_value"]),
                    "output_psi": float(s["psi"].iloc[0]) if not s.empty else np.nan,
                    "features_red": int((f["light"] == "red").sum()), "features_amber": int((f["light"] == "amber").sum()),
                    "sensitive_groups_flagged": int((fair["calibration_flag"].fillna("") != "").sum()),
                    "min_air": float(fair["adverse_impact_ratio"].min()) if fair["adverse_impact_ratio"].notna().any() else np.nan,
                    "min_group_auc_gap": float(fair["auc_gap_vs_all"].min()) if fair["auc_gap_vs_all"].notna().any() else np.nan})
        rows.append(row)
    return pd.DataFrame(rows)


def comparison_checks(h2h: dict[str, Any], spec: ComparisonSpec, n_sample: int,
                      published_agreement: dict[str, float] | None = None) -> pd.DataFrame:
    rows = []

    def add(check, claimed, recounted, source, tolerance):
        ok = claimed is not None and recounted is not None and abs(float(claimed) - float(recounted)) <= tolerance
        rows.append({"check": check, "claimed": claimed, "recounted": recounted, "source": source, "ok": bool(ok)})

    swap = h2h["swap_set"].set_index("cell")
    add("swap-set cells cover the sample", n_sample, int(swap["n"].sum()), "recount", 0)
    add("champion rejects = both reject + champion only", round(h2h["decisions"]["champion"]["reject_rate"] * n_sample),
        int(swap.loc["both reject", "n"] + swap.loc["champion only rejects (challenger accepts)", "n"]), "recount", 0)
    add("challenger rejects = both reject + challenger only", round(h2h["decisions"]["challenger"]["reject_rate"] * n_sample),
        int(swap.loc["both reject", "n"] + swap.loc["challenger only rejects (champion accepts)", "n"]), "recount", 0)
    if published_agreement and "validation" in h2h["agreement"]:
        mine = h2h["agreement"]["validation"]
        for key in ("spearman", "top_overlap", "bad_rate_only_challenger_top", "bad_rate_only_champion_top"):
            if key in published_agreement:
                add(f"published agreement {key} (validation)", published_agreement[key], mine[key], "development run", spec.abs_tolerance_metric)
    return pd.DataFrame(rows)


def published_agreement_from_train_artifacts(train_dir) -> dict[str, float] | None:
    import json
    from pathlib import Path

    path = Path(train_dir) / "challenger_performance.json"
    if not path.exists():
        return None
    return dict(json.loads(path.read_text(encoding="utf-8")).get("agreement", {})) or None


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _p(v):
    return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.2%}"


def comparison_markdown(table: pd.DataFrame, h2h: dict[str, Any], checks: pd.DataFrame, spec: ComparisonSpec) -> str:
    c, g = table.set_index("model").loc["champion"], table.set_index("model").loc["challenger"]
    lines = [f"## So sánh hai mô hình ({spec.sample})", "", "| | Champion | Challenger |", "|---|---:|---:|",
             f"| AUC [95%] | {c['auc']:.4f} {c['auc_ci']} | {g['auc']:.4f} {g['auc_ci']} |",
             f"| Gini [95%] | {c['gini']:.4f} {c['gini_ci']} | {g['gini']:.4f} {g['gini_ci']} |",
             f"| KS [95%], tại | {c['ks']:.4f} {c['ks_ci']}, {c['ks_cut']} | {g['ks']:.4f} {g['ks_ci']}, {g['ks_cut']} |",
             f"| Brier | {c['brier']:.5f} | {g['brier']:.5f} |",
             f"| Slope (SE) | {c['slope']:.3f} ({c['slope_se']:.3f}) | {g['slope']:.3f} ({g['slope_se']:.3f}) |",
             f"| Bad rate / PD | {c['ratio_bad_rate_to_pd']:.3f} | {g['ratio_bad_rate_to_pd']:.3f} |",
             f"| Band PD bị cờ, HL p | {c['bands_flagged']}, {c['hl_p_value']:.3f} | {g['bands_flagged']}, {g['hl_p_value']:.3f} |",
             f"| PSI đầu ra vs application_test | {c['output_psi']:.4f} | {g['output_psi']:.4f} |",
             f"| Biến top-5 đỏ / vàng | {c['features_red']} / {c['features_amber']} | {g['features_red']} / {g['features_amber']} |",
             f"| Nhóm nhạy cảm lệch hiệu chỉnh | {c['sensitive_groups_flagged']} | {g['sensitive_groups_flagged']} |",
             f"| AIR thấp nhất | {c['min_air']:.2f} | {g['min_air']:.2f} |",
             f"| Lệch AUC nhóm lớn nhất | {c['min_group_auc_gap']:+.3f} | {g['min_group_auc_gap']:+.3f} |"]
    lines += ["", "### Hai mô hình xếp hạng giống nhau đến đâu", "",
              "| Tập | Spearman | Trùng top 10% | Bad rate: cả hai top / chỉ champion / chỉ challenger |", "|---|---:|---:|---|"]
    for sample, a in h2h["agreement"].items():
        lines.append(f"| {sample} | {a['spearman']:.4f} | {a['top_overlap']:.1%} | {_p(a['bad_rate_both_top'])} / "
                     f"{_p(a['bad_rate_only_champion_top'])} / {_p(a['bad_rate_only_challenger_top'])} ({a['n_only_each']:,} hồ sơ mỗi bên) |")
    lines += ["", f"### Hoán đổi quyết định tại PD > {spec.reject_pd_above:.0%}", "",
              "| Ô | Hồ sơ | Tỷ trọng | Bad | Bad rate |", "|---|---:|---:|---:|---:|"]
    for r in h2h["swap_set"].itertuples(index=False):
        lines.append(f"| {r.cell} | {r.n:,} | {r.share:.1%} | {r.bads:,} | {_p(r.bad_rate)} |")
    d, e = h2h["decisions"], h2h["equal_volume"]
    lines += ["", "| Quyết định | Tỷ lệ từ chối | Bad rate trong nhóm duyệt | Bad bị chặn |", "|---|---:|---:|---:|",
              f"| Champion tại ngưỡng | {d['champion']['reject_rate']:.1%} | {_p(d['champion']['bad_rate_approved'])} | {_p(d['champion']['bads_caught_share'])} |",
              f"| Challenger tại ngưỡng | {d['challenger']['reject_rate']:.1%} | {_p(d['challenger']['bad_rate_approved'])} | {_p(d['challenger']['bads_caught_share'])} |",
              f"| Challenger, cùng số từ chối với champion | {e['challenger_at_champion_volume']['reject_rate']:.1%} | "
              f"{_p(e['challenger_at_champion_volume']['bad_rate_approved'])} | {_p(e['challenger_at_champion_volume']['bads_caught_share'])} |",
              f"| Champion, cùng số từ chối với challenger | {e['champion_at_challenger_volume']['reject_rate']:.1%} | "
              f"{_p(e['champion_at_challenger_volume']['bad_rate_approved'])} | {_p(e['champion_at_challenger_volume']['bads_caught_share'])} |"]
    failed = checks[~checks["ok"]]
    lines += ["", f"Kiểm tra: {len(checks)}; lệch: {len(failed)}."]
    return "\n".join(lines) + "\n"
