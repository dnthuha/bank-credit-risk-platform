"""Stage 6.3: discrimination recomputed by the validator.

AUC, Gini and KS of both models on every split that has a target, from the
rescored outputs of `score_with_package` (never from a published file).

What is checked, beyond the numbers themselves
    two AUCs        the engine's rank-based AUC (validation.metrics) is
                    confirmed by a second, pair-counting implementation
                    written here; they must agree to 1e-12;
    Gini identity   Gini = 2 * AUC - 1 holds for every recomputed row, and
                    for every published pair it is compared with;
    KS location     the score (or PD) at which the gap between the
                    cumulative good and bad distributions is largest, with
                    the share of applicants and of bads at or below it;
    ties            scorecards give integer scores, so many applicants tie.
                    The share of tied pairs is reported because AUC counts
                    a tie as one half and a report should say so;
    published       when the development run's performance figures are
                    available (artifacts/<run>/train/*.json), each is
                    compared with the recomputed value within a tolerance.

Risk convention: higher risk = more likely bad. The scorecard's risk is
minus its score; the challenger's is its PD. Platt calibration is
monotone, so calibrated and raw PD rank identically and AUC does not care
which is used; KS thresholds are reported in the model's own units.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from credit_risk.validation.metrics import _binary_target_and_score, auc, ks

MODELS = {"champion": ("score", -1.0), "challenger": ("pd_challenger", 1.0)}   # column, sign to make it a risk
SPLITS_WITH_TARGET = ("train", "validation", "calibration", "test")
METRICS = ("auc", "gini", "ks")


class DiscriminationError(ValueError):
    pass


@dataclass(frozen=True)
class DiscriminationSpec:
    metrics: tuple[str, ...] = METRICS
    headline_samples: tuple[str, ...] = ("validation", "test")
    abs_tolerance_metric: float = 1e-6     # recomputed vs published
    agreement_tolerance: float = 1e-12     # two AUC implementations

    @classmethod
    def from_config(cls, validation: dict[str, Any]) -> DiscriminationSpec:
        raw = validation.get("discrimination") or {}
        metrics = tuple(str(m) for m in raw.get("metrics", METRICS))
        unknown = set(metrics) - set(METRICS)
        if unknown or not metrics:
            raise DiscriminationError(f"discrimination.metrics must be a non-empty subset of {list(METRICS)}, got {list(metrics)}")
        samples = tuple(str(s) for s in raw.get("samples", ("validation", "test")))
        unknown = set(samples) - set(SPLITS_WITH_TARGET)
        if unknown:
            raise DiscriminationError(f"discrimination.samples: unknown split {sorted(unknown)}")
        return cls(metrics=metrics, headline_samples=samples,
                   abs_tolerance_metric=float(raw.get("abs_tolerance_metric", 1e-6)))


# ---------------------------------------------------------------------------
# A second AUC, from pair counting, to confirm the engine's rank-based one
# ---------------------------------------------------------------------------

def auc_by_pairs(y_true, risk) -> tuple[float, float]:
    """AUC as (concordant + tied / 2) / (bads x goods), and the share of tied bad-good pairs.

    Sort by risk once; for each distinct risk value, the goods strictly below it are
    concordant with the bads at it, the goods at it tie. O(n log n), no ranks.
    """
    y, s = _binary_target_and_score(y_true, risk)
    order = np.argsort(s, kind="mergesort")
    s, y = s[order], y[order]
    ends = np.r_[np.nonzero(np.diff(s))[0], len(s) - 1]
    starts = np.r_[0, ends[:-1] + 1]
    cum_good = np.cumsum(1 - y)
    goods_below = np.where(starts > 0, cum_good[starts - 1], 0)
    goods_at = cum_good[ends] - goods_below
    bads_at = (ends - starts + 1) - goods_at
    n_bad, n_good = y.sum(), len(y) - y.sum()
    concordant = float((bads_at * goods_below).sum())
    tied = float((bads_at * goods_at).sum())
    pairs = float(n_bad * n_good)
    return (concordant + tied / 2) / pairs, tied / pairs


def ks_location(y_true, risk) -> dict[str, float]:
    """KS statistic with where it happens: the risk threshold and the shares at or below it."""
    y, s = _binary_target_and_score(y_true, risk)
    result = ks(y, s)
    below = s <= result.threshold
    return {
        "ks": result.statistic,
        "ks_threshold_risk": result.threshold,
        "share_at_or_below": float(below.mean()),
        "bads_at_or_below": float(y[below].sum() / y.sum()),
        "goods_at_or_below": float((1 - y[below]).sum() / (len(y) - y.sum())),
    }


# ---------------------------------------------------------------------------
# Recompute for both models on every split
# ---------------------------------------------------------------------------

def recompute_discrimination(scored: pd.DataFrame, spec: DiscriminationSpec,
                             models: dict[str, tuple[str, float]] = MODELS) -> pd.DataFrame:
    """One row per (model, split): n, bads, auc, gini, ks and where KS happens.

    `scored` holds TARGET, split and the model output columns (from score_with_package joined
    to the applicants). Rows without a split (application_test) have no target and are skipped.
    """
    for column in ("TARGET", "split", *(c for c, _ in models.values())):
        if column not in scored.columns:
            raise DiscriminationError(f"scored frame lacks '{column}'")
    rows = []
    for name, (column, sign) in models.items():
        for split in SPLITS_WITH_TARGET:
            part = scored[scored["split"] == split]
            if part.empty:
                continue
            y = part["TARGET"].to_numpy()
            if np.isnan(y.astype(float)).any():
                raise DiscriminationError(f"split {split} has applicants without TARGET")
            y = y.astype(int)
            risk = sign * part[column].to_numpy(dtype=float)
            a = auc(y, risk)
            a2, tied = auc_by_pairs(y, risk)
            if abs(a - a2) > spec.agreement_tolerance:
                raise DiscriminationError(f"{name}/{split}: the two AUC implementations disagree ({a} vs {a2})")
            loc = ks_location(y, risk)
            threshold_in_units = sign * loc["ks_threshold_risk"]   # back to score points or PD
            rows.append({
                "model": name, "split": split, "headline": split in spec.headline_samples,
                "n": int(len(y)), "bads": int(y.sum()), "bad_rate": float(y.mean()),
                "auc": a, "gini": 2 * a - 1, "auc_by_pairs": a2, "tied_pair_share": tied,
                "ks": loc["ks"], "ks_at": threshold_in_units,
                "ks_cut": ("score >= " if sign < 0 else "pd <= ") + (f"{threshold_in_units:.0f}" if sign < 0 else f"{threshold_in_units:.4f}"),
                "share_at_or_below_risk": loc["share_at_or_below"],
                "bads_at_or_below_risk": loc["bads_at_or_below"],
                "goods_at_or_below_risk": loc["goods_at_or_below"],
            })
    if not rows:
        raise DiscriminationError("no split with a target to evaluate")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Checks against what development published
# ---------------------------------------------------------------------------

def discrimination_checks(recomputed: pd.DataFrame, spec: DiscriminationSpec,
                          published: dict[str, dict[str, dict[str, float]]] | None = None) -> pd.DataFrame:
    """Gini identity on every row and, when given, published vs recomputed per metric.

    `published` is {model: {split: {"auc": .., "gini": .., "ks": ..}}}, as the development run
    wrote it (champion: the integer-points figures; challenger: its own).
    """
    rows = []

    def add(check, claimed, recounted, source, tolerance):
        ok = claimed is not None and recounted is not None and abs(float(claimed) - float(recounted)) <= tolerance
        rows.append({"check": check, "claimed": claimed, "recounted": recounted, "source": source, "ok": bool(ok)})

    for r in recomputed.itertuples(index=False):
        add(f"gini = 2*auc - 1 ({r.model}, {r.split})", 2 * r.auc - 1, r.gini, "recount", 1e-12)
        add(f"two AUC implementations agree ({r.model}, {r.split})", r.auc, r.auc_by_pairs, "recount",
            spec.agreement_tolerance)
    for model, by_split in (published or {}).items():
        for split, metrics in by_split.items():
            ours = recomputed[(recomputed["model"] == model) & (recomputed["split"] == split)]
            if ours.empty:
                continue
            row = ours.iloc[0]
            if "auc" in metrics and "gini" in metrics:
                add(f"published gini = 2*auc - 1 ({model}, {split})", 2 * metrics["auc"] - 1, metrics["gini"],
                    "development run", 1e-9)
            for metric in spec.metrics:
                if metric in metrics:
                    add(f"published {metric} ({model}, {split})", metrics[metric], float(row[metric]),
                        "development run", spec.abs_tolerance_metric)
    return pd.DataFrame(rows)


def discrimination_markdown(recomputed: pd.DataFrame, checks: pd.DataFrame) -> str:
    lines = ["## Sức phân biệt (tính lại)", "",
             "| Mô hình | Tập | Hồ sơ | Bad | AUC | Gini | KS | KS tại | Cặp hòa |",
             "|---|---|---:|---:|---:|---:|---:|---|---:|"]
    for r in recomputed.itertuples(index=False):
        mark = "**" if r.headline else ""
        lines.append(f"| {r.model} | {mark}{r.split}{mark} | {r.n:,} | {r.bads:,} | {r.auc:.4f} | {r.gini:.4f} | "
                     f"{r.ks:.4f} | {r.ks_cut} ({r.share_at_or_below_risk:.1%} hồ sơ ít rủi ro hơn, "
                     f"{r.bads_at_or_below_risk:.1%} bad) | {r.tied_pair_share:.1%} |")
    failed = checks[~checks["ok"]]
    lines += ["", f"Kiểm tra: {len(checks)}; lệch: {len(failed)}."]
    if not failed.empty:
        lines += ["", "| Kiểm tra | Công bố | Tính lại | Nguồn |", "|---|---:|---:|---|"]
        for r in failed.itertuples(index=False):
            lines.append(f"| {r.check} | {r.claimed:.6f} | {r.recounted:.6f} | {r.source} |")
    return "\n".join(lines) + "\n"


def published_from_train_artifacts(train_dir) -> dict[str, dict[str, dict[str, float]]]:
    """Read the development run's performance figures, if the artifacts are on this machine."""
    import json
    from pathlib import Path

    train_dir = Path(train_dir)
    out: dict[str, dict[str, dict[str, float]]] = {}
    champion = train_dir / "performance.json"
    if champion.exists():
        perf = json.loads(champion.read_text(encoding="utf-8")).get("performance", {})
        out["champion"] = {split: dict(v["points"]) for split, v in perf.items() if "points" in v}
    challenger = train_dir / "challenger_performance.json"
    if challenger.exists():
        perf = json.loads(challenger.read_text(encoding="utf-8")).get("performance", {})
        out["challenger"] = {split: dict(v) for split, v in perf.items()}
    return out
