"""Stage 6.6: how sure the discrimination figures are.

A model with AUC 0.74 and a 95% interval [0.71, 0.77] is a different thing
from one with AUC 0.74 and [0.65, 0.83]. The validator bootstraps:

    stratified     goods and bads are resampled separately, with replacement,
                   so every resample has the split's own number of bads;
    paired         both models are evaluated on the same resampled rows, so
                   the interval for "challenger minus champion" reflects the
                   two models' disagreement, not two independent noises;
    seeded         `validation.yaml -> seed` (2026, chosen to differ from the
                   development seed) through numpy's default_rng; the same
                   seed gives the same intervals on any machine;
    percentile     the interval is the (alpha/2, 1 - alpha/2) percentiles of
                   the bootstrap distribution, as configured.

Metrics: AUC, Gini (= 2 AUC - 1) and KS, computed from counts per distinct
risk value, which is exact and makes a resample cost O(n). Point estimates
are the same functions on the unresampled split and must equal Stage 6.3.

Also bootstrapped: the gap between the validation and the test split for
each model (independent resamples of the two splits), because the
development sample was used to choose features and configurations and
Stage 6.3 saw a lower Gini on test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from credit_risk.validation.metrics import _binary_target_and_score

MODELS = {"champion": ("score", -1.0), "challenger": ("pd_challenger", 1.0)}
METRICS = ("auc", "gini", "ks")


class UncertaintyError(ValueError):
    pass


@dataclass(frozen=True)
class UncertaintySpec:
    resamples: int = 1000
    confidence_level: float = 0.95
    method: str = "percentile"
    stratified: bool = True
    seed: int = 2026
    samples: tuple[str, ...] = ("validation", "test")

    @classmethod
    def from_config(cls, validation: dict[str, Any]) -> UncertaintySpec:
        raw = validation.get("uncertainty") or {}
        spec = cls(
            resamples=int(raw.get("bootstrap_resamples", 1000)),
            confidence_level=float(raw.get("confidence_level", 0.95)),
            method=str(raw.get("method", "percentile")),
            stratified=bool(raw.get("stratified", True)),
            seed=int(validation.get("seed", 2026)),
            samples=tuple(str(s) for s in raw.get("samples", ("validation", "test"))),
        )
        if spec.resamples < 2:
            raise UncertaintyError("uncertainty.bootstrap_resamples must be at least 2")
        if not 0 < spec.confidence_level < 1:
            raise UncertaintyError("uncertainty.confidence_level must be in (0, 1)")
        if spec.method != "percentile":
            raise UncertaintyError(f"uncertainty.method: only 'percentile' is implemented, got {spec.method!r}")
        if not spec.samples:
            raise UncertaintyError("uncertainty.samples is empty")
        return spec

    @property
    def alpha(self) -> float:
        return 1 - self.confidence_level


# ---------------------------------------------------------------------------
# Metrics from counts per distinct risk value
# ---------------------------------------------------------------------------

def _codes(risk: np.ndarray) -> tuple[np.ndarray, int]:
    """Ordinal code of each row's risk (0 = lowest distinct value) and the number of distinct values."""
    _, inverse = np.unique(risk, return_inverse=True)
    return inverse.astype(np.int64), int(inverse.max()) + 1


def auc_ks_from_counts(bad_counts: np.ndarray, good_counts: np.ndarray) -> tuple[float, float]:
    """AUC (ties one half) and KS from the number of bads and goods at each distinct risk value,
    ordered from lowest to highest risk."""
    n_bad, n_good = bad_counts.sum(), good_counts.sum()
    if n_bad == 0 or n_good == 0:
        raise UncertaintyError("a resample has no goods or no bads")
    cum_good = np.cumsum(good_counts)
    goods_below = cum_good - good_counts
    auc = float((bad_counts * (goods_below + good_counts / 2.0)).sum() / (n_bad * n_good))
    ks = float(np.max(np.abs(cum_good / n_good - np.cumsum(bad_counts) / n_bad)))
    return auc, ks


def point_estimates(y: np.ndarray, risk: np.ndarray) -> dict[str, float]:
    codes, n_codes = _codes(risk)
    auc, ks = auc_ks_from_counts(np.bincount(codes[y == 1], minlength=n_codes),
                                 np.bincount(codes[y == 0], minlength=n_codes))
    return {"auc": auc, "gini": 2 * auc - 1, "ks": ks}


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

@dataclass
class SplitBootstrap:
    split: str
    n: int
    bads: int
    estimates: dict[str, dict[str, float]]          # model -> metric -> point estimate
    draws: dict[str, dict[str, np.ndarray]]         # model -> metric -> resampled values


def bootstrap_split(y, risks: dict[str, np.ndarray], spec: UncertaintySpec, rng: np.random.Generator) -> SplitBootstrap:
    """Resample one split `spec.resamples` times; evaluate every model on the same rows each time."""
    y = np.asarray(y)
    for name, risk in risks.items():
        _binary_target_and_score(y, risk)
    y = y.astype(int)
    bad_idx, good_idx = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    coded = {name: _codes(risk) for name, risk in risks.items()}
    draws = {name: {m: np.empty(spec.resamples) for m in METRICS} for name in risks}
    for b in range(spec.resamples):
        if spec.stratified:
            rows_bad = rng.choice(bad_idx, size=len(bad_idx), replace=True)
            rows_good = rng.choice(good_idx, size=len(good_idx), replace=True)
        else:
            rows = rng.choice(len(y), size=len(y), replace=True)
            rows_bad, rows_good = rows[y[rows] == 1], rows[y[rows] == 0]
        for name, (codes, n_codes) in coded.items():
            auc, ks = auc_ks_from_counts(np.bincount(codes[rows_bad], minlength=n_codes),
                                         np.bincount(codes[rows_good], minlength=n_codes))
            draws[name]["auc"][b], draws[name]["gini"][b], draws[name]["ks"][b] = auc, 2 * auc - 1, ks
    estimates = {name: point_estimates(y, risk) for name, risk in risks.items()}
    return SplitBootstrap(split="", n=int(len(y)), bads=int(y.sum()), estimates=estimates, draws=draws)


def _interval(values: np.ndarray, spec: UncertaintySpec) -> tuple[float, float, float]:
    low, high = np.quantile(values, [spec.alpha / 2, 1 - spec.alpha / 2])
    return float(low), float(high), float(values.std(ddof=1))


def recompute_uncertainty(scored: pd.DataFrame, spec: UncertaintySpec,
                          models: dict[str, tuple[str, float]] = MODELS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(intervals, differences).

    intervals:    one row per (split, model, metric): estimate, ci_low, ci_high, se, n, bads.
    differences:  paired "challenger - champion" per split and metric, and the independent
                  "<sample A> - <sample B>" gap between configured samples per model and metric.
    """
    for column in ("TARGET", "split", *(c for c, _ in models.values())):
        if column not in scored.columns:
            raise UncertaintyError(f"scored frame lacks '{column}'")
    rng = np.random.default_rng(spec.seed)
    per_split: dict[str, SplitBootstrap] = {}
    for split in spec.samples:
        part = scored[scored["split"] == split]
        if part.empty:
            raise UncertaintyError(f"split {split} has no rows")
        y = part["TARGET"].to_numpy(dtype=float)
        if np.isnan(y).any():
            raise UncertaintyError(f"split {split} has applicants without TARGET")
        risks = {name: sign * part[column].to_numpy(dtype=float) for name, (column, sign) in models.items()}
        result = bootstrap_split(y.astype(int), risks, spec, rng)
        result.split = split
        per_split[split] = result

    rows = []
    for split, result in per_split.items():
        for model in models:
            for metric in METRICS:
                low, high, se = _interval(result.draws[model][metric], spec)
                rows.append({"split": split, "model": model, "metric": metric, "n": result.n, "bads": result.bads,
                             "estimate": result.estimates[model][metric], "ci_low": low, "ci_high": high, "se": se,
                             "half_width": (high - low) / 2, "resamples": spec.resamples})
    intervals = pd.DataFrame(rows)

    diffs = []
    names = list(models)
    if len(names) == 2:
        first, second = names
        for split, result in per_split.items():
            for metric in METRICS:
                d = result.draws[second][metric] - result.draws[first][metric]
                low, high, se = _interval(d, spec)
                diffs.append({"comparison": f"{second} - {first}", "split": split, "model": "both", "metric": metric,
                              "paired": True, "estimate": result.estimates[second][metric] - result.estimates[first][metric],
                              "ci_low": low, "ci_high": high, "se": se, "excludes_zero": bool(low > 0 or high < 0)})
    splits = list(per_split)
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            a, b = splits[i], splits[j]
            for model in models:
                for metric in METRICS:
                    d = per_split[a].draws[model][metric] - per_split[b].draws[model][metric]
                    low, high, se = _interval(d, spec)
                    diffs.append({"comparison": f"{a} - {b}", "split": f"{a}, {b}", "model": model, "metric": metric,
                                  "paired": False,
                                  "estimate": per_split[a].estimates[model][metric] - per_split[b].estimates[model][metric],
                                  "ci_low": low, "ci_high": high, "se": se, "excludes_zero": bool(low > 0 or high < 0)})
    return intervals, pd.DataFrame(diffs)


def uncertainty_checks(intervals: pd.DataFrame, discrimination: pd.DataFrame | None = None) -> pd.DataFrame:
    """Every interval contains its point estimate, and the estimate equals Stage 6.3's figure."""
    rows = []

    def add(check, claimed, recounted, source, tolerance):
        ok = claimed is not None and recounted is not None and abs(float(claimed) - float(recounted)) <= tolerance
        rows.append({"check": check, "claimed": claimed, "recounted": recounted, "source": source, "ok": bool(ok)})

    for r in intervals.itertuples(index=False):
        inside = r.ci_low <= r.estimate <= r.ci_high
        rows.append({"check": f"interval contains the estimate ({r.model}, {r.split}, {r.metric})",
                     "claimed": r.estimate, "recounted": f"[{r.ci_low:.6f}, {r.ci_high:.6f}]", "source": "recount", "ok": bool(inside)})
        if discrimination is not None:
            match = discrimination[(discrimination["model"] == r.model) & (discrimination["split"] == r.split)]
            if not match.empty:
                add(f"estimate equals Stage 6.3 ({r.model}, {r.split}, {r.metric})", float(match.iloc[0][r.metric]),
                    r.estimate, "discrimination", 1e-12)
    return pd.DataFrame(rows)


def uncertainty_markdown(intervals: pd.DataFrame, differences: pd.DataFrame, checks: pd.DataFrame,
                         spec: UncertaintySpec) -> str:
    pct = f"{spec.confidence_level:.0%}"
    lines = [f"## Độ bất định (bootstrap {'phân tầng' if spec.stratified else 'thường'}, "
             f"{spec.resamples:,} lần, seed {spec.seed}, khoảng {pct} percentile)", "",
             f"| Tập | Mô hình | Chỉ số | Ước lượng | Khoảng {pct} | ± | SE |", "|---|---|---|---:|---|---:|---:|"]
    for r in intervals.itertuples(index=False):
        lines.append(f"| {r.split} | {r.model} | {r.metric} | {r.estimate:.4f} | [{r.ci_low:.4f}, {r.ci_high:.4f}] | "
                     f"{r.half_width:.4f} | {r.se:.4f} |")
    lines += ["", "### Chênh lệch", "", f"| So sánh | Tập | Mô hình | Chỉ số | Chênh | Khoảng {pct} | Khác 0? |",
              "|---|---|---|---|---:|---|---|"]
    for r in differences.itertuples(index=False):
        kind = " (ghép đôi)" if r.paired else ""
        lines.append(f"| {r.comparison}{kind} | {r.split} | {r.model} | {r.metric} | {r.estimate:+.4f} | "
                     f"[{r.ci_low:+.4f}, {r.ci_high:+.4f}] | {'có' if r.excludes_zero else 'không'} |")
    failed = checks[~checks["ok"]]
    lines += ["", f"Kiểm tra: {len(checks)}; lệch: {len(failed)}."]
    return "\n".join(lines) + "\n"
