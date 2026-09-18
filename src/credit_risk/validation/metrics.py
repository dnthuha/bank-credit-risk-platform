"""Discrimination, calibration and stability metrics, implemented from scratch.

These are the validation engine's own implementations (numpy only), so Module 2
never relies on the library code Module 1 used during development.

Convention: `score` is a RISK score - higher means more likely to be bad (y = 1).
Pass PD, or the negative of scorecard points.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def _binary_target_and_score(y_true, score) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true)
    s = np.asarray(score, dtype=float)
    if y.ndim != 1 or y.shape != s.shape:
        raise ValueError("y_true and score must be 1-D arrays of the same length")
    if np.isnan(s).any():
        raise ValueError("score contains NaN")
    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("y_true must contain only 0 and 1")
    y = y.astype(int)
    if y.sum() == 0 or y.sum() == len(y):
        raise ValueError("y_true must contain both goods (0) and bads (1)")
    return y, s


def auc(y_true, score) -> float:
    """P(score of a random bad > score of a random good); ties count one half."""
    y, s = _binary_target_and_score(y_true, score)
    ranks = rankdata(s)  # average ranks handle ties
    n_bad = y.sum()
    n_good = len(y) - n_bad
    return float((ranks[y == 1].sum() - n_bad * (n_bad + 1) / 2) / (n_bad * n_good))


def gini(y_true, score) -> float:
    return 2.0 * auc(y_true, score) - 1.0


@dataclass(frozen=True)
class KSResult:
    statistic: float
    threshold: float  # score at which the gap is largest (cumulative: score <= threshold)


def ks(y_true, score) -> KSResult:
    """Maximum distance between the cumulative score distributions of goods and bads."""
    y, s = _binary_target_and_score(y_true, score)
    order = np.argsort(s, kind="mergesort")
    s_sorted, y_sorted = s[order], y[order]
    # Evaluate only at the last position of each distinct score, so ties move together.
    ends = np.r_[np.nonzero(np.diff(s_sorted))[0], len(s_sorted) - 1]
    cum_bad = np.cumsum(y_sorted)[ends] / y.sum()
    cum_good = np.cumsum(1 - y_sorted)[ends] / (len(y) - y.sum())
    gap = np.abs(cum_good - cum_bad)
    best = int(np.argmax(gap))
    return KSResult(float(gap[best]), float(s_sorted[ends[best]]))


def brier(y_true, pd_hat) -> float:
    y, p = _binary_target_and_score(y_true, pd_hat)
    if (p < 0).any() or (p > 1).any():
        raise ValueError("predicted probabilities must lie in [0, 1]")
    return float(np.mean((p - y) ** 2))


# ---------------------------------------------------------------------------
# Population Stability Index
# ---------------------------------------------------------------------------

MISSING_BIN = "missing"


def psi_from_proportions(reference_pct, current_pct, epsilon: float) -> float:
    """PSI = sum((cur - ref) * ln(cur / ref)); zero shares are replaced by epsilon."""
    ref = np.asarray(reference_pct, dtype=float)
    cur = np.asarray(current_pct, dtype=float)
    if ref.shape != cur.shape:
        raise ValueError("reference and current distributions must have the same bins")
    ref = np.where(ref == 0, epsilon, ref)
    cur = np.where(cur == 0, epsilon, cur)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def reference_quantile_edges(reference, n_bins: int) -> np.ndarray:
    """Bin edges from the reference sample only: [-inf, q1, ..., q_{n-1}, inf]."""
    ref = np.asarray(reference, dtype=float)
    ref = ref[~np.isnan(ref)]
    if ref.size == 0:
        raise ValueError("reference sample has no non-missing values")
    inner = np.unique(np.quantile(ref, np.linspace(0, 1, n_bins + 1)[1:-1]))
    return np.r_[-np.inf, inner, np.inf]


@dataclass(frozen=True)
class PSIResult:
    value: float
    table: pd.DataFrame
    edges: np.ndarray


def psi(reference, current, n_bins: int = 10, epsilon: float = 1e-4, edges=None) -> PSIResult:
    """PSI of `current` against `reference`.

    Edges are learned on the reference sample (or passed in) and reused as-is
    for the current sample; re-binning the current sample would bias PSI down.
    Missing values get their own bin.
    """
    edges = reference_quantile_edges(reference, n_bins) if edges is None else np.asarray(edges, float)
    inner = edges[1:-1]
    n_value_bins = len(edges) - 1

    def distribution(values) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(values, dtype=float)
        if x.size == 0:
            raise ValueError("cannot compute PSI on an empty sample")
        missing = np.isnan(x)
        idx = np.searchsorted(inner, x[~missing], side="right")
        counts = np.r_[np.bincount(idx, minlength=n_value_bins), missing.sum()]
        return counts, counts / x.size

    ref_counts, ref_pct = distribution(reference)
    cur_counts, cur_pct = distribution(current)
    labels = [f"[{edges[i]:.6g}, {edges[i + 1]:.6g})" for i in range(n_value_bins)] + [MISSING_BIN]

    table = pd.DataFrame(
        {
            "bin": labels,
            "reference_count": ref_counts,
            "reference_pct": ref_pct,
            "current_count": cur_counts,
            "current_pct": cur_pct,
        }
    )
    if ref_counts[-1] == 0 and cur_counts[-1] == 0:
        table = table.iloc[:-1]  # no missing values anywhere: drop the empty missing bin
    r = np.where(table["reference_pct"] == 0, epsilon, table["reference_pct"])
    c = np.where(table["current_pct"] == 0, epsilon, table["current_pct"])
    table["psi_contribution"] = (c - r) * np.log(c / r)
    return PSIResult(float(table["psi_contribution"].sum()), table.reset_index(drop=True), edges)
