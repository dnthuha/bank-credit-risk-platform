"""Stage 9.2: early-warning metrics on a rare event.

The event rate is about 0.1-1%, so ROC AUC alone says little: PR AUC is read
against the event rate (a random score's PR AUC), and the operating language
is the monthly list. Every month the operations team works the top `share` of
loans by score; for a set of months,

    capture   = events in the monthly lists / all events
    precision = events in the monthly lists / loans in the lists
    lift      = precision / event rate

Ranking is within each month (the list is rebuilt every month), ties broken by
loan_id so the same input always gives the same list. ROC AUC and Brier come
from the validation engine's own implementations.
"""

from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd

from credit_risk.validation.metrics import auc, brier

_TOP = re.compile(r"^(lift|capture|precision)_top_(\d+(?:\.\d+)?)pct$")


def average_precision(y, score) -> float:
    """Area under the precision-recall step curve; tied scores enter together."""
    y = np.asarray(y, dtype=int)
    s = np.asarray(score, dtype=float)
    if y.sum() == 0:
        raise ValueError("average precision needs at least one event")
    order = np.argsort(-s, kind="mergesort")
    s, y = s[order], y[order]
    ends = np.r_[np.nonzero(np.diff(s))[0], len(s) - 1]       # last row of each distinct score
    tp = np.cumsum(y)[ends]
    precision = tp / (ends + 1)
    recall = tp / y.sum()
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def monthly_rank(frame: pd.DataFrame, score: str, period: str = "period", key: str = "loan_id") -> tuple[np.ndarray, np.ndarray]:
    """(0-based rank within the month by score, ties by key; rows in that month), aligned with `frame`."""
    order = frame.sort_values([period, score, key], ascending=[True, False, True], kind="mergesort")
    rank = pd.Series(order.groupby(period, sort=False).cumcount().to_numpy(), index=order.index)
    n = pd.Series(order.groupby(period, sort=False)[score].transform("size").to_numpy(), index=order.index)
    return rank.reindex(frame.index).to_numpy(), n.reindex(frame.index).to_numpy()


def flags_from_rank(rank: np.ndarray, n: np.ndarray, share: float) -> np.ndarray:
    if not 0 < share < 1:
        raise ValueError(f"share must be in (0, 1), got {share}")
    return rank < np.ceil(share * n - 1e-9)


def monthly_flags(frame: pd.DataFrame, score: str, share: float, period: str = "period", key: str = "loan_id") -> np.ndarray:
    """True for the top ceil(share * n) rows of each month by score (ties by key)."""
    if not 0 < share < 1:
        raise ValueError(f"share must be in (0, 1), got {share}")
    return flags_from_rank(*monthly_rank(frame, score, period, key), share)


def list_performance(y, flagged) -> dict[str, float]:
    y, flagged = np.asarray(y, dtype=int), np.asarray(flagged, dtype=bool)
    events, rows, listed = int(y.sum()), len(y), int(flagged.sum())
    caught = int(y[flagged].sum())
    rate = events / rows if rows else math.nan
    precision = caught / listed if listed else math.nan
    return {"listed": listed, "caught": caught, "capture": caught / events if events else math.nan,
            "precision": precision, "lift": precision / rate if rate else math.nan}


def monthly_list_table(frame: pd.DataFrame, score: str, share: float, target: str = "y", key: str = "loan_id",
                       flagged: np.ndarray | None = None) -> pd.DataFrame:
    """One row per month: loans, events, listed, caught, capture, precision, lift."""
    flagged = monthly_flags(frame, score, share, key=key) if flagged is None else flagged
    rows = []
    for period, idx in frame.groupby("period", sort=True).indices.items():
        y = frame[target].to_numpy()[idx]
        rows.append({"period": period, "loans": len(idx), "events": int(y.sum()), **list_performance(y, flagged[idx])})
    return pd.DataFrame(rows)


def evaluate(frame: pd.DataFrame, score: str, metrics: list[str], target: str = "y", key: str = "loan_id") -> dict[str, float]:
    """The configured metrics on one sample (a split, a segment); `score` is a PD in [0, 1]."""
    y = frame[target].to_numpy(dtype=int)
    p = frame[score].to_numpy(dtype=float)
    events = int(y.sum())
    out: dict[str, float] = {"rows": len(y), "events": events, "event_rate": events / len(y) if len(y) else math.nan,
                             "mean_pd": float(p.mean()) if len(p) else math.nan}
    both = 0 < events < len(y)
    ranked = None                                             # one sort serves every top-k metric
    for name in metrics:
        if name == "roc_auc":
            out[name] = auc(y, p) if both else math.nan
        elif name == "pr_auc":
            out[name] = average_precision(y, p) if events else math.nan
        elif name == "brier":
            out[name] = brier(y, p) if both else math.nan
        elif m := _TOP.match(name):
            kind, pct = m.group(1), float(m.group(2))
            if events and ranked is None:
                ranked = monthly_rank(frame, score, key=key)
            out[name] = list_performance(y, flags_from_rank(*ranked, pct / 100))[kind] if events else math.nan
        else:
            raise ValueError(f"unknown EWS metric {name!r}")
    return out


def calibration_table(frame: pd.DataFrame, score: str, bins: int = 10, target: str = "y") -> pd.DataFrame:
    """Mean PD against the observed rate by PD decile of the sample."""
    q = pd.qcut(frame[score].rank(method="first"), bins, labels=False)
    g = frame.groupby(q)
    return pd.DataFrame({"decile": np.arange(1, bins + 1), "rows": g.size().to_numpy(),
                         "mean_pd": g[score].mean().to_numpy(), "observed_rate": g[target].mean().to_numpy()})
