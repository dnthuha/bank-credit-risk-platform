"""Stage 9.3: the watchlist for the scoring month, with reason codes.

Capacity rule (ews.alert): the operations team can work `capacity_share` of the
loans each month, so the list is the top ceil(capacity_share * n) loans of the
scoring month by the selected model's PD (ties by loan_id). Each listed loan
carries the `reason_codes` features that push its score up the most against an
average loan (TreeSHAP for LightGBM, coefficient minus train mean for the
logistic model), written as `feature=value`.

How well such a list works is measured on the labelled months by the same rule
(evaluation.monthly): capture, precision and lift of every monthly list.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from credit_risk.ews.metrics import monthly_flags

LIST_COLUMNS = ["rank", "loan_id", "period", "pd", "bucket_at_t", "fico_band", "ltv_band", "vintage_year", "upb"]


def _value(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "missing"
    if isinstance(v, (float, np.floating)):
        return f"{v:.4g}"
    return str(v)


def build_watchlist(frame: pd.DataFrame, model, share: float, reason_codes: int) -> pd.DataFrame:
    """`frame`: the scoring rows with their features. Returns the list, rank 1 = highest PD."""
    if frame.empty:
        raise ValueError("no scoring rows: nothing to put on the watchlist")
    scored = frame.copy()
    scored["pd"] = model.predict_pd(scored)
    scored["period"] = pd.to_datetime(scored["period"])
    listed = scored[monthly_flags(scored, "pd", share, key="loan_id")]
    listed = listed.sort_values(["pd", "loan_id"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
    listed.insert(0, "rank", np.arange(1, len(listed) + 1))
    contrib = model.contributions(listed)
    names = list(contrib.columns)
    values = contrib.to_numpy()
    by_name = np.argsort(names, kind="stable")                     # equal pushes: alphabetical, so stable
    order = by_name[np.argsort(-values[:, by_name], axis=1, kind="stable")]
    out = listed[LIST_COLUMNS].copy()
    for k in range(reason_codes):
        col, push = [], []
        for i, row_order in enumerate(order):
            j = row_order[k] if k < len(row_order) else None
            if j is not None and values[i, j] > 0:
                col.append(f"{names[j]}={_value(listed.iloc[i][names[j]])}")
                push.append(float(values[i, j]))
            else:
                col.append("")
                push.append(math.nan)
        out[f"reason_{k + 1}"] = col
        out[f"push_{k + 1}"] = push
    return out


def watchlist_summary(watchlist: pd.DataFrame, scoring_rows: int, share: float) -> dict:
    reasons = pd.concat([watchlist[c] for c in watchlist.columns if c.startswith("reason_")], ignore_index=True)
    reasons = reasons[reasons != ""].str.split("=").str[0].value_counts()
    return {
        "scoring_rows": int(scoring_rows), "listed": int(len(watchlist)), "capacity_share": share,
        "listed_share": len(watchlist) / scoring_rows if scoring_rows else math.nan,
        "expected_events": float(watchlist["pd"].sum()), "mean_pd": float(watchlist["pd"].mean()),
        "min_pd": float(watchlist["pd"].min()), "upb": float(watchlist["upb"].sum()),
        "by_bucket": watchlist["bucket_at_t"].value_counts().to_dict(),
        "by_fico_band": watchlist["fico_band"].value_counts().to_dict(),
        "top_reasons": {k: int(v) for k, v in reasons.head(10).items()},
    }
