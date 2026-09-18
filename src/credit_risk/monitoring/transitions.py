"""Roll rates and migration matrices from a loan-month panel."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from credit_risk.monitoring.buckets import month_index


def transition_matrix(
    panel: pd.DataFrame,
    state_col: str = "bucket",
    loan_col: str = "loan_id",
    period_col: str = "period",
    states: Sequence[str] | None = None,
    normalize: bool = True,
) -> pd.DataFrame:
    """One-month state transitions.

    Only pairs (month m, month m+1) of the same loan are counted: a gap in
    reporting breaks the pair instead of being bridged. Rows with an unknown
    state are dropped. With `normalize`, each row sums to 1 (a row with no
    observations stays NaN rather than pretending to be 0).
    """
    df = panel[[loan_col, period_col, state_col]].copy()
    df["_m"] = month_index(df[period_col]).to_numpy()
    if df.duplicated([loan_col, "_m"]).any():
        raise ValueError("panel has duplicated (loan, month) rows")

    following = df[[loan_col, "_m", state_col]].rename(columns={state_col: "to_state"})
    following["_m"] -= 1
    pairs = (
        df.rename(columns={state_col: "from_state"})
        .merge(following, on=[loan_col, "_m"], how="inner")
        .dropna(subset=["from_state", "to_state"])
    )
    counts = pd.crosstab(pairs["from_state"], pairs["to_state"])
    if states is not None:
        counts = counts.reindex(index=list(states), columns=list(states), fill_value=0)
    counts.index.name, counts.columns.name = "from_state", "to_state"
    if not normalize:
        return counts
    totals = counts.sum(axis=1)
    return counts.div(totals.where(totals > 0), axis=0)
