"""Three-month early-warning label on a loan-month panel.

For each loan and observation month t:
  y = 1  if, within t+1 .. t+h, the loan enters 90+ DPD (RA included) or is
         liquidated (zero balance code in the liquidation group);
  y = 0  if the whole window is observed without that event, or the loan
         voluntarily pays off before any event;
  excluded (y = <NA>, exclusion_reason set) when the outcome is not knowable
         or the row is not part of the cohort.

Months are scanned in order t+1, t+2, t+3, and the first informative month
decides the label. This is the pandas reference implementation that the unit
tests pin down; a DuckDB version for the full panel must reproduce it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from credit_risk.monitoring.buckets import DPD90PLUS, month_index, status_to_bucket

INACTIVE_AT_T = "inactive_at_t"
ALREADY_90PLUS = "already_90plus_at_t"
STATUS_NOT_AVAILABLE = "status_not_available_at_t"
PRIOR_90PLUS = "prior_90plus_excluded"
INCOMPLETE_WINDOW = "incomplete_performance_window"


def build_ews_labels(
    panel: pd.DataFrame,
    definitions: dict,
    horizon: int | None = None,
    loan_col: str = "loan_id",
    period_col: str = "period",
    status_col: str = "delinquency_status",
    zero_balance_col: str = "zero_balance_code",
) -> pd.DataFrame:
    mortgage = definitions["mortgage"]
    zb_groups = mortgage["zero_balance_codes"]
    horizon = horizon or definitions["windows"]["performance_months"]

    df = panel[[loan_col, period_col, status_col, zero_balance_col]].copy()
    df["_m"] = month_index(df[period_col]).to_numpy()
    if df.duplicated([loan_col, "_m"]).any():
        raise ValueError("panel has duplicated (loan, month) rows")

    df["bucket"] = status_to_bucket(df[status_col], mortgage["delinquency_status_codes"]).to_numpy()
    zb = pd.Series(df[zero_balance_col], dtype="string").str.strip().replace("", pd.NA)
    df["_bad"] = (df["bucket"].eq(DPD90PLUS).fillna(False) | zb.isin(zb_groups["liquidation"]).fillna(False)).astype(bool)
    df["_payoff"] = zb.isin(zb_groups["voluntary_payoff"]).fillna(False).astype(bool)
    df["_terminal"] = zb.notna().astype(bool)
    df["_known"] = (df["bucket"].notna() | df["_bad"] | df["_terminal"]).astype(bool)

    first_terminal = df.loc[df["_terminal"], [loan_col, "_m"]].groupby(loan_col)["_m"].min()
    first_bad = df.loc[df["_bad"], [loan_col, "_m"]].groupby(loan_col)["_m"].min()
    terminated_before = df["_m"] > df[loan_col].map(first_terminal).fillna(np.inf)
    ever_bad_before = df["_m"] > df[loan_col].map(first_bad).fillna(np.inf)

    out = df[[loan_col, period_col, "bucket"]].rename(columns={"bucket": "bucket_at_t"}).copy()
    out["ever_90plus_before_t"] = ever_bad_before.to_numpy()
    y = pd.Series(pd.NA, index=df.index, dtype="Int8")
    reason = pd.Series(pd.NA, index=df.index, dtype="string")
    event_offset = pd.Series(pd.NA, index=df.index, dtype="Int8")

    # Cohort exclusions at t, in order of precedence.
    decided = pd.Series(False, index=df.index)
    exclusions_at_t = [
        (df["_terminal"] | terminated_before, INACTIVE_AT_T),
        (df["bucket"].eq(DPD90PLUS).fillna(False).astype(bool), ALREADY_90PLUS),
        (df["bucket"].isna(), STATUS_NOT_AVAILABLE),
    ]
    if not mortgage.get("redefault_after_cure_counts_as_event", True):
        exclusions_at_t.append((ever_bad_before, PRIOR_90PLUS))
    for mask, label in exclusions_at_t:
        hit = mask & ~decided
        reason[hit] = label
        decided |= hit

    # Scan the performance window month by month.
    future = df[[loan_col, "_m", "_bad", "_payoff", "_terminal", "_known"]]
    for k in range(1, horizon + 1):
        shifted = future.assign(_m=future["_m"] - k)
        month_k = df[[loan_col, "_m"]].merge(shifted, on=[loan_col, "_m"], how="left")
        present = month_k["_bad"].notna().to_numpy()
        bad = present & month_k["_bad"].fillna(False).astype(bool).to_numpy()
        payoff = present & month_k["_payoff"].fillna(False).astype(bool).to_numpy()
        other_exit = present & month_k["_terminal"].fillna(False).astype(bool).to_numpy() & ~bad & ~payoff
        unknown = present & ~month_k["_known"].fillna(False).astype(bool).to_numpy()

        open_rows = ~decided.to_numpy()
        is_event = open_rows & bad
        y[is_event] = 1
        event_offset[is_event] = k
        is_payoff = open_rows & ~bad & payoff
        y[is_payoff] = 0
        is_incomplete = open_rows & ~bad & ~payoff & (~present | other_exit | unknown)
        reason[is_incomplete] = INCOMPLETE_WINDOW
        decided |= pd.Series(is_event | is_payoff | is_incomplete, index=df.index)

    y[~decided] = 0
    out["y"] = y
    out["event_month_offset"] = event_offset
    out["exclusion_reason"] = reason
    return out.reset_index(drop=True)
