"""Map raw delinquency status codes to the buckets defined in definitions.yaml."""

from __future__ import annotations

import pandas as pd

CURRENT, DPD30, DPD60, DPD90PLUS = "current", "dpd30", "dpd60", "dpd90plus"
BUCKET_ORDER = [CURRENT, DPD30, DPD60, DPD90PLUS]


def month_index(period) -> pd.Series:
    """Months since year 0, so consecutive reporting months differ by exactly 1."""
    p = pd.to_datetime(pd.Series(period))
    return p.dt.year * 12 + p.dt.month


def status_to_bucket(status, codes: dict) -> pd.Series:
    """Freddie Mac status ("00", "01", ..., "RA") -> bucket name; unknown / NULL -> <NA>.

    `codes` is definitions["mortgage"]["delinquency_status_codes"].
    """
    s = pd.Series(status, dtype="string").str.strip()
    out = pd.Series(pd.NA, index=s.index, dtype="string")
    out[s.isin(codes["current"]).fillna(False).astype(bool)] = CURRENT
    out[s.isin(codes["dpd30"]).fillna(False).astype(bool)] = DPD30
    out[s.isin(codes["dpd60"]).fillna(False).astype(bool)] = DPD60
    numeric = pd.to_numeric(s, errors="coerce")
    out[(numeric >= codes["dpd90plus_min_numeric_code"]).fillna(False).astype(bool)] = DPD90PLUS
    out[s.eq(codes["reo_acquisition"]).fillna(False).astype(bool)] = DPD90PLUS
    return out
