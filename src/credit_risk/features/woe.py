"""Weight of Evidence and Information Value for already-binned features.

WoE_i = ln(%Good_i / %Bad_i)          (positive = bin richer in goods)
IV    = sum((%Good_i - %Bad_i) * WoE_i)

The table must be fitted on the training sample only, then applied elsewhere
with `woe_transform`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MISSING_BIN = "__missing__"


@dataclass(frozen=True)
class WoETable:
    table: pd.DataFrame  # one row per bin
    iv: float
    zero_count_adjusted: bool

    def mapping(self) -> dict:
        return dict(zip(self.table["bin"], self.table["woe"]))


def woe_from_counts(n_good, n_bad, zero_count_adjustment: float = 0.5):
    """WoE and IV contribution per bin, from good / bad counts.

    If any bin has zero goods or zero bads, `zero_count_adjustment` is added to
    the good and bad counts of every bin, so WoE stays finite and all bins are
    treated alike. Returns (woe, iv_contribution, pct_good, pct_bad, adjusted).
    """
    good = np.asarray(n_good, dtype=float)
    bad = np.asarray(n_bad, dtype=float)
    if good.sum() == 0 or bad.sum() == 0:
        raise ValueError("WoE needs both goods and bads in the sample")
    adjusted = bool(((good == 0) | (bad == 0)).any())
    if adjusted:
        good = good + zero_count_adjustment
        bad = bad + zero_count_adjustment
    pct_good = good / good.sum()
    pct_bad = bad / bad.sum()
    woe = np.log(pct_good / pct_bad)
    return woe, (pct_good - pct_bad) * woe, pct_good, pct_bad, adjusted


def _bin_labels(bins) -> pd.Series:
    s = pd.Series(bins, dtype="object")
    return s.where(s.notna(), MISSING_BIN)


def woe_iv_table(bins, y, zero_count_adjustment: float = 0.5) -> WoETable:
    """y: 1 = bad, 0 = good.

    If any bin has zero goods or zero bads, `zero_count_adjustment` is added to
    the good and bad counts of every bin, so WoE stays finite and all bins are
    treated alike.
    """
    labels = _bin_labels(bins)
    target = pd.Series(np.asarray(y))
    if len(labels) != len(target):
        raise ValueError("bins and y must have the same length")
    if not set(target.unique()).issubset({0, 1}):
        raise ValueError("y must contain only 0 and 1")

    grouped = (
        pd.DataFrame({"bin": labels.to_numpy(), "y": target.to_numpy()})
        .groupby("bin", sort=False)["y"]
        .agg(n="size", n_bad="sum")
    )
    grouped["n_good"] = grouped["n"] - grouped["n_bad"]
    woe, iv_contribution, pct_good, pct_bad, adjusted = woe_from_counts(
        grouped["n_good"], grouped["n_bad"], zero_count_adjustment
    )

    table = pd.DataFrame(
        {
            "bin": grouped.index,
            "n": grouped["n"].to_numpy(),
            "n_good": grouped["n_good"].to_numpy(),
            "n_bad": grouped["n_bad"].to_numpy(),
            "bad_rate": (grouped["n_bad"] / grouped["n"]).to_numpy(),
            "pct_good": pct_good,
            "pct_bad": pct_bad,
            "woe": woe,
            "iv_contribution": iv_contribution,
        }
    )
    return WoETable(table, float(iv_contribution.sum()), adjusted)


def woe_transform(bins, woe_table: WoETable) -> np.ndarray:
    """Replace each bin label by its fitted WoE. Unseen bins are an error, not a silent 0."""
    labels = _bin_labels(bins)
    mapping = woe_table.mapping()
    unseen = set(labels.unique()) - set(mapping)
    if unseen:
        raise ValueError(f"bins not present in the fitted WoE table: {sorted(map(str, unseen))}")
    return labels.map(mapping).to_numpy(dtype=float)
