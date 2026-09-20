"""Population split for Module 1 (Home Credit).

Assigned once, written to a table, and reused by every later step, so that the
scorecard, the LightGBM challenger and the independent validation all work on
exactly the same population.

Home Credit has no application date (see definitions.yaml), so the split is
stratified random rather than out-of-time. The assignment is deterministic:
rows are sorted by id first, so the result depends only on the ids, the strata
and the seed - never on the order in which rows arrive.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

SPLIT_COLUMN = "split"
SUPPORTED_METHODS = {"stratified_random"}
_FRACTION_TOLERANCE = 1e-9


class SplitError(ValueError):
    pass


@dataclass(frozen=True)
class SplitSpec:
    id_column: str
    stratify_on: str | None
    fractions: dict[str, float]
    seed: int

    @classmethod
    def from_config(cls, model_dev: dict[str, Any]) -> SplitSpec:
        split = model_dev.get("split") or {}
        method = split.get("method")
        if method not in SUPPORTED_METHODS:
            raise SplitError(f"split.method must be one of {sorted(SUPPORTED_METHODS)}, got {method!r}")
        fractions = split.get("fractions") or {}
        if not fractions:
            raise SplitError("split.fractions is empty")
        if any(v <= 0 for v in fractions.values()):
            raise SplitError(f"split.fractions must all be > 0: {fractions}")
        total = sum(fractions.values())
        if abs(total - 1.0) > _FRACTION_TOLERANCE:
            raise SplitError(f"split.fractions must sum to 1, got {total}")
        id_column = split.get("id_column")
        if not id_column:
            raise SplitError("split.id_column is required")
        return cls(
            id_column=id_column,
            stratify_on=split.get("stratify_on"),
            fractions={str(k): float(v) for k, v in fractions.items()},
            seed=int(model_dev.get("seed", 0)),
        )

    @property
    def names(self) -> list[str]:
        return list(self.fractions)


def _counts(n: int, fractions: dict[str, float]) -> dict[str, int]:
    """Largest remainder: floor every share, then hand out the leftover rows."""
    exact = {name: n * share for name, share in fractions.items()}
    counts = {name: int(np.floor(value)) for name, value in exact.items()}
    order = sorted(fractions, key=lambda name: (-(exact[name] - counts[name]), list(fractions).index(name)))
    for name in order[: n - sum(counts.values())]:
        counts[name] += 1
    return counts


def assign_splits(frame: pd.DataFrame, spec: SplitSpec) -> pd.DataFrame:
    """Return one row per id: [id_column, split]."""
    needed = [spec.id_column] + ([spec.stratify_on] if spec.stratify_on else [])
    missing = [c for c in needed if c not in frame.columns]
    if missing:
        raise SplitError(f"columns missing from the population: {missing}")

    df = frame.loc[:, needed]
    if df[spec.id_column].isna().any():
        raise SplitError(f"{spec.id_column} contains nulls")
    if df[spec.id_column].duplicated().any():
        raise SplitError(f"{spec.id_column} is not unique; the split needs one row per id")
    if spec.stratify_on and df[spec.stratify_on].isna().any():
        raise SplitError(f"{spec.stratify_on} contains nulls; stratification would be undefined")
    if df.empty:
        raise SplitError("the population is empty")

    df = df.sort_values(spec.id_column, kind="mergesort").reset_index(drop=True)
    rng = np.random.default_rng(spec.seed)

    parts = []
    groups = (
        [(value, group) for value, group in df.groupby(spec.stratify_on, sort=True)]
        if spec.stratify_on
        else [(None, df)]
    )
    for _, group in groups:
        counts = _counts(len(group), spec.fractions)
        labels = np.repeat(list(counts), list(counts.values()))
        assigned = np.empty(len(group), dtype=object)
        assigned[rng.permutation(len(group))] = labels  # shuffle the labels, not the ids
        parts.append(pd.DataFrame({spec.id_column: group[spec.id_column].to_numpy(), SPLIT_COLUMN: assigned}))

    result = pd.concat(parts, ignore_index=True).sort_values(spec.id_column, kind="mergesort")
    return result.reset_index(drop=True)


def fingerprint(splits: pd.DataFrame, spec: SplitSpec) -> str:
    """Hash of the (id, split) pairs: two runs match only if every id landed in the same split."""
    ordered = splits.sort_values(spec.id_column, kind="mergesort")
    payload = "\n".join(f"{i}:{s}" for i, s in zip(ordered[spec.id_column], ordered[SPLIT_COLUMN]))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def split_summary(splits: pd.DataFrame, population: pd.DataFrame, spec: SplitSpec) -> dict[str, Any]:
    merged = splits.merge(population, on=spec.id_column, how="left", validate="one_to_one")
    n = len(merged)
    rows = []
    for name in spec.names:
        part = merged[merged[SPLIT_COLUMN] == name]
        row = {
            "split": name,
            "target_share": spec.fractions[name],
            "rows": len(part),
            "share": len(part) / n if n else 0.0,
        }
        if spec.stratify_on:
            row["bad_rate"] = float(part[spec.stratify_on].mean()) if len(part) else None
        rows.append(row)
    summary = {
        "method": "stratified_random",
        "seed": spec.seed,
        "id_column": spec.id_column,
        "stratify_on": spec.stratify_on,
        "rows": n,
        "fingerprint": fingerprint(splits, spec),
        "splits": rows,
    }
    if spec.stratify_on:
        summary["bad_rate_overall"] = float(merged[spec.stratify_on].mean())
    return summary
