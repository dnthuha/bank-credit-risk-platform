"""Stage 2.4 - binning engine: fit on train, apply anywhere.

Numeric features
    1. Pre-bin at `initial_quantile_bins` quantiles of the train values. Edges are
       observed values and intervals are right-closed: (lower, upper].
    2. Merge bins smaller than `min_bin_share` of the train rows into the
       neighbour with the closer bad rate.
    3. If `prefer_monotonic_trend`, merge adjacent bins that break the dominant
       trend until the bad rate is monotonic. Features listed in
       `allow_non_monotonic` - each with a business rationale - keep their shape;
       a monotonic fit that keeps less than `review_iv_retention` of the pre-bin
       IV is flagged for exactly that review.
    4. Merge the most similar adjacent pair until at most `max_bins` remain.
    Missing values keep a bin of their own; `max_bins` counts non-missing bins.

Categorical features
    Categories below `rare_category_share` are pooled into __rare__; groups are
    ordered by bad rate and merged with steps 2 and 4. Ordering by bad rate
    makes the trend monotonic by construction.

Values the train sample never showed - a missing value when train had none, a
category never seen - get WoE 0, the population average. It is the neutral
choice: an unknown value neither earns nor loses points.

Everything is deterministic: no sampling and no random tie-breaks, so the same
train rows always give the same table. The table records the IV of the 20
pre-bins next to the final IV, so a reviewer sees what monotonic merging cost.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from credit_risk.features.woe import woe_from_counts

MISSING = "__missing__"
RARE = "__rare__"
UNSEEN = "__unseen__"


class BinningError(ValueError):
    pass


@dataclass(frozen=True)
class BinningSpec:
    initial_bins: int
    max_bins: int
    min_bin_share: float
    monotonic: bool
    missing_own_bin: bool
    rare_share: float
    zero_count_adjustment: float
    # feature -> business rationale; these keep their shape instead of being forced monotonic
    allow_non_monotonic: dict[str, str] = field(default_factory=dict)
    # below this share of pre-bin IV kept, a monotonic fit is flagged for review
    review_iv_retention: float = 0.6

    @classmethod
    def from_config(cls, model_dev: dict[str, Any], definitions: dict[str, Any]) -> BinningSpec:
        raw = model_dev.get("binning") or {}
        spec = cls(
            initial_bins=int(raw.get("initial_quantile_bins", 20)),
            max_bins=int(raw.get("max_bins", 8)),
            min_bin_share=float(raw.get("min_bin_share", 0.05)),
            monotonic=bool(raw.get("prefer_monotonic_trend", True)),
            missing_own_bin=bool(raw.get("missing_as_own_bin", True)),
            rare_share=float(raw.get("rare_category_share", 0.01)),
            zero_count_adjustment=float(
                definitions.get("metric_thresholds", {}).get("woe", {}).get("zero_count_adjustment", 0.5)
            ),
            allow_non_monotonic={str(k): str(v) for k, v in (raw.get("allow_non_monotonic") or {}).items()},
            review_iv_retention=float(raw.get("review_iv_retention", 0.6)),
        )
        for feature, rationale in spec.allow_non_monotonic.items():
            if not rationale.strip():
                raise BinningError(f"allow_non_monotonic.{feature} needs a business rationale")
        if spec.initial_bins < 2 or spec.max_bins < 2:
            raise BinningError("initial_quantile_bins and max_bins must be at least 2")
        if not 0 < spec.min_bin_share < 0.5:
            raise BinningError("min_bin_share must be between 0 and 0.5")
        if not spec.missing_own_bin:
            raise BinningError("missing_as_own_bin: false is not supported; missing always gets its own bin")
        return spec


# ---------------------------------------------------------------------------
# Segment arithmetic: bins are adjacent segments with counts; merging is the only move
# ---------------------------------------------------------------------------

@dataclass
class _Segments:
    n: list[float]
    bad: list[float]
    keys: list[Any]  # numeric: upper edge of each segment except the last; categorical: category lists

    def rates(self) -> np.ndarray:
        return np.asarray(self.bad) / np.asarray(self.n)

    def merge(self, i: int) -> None:
        """Merge segment i with segment i + 1."""
        self.n[i] += self.n.pop(i + 1)
        self.bad[i] += self.bad.pop(i + 1)
        if isinstance(self.keys[i], list):  # categorical groups
            self.keys[i] = self.keys[i] + self.keys.pop(i + 1)
        else:  # numeric: drop the edge between i and i + 1
            self.keys.pop(i)

    def __len__(self) -> int:
        return len(self.n)


def _merge_small(seg: _Segments, n_total: float, min_share: float) -> None:
    while len(seg) > 1:
        shares = np.asarray(seg.n) / n_total
        i = int(np.argmin(shares))
        if shares[i] >= min_share:
            return
        rates = seg.rates()
        if i == 0:
            seg.merge(0)
        elif i == len(seg) - 1:
            seg.merge(i - 1)
        else:
            left, right = abs(rates[i] - rates[i - 1]), abs(rates[i] - rates[i + 1])
            if left < right or (left == right and seg.n[i - 1] <= seg.n[i + 1]):
                seg.merge(i - 1)
            else:
                seg.merge(i)


def _trend(seg: _Segments) -> str:
    """Direction of the bad rate over bin order, weighted by bin size."""
    rates, n = seg.rates(), np.asarray(seg.n)
    order = np.arange(len(rates))
    covariance = np.sum(n * (order - np.average(order, weights=n)) * (rates - np.average(rates, weights=n)))
    return "decreasing" if covariance < 0 else "increasing"


def _enforce_monotonic(seg: _Segments, trend: str) -> None:
    while len(seg) > 1:
        rates = seg.rates()
        steps = np.diff(rates)
        broken = np.flatnonzero(steps < 0) if trend == "increasing" else np.flatnonzero(steps > 0)
        if broken.size == 0:
            return
        seg.merge(int(broken[0]))


def _merge_to_max(seg: _Segments, max_bins: int) -> None:
    while len(seg) > max_bins:
        seg.merge(int(np.argmin(np.abs(np.diff(seg.rates())))))


def _iv(n: list[float], bad: list[float], adjustment: float) -> float:
    n_arr, bad_arr = np.asarray(n, dtype=float), np.asarray(bad, dtype=float)
    keep = n_arr > 0
    if keep.sum() < 2:
        return 0.0
    good = n_arr[keep] - bad_arr[keep]
    if good.sum() == 0 or bad_arr[keep].sum() == 0:
        return 0.0
    return float(woe_from_counts(good, bad_arr[keep], adjustment)[1].sum())


def _fmt(value: float) -> str:
    return f"{value:.6g}"


# ---------------------------------------------------------------------------
# One fitted feature
# ---------------------------------------------------------------------------

@dataclass
class FeatureBinning:
    feature: str
    kind: str                      # numeric | categorical
    bins: list[dict[str, Any]]     # one row per bin, missing / unseen rows included
    iv: float
    prebin_iv: float
    trend: str | None = None
    edges: list[float] = field(default_factory=list)          # numeric: inner right-closed edges
    groups: list[list[str]] = field(default_factory=list)     # categorical: categories per bin
    notes: list[str] = field(default_factory=list)

    # -- scoring ---------------------------------------------------------
    def _lookup(self) -> dict[str, float]:
        return {row["label"]: row["woe"] for row in self.bins}

    def assign(self, values) -> np.ndarray:
        """Bin label for every value, using only what was fitted."""
        labels = [row["label"] for row in self.bins if row["label"] not in (MISSING, UNSEEN)]
        has_missing = any(row["label"] == MISSING for row in self.bins)
        missing_label = MISSING if has_missing else UNSEEN

        if self.kind == "numeric":
            x = pd.to_numeric(pd.Series(values), errors="raise").to_numpy(dtype=float)
            out = np.empty(len(x), dtype=object)
            missing = np.isnan(x)
            idx = np.searchsorted(np.asarray(self.edges, dtype=float), x[~missing], side="left")
            out[~missing] = np.asarray(labels, dtype=object)[idx]
            out[missing] = missing_label
            return out

        s = pd.Series(values, dtype="object")
        mapping = {cat: label for label, group in zip(labels, self.groups) for cat in group}
        rare_label = mapping.get(RARE, UNSEEN)
        out = s.map(lambda v: missing_label if pd.isna(v) else mapping.get(str(v), rare_label))
        return out.to_numpy(dtype=object)

    def transform(self, values) -> np.ndarray:
        """WoE for every value; unseen values score 0 (the population average)."""
        lookup = self._lookup()
        return np.array([lookup.get(label, 0.0) for label in self.assign(values)], dtype=float)

    # -- persistence -----------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "kind": self.kind,
            "trend": self.trend,
            "iv": self.iv,
            "prebin_iv": self.prebin_iv,
            "edges": self.edges,
            "groups": self.groups,
            "notes": self.notes,
            "bins": self.bins,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> FeatureBinning:
        return cls(
            feature=raw["feature"], kind=raw["kind"], bins=raw["bins"], iv=raw["iv"],
            prebin_iv=raw["prebin_iv"], trend=raw.get("trend"), edges=raw.get("edges", []),
            groups=raw.get("groups", []), notes=raw.get("notes", []),
        )


def _table(labels: list[str], n: list[float], bad: list[float], spec: BinningSpec,
           n_missing: float, bad_missing: float) -> tuple[list[dict[str, Any]], float, list[str]]:
    """Final bin rows with WoE / IV; the missing bin (if any) is the last row."""
    notes: list[str] = []
    all_labels, all_n, all_bad = list(labels), list(n), list(bad)
    if n_missing > 0:
        all_labels.append(MISSING)
        all_n.append(n_missing)
        all_bad.append(bad_missing)

    n_arr, bad_arr = np.asarray(all_n, dtype=float), np.asarray(all_bad, dtype=float)
    good_arr = n_arr - bad_arr
    rows = []
    if len(all_labels) >= 2 and good_arr.sum() > 0 and bad_arr.sum() > 0:
        woe, contrib, _, _, adjusted = woe_from_counts(good_arr, bad_arr, spec.zero_count_adjustment)
        if adjusted:
            notes.append(f"a bin had 0 goods or 0 bads: {spec.zero_count_adjustment} added to every bin")
    else:
        woe, contrib = np.zeros(len(all_labels)), np.zeros(len(all_labels))
        notes.append("a single bin (or a single class): no discriminatory power, WoE set to 0")

    n_total = n_arr.sum()
    for i, label in enumerate(all_labels):
        rows.append({
            "label": label,
            "n": int(n_arr[i]),
            "share": float(n_arr[i] / n_total),
            "n_good": int(good_arr[i]),
            "n_bad": int(bad_arr[i]),
            "bad_rate": float(bad_arr[i] / n_arr[i]),
            "woe": float(woe[i]),
            "iv_contribution": float(contrib[i]),
        })
    if n_missing == 0:
        # Keep a missing row so scoring never meets an unknown label; it scores the average.
        rows.append({"label": MISSING, "n": 0, "share": 0.0, "n_good": 0, "n_bad": 0,
                     "bad_rate": None, "woe": 0.0, "iv_contribution": 0.0})
        notes.append("no missing values in train: missing scores WoE 0")
    elif n_missing / n_total < spec.min_bin_share:
        notes.append(f"missing bin holds {n_missing / n_total:.2%} of train, under min_bin_share: its WoE is noisy")
    return rows, float(np.sum(contrib)), notes


def fit_numeric(feature: str, values, y, spec: BinningSpec) -> FeatureBinning:
    x = pd.to_numeric(pd.Series(values), errors="raise").to_numpy(dtype=float)
    target = np.asarray(y, dtype=float)
    missing = np.isnan(x)
    xv, yv = x[~missing], target[~missing]
    n_total = float(len(x))
    n_missing, bad_missing = float(missing.sum()), float(target[missing].sum())

    if xv.size == 0:
        rows, iv, notes = _table([], [], [], spec, n_missing, bad_missing)
        return FeatureBinning(feature, "numeric", rows, iv, iv, None, [], [], notes + ["all values missing in train"])

    quantiles = np.linspace(0, 1, spec.initial_bins + 1)[1:-1]
    edges = np.unique(np.quantile(xv, quantiles, method="higher"))
    edges = edges[edges < xv.max()]  # an edge at the maximum would leave the last bin empty
    idx = np.searchsorted(edges, xv, side="left")
    n = np.bincount(idx, minlength=edges.size + 1).astype(float)
    bad = np.bincount(idx, weights=yv, minlength=edges.size + 1)
    seg = _Segments(list(n), list(bad), list(edges))

    prebin_iv = _iv(seg.n + [n_missing], seg.bad + [bad_missing], spec.zero_count_adjustment)
    _merge_small(seg, n_total, spec.min_bin_share)
    trend = _trend(seg) if len(seg) > 1 else None
    extra_notes = []
    if feature in spec.allow_non_monotonic:
        trend = "non_monotonic" if trend else None
        extra_notes.append(f"monotonic trend not enforced: {spec.allow_non_monotonic[feature]}")
    elif spec.monotonic and trend:
        _enforce_monotonic(seg, trend)
    _merge_to_max(seg, spec.max_bins)

    bounds = [-math.inf] + [float(e) for e in seg.keys] + [math.inf]
    labels = [
        f"({'-inf' if lo == -math.inf else _fmt(lo)}, {'inf' if hi == math.inf else _fmt(hi)}]"
        for lo, hi in zip(bounds[:-1], bounds[1:])
    ]
    rows, iv, notes = _table(labels, seg.n, seg.bad, spec, n_missing, bad_missing)
    if trend not in (None, "non_monotonic") and prebin_iv >= 0.02 and iv < spec.review_iv_retention * prebin_iv:
        extra_notes.append(
            f"monotonic merging kept {iv / prebin_iv:.0%} of the pre-bin IV ({prebin_iv:.4f} -> {iv:.4f}): "
            "review the shape; allow_non_monotonic needs a business rationale"
        )
    return FeatureBinning(feature, "numeric", rows, iv, prebin_iv, trend if len(seg) > 1 else None,
                          [float(e) for e in seg.keys], [], extra_notes + notes)


def fit_categorical(feature: str, values, y, spec: BinningSpec) -> FeatureBinning:
    s = pd.Series(values, dtype="object").reset_index(drop=True)
    target = pd.Series(np.asarray(y, dtype=float))
    missing = s.isna()
    n_total = float(len(s))
    n_missing, bad_missing = float(missing.sum()), float(target[missing].sum())

    present = pd.DataFrame({"c": s[~missing].astype(str), "y": target[~missing]})
    if present.empty:
        rows, iv, notes = _table([], [], [], spec, n_missing, bad_missing)
        return FeatureBinning(feature, "categorical", rows, iv, iv, None, [], [], notes + ["all values missing in train"])

    stats = present.groupby("c")["y"].agg(n="size", bad="sum")
    rare = sorted(stats.index[stats["n"] / n_total < spec.rare_share])
    groups: list[tuple[list[str], float, float]] = [
        ([cat], float(row.n), float(row.bad)) for cat, row in stats.drop(index=rare).iterrows()
    ]
    notes = []
    if rare:
        groups.append(([RARE] + rare, float(stats.loc[rare, "n"].sum()), float(stats.loc[rare, "bad"].sum())))
        notes.append(f"{len(rare)} categories under {spec.rare_share:.0%} of train pooled into {RARE}")
    # Order by bad rate, ties broken by the category name so the order never depends on input order.
    groups.sort(key=lambda g: (g[2] / g[1], g[0][0]))
    seg = _Segments([g[1] for g in groups], [g[2] for g in groups], [g[0] for g in groups])

    prebin_iv = _iv(seg.n + [n_missing], seg.bad + [bad_missing], spec.zero_count_adjustment)
    _merge_small(seg, n_total, spec.min_bin_share)
    _merge_to_max(seg, spec.max_bins)

    counts = stats["n"].to_dict()
    rare_set = set(rare)
    final_groups = [sorted(g, key=lambda c: (c == RARE, -counts.get(c, 0), c)) for g in seg.keys]
    labels = []
    for i, group in enumerate(final_groups):
        named = [c for c in group if c != RARE and c not in rare_set]
        text = " | ".join(named[:3]) + (f" | +{len(named) - 3}" if len(named) > 3 else "")
        if RARE in group:
            pooled = f"{RARE}({len(rare_set)})"
            text = f"{text} | {pooled}" if text else pooled
        labels.append(f"[{i}] {text}")  # the index keeps labels unique
    rows, iv, table_notes = _table(labels, seg.n, seg.bad, spec, n_missing, bad_missing)
    return FeatureBinning(feature, "categorical", rows, iv, prebin_iv,
                          "by_bad_rate" if len(seg) > 1 else None, [], final_groups, notes + table_notes)


# ---------------------------------------------------------------------------
# All features
# ---------------------------------------------------------------------------

def is_categorical(series: pd.Series) -> bool:
    return (
        series.dtype == object
        or isinstance(series.dtype, pd.StringDtype)
        or pd.api.types.is_bool_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
    )


def fit_feature(feature: str, values: pd.Series, y, spec: BinningSpec) -> FeatureBinning:
    if is_categorical(values):
        as_text = values.astype("object").where(values.notna(), None).map(lambda v: v if v is None else str(v))
        return fit_categorical(feature, as_text, y, spec)
    return fit_numeric(feature, values, y, spec)


def fit_all(frame: pd.DataFrame, features: list[str], target: str, spec: BinningSpec) -> dict[str, FeatureBinning]:
    """Fit every feature on `frame`. The caller passes train rows only."""
    if frame[target].isna().any():
        raise BinningError(f"{target} has nulls in the fitting sample")
    y = frame[target].to_numpy(dtype=float)
    if not set(np.unique(y)) <= {0.0, 1.0}:
        raise BinningError(f"{target} must be 0 / 1")
    return {feature: fit_feature(feature, frame[feature], y, spec) for feature in features}


def binning_payload(binnings: dict[str, FeatureBinning], spec: BinningSpec, meta: dict[str, Any]) -> dict[str, Any]:
    features = {name: b.to_dict() for name, b in binnings.items()}
    body = json.dumps(features, sort_keys=True, default=str)
    return {
        **meta,
        "spec": spec.__dict__,
        "fingerprint": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        "features": features,
    }


def load_binnings(payload: dict[str, Any]) -> dict[str, FeatureBinning]:
    return {name: FeatureBinning.from_dict(raw) for name, raw in payload["features"].items()}


def binning_frame(binnings: dict[str, FeatureBinning]) -> pd.DataFrame:
    """Flat table, one row per feature x bin, for review."""
    rows = []
    for name, b in binnings.items():
        for i, row in enumerate(b.bins):
            rows.append({"feature": name, "kind": b.kind, "trend": b.trend, "feature_iv": b.iv,
                         "bin_order": i, **row})
    return pd.DataFrame(rows)
