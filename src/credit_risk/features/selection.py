"""Stage 2.6 - the shortlist: which binned features may enter the scorecard.

Rules run in a fixed order and every feature leaves with the first reason that
removed it, so the table reads as an audit trail:

1. policy      - excluded by a documented policy decision (e.g. a protected attribute)
2. iv_min      - IV on train below `iv_min`
3. flags       - a 2.5 review flag listed in `drop_on_flags`
4. correlation - greedy by IV: a feature is kept only if, against every feature
                 already kept, |r| of the WoE values is at most `max_abs_correlation`
                 and, for two numeric features, |Spearman rho| of the raw values is at
                 most `max_abs_rank_correlation_raw`. The second test catches two
                 measures of the same quantity whose bins differ in shape (so their
                 WoE correlation looks low) - a scorecard should not score one
                 quantity twice, in two different ways.
5. vif         - while the largest VIF of the kept set exceeds `max_vif`, drop it

Correlation and VIF are computed on the WoE-transformed train sample, which is
what the logistic regression of Stage 3 will see. Nothing here refits a bin, and
only train rows are used: 2.5 already did the out-of-sample checks.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from credit_risk.features.binning import FeatureBinning


class SelectionError(ValueError):
    pass


@dataclass(frozen=True)
class SelectionSpec:
    iv_min: float
    max_abs_correlation: float
    max_vif: float
    drop_on_flags: tuple[str, ...]
    max_abs_rank_correlation_raw: float = 0.9
    policy_exclude: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_config(cls, model_dev: dict[str, Any], definitions: dict[str, Any]) -> SelectionSpec:
        raw = model_dev.get("feature_selection") or {}
        spec = cls(
            iv_min=float(raw.get("iv_min", definitions["metric_thresholds"]["iv"]["min_useful"])),
            max_abs_correlation=float(raw.get("max_abs_correlation", 0.7)),
            max_vif=float(raw.get("max_vif", 5.0)),
            max_abs_rank_correlation_raw=float(raw.get("max_abs_rank_correlation_raw", 0.9)),
            drop_on_flags=tuple(raw.get("drop_on_flags") or ()),
            policy_exclude={str(k): str(v) for k, v in (raw.get("policy_exclude") or {}).items()},
        )
        if not 0 < spec.max_abs_correlation < 1:
            raise SelectionError("max_abs_correlation must be between 0 and 1")
        if spec.max_vif <= 1:
            raise SelectionError("max_vif must be above 1")
        for feature, reason in spec.policy_exclude.items():
            if not reason.strip():
                raise SelectionError(f"policy_exclude.{feature} needs a reason")
        return spec


def _vif(woe: np.ndarray) -> np.ndarray:
    """VIF_j = 1 / (1 - R^2_j), read off the diagonal of the inverse correlation matrix."""
    corr = np.corrcoef(woe, rowvar=False)
    return np.diag(np.linalg.pinv(corr))


def _rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rho on the rows where both are present (inputs are already ranks)."""
    both = ~(np.isnan(a) | np.isnan(b))
    if both.sum() < 3:
        return 0.0
    x, y = a[both], b[both]
    if x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def select_features(report: pd.DataFrame, binnings: dict[str, FeatureBinning], train: pd.DataFrame,
                    spec: SelectionSpec) -> pd.DataFrame:
    """One row per feature: kept or dropped, at which step, and why."""
    rows = report.set_index("feature")
    missing = set(rows.index) - set(binnings)
    if missing:
        raise SelectionError(f"report features without a fitted binning: {sorted(missing)}")

    decision: dict[str, tuple[str, str]] = {}  # feature -> (step, reason)

    for feature in rows.index:
        if feature in spec.policy_exclude:
            decision[feature] = ("policy", spec.policy_exclude[feature])

    for feature, row in rows.iterrows():
        if feature in decision:
            continue
        if row["iv_train"] < spec.iv_min:
            decision[feature] = ("iv_min", f"IV {row['iv_train']:.4f} < {spec.iv_min}")
            continue
        flags = [f for f in str(row.get("review_flags") or "").split(";") if f]
        hit = [f for f in spec.drop_on_flags if f in flags]
        if hit:
            detail = {
                "shift_vs_current": f"PSI vs application_test {row['psi_current']:.3f}",
                "trend_not_confirmed": f"bin order rank corr on validation {row['bad_rate_rank_corr']:.2f}",
                "iv_drops_out_of_sample": f"validation keeps {row['iv_retention']:.0%} of the train IV",
                "unstable_vs_validation": f"PSI vs validation {row['psi_validation']:.3f}",
                "leakage_suspect": f"IV {row['iv_train']:.3f}",
            }.get(hit[0], "")
            decision[feature] = ("flags", f"{hit[0]}: {detail}".rstrip(": "))

    # Correlation, greedy by IV (ties broken by name so the order never depends on input order).
    candidates = sorted((f for f in rows.index if f not in decision),
                        key=lambda f: (-rows.loc[f, "iv_train"], f))
    woe = {f: binnings[f].transform(train[f]) for f in candidates}
    # Ranks of the raw numeric values (NaN stays NaN); Pearson on ranks is Spearman.
    ranks = {f: pd.to_numeric(train[f], errors="coerce").rank().to_numpy(dtype=float)
             for f in candidates if binnings[f].kind == "numeric"}
    kept: list[str] = []
    for feature in candidates:
        reason = None
        partner, strongest = None, 0.0
        for other in kept:
            r = float(np.corrcoef(woe[feature], woe[other])[0, 1])
            if abs(r) > abs(strongest):
                partner, strongest = other, r
        if partner is not None and abs(strongest) > spec.max_abs_correlation:
            reason = f"|r| of WoE = {abs(strongest):.2f} with {partner} (higher IV)"
        elif feature in ranks:
            for other in kept:
                if other in ranks:
                    rho = _rank_corr(ranks[feature], ranks[other])
                    if abs(rho) > spec.max_abs_rank_correlation_raw:
                        reason = (f"|Spearman rho| of raw values = {abs(rho):.2f} with {other} "
                                  "(higher IV): the same quantity measured twice")
                        break
        if reason:
            decision[feature] = ("correlation", reason)
        else:
            kept.append(feature)

    # VIF on what is left: drop the worst offender until everything is under the limit.
    vifs: dict[str, float] = {}
    while len(kept) > 1:
        values = _vif(np.column_stack([woe[f] for f in kept]))
        vifs = dict(zip(kept, values))
        worst = max(kept, key=lambda f: (vifs[f], f))
        if vifs[worst] <= spec.max_vif:
            break
        decision[worst] = ("vif", f"VIF {vifs[worst]:.1f} > {spec.max_vif}")
        kept.remove(worst)

    out = []
    for feature, row in rows.iterrows():
        step, reason = decision.get(feature, ("selected", ""))
        out.append({
            "feature": feature,
            "selected": feature in kept and feature not in decision,
            "step": step,
            "reason": reason,
            "iv_train": row["iv_train"],
            "iv_validation": row["iv_validation"],
            "psi_current": row["psi_current"],
            "vif": vifs.get(feature) if feature in kept else None,
            "kind": row["kind"],
            "trend": row["trend"],
        })
    frame = pd.DataFrame(out)
    order = {"selected": 0, "vif": 1, "correlation": 2, "flags": 3, "policy": 4, "iv_min": 5}
    frame["_order"] = frame["step"].map(order)
    return frame.sort_values(["_order", "iv_train"], ascending=[True, False]).drop(columns="_order").reset_index(drop=True)


def shortlist_payload(decisions: pd.DataFrame, spec: SelectionSpec, meta: dict[str, Any]) -> dict[str, Any]:
    selected = decisions.loc[decisions["selected"], "feature"].tolist()
    return {
        **meta,
        "spec": {**spec.__dict__, "drop_on_flags": list(spec.drop_on_flags)},
        "selected": selected,
        "fingerprint": hashlib.sha256(json.dumps(selected).encode("utf-8")).hexdigest(),
    }


def shortlist_markdown(decisions: pd.DataFrame, spec: SelectionSpec, meta: dict[str, Any]) -> str:
    selected = decisions[decisions["selected"]]
    lines = [
        "# Feature shortlist - home_credit",
        "",
        f"{len(selected)} of {len(decisions)} binned features selected. Rules, in order: policy exclusions, "
        f"IV >= {spec.iv_min}, review flags {list(spec.drop_on_flags)}, |r| of WoE <= {spec.max_abs_correlation} "
        f"and |Spearman rho| of raw numeric values <= {spec.max_abs_rank_correlation_raw} (greedy by IV), "
        f"VIF <= {spec.max_vif}. Computed on the WoE-transformed train sample "
        f"({meta['rows_train']:,} rows); binning fingerprint {meta['binning_fingerprint'][:12]}.",
        "",
        "| Step | Features |",
        "|---|---:|",
    ]
    for step in ["selected", "policy", "iv_min", "flags", "correlation", "vif"]:
        lines.append(f"| {step} | {int((decisions['step'] == step).sum())} |")

    lines += ["", "## Selected", "", "| Feature | Kind | Trend | IV train | IV validation | PSI current | VIF |",
              "|---|---|---|---:|---:|---:|---:|"]
    for r in selected.itertuples(index=False):
        lines.append(f"| `{r.feature}` | {r.kind} | {r.trend} | {r.iv_train:.4f} | {r.iv_validation:.4f} | "
                     f"{r.psi_current:.4f} | {r.vif:.2f} |")

    dropped = decisions[~decisions["selected"] & (decisions["step"] != "iv_min")]
    lines += ["", "## Dropped after the IV screen", "", "| Feature | Step | IV train | Reason |", "|---|---|---:|---|"]
    for r in dropped.itertuples(index=False):
        lines.append(f"| `{r.feature}` | {r.step} | {r.iv_train:.4f} | {r.reason} |")
    lines += ["", f"{int((decisions['step'] == 'iv_min').sum())} further features were dropped for IV < {spec.iv_min}; "
              "see shortlist.csv."]
    return "\n".join(lines) + "\n"
