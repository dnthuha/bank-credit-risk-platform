"""Stage 2.5 - out-of-sample IV and stability of every binned feature.

The bins were fixed on train in 2.4; here they are only applied, never refitted.

* Validation: goods and bads are counted per train bin, giving a validation WoE
  and IV. A feature whose IV collapses out of sample, or whose bin ordering does
  not repeat, has learned noise.
* PSI of the bin distribution: train vs validation (a random split, so it should
  be close to 0) and train vs application_test, the unlabelled current sample
  that Module 2 uses. Only distributions are compared, never labels.

Nothing is selected here - features only collect flags; 2.6 decides. The
calibration and test splits are not touched: they stay clean for calibration
and for the final evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from credit_risk.features.binning import MISSING, UNSEEN, FeatureBinning
from credit_risk.features.woe import woe_from_counts
from credit_risk.validation.metrics import psi_from_proportions


@dataclass(frozen=True)
class ReportThresholds:
    iv_min: float
    leakage_suspect: float
    psi_stable: float
    psi_significant: float
    psi_epsilon: float
    min_iv_retention: float
    min_rank_corr: float
    zero_count_adjustment: float

    @classmethod
    def from_config(cls, definitions: dict[str, Any], model_dev: dict[str, Any]) -> ReportThresholds:
        thresholds = definitions["metric_thresholds"]
        selection = model_dev.get("feature_selection", {})
        return cls(
            iv_min=float(thresholds["iv"]["min_useful"]),
            leakage_suspect=float(thresholds["iv"]["leakage_suspect_above"]),
            psi_stable=float(thresholds["psi"]["stable_below"]),
            psi_significant=float(thresholds["psi"]["significant_above"]),
            psi_epsilon=float(thresholds["psi"]["zero_bin_epsilon"]),
            min_iv_retention=float(selection.get("min_iv_retention_validation", 0.5)),
            min_rank_corr=float(selection.get("min_bad_rate_rank_corr", 0.8)),
            zero_count_adjustment=float(thresholds["woe"]["zero_count_adjustment"]),
        )


def iv_band(iv: float, th: ReportThresholds) -> str:
    if iv > th.leakage_suspect:
        return "suspicious"
    if iv >= 0.30:
        return "strong"
    if iv >= 0.10:
        return "medium"
    if iv >= th.iv_min:
        return "weak"
    return "useless"


def _count(binning: FeatureBinning, values, y=None) -> pd.DataFrame:
    labels = pd.Series(binning.assign(values), name="label")
    if y is None:
        return labels.value_counts().rename("n").to_frame()
    frame = pd.DataFrame({"label": labels, "y": np.asarray(y, dtype=float)})
    return frame.groupby("label")["y"].agg(n="size", bad="sum")


def _feature(binning: FeatureBinning, validation: pd.DataFrame, current: pd.DataFrame,
             target: str, th: ReportThresholds) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    train = pd.DataFrame(binning.bins).set_index("label")[["n", "n_bad", "woe", "bad_rate"]]
    val = _count(binning, validation[binning.feature], validation[target])
    cur = _count(binning, current[binning.feature])

    labels = list(train.index) + [x for x in list(val.index) + list(cur.index) if x not in train.index]
    labels = list(dict.fromkeys(labels))  # unseen buckets, if any, go last
    t_n = train["n"].reindex(labels, fill_value=0).astype(float)
    v_n = val["n"].reindex(labels, fill_value=0).astype(float)
    v_bad = val["bad"].reindex(labels, fill_value=0).astype(float)
    c_n = cur["n"].reindex(labels, fill_value=0).astype(float)

    # Validation WoE / IV on the train bins; bins empty in both samples carry no information.
    keep = (t_n > 0) | (v_n > 0)
    v_good = v_n - v_bad
    woe_val = pd.Series(np.nan, index=labels)
    iv_val = 0.0
    if keep.sum() >= 2 and v_good[keep].sum() > 0 and v_bad[keep].sum() > 0:
        woe, contrib, *_ = woe_from_counts(v_good[keep], v_bad[keep], th.zero_count_adjustment)
        woe_val[keep[keep].index] = woe
        iv_val = float(np.sum(contrib))

    # Does the bad-rate ordering of the bins repeat out of sample?
    both = (t_n > 0) & (v_n > 0)
    rank_corr = np.nan
    if both.sum() >= 3:
        rank_corr = float(spearmanr(train["bad_rate"].reindex(labels)[both], (v_bad / v_n)[both]).statistic)

    share = lambda n: (n / n.sum()).to_numpy() if n.sum() else np.zeros(len(n))  # noqa: E731
    psi_val = psi_from_proportions(share(t_n), share(v_n), th.psi_epsilon)
    psi_cur = psi_from_proportions(share(t_n), share(c_n), th.psi_epsilon)
    woe_train = train["woe"].reindex(labels)
    shift = (woe_val - woe_train)[both]

    iv_train = float(binning.iv)
    retention = iv_val / iv_train if iv_train > 0 else np.nan
    flags = []
    if iv_train < th.iv_min:
        flags.append("weak")
    if iv_train > th.leakage_suspect:
        flags.append("leakage_suspect")
    if iv_train >= th.iv_min and retention < th.min_iv_retention:
        flags.append("iv_drops_out_of_sample")
    if iv_train >= th.iv_min and not np.isnan(rank_corr) and rank_corr < th.min_rank_corr:
        flags.append("trend_not_confirmed")
    if psi_val >= th.psi_stable:
        flags.append("unstable_vs_validation")
    if psi_cur >= th.psi_significant:
        flags.append("shift_vs_current")
    elif psi_cur >= th.psi_stable:
        flags.append("shift_vs_current_watch")
    if any("review" in note for note in binning.notes):
        flags.append("binning_review")

    missing = train.loc[MISSING, "n"] if MISSING in train.index else 0
    row = {
        "feature": binning.feature,
        "kind": binning.kind,
        "trend": binning.trend,
        "bins": int(((t_n > 0) & (pd.Index(labels) != MISSING)).sum()),
        "missing_share_train": float(missing / t_n.sum()),
        "iv_train": iv_train,
        "iv_validation": iv_val,
        "iv_retention": retention,
        "iv_band": iv_band(iv_train, th),
        "bad_rate_rank_corr": rank_corr,
        "max_abs_woe_shift": float(shift.abs().max()) if len(shift) else np.nan,
        "psi_validation": psi_val,
        "psi_current": psi_cur,
        "review_flags": ";".join(flags),
    }
    bins = [
        {
            "feature": binning.feature,
            "label": label,
            "share_train": float(t_n[label] / t_n.sum()),
            "share_validation": float(v_n[label] / v_n.sum()) if v_n.sum() else 0.0,
            "share_current": float(c_n[label] / c_n.sum()) if c_n.sum() else 0.0,
            "bad_rate_train": None if pd.isna(train["bad_rate"].get(label)) else float(train["bad_rate"][label]),
            "bad_rate_validation": float(v_bad[label] / v_n[label]) if v_n[label] else None,
            "woe_train": None if label not in train.index else float(train.loc[label, "woe"]),
            "woe_validation": None if pd.isna(woe_val[label]) else float(woe_val[label]),
            "unseen_in_train": label == UNSEEN or label not in train.index,
        }
        for label in labels
    ]
    return row, bins


def iv_report(binnings: dict[str, FeatureBinning], validation: pd.DataFrame, current: pd.DataFrame,
              target: str, th: ReportThresholds) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One row per feature, plus one row per feature x bin."""
    if validation[target].isna().any():
        raise ValueError(f"{target} has nulls in the validation sample")
    rows, bins = [], []
    for binning in binnings.values():
        row, bin_rows = _feature(binning, validation, current, target, th)
        rows.append(row)
        bins.extend(bin_rows)
    features = pd.DataFrame(rows).sort_values(["iv_train", "feature"], ascending=[False, True])
    return features.reset_index(drop=True), pd.DataFrame(bins)


def iv_report_markdown(features: pd.DataFrame, th: ReportThresholds, meta: dict[str, Any]) -> str:
    def fmt(value, digits=4):
        return "" if value is None or (isinstance(value, float) and np.isnan(value)) else f"{value:.{digits}f}"

    lines = [
        "# IV and stability report - home_credit",
        "",
        f"Bins fitted on train ({meta['rows_train']:,} rows) and applied unchanged to validation "
        f"({meta['rows_validation']:,} rows) and application_test ({meta['rows_current']:,} rows, no labels). "
        "Calibration and test splits are not used.",
        "",
        f"Thresholds: IV useful >= {th.iv_min}, leakage suspect > {th.leakage_suspect}; "
        f"PSI stable < {th.psi_stable}, significant >= {th.psi_significant}; "
        f"IV retention on validation >= {th.min_iv_retention:.0%}; bad-rate rank correlation >= {th.min_rank_corr}.",
        "",
        "## IV bands (train)",
        "",
        "| Band | Features |",
        "|---|---:|",
    ]
    for band in ["suspicious", "strong", "medium", "weak", "useless"]:
        lines.append(f"| {band} | {int((features['iv_band'] == band).sum())} |")

    flag_counts = features["review_flags"].str.split(";").explode()
    flag_counts = flag_counts[flag_counts != ""].value_counts()
    lines += ["", "## Flags", "", "| Flag | Features |", "|---|---:|"]
    lines += [f"| {flag} | {count} |" for flag, count in flag_counts.items()] or ["| none | 0 |"]

    useful = features[features["iv_train"] >= th.iv_min]
    lines += [
        "",
        f"## Features with IV >= {th.iv_min} on train ({len(useful)})",
        "",
        "| Feature | Bins | IV train | IV validation | Kept | Rank corr | PSI validation | PSI current | Flags |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in useful.itertuples(index=False):
        lines.append(
            f"| `{r.feature}` | {r.bins} | {fmt(r.iv_train)} | {fmt(r.iv_validation)} | "
            f"{fmt(r.iv_retention * 100, 0)}% | {fmt(r.bad_rate_rank_corr, 2)} | {fmt(r.psi_validation)} | "
            f"{fmt(r.psi_current)} | {r.review_flags.replace(';', ', ')} |"
        )
    return "\n".join(lines) + "\n"
