"""Stage 9.2: score every dataset row with both models, then evaluate.

Scoring streams the dataset in batches (20M+ rows do not fit in pandas with
their features on this laptop) and writes `ews_scores.parquet`: the context
columns, a 64-bit hash of loan_id (`key`, the tie-break of the monthly list)
and one PD per model. Evaluation then reads one split at a time:

    by_split      the configured metrics for each model on each split
    monthly       the monthly list of the selected model: loans, events,
                  listed, caught, capture, precision, lift, mean PD
    calibration   mean PD against the observed rate by PD decile, per split
    segments      within validation / stress / out-of-time, for each segment
                  dimension and level: rows, events, rates, mean PD, ROC / PR
                  AUC and how much of the segment the overall monthly list
                  catches (the list is not rebuilt per segment)
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from credit_risk.ews import metrics as mt
from credit_risk.ews.features import CONTEXT, SCORING

SCORES_FILE = "ews_scores.parquet"
EVALUATED = ("validation", "stress_test", "out_of_time")


def pd_column(model: str) -> str:
    return f"pd_{model}"


def score_dataset(dataset: Path, out: Path, models: list, features: list[str], batch_rows: int = 500_000) -> Path:
    """Stream the dataset, write context + key + one PD per model."""
    source = pq.ParquetFile(dataset)
    columns = [c for c in CONTEXT + ["scoring_row"] if c in source.schema_arrow.names] + features
    tmp = out.with_suffix(".tmp.parquet")
    writer = None
    try:
        for batch in source.iter_batches(batch_size=batch_rows, columns=list(dict.fromkeys(columns))):
            frame = batch.to_pandas()
            scored = frame[[c for c in CONTEXT + ["scoring_row"] if c in frame]].copy()
            scored["key"] = pd.util.hash_pandas_object(frame["loan_id"], index=False).to_numpy()
            for model in models:
                scored[pd_column(model.name)] = model.predict_pd(frame).astype(np.float64)
            table = pa.Table.from_pandas(scored, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(tmp, table.schema, compression="zstd")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    tmp.replace(out)
    return out


def read_split(scores: Path, split: str, columns: list[str]) -> pd.DataFrame:
    frame = pq.read_table(scores, columns=columns, filters=[("split", "=", split)]).to_pandas()
    frame["period"] = pd.to_datetime(frame["period"])
    # the dataset's row order is not fixed (DuckDB writes in parallel): sort, so tied scores always fall alike
    return frame.sort_values(["period", "key"], kind="mergesort").reset_index(drop=True)


def evaluate_scores(scores: Path, models: list[str], selected: str, metrics: list[str], segments: list[str],
                    capacity_share: float) -> dict[str, pd.DataFrame]:
    pds = [pd_column(m) for m in models]
    base_cols = ["period", "key", "y", "split", *segments] + pds
    by_split, monthly, calibration, seg_rows = [], [], [], []
    for split in ("train", *EVALUATED):
        frame = read_split(scores, split, list(dict.fromkeys(base_cols)))
        if frame.empty or frame["y"].sum() == 0:
            continue
        frame["y"] = frame["y"].astype(int)
        for m, col in zip(models, pds):
            row = mt.evaluate(frame, col, metrics, key="key")
            by_split.append({"split": split, "model": m, **row})
            if split != "train":
                calibration.append(mt.calibration_table(frame, col).assign(split=split, model=m))
        if split == "train":
            continue                                        # train is a weighted sample, not a population
        sel = pd_column(selected)
        flagged = mt.monthly_flags(frame, sel, capacity_share, key="key")
        table = mt.monthly_list_table(frame, sel, capacity_share, key="key", flagged=flagged)
        table["mean_pd"] = frame.groupby("period")[sel].mean().reindex(table["period"]).to_numpy()
        monthly.append(table.assign(split=split))
        for dim in segments:
            for level, idx in frame.groupby(dim, sort=True, observed=True).indices.items():
                part = frame.iloc[idx]
                y, p = part["y"].to_numpy(), part[sel].to_numpy()
                events = int(y.sum())
                lst = mt.list_performance(y, flagged[idx])
                seg_rows.append({
                    "split": split, "dimension": dim, "level": str(level), "rows": len(idx), "events": events,
                    "event_rate": events / len(idx), "mean_pd": float(p.mean()),
                    "roc_auc": mt.auc(y, p) if 0 < events < len(y) else math.nan,
                    "pr_auc": mt.average_precision(y, p) if events else math.nan,
                    "share_of_list": lst["listed"] / max(int(flagged.sum()), 1),
                    "capture": lst["capture"], "precision": lst["precision"],
                })
    return {"by_split": pd.DataFrame(by_split), "monthly": pd.concat(monthly, ignore_index=True) if monthly else pd.DataFrame(),
            "calibration": pd.concat(calibration, ignore_index=True) if calibration else pd.DataFrame(),
            "segments": pd.DataFrame(seg_rows)}


def select_model(by_split: pd.DataFrame, metric: str, order: list[str]) -> str:
    """The model with the higher validation metric; the earlier in `order` on a tie."""
    v = by_split[by_split["split"] == "validation"].set_index("model")[metric]
    return max(order, key=lambda m: (v.get(m, -np.inf), -order.index(m)))


def scoring_frame(scores: Path, columns: list[str]) -> pd.DataFrame:
    frame = pq.read_table(scores, columns=columns, filters=[("scoring_row", "=", True)]).to_pandas()
    return frame.reset_index(drop=True)

