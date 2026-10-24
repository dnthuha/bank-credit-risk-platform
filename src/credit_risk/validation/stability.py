"""Stage 6.5: stability (PSI) measured by the validator.

Reference is the train split: the sample the models learnt on. Every other
sample is compared with it on fixed bins learnt from the reference only:

    test / validation / calibration   random splits of the same period, so
                                      their PSI should be near zero: a sanity
                                      check of the method and of the split;
    application_test                  the population the models would score
                                      next; the one PSI that can be large.

What is measured
    score / PD   the champion's integer score and the challenger's calibrated
                 PD, on `psi_n_bins` quantile bins of the reference, overall
                 and within each contract type (PROJECT_SCOPE #1: application_test
                 has far fewer revolving loans than train);
    features     the `top_features_for_psi` most influential inputs of each
                 model: for the scorecard, by points range (the most a feature
                 can move a score); for the challenger, by the booster's gain.
                 Numeric features use reference quantile bins with a missing
                 bin; categorical features use the reference's categories,
                 unseen values pooled as "other".

Traffic light from definitions.metric_thresholds.psi: green below
`stable_below`, red at or above `significant_above`, amber between.

Published PSI figures (development's performance.json: score PSI vs
application_test overall and by contract; challenger_performance.json: PD
PSI) are compared with the recomputed values when the artifacts exist.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from credit_risk.validation.metrics import MISSING_BIN, psi, psi_from_proportions
from credit_risk.validation.package import LoadedPackage

REFERENCE = "train"
CURRENT_SAMPLES = ("test", "validation", "calibration", "application_test")
OTHER = "other"


class StabilityError(ValueError):
    pass


@dataclass(frozen=True)
class StabilitySpec:
    n_bins: int = 10
    top_features: int = 5
    stable_below: float = 0.10
    significant_above: float = 0.25
    epsilon: float = 1e-4
    contract_column: str = "NAME_CONTRACT_TYPE"
    abs_tolerance_metric: float = 1e-6

    @classmethod
    def from_config(cls, validation: dict[str, Any], definitions: dict[str, Any]) -> StabilitySpec:
        raw = validation.get("stability") or {}
        thresholds = (definitions.get("metric_thresholds") or {}).get("psi") or {}
        spec = cls(
            n_bins=int(raw.get("psi_n_bins", 10)),
            top_features=int(raw.get("top_features_for_psi", 5)),
            stable_below=float(thresholds.get("stable_below", 0.10)),
            significant_above=float(thresholds.get("significant_above", 0.25)),
            epsilon=float(thresholds.get("zero_bin_epsilon", 1e-4)),
            contract_column=str(raw.get("contract_column", "NAME_CONTRACT_TYPE")),
            abs_tolerance_metric=float(raw.get("abs_tolerance_metric", 1e-6)),
        )
        if spec.n_bins < 2 or spec.top_features < 0:
            raise StabilityError("stability.psi_n_bins must be >= 2 and top_features_for_psi >= 0")
        if not 0 < spec.stable_below <= spec.significant_above:
            raise StabilityError("metric_thresholds.psi: need 0 < stable_below <= significant_above")
        return spec

    def light(self, value: float) -> str:
        if value < self.stable_below:
            return "green"
        return "red" if value >= self.significant_above else "amber"


# ---------------------------------------------------------------------------
# Which features to watch
# ---------------------------------------------------------------------------

def influential_features(package: LoadedPackage, top: int) -> pd.DataFrame:
    """The `top` most influential inputs of each model, with the measure used to rank them."""
    card = package.champion
    rows = [{"model": "champion", "feature": f, "measure": "points_range",
             "value": float(max(card.points[f].values()) - min(card.points[f].values()))} for f in card.features]
    champion = pd.DataFrame(rows).sort_values(["value", "feature"], ascending=[False, True]).head(top)
    booster = package.challenger.booster
    gain = booster.feature_importance(importance_type="gain", iteration=package.challenger.best_iteration)
    names = booster.feature_name()
    total = float(gain.sum()) or 1.0
    rows = [{"model": "challenger", "feature": f, "measure": "gain_share", "value": float(g) / total}
            for f, g in zip(names, gain)]
    challenger = pd.DataFrame(rows).sort_values(["value", "feature"], ascending=[False, True]).head(top)
    out = pd.concat([champion, challenger], ignore_index=True)
    out["rank"] = out.groupby("model").cumcount() + 1
    return out


# ---------------------------------------------------------------------------
# PSI of one quantity, numeric or categorical
# ---------------------------------------------------------------------------

def categorical_psi(reference, current, epsilon: float) -> tuple[float, pd.DataFrame]:
    """PSI on category shares; categories fixed by the reference, unseen ones pooled as `other`,
    missing values as their own category."""
    ref = pd.Series(reference, dtype="object").where(lambda s: s.notna(), MISSING_BIN).astype(str)
    cur = pd.Series(current, dtype="object").where(lambda s: s.notna(), MISSING_BIN).astype(str)
    categories = list(ref.value_counts().index)
    cur = cur.where(cur.isin(categories), OTHER)
    labels = categories + [OTHER]
    ref_counts = ref.value_counts().reindex(labels, fill_value=0)
    cur_counts = cur.value_counts().reindex(labels, fill_value=0)
    table = pd.DataFrame({"bin": labels, "reference_count": ref_counts.to_numpy(), "reference_pct": (ref_counts / len(ref)).to_numpy(),
                          "current_count": cur_counts.to_numpy(), "current_pct": (cur_counts / len(cur)).to_numpy()})
    value = psi_from_proportions(table["reference_pct"], table["current_pct"], epsilon)
    return value, table


def _is_categorical(values: pd.Series) -> bool:
    return not pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values)


def one_psi(reference: pd.Series, current: pd.Series, spec: StabilitySpec) -> tuple[float, pd.DataFrame]:
    if len(reference) == 0 or len(current) == 0:
        raise StabilityError("PSI needs a non-empty reference and current sample")
    if _is_categorical(reference):
        return categorical_psi(reference, current, spec.epsilon)
    result = psi(reference.to_numpy(dtype=float), current.to_numpy(dtype=float), spec.n_bins, spec.epsilon)
    return result.value, result.table


# ---------------------------------------------------------------------------
# All of it
# ---------------------------------------------------------------------------

def recompute_stability(scored: pd.DataFrame, package: LoadedPackage, spec: StabilitySpec
                        ) -> tuple[pd.DataFrame, dict[tuple, pd.DataFrame], pd.DataFrame]:
    """(table, bin_tables, features): PSI of score, PD and influential features vs the train split.

    `scored` holds split, population, the contract column, score, pd_challenger and every model
    input. Rows of application_test have population "test" and no split.
    """
    for column in ("split", "population", spec.contract_column, "score", "pd_challenger"):
        if column not in scored.columns:
            raise StabilityError(f"scored frame lacks '{column}'")
    sample = scored["split"].where(scored["split"].notna(), scored["population"].map({"test": "application_test"}))
    reference = scored[sample == REFERENCE]
    if reference.empty:
        raise StabilityError("no train rows to use as the PSI reference")
    features = influential_features(package, spec.top_features)
    watched = features.drop_duplicates("feature")
    missing = [f for f in watched["feature"] if f not in scored.columns]
    if missing:
        raise StabilityError(f"scored frame lacks influential features {missing}")

    rows, tables = [], {}

    def record(kind, model, name, current_name, ref: pd.Series, cur: pd.Series, segment="all"):
        value, table = one_psi(ref, cur, spec)
        rows.append({"kind": kind, "model": model, "name": name, "current": current_name, "segment": segment,
                     "n_reference": int(len(ref)), "n_current": int(len(cur)),
                     "missing_reference": float(ref.isna().mean()), "missing_current": float(cur.isna().mean()),
                     "psi": value, "light": spec.light(value)})
        tables[(kind, model, name, current_name, segment)] = table

    for current_name in CURRENT_SAMPLES:
        current = scored[sample == current_name]
        if current.empty:
            continue
        for model, column in (("champion", "score"), ("challenger", "pd_challenger")):
            record("output", model, column, current_name, reference[column], current[column])
            for contract in sorted(set(reference[spec.contract_column].dropna()) & set(current[spec.contract_column].dropna())):
                record("output", model, column, current_name,
                       reference.loc[reference[spec.contract_column] == contract, column],
                       current.loc[current[spec.contract_column] == contract, column], segment=str(contract))
        for f in features.itertuples(index=False):
            record("feature", f.model, f.feature, current_name, reference[f.feature], current[f.feature])
    return pd.DataFrame(rows), tables, features


def stability_checks(table: pd.DataFrame, spec: StabilitySpec,
                     published: dict[str, dict[str, float]] | None = None) -> pd.DataFrame:
    """Sanity on the method (same-period splits must be stable) and published vs recomputed.

    `published` is {"champion": {"all": .., "<contract>": ..}, "challenger": {...}}: score / PD PSI vs
    application_test as development reported it.
    """
    rows = []

    def add(check, claimed, recounted, source, tolerance):
        ok = claimed is not None and recounted is not None and abs(float(claimed) - float(recounted)) <= tolerance
        rows.append({"check": check, "claimed": claimed, "recounted": recounted, "source": source, "ok": bool(ok)})

    same_period = table[(table["segment"] == "all") & (table["current"] != "application_test")]
    for r in same_period.itertuples(index=False):
        add(f"same-period split is stable: {r.model} {r.name} vs {r.current} (PSI < {spec.stable_below:g})",
            0.0, r.psi, "recount", spec.stable_below)
    for model, by_segment in (published or {}).items():
        ours = table[(table["kind"] == "output") & (table["model"] == model) & (table["current"] == "application_test")]
        for segment, claimed in by_segment.items():
            mine = ours[ours["segment"] == segment]
            if not mine.empty:
                add(f"published PSI vs application_test ({model}, {segment})", claimed, float(mine["psi"].iloc[0]),
                    "development run", spec.abs_tolerance_metric)
    return pd.DataFrame(rows)


def published_from_train_artifacts(train_dir) -> dict[str, dict[str, float]]:
    import json
    from pathlib import Path

    train_dir = Path(train_dir)
    out = {}
    champion = train_dir / "performance.json"
    if champion.exists():
        out["champion"] = dict(json.loads(champion.read_text(encoding="utf-8")).get("score_psi_vs_application_test", {}))
    challenger = train_dir / "challenger_performance.json"
    if challenger.exists():
        out["challenger"] = dict(json.loads(challenger.read_text(encoding="utf-8")).get("pd_psi_vs_application_test", {}))
    return out


LIGHT_VI = {"green": "xanh", "amber": "vàng", "red": "đỏ"}


def stability_markdown(table: pd.DataFrame, features: pd.DataFrame, checks: pd.DataFrame, spec: StabilitySpec,
                       bin_tables: dict[tuple, pd.DataFrame] | None = None) -> str:
    lines = ["## Độ ổn định (PSI, tham chiếu = train)", "",
             f"Đèn: xanh < {spec.stable_below:g}, đỏ >= {spec.significant_above:g}, vàng ở giữa. "
             f"Bin cố định học trên train ({spec.n_bins} bin phân vị; biến phân loại theo nhóm của train).", "",
             "### Điểm và PD", "", "| Mô hình | Đầu ra | So với | Phân khúc | Hồ sơ | PSI | Đèn |", "|---|---|---|---|---:|---:|---|"]
    outputs = table[table["kind"] == "output"]
    for r in outputs.itertuples(index=False):
        lines.append(f"| {r.model} | {r.name} | {r.current} | {r.segment} | {r.n_current:,} | {r.psi:.4f} | {LIGHT_VI[r.light]} |")
    feats = table[table["kind"] == "feature"]
    same_period = feats[feats["current"] != "application_test"]
    lines += ["", f"### {spec.top_features} biến ảnh hưởng nhất mỗi mô hình, so với application_test", ""]
    if not same_period.empty:
        lines += [f"Trên các tập cùng kỳ (test / validation / calibration) PSI lớn nhất của các biến này là "
                  f"{same_period['psi'].max():.4f}, tất cả {LIGHT_VI[spec.light(same_period['psi'].max())]}.", ""]
    lines += ["| Mô hình | # | Biến | Thước đo | Thiếu (train / application_test) | PSI | Đèn |", "|---|---:|---|---|---|---:|---|"]
    rank = {(f.model, f.feature): (f.rank, f.measure, f.value) for f in features.itertuples(index=False)}
    for r in feats[feats["current"] == "application_test"].itertuples(index=False):
        k, measure, value = rank[(r.model, r.name)]
        lines.append(f"| {r.model} | {k} | `{r.name}` | {measure} = {value:.3g} | "
                     f"{r.missing_reference:.1%} / {r.missing_current:.1%} | {r.psi:.4f} | {LIGHT_VI[r.light]} |")
    worst = table[(table["current"] == "application_test") & (table["light"] != "green")]
    if bin_tables is not None and not worst.empty:
        for r in worst.sort_values("psi", ascending=False).head(3).itertuples(index=False):
            bins = bin_tables[(r.kind, r.model, r.name, r.current, r.segment)]
            lines += ["", f"### Chi tiết: {r.model} {r.name}, {r.segment}, so với application_test (PSI {r.psi:.4f})", "",
                      "| Bin | Train | Hiện tại |", "|---|---:|---:|"]
            for b in bins.itertuples(index=False):
                lines.append(f"| {str(b.bin).replace('|', '/')} | {b.reference_pct:.1%} | {b.current_pct:.1%} |")
    failed = checks[~checks["ok"]]
    lines += ["", f"Kiểm tra: {len(checks)}; lệch: {len(failed)}."]
    if not failed.empty:
        lines += ["", "| Kiểm tra | Công bố | Tính lại | Nguồn |", "|---|---:|---:|---|"]
        for r in failed.itertuples(index=False):
            lines.append(f"| {r.check} | {float(r.claimed):.6f} | {float(r.recounted):.6f} | {r.source} |")
    return "\n".join(lines) + "\n"
