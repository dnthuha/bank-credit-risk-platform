"""Stage 6.7: performance by segment, including the fairness review.

An overall AUC can hide a segment where the model does not work. For every
segment in `validation.yaml -> segments`, every group of it, and both
models, the validator recomputes on the configured samples (test by default):

    size            n, bads, bad rate;
    discrimination  AUC, Gini, KS within the group (only when the group has
                    at least `min_group_size` applicants and both outcomes);
    calibration     mean PD against the bad rate, and a z-score of observed
                    against expected bads (as in Stage 6.4) that flags a group
                    the model systematically under- or over-predicts;
    decision        the share of the group the reject rule would decline
                    (`reject_rule.pd_above`: calibrated PD above a threshold,
                    the same rule for both models), and the adverse impact
                    ratio: the group's approval rate over the approval rate
                    of the most-approved sizable group (1 = equal; below 0.8
                    is the common four-fifths reference).

Derived segment columns
    score_band                  equal-count bands of the champion's score with
                                edges learnt on the train split;
    ext_source_missing_pattern  which of EXT_SOURCE_1/2/3 are missing;
    age_band                    years from DAYS_BIRTH, cut at `age_bands_years`.

Fairness review (PROJECT_SCOPE #7): CODE_GENDER, NAME_FAMILY_STATUS and
DAYS_BIRTH stayed in the scorecard, so for the segments marked
`fairness_review: true` the report must show discrimination, calibration
and reject rate per group. This module computes them; it does not decide
whether a gap is acceptable (that is Stage 7's conclusion rules), but it
flags every gap above the configured limits so none is missed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from credit_risk.validation.metrics import auc, ks

MODELS = ("champion", "challenger")
EXT_SOURCES = ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3")
DERIVED = {"score_band", "ext_source_missing_pattern", "age_band"}


class SegmentError(ValueError):
    pass


@dataclass(frozen=True)
class Segment:
    name: str
    column: str
    fairness_review: bool = False


@dataclass(frozen=True)
class SegmentSpec:
    segments: tuple[Segment, ...]
    samples: tuple[str, ...] = ("test",)
    min_group_size: int = 500
    score_bands: int = 10
    age_bands_years: tuple[float, ...] = (30, 45, 60)
    reject_pd_above: float = 0.10
    band_z_limit: float = 2.0
    age_column: str = "DAYS_BIRTH"
    extra_columns: tuple[str, ...] = field(default=())

    @classmethod
    def from_config(cls, validation: dict[str, Any]) -> SegmentSpec:
        raw_segments = validation.get("segments") or []
        if not raw_segments:
            raise SegmentError("validation.segments is empty")
        opt = validation.get("segment_options") or {}
        age_column = str(opt.get("age_column", "DAYS_BIRTH"))
        segments = []
        for raw in raw_segments:
            if not raw.get("name") or not raw.get("column"):
                raise SegmentError(f"every segment needs name and column, got {raw}")
            column = str(raw["column"])
            if column == age_column:      # an age in days is a continuous value: segment on the derived band
                column = "age_band"
            segments.append(Segment(str(raw["name"]), column, bool(raw.get("fairness_review", False))))
        if len({s.name for s in segments}) != len(segments):
            raise SegmentError("segment names must be unique")
        rule = opt.get("reject_rule") or {}
        spec = cls(
            segments=tuple(segments),
            samples=tuple(str(s) for s in opt.get("samples", ("test",))),
            min_group_size=int(opt.get("min_group_size", 500)),
            score_bands=int(opt.get("score_bands", 10)),
            age_bands_years=tuple(float(v) for v in opt.get("age_bands_years", (30, 45, 60))),
            reject_pd_above=float(rule.get("pd_above", 0.10)),
            band_z_limit=float(opt.get("band_z_limit", 2.0)),
            age_column=age_column,
        )
        if spec.min_group_size < 2 or spec.score_bands < 2:
            raise SegmentError("segment_options: min_group_size and score_bands must be >= 2")
        if not 0 < spec.reject_pd_above < 1:
            raise SegmentError("segment_options.reject_rule.pd_above must be in (0, 1)")
        if list(spec.age_bands_years) != sorted(set(spec.age_bands_years)):
            raise SegmentError("segment_options.age_bands_years must be strictly increasing")
        return spec

    def split_by_availability(self, available_columns) -> tuple["SegmentSpec", list[Segment]]:
        """(spec with the segments whose columns the data holds, segments it cannot compute).

        A configured segment the data cannot support is not silently dropped: the caller
        records it as a failed check so the report shows what was asked for and not done.
        """
        available = set(available_columns)
        usable, missing = [], []
        for segment in self.segments:
            needs = {segment.column} if segment.column not in DERIVED else (
                {self.age_column} if segment.column == "age_band" else set(EXT_SOURCES) if segment.column == "ext_source_missing_pattern" else set())
            (usable if needs <= available else missing).append(segment)
        if not usable:
            raise SegmentError(f"none of the configured segments can be computed; missing columns for {[s.name for s in missing]}")
        return SegmentSpec(**{**self.__dict__, "segments": tuple(usable)}), missing

    @property
    def source_columns(self) -> list[str]:
        """Columns to load from features.parquet: raw segment columns plus what derived ones need."""
        columns = [s.column for s in self.segments if s.column not in DERIVED]
        if any(s.column == "age_band" for s in self.segments):
            columns.append(self.age_column)
        if any(s.column == "ext_source_missing_pattern" for s in self.segments):
            columns.extend(EXT_SOURCES)
        return list(dict.fromkeys(columns))


# ---------------------------------------------------------------------------
# Derived segment columns
# ---------------------------------------------------------------------------

def score_band_edges(train_score, n_bands: int) -> np.ndarray:
    """Inner edges of equal-count score bands, learnt on the train split only."""
    s = np.asarray(train_score, dtype=float)
    return np.unique(np.quantile(s, np.linspace(0, 1, n_bands + 1)[1:-1]))


def assign_score_band(score, edges: np.ndarray) -> pd.Series:
    """Band 1 = lowest scores (riskiest). Labels carry the score range so a reader needs no key."""
    idx = np.searchsorted(edges, np.asarray(score, dtype=float), side="right")
    bounds = np.r_[-np.inf, edges, np.inf]
    labels = [f"{i + 1:02d}: " + (f"< {bounds[i + 1]:.0f}" if i == 0 else f">= {bounds[i]:.0f}" if i == len(edges) else
                                   f"{bounds[i]:.0f}-{bounds[i + 1]:.0f}") for i in range(len(edges) + 1)]
    return pd.Series(pd.Categorical.from_codes(idx, categories=labels), index=getattr(score, "index", None))


def ext_source_missing_pattern(frame: pd.DataFrame) -> pd.Series:
    missing = [frame[c].isna() for c in EXT_SOURCES]
    labels = []
    for flags in zip(*missing):
        gone = [str(i + 1) for i, m in enumerate(flags) if m]
        labels.append("none missing" if not gone else "missing " + "+".join(gone))
    return pd.Series(labels, index=frame.index, dtype="object")


def age_band(days_birth, edges_years: tuple[float, ...]) -> pd.Series:
    years = -np.asarray(days_birth, dtype=float) / 365.25
    bounds = [-np.inf, *edges_years, np.inf]
    labels = [f"< {edges_years[0]:.0f}"] + [f"{a:.0f}-{b:.0f}" for a, b in zip(edges_years[:-1], edges_years[1:])] + [f">= {edges_years[-1]:.0f}"]
    return pd.cut(years, bins=bounds, labels=labels, right=False).astype(str)


def add_derived_columns(scored: pd.DataFrame, spec: SegmentSpec, train_mask: np.ndarray) -> pd.DataFrame:
    out = scored.copy()
    wanted = {s.column for s in spec.segments}
    if "score_band" in wanted:
        if "score" not in out.columns:
            raise SegmentError("score_band needs the champion score")
        edges = score_band_edges(out.loc[train_mask, "score"], spec.score_bands)
        out["score_band"] = assign_score_band(out["score"], edges)
    if "ext_source_missing_pattern" in wanted:
        missing = [c for c in EXT_SOURCES if c not in out.columns]
        if missing:
            raise SegmentError(f"ext_source_missing_pattern needs {missing}")
        out["ext_source_missing_pattern"] = ext_source_missing_pattern(out)
    if "age_band" in wanted:
        if spec.age_column not in out.columns:
            raise SegmentError(f"age_band needs {spec.age_column}")
        out["age_band"] = age_band(out[spec.age_column], spec.age_bands_years)
    for s in spec.segments:
        if s.column not in out.columns:
            raise SegmentError(f"segment {s.name}: column {s.column} is not in the scored frame")
    return out


# ---------------------------------------------------------------------------
# Per group
# ---------------------------------------------------------------------------

def group_metrics(y: np.ndarray, pd_hat: np.ndarray, risk: np.ndarray, spec: SegmentSpec) -> dict[str, Any]:
    n, bads = int(len(y)), int(y.sum())
    expected = float(pd_hat.sum())
    variance = float((pd_hat * (1 - pd_hat)).sum())
    z = (bads - expected) / np.sqrt(variance) if variance > 0 else np.nan
    out = {"n": n, "bads": bads, "bad_rate": bads / n if n else np.nan, "mean_pd": float(pd_hat.mean()) if n else np.nan,
           "expected_bads": expected, "calibration_z": float(z),
           "calibration_flag": "PD thấp (thiếu)" if z > spec.band_z_limit else "PD cao (thừa)" if z < -spec.band_z_limit else "",
           "reject_rate": float((pd_hat > spec.reject_pd_above).mean()) if n else np.nan,
           "auc": np.nan, "gini": np.nan, "ks": np.nan, "note": ""}
    out["ratio"] = out["bad_rate"] / out["mean_pd"] if out["mean_pd"] else np.nan
    if n < spec.min_group_size:
        out["note"] = f"ít quan sát (< {spec.min_group_size})"
    elif bads == 0 or bads == n:
        out["note"] = "một kết cục duy nhất"
    else:
        a = auc(y, risk)
        out.update({"auc": a, "gini": 2 * a - 1, "ks": ks(y, risk).statistic})
    return out


def recompute_segments(scored: pd.DataFrame, spec: SegmentSpec) -> pd.DataFrame:
    """One row per (sample, segment, group, model). `scored` must already hold the segment columns
    (see add_derived_columns), TARGET, split, score, pd_champion and pd_challenger."""
    for column in ("TARGET", "split", "score", "pd_champion", "pd_challenger"):
        if column not in scored.columns:
            raise SegmentError(f"scored frame lacks '{column}'")
    risk_of = {"champion": lambda part: -part["score"].to_numpy(dtype=float),
               "challenger": lambda part: part["pd_challenger"].to_numpy(dtype=float)}
    rows = []
    for sample in spec.samples:
        part = scored[scored["split"] == sample]
        if part.empty:
            raise SegmentError(f"split {sample} has no rows")
        y_all = part["TARGET"].to_numpy(dtype=float)
        if np.isnan(y_all).any():
            raise SegmentError(f"split {sample} has applicants without TARGET")
        for segment in spec.segments:
            values = part[segment.column]
            groups = values.astype("object").where(values.notna(), "(missing)")
            for model in MODELS:
                overall = group_metrics(y_all.astype(int), part[f"pd_{model}"].to_numpy(dtype=float), risk_of[model](part), spec)
                rows.append({"sample": sample, "segment": segment.name, "fairness_review": segment.fairness_review,
                             "group": "(all)", "model": model, **overall, "share": 1.0})
                for name, idx in groups.groupby(groups, sort=True).groups.items():
                    g = part.loc[idx]
                    m = group_metrics(g["TARGET"].to_numpy(dtype=int), g[f"pd_{model}"].to_numpy(dtype=float), risk_of[model](g), spec)
                    rows.append({"sample": sample, "segment": segment.name, "fairness_review": segment.fairness_review,
                                 "group": str(name), "model": model, **m, "share": len(g) / len(part)})
    table = pd.DataFrame(rows)
    # Gaps against the segment's overall row and the adverse impact ratio against the least-declined group.
    table["auc_gap_vs_all"] = np.nan
    table["reject_rate_vs_all"] = np.nan
    table["adverse_impact_ratio"] = np.nan
    for (sample, segment, model), idx in table.groupby(["sample", "segment", "model"]).groups.items():
        block = table.loc[idx]
        overall = block[block["group"] == "(all)"].iloc[0]
        members = block[block["group"] != "(all)"]
        table.loc[members.index, "auc_gap_vs_all"] = members["auc"] - overall["auc"]
        table.loc[members.index, "reject_rate_vs_all"] = members["reject_rate"] - overall["reject_rate"]
        # Four-fifths rule: approval rate of the group / approval rate of the most-approved sizable group.
        sizable = members[members["n"] >= spec.min_group_size]
        if not sizable.empty and (1 - sizable["reject_rate"]).max() > 0:
            table.loc[members.index, "adverse_impact_ratio"] = (1 - members["reject_rate"]) / (1 - sizable["reject_rate"]).max()
    return table


def segment_checks(table: pd.DataFrame, scored: pd.DataFrame, spec: SegmentSpec) -> pd.DataFrame:
    """Groups partition each sample; the champion's bad rate falls with its score band."""
    rows = []
    for (sample, segment, model), idx in table.groupby(["sample", "segment", "model"]).groups.items():
        block = table.loc[idx]
        members = block[block["group"] != "(all)"]
        total = int((scored["split"] == sample).sum())
        rows.append({"check": f"groups cover the sample ({segment}, {model}, {sample})", "claimed": total,
                     "recounted": int(members["n"].sum()), "source": "recount", "ok": int(members["n"].sum()) == total})
        if segment == "score_band" and model == "champion":
            rates = members.sort_values("group")["bad_rate"].to_numpy()
            rows.append({"check": f"bad rate falls with score band (champion, {sample})", "claimed": "monotone",
                         "recounted": "monotone" if np.all(np.diff(rates) <= 0) else "not monotone", "source": "recount",
                         "ok": bool(np.all(np.diff(rates) <= 0))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _f(v, fmt):
    return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else format(v, fmt)


def segments_markdown(table: pd.DataFrame, checks: pd.DataFrame, spec: SegmentSpec) -> str:
    lines = ["## Theo phân khúc", "",
             f"Tập: {', '.join(spec.samples)}. Nhóm dưới {spec.min_group_size} hồ sơ không tính AUC. "
             f"Từ chối = PD đã calibrate > {spec.reject_pd_above:.0%}. Cờ hiệu chỉnh khi |z| > {spec.band_z_limit:g}. "
             f"AIR = tỷ lệ được duyệt của nhóm / nhóm được duyệt nhiều nhất (quy tắc 4/5: dưới 0.8 cần lưu ý)."]
    for sample in spec.samples:
        for segment in spec.segments:
            block = table[(table["sample"] == sample) & (table["segment"] == segment.name)]
            title = f"### {segment.name} (`{segment.column}`), {sample}" + (" — **rà soát công bằng**" if segment.fairness_review else "")
            lines += ["", title, "", "| Nhóm | Mô hình | Hồ sơ | Tỷ trọng | Bad rate | PD TB | Tỷ số | z | Cờ | AUC | Gini | Từ chối | AIR | Ghi chú |",
                      "|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---|"]
            for r in block.itertuples(index=False):
                lines.append(f"| {r.group} | {r.model} | {r.n:,} | {r.share:.1%} | {_f(r.bad_rate, '.2%')} | {_f(r.mean_pd, '.2%')} | "
                             f"{_f(r.ratio, '.3f')} | {_f(r.calibration_z, '+.2f')} | {r.calibration_flag} | {_f(r.auc, '.4f')} | "
                             f"{_f(r.gini, '.4f')} | {_f(r.reject_rate, '.1%')} | {_f(r.adverse_impact_ratio, '.2f')} | {r.note} |")
    fair = table[(table["fairness_review"]) & (table["group"] != "(all)")]
    flagged = fair[(fair["calibration_flag"] != "") | (fair["adverse_impact_ratio"] < 0.8) | (fair["auc_gap_vs_all"].abs() > 0.02)]
    lines += ["", "### Điểm cần lưu ý trong rà soát công bằng", ""]
    if flagged.empty:
        lines.append("Không nhóm nhạy cảm nào bị cờ hiệu chỉnh, AIR < 0.8 hay AUC lệch > 0.02 so với tổng thể.")
    else:
        for r in flagged.itertuples(index=False):
            why = []
            if r.calibration_flag:
                why.append(f"hiệu chỉnh: {r.calibration_flag}, z {r.calibration_z:+.2f}")
            if not np.isnan(r.adverse_impact_ratio) and r.adverse_impact_ratio < 0.8:
                why.append(f"AIR {r.adverse_impact_ratio:.2f}")
            if not np.isnan(r.auc_gap_vs_all) and abs(r.auc_gap_vs_all) > 0.02:
                why.append(f"AUC lệch {r.auc_gap_vs_all:+.3f} so với tổng thể")
            lines.append(f"- {r.segment} = {r.group}, {r.model}, {r.sample}: " + "; ".join(why) + ".")
    failed = checks[~checks["ok"]]
    lines += ["", f"Kiểm tra: {len(checks)}; lệch: {len(failed)}."]
    if not failed.empty:
        for r in failed.itertuples(index=False):
            lines.append(f"- LỆCH: {r.check}: công bố {r.claimed}, tính lại {r.recounted}.")
    return "\n".join(lines) + "\n"
