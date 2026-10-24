"""Stage 6.8: implementation check.

Does the frozen package, run again by someone else, give the numbers that
were published? A seeded random sample of applicants (`validation.yaml ->
implementation_check.sample_size`, drawn across every population and split)
is rescored from the package alone (`score_with_package`) and compared row
by row with `calibrated_scores.parquet`:

    score            must be identical (integers);
    PD, both models  within `abs_tolerance_pd` (1e-9 by default);
    reason codes     the published reason_1..k of each model must equal the
                     codes the package produces for that applicant, in order.

Every difference is kept with its applicant id, so a non-zero count can be
explained or escalated. The check passes only when there is none.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from credit_risk.validation.package import ID, LoadedPackage, score_with_package


class ImplementationError(ValueError):
    pass


@dataclass(frozen=True)
class ImplementationSpec:
    sample_size: int = 5000
    abs_tolerance_pd: float = 1e-9
    seed: int = 2026

    @classmethod
    def from_config(cls, validation: dict[str, Any]) -> ImplementationSpec:
        raw = validation.get("implementation_check") or {}
        spec = cls(sample_size=int(raw.get("sample_size", 5000)),
                   abs_tolerance_pd=float(raw.get("abs_tolerance_pd", 1e-9)),
                   seed=int(validation.get("seed", 2026)))
        if spec.sample_size < 1 or spec.abs_tolerance_pd < 0:
            raise ImplementationError("implementation_check: sample_size must be >= 1 and abs_tolerance_pd >= 0")
        return spec


def draw_sample(published: pd.DataFrame, spec: ImplementationSpec) -> np.ndarray:
    """Applicant ids to rescore: a seeded draw without replacement, spread over population and split
    in proportion to their size (each stratum gets at least one applicant)."""
    strata = published["split"].where(published["split"].notna(), "application_test")
    rng = np.random.default_rng(spec.seed)
    n = min(spec.sample_size, len(published))
    sizes = strata.value_counts()
    quota = {name: max(1, int(round(n * count / len(published)))) for name, count in sizes.items()}
    chosen = []
    for name, k in quota.items():
        ids = published.loc[strata == name, ID].to_numpy()
        chosen.append(rng.choice(ids, size=min(k, len(ids)), replace=False))
    return np.sort(np.concatenate(chosen))


def reason_columns(published: pd.DataFrame, model: str) -> list[str]:
    columns = [c for c in published.columns if c.startswith(f"{model}_reason_")]
    return sorted(columns, key=lambda c: int(c.rsplit("_", 1)[1]))


def package_reason_codes(package: LoadedPackage, frame: pd.DataFrame, model: str, k: int) -> list[list[str | None]]:
    if model == "champion":
        codes = package.champion.reason_codes(package.champion.points_frame(frame))
    else:
        codes = package.challenger.reason_codes(package.challenger.contributions(frame))
    return [[c[i] if i < len(c) else None for i in range(k)] for c in codes]


def check_implementation(package: LoadedPackage, applicants: pd.DataFrame, spec: ImplementationSpec
                         ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """(summary, differences, info).

    `applicants` is the scoring frame (every model input, keyed by SK_ID_CURR) for at least the
    sampled ids. summary: one row per compared quantity with the number of differences.
    differences: one row per (applicant, quantity) that differs, with both values.
    """
    ids = draw_sample(package.published, spec)
    frame = applicants.set_index(ID, drop=False).loc[ids].reset_index(drop=True)
    if len(frame) != len(ids) or frame[ID].duplicated().any():
        raise ImplementationError("the scoring frame does not hold every sampled applicant exactly once")
    published = package.published.set_index(ID).loc[ids].reset_index()
    rescored = score_with_package(package, frame)
    assert np.array_equal(rescored[ID].to_numpy(), published[ID].to_numpy())

    rows, diffs = [], []

    def compare(name, mine, theirs, tolerance=None):
        mine_arr, theirs_arr = np.asarray(mine), np.asarray(theirs)
        if tolerance is None:
            unequal = ~((mine_arr == theirs_arr) | (pd.isna(mine_arr) & pd.isna(theirs_arr)))
            max_abs_diff = np.nan
        else:
            gap = np.abs(mine_arr.astype(float) - theirs_arr.astype(float))
            unequal = gap > tolerance
            max_abs_diff = float(gap.max()) if len(gap) else 0.0
        rows.append({"quantity": name, "compared": int(len(mine_arr)), "differences": int(unequal.sum()),
                     "tolerance": "exact" if tolerance is None else tolerance, "max_abs_diff": max_abs_diff,
                     "ok": not unequal.any()})
        for i in np.flatnonzero(unequal):
            diffs.append({ID: int(published[ID].iloc[i]), "quantity": name, "package": mine_arr[i], "published": theirs_arr[i]})

    compare("score", rescored["score"], published["score"])
    compare("pd_champion", rescored["pd_champion"], published["pd_champion"], spec.abs_tolerance_pd)
    compare("pd_challenger", rescored["pd_challenger"], published["pd_challenger"], spec.abs_tolerance_pd)
    for model in ("champion", "challenger"):
        columns = reason_columns(published, model)
        if not columns:
            continue
        codes = package_reason_codes(package, frame, model, len(columns))
        for j, column in enumerate(columns):
            compare(column, [c[j] for c in codes], published[column].where(published[column].notna(), None))

    summary = pd.DataFrame(rows)
    info = {"sample_size": int(len(ids)), "seed": spec.seed, "abs_tolerance_pd": spec.abs_tolerance_pd,
            "by_stratum": published["split"].where(published["split"].notna(), "application_test").value_counts().to_dict(),
            "ok": bool(summary["ok"].all()), "differences": int(summary["differences"].sum())}
    return summary, pd.DataFrame(diffs, columns=[ID, "quantity", "package", "published"]), info


def implementation_markdown(summary: pd.DataFrame, differences: pd.DataFrame, info: dict[str, Any]) -> str:
    strata = ", ".join(f"{k} {v:,}" for k, v in info["by_stratum"].items())
    lines = ["## Đối chiếu triển khai", "",
             f"**Kết luận:** {'khớp hoàn toàn' if info['ok'] else str(info['differences']) + ' chênh lệch chưa giải thích'} "
             f"trên {info['sample_size']:,} hồ sơ chấm lại từ gói (seed {info['seed']}; {strata}). "
             f"Dung sai PD {info['abs_tolerance_pd']:g}; score và mã lý do phải trùng tuyệt đối.", "",
             "| Đại lượng | So sánh | Lệch | Dung sai | Lệch lớn nhất | Kết quả |", "|---|---:|---:|---|---:|---|"]
    for r in summary.itertuples(index=False):
        worst = "-" if r.tolerance == "exact" else format(r.max_abs_diff, ".2e")
        lines.append(f"| {r.quantity} | {r.compared:,} | {r.differences:,} | {r.tolerance} | {worst} | "
                     f"{'OK' if r.ok else '**LỆCH**'} |")
    if not differences.empty:
        lines += ["", "Chênh lệch đầu tiên (tối đa 20):", "", "| SK_ID_CURR | Đại lượng | Gói | Công bố |", "|---:|---|---|---|"]
        for r in differences.head(20).itertuples(index=False):
            lines.append(f"| {r.SK_ID_CURR} | {r.quantity} | {r.package} | {r.published} |")
    return "\n".join(lines) + "\n"
