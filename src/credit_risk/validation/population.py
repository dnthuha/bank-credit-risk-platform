"""Stage 6.2: population reconciliation.

Before any metric is recomputed, the validator recounts the population the
model card describes and compares it with what the package and the
development run claim. Every number is counted again from the processed
data (application_train / application_test, features.parquet,
splits.parquet) in DuckDB, never copied from a report.

Claims and where they come from
    scorecard payload      rows_train, split_fingerprint
    challenger payload     rows_train, rows_validation, split_fingerprint
    package.json           split_fingerprint
    calibrated_scores      rows per split and per population (what was scored)
    model_dev.split        target fractions 60 / 10 / 15 / 15
    splits_summary.json    rows and bad rate per split (optional: the features
                           run's summary, when the caller can locate it)

Exclusions
    The model card states the population is all of application_train with no
    exclusion. The recount therefore also checks that every applicant has a
    TARGET, sits in exactly one split, and appears exactly once in
    features.parquet and in the published scores.

A difference larger than the configured tolerance (`validation.yaml` ->
population_reconciliation) is a failed check. The result keeps every check,
passed or failed, so the report can show the reconciliation in full.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from credit_risk.data.db import sql_str
from credit_risk.features.split import SPLIT_COLUMN, SplitSpec, fingerprint as split_fingerprint
from credit_risk.validation.package import ID, LoadedPackage

SPLITS = ("train", "validation", "calibration", "test")


class PopulationError(ValueError):
    pass


@dataclass(frozen=True)
class Tolerances:
    rows: int = 0
    bad_rate: float = 1e-6
    share: float = 1e-3     # recounted share vs configured fraction (stratified rounding)

    @classmethod
    def from_config(cls, validation: dict[str, Any]) -> Tolerances:
        raw = validation.get("population_reconciliation") or {}
        tol = cls(rows=int(raw.get("abs_tolerance_rows", 0)),
                  bad_rate=float(raw.get("abs_tolerance_bad_rate", 1e-6)),
                  share=float(raw.get("abs_tolerance_share", 1e-3)))
        if tol.rows < 0 or tol.bad_rate < 0 or tol.share < 0:
            raise PopulationError("population_reconciliation tolerances must be >= 0")
        return tol


@dataclass(frozen=True)
class PopulationResult:
    checks: pd.DataFrame        # check, claimed, recounted, source, ok
    recount: dict[str, Any]     # the validator's own counts, for the report
    tolerances: Tolerances

    @property
    def ok(self) -> bool:
        return bool(self.checks["ok"].all())

    @property
    def failures(self) -> pd.DataFrame:
        return self.checks[~self.checks["ok"]]

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "recount": self.recount, "checks": self.checks.to_dict(orient="records")}


# ---------------------------------------------------------------------------
# Recount
# ---------------------------------------------------------------------------

def recount_population(con: duckdb.DuckDBPyConnection, folder: Path, spec: SplitSpec) -> dict[str, Any]:
    """Count the development population again from the processed files in `folder`."""
    folder = Path(folder)
    paths = {name: folder / f"{name}.parquet" for name in ("application_train", "application_test", "features", "splits")}
    missing = [p.name for p in paths.values() if not p.exists()]
    if missing:
        raise PopulationError(f"{folder}: missing {missing}")
    train, test, feats, splits = (sql_str(paths[n]) for n in ("application_train", "application_test", "features", "splits"))

    def one(sql: str) -> tuple:
        return con.execute(sql).fetchone()

    raw_train_rows, raw_train_target_rows, raw_bad_rate, raw_train_distinct = one(
        f"SELECT count(*), count(TARGET), avg(TARGET), count(DISTINCT {ID}) FROM read_parquet({train})")
    raw_test_rows, raw_test_distinct = one(f"SELECT count(*), count(DISTINCT {ID}) FROM read_parquet({test})")

    per_population = {pop: {"rows": int(n), "rows_with_target": int(t), "distinct_ids": int(d)} for pop, n, t, d in con.execute(
        f"SELECT population, count(*), count(TARGET), count(DISTINCT {ID}) FROM read_parquet({feats}) GROUP BY 1").fetchall()}

    per_split = {}
    for name, n, bad, distinct in con.execute(
            f"SELECT s.{SPLIT_COLUMN}, count(*), avg(t.TARGET), count(DISTINCT s.{ID}) "
            f"FROM read_parquet({splits}) s LEFT JOIN read_parquet({train}) t USING ({ID}) GROUP BY 1").fetchall():
        per_split[name] = {"rows": int(n), "bad_rate": None if bad is None else float(bad), "distinct_ids": int(distinct)}
    split_rows = int(sum(v["rows"] for v in per_split.values()))

    # Exclusions: anything that keeps an applicant out of the stated population.
    exclusions = {
        "train_rows_without_target": int(raw_train_rows - raw_train_target_rows),
        "duplicate_ids_application_train": int(raw_train_rows - raw_train_distinct),
        "duplicate_ids_application_test": int(raw_test_rows - raw_test_distinct),
        "train_applicants_without_split": int(one(
            f"SELECT count(*) FROM read_parquet({train}) t LEFT JOIN read_parquet({splits}) s USING ({ID}) "
            f"WHERE s.{SPLIT_COLUMN} IS NULL")[0]),
        "split_ids_not_in_application_train": int(one(
            f"SELECT count(*) FROM read_parquet({splits}) s LEFT JOIN read_parquet({train}) t USING ({ID}) "
            f"WHERE t.{ID} IS NULL")[0]),
        "ids_in_more_than_one_split": int(one(
            f"SELECT count(*) FROM (SELECT {ID} FROM read_parquet({splits}) GROUP BY 1 HAVING count(*) > 1)")[0]),
        "train_applicants_missing_from_features": int(one(
            f"SELECT count(*) FROM read_parquet({train}) t LEFT JOIN read_parquet({feats}) f USING ({ID}) WHERE f.{ID} IS NULL")[0]),
        "test_applicants_missing_from_features": int(one(
            f"SELECT count(*) FROM read_parquet({test}) t LEFT JOIN read_parquet({feats}) f USING ({ID}) WHERE f.{ID} IS NULL")[0]),
        "feature_rows_not_in_any_application_file": int(one(
            f"SELECT count(*) FROM read_parquet({feats}) f "
            f"LEFT JOIN read_parquet({train}) a USING ({ID}) LEFT JOIN read_parquet({test}) b USING ({ID}) "
            f"WHERE a.{ID} IS NULL AND b.{ID} IS NULL")[0]),
    }

    splits_frame = con.execute(f"SELECT {ID}, {SPLIT_COLUMN} FROM read_parquet({splits})").df()
    return {
        "application_train_rows": int(raw_train_rows),
        "application_test_rows": int(raw_test_rows),
        "bad_rate_overall": float(raw_bad_rate),
        "features_by_population": per_population,
        "splits": per_split,
        "split_rows": split_rows,
        "split_fingerprint": split_fingerprint(splits_frame, spec),
        "exclusions": exclusions,
    }


# ---------------------------------------------------------------------------
# Reconcile
# ---------------------------------------------------------------------------

def _within(claimed, recounted, tolerance: float) -> bool:
    if claimed is None or recounted is None:
        return False
    try:
        return abs(float(claimed) - float(recounted)) <= tolerance
    except (TypeError, ValueError):
        return claimed == recounted


def reconcile_population(package: LoadedPackage, recount: dict[str, Any], spec: SplitSpec,
                         tolerances: Tolerances, split_summary: dict[str, Any] | None = None) -> PopulationResult:
    """Compare every published population figure with the recount."""
    rows: list[dict[str, Any]] = []

    def add(check: str, claimed, recounted, source: str, tolerance: float | None = None) -> None:
        ok = _within(claimed, recounted, tolerance) if tolerance is not None else claimed == recounted
        rows.append({"check": check, "claimed": claimed, "recounted": recounted, "source": source, "ok": bool(ok)})

    scorecard = package.meta  # package.json claims
    champion_payload = _champion_payload(package)
    challenger_payload = _challenger_payload(package)
    per_split = recount["splits"]
    rows_of = lambda name: per_split.get(name, {}).get("rows")  # noqa: E731

    # 1. What the models say they were fitted and evaluated on.
    add("rows_train (scorecard)", champion_payload.get("rows_train"), rows_of("train"), "champion.json", tolerances.rows)
    add("rows_train (challenger)", challenger_payload.get("rows_train"), rows_of("train"), "challenger.json", tolerances.rows)
    add("rows_validation (challenger)", challenger_payload.get("rows_validation"), rows_of("validation"),
        "challenger.json", tolerances.rows)
    for source, claimed in (("champion.json", champion_payload.get("split_fingerprint")),
                            ("challenger.json", challenger_payload.get("split_fingerprint")),
                            ("package.json", scorecard.get("split_fingerprint"))):
        add(f"split fingerprint ({source})", claimed, recount["split_fingerprint"], source)

    # 2. What was scored: the published file must cover the population exactly once.
    published = package.published
    pub_by_split = published[SPLIT_COLUMN].value_counts(dropna=False)
    for name in SPLITS:
        add(f"published rows, split {name}", int(pub_by_split.get(name, 0)), rows_of(name) or 0,
            "calibrated_scores.parquet", tolerances.rows)
    add("published rows without split (application_test)", int(published[SPLIT_COLUMN].isna().sum()),
        recount["application_test_rows"], "calibrated_scores.parquet", tolerances.rows)
    add("published rows, total", int(len(published)),
        recount["application_train_rows"] + recount["application_test_rows"], "calibrated_scores.parquet", tolerances.rows)
    add("published duplicate ids", int(published[ID].duplicated().sum()), 0, "calibrated_scores.parquet")

    # 3. The stated split design: sizes against the configured fractions, bad rate stratified.
    total = recount["split_rows"]
    for name in SPLITS:
        target = spec.fractions.get(name)
        share = (rows_of(name) or 0) / total if total else None
        add(f"share of split {name} vs configured fraction", target, None if share is None else round(share, 6),
            "model_dev.yaml", tolerances.share)
    if spec.stratify_on:
        overall = recount["bad_rate_overall"]
        for name in SPLITS:
            add(f"bad rate of split {name} vs overall (stratified)", round(overall, 6),
                None if per_split.get(name, {}).get("bad_rate") is None else round(per_split[name]["bad_rate"], 6),
                "recount", max(tolerances.bad_rate, 1.0 / max(rows_of(name) or 1, 1)))

    # 4. Exclusions: the model card says there are none.
    for key, value in recount["exclusions"].items():
        add(f"exclusion: {key}", 0, value, "model card (no exclusions)")

    # 5. The development run's own split summary, when available.
    if split_summary:
        add("rows (splits_summary)", split_summary.get("rows"), total, "splits_summary.json", tolerances.rows)
        add("split fingerprint (splits_summary)", split_summary.get("fingerprint"), recount["split_fingerprint"],
            "splits_summary.json")
        if "bad_rate_overall" in split_summary:
            add("bad rate overall (splits_summary)", split_summary["bad_rate_overall"], recount["bad_rate_overall"],
                "splits_summary.json", tolerances.bad_rate)
        for entry in split_summary.get("splits", []):
            name = entry.get("split")
            add(f"rows of split {name} (splits_summary)", entry.get("rows"), rows_of(name), "splits_summary.json",
                tolerances.rows)
            if "bad_rate" in entry:
                add(f"bad rate of split {name} (splits_summary)", entry["bad_rate"],
                    per_split.get(name, {}).get("bad_rate"), "splits_summary.json", tolerances.bad_rate)

    return PopulationResult(checks=pd.DataFrame(rows), recount=recount, tolerances=tolerances)


def _champion_payload(package: LoadedPackage) -> dict[str, Any]:
    import json
    return json.loads((package.folder / "champion.json").read_text(encoding="utf-8"))["scorecard"]


def _challenger_payload(package: LoadedPackage) -> dict[str, Any]:
    import json
    return json.loads((package.folder / "challenger.json").read_text(encoding="utf-8"))["challenger"]


def population_markdown(result: PopulationResult) -> str:
    """The reconciliation as the report shows it: verdict, recount, then every check."""
    r = result.recount
    lines = [
        "## Đối chiếu dân số",
        "",
        f"**Kết luận:** {'khớp' if result.ok else f'{len(result.failures)} chênh lệch chưa giải thích'} "
        f"({len(result.checks)} kiểm tra, dung sai {result.tolerances.rows} dòng / {result.tolerances.bad_rate:g} bad rate).",
        "",
        f"- application_train: {r['application_train_rows']:,} hồ sơ, bad rate {r['bad_rate_overall']:.4%}; "
        f"application_test: {r['application_test_rows']:,} hồ sơ (không có TARGET).",
        f"- Split fingerprint tính lại: `{r['split_fingerprint'][:12]}`.",
        "",
        "| Split | Hồ sơ | Bad rate |",
        "|---|---:|---:|",
    ]
    for name in SPLITS:
        part = r["splits"].get(name)
        if part:
            bad = "" if part["bad_rate"] is None else f"{part['bad_rate']:.4%}"
            lines.append(f"| {name} | {part['rows']:,} | {bad} |")
    lines += ["", "| Kiểm tra | Công bố | Đếm lại | Nguồn | Kết quả |", "|---|---:|---:|---|---|"]
    for row in result.checks.itertuples(index=False):
        lines.append(f"| {row.check} | {_fmt(row.claimed)} | {_fmt(row.recounted)} | {row.source} | "
                     f"{'OK' if row.ok else '**LỆCH**'} |")
    return "\n".join(lines) + "\n"


def _fmt(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6f}" if value < 1 else f"{value:,.2f}"
    if isinstance(value, int):
        return f"{value:,}"
    text = str(value)
    return f"`{text[:12]}`" if len(text) == 64 else text
