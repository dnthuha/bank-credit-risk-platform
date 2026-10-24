"""Stage 6.9: the validation engine, end to end.

One call runs Stage 6.1 to 6.8 in order on the frozen package and writes two
files the pipeline step records:

    metrics.json           machine-readable: every table of every module as
                           records, every check with its verdict, the
                           findings list, and the provenance (package and
                           split fingerprints, seed, config hash, versions,
                           sample sizes);
    validation_report.md   the same, for a reader, section by section.

Order and stopping rule
    package -> population -> discrimination -> calibration -> stability ->
    uncertainty -> segments -> implementation. A package that fails its
    integrity checks or a population that does not reconcile stops the run:
    every later number would be about a model or a sample other than the
    one the model card describes.

Findings are collected, not judged: each failed check and each flagged item
(red PSI, a mis-calibrated band or group, a fairness gap) becomes one
finding with a severity hint. The approve / remediate conclusion is Stage
7's job (`validation.yaml -> conclusion_rules`).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from credit_risk.config import ProjectConfig
from credit_risk.data.db import connect
from credit_risk.features.split import SplitSpec
from credit_risk.settings import Settings
from credit_risk.validation import calibration_check as cal
from credit_risk.validation import discrimination as disc
from credit_risk.validation import implementation as impl
from credit_risk.validation import population as pop
from credit_risk.validation import segments as seg
from credit_risk.validation import stability as stab
from credit_risk.validation import uncertainty as unc
from credit_risk.validation.package import ID, LoadedPackage, PackageError, load_package, load_scoring_frame, score_with_package

log = logging.getLogger(__name__)

MODULES = ("package", "population", "discrimination", "calibration", "stability", "uncertainty", "segments", "implementation")


class ValidationFailed(RuntimeError):
    """The run stopped: the package or the population is not what the model card describes."""


@dataclass
class ValidationResult:
    run_id: str
    package: LoadedPackage
    sections: dict[str, dict[str, Any]] = field(default_factory=dict)   # module -> tables / info
    checks: pd.DataFrame = field(default_factory=pd.DataFrame)          # module, check, claimed, recounted, source, ok
    findings: list[dict[str, Any]] = field(default_factory=list)
    markdown: dict[str, str] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return bool(self.checks["ok"].all()) if len(self.checks) else True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, pd.DataFrame):
        return _records(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, Path):
        return obj.as_posix()
    return obj


def find_split_summary(artifacts_dir: Path, split_fingerprint: str) -> dict[str, Any] | None:
    """The features run's splits_summary.json whose fingerprint is the package's, newest first."""
    if not artifacts_dir.exists():
        return None
    for run in sorted(artifacts_dir.iterdir(), reverse=True):
        path = run / "features" / "splits_summary.json"
        if path.exists():
            try:
                summary = json.loads(path.read_text(encoding="utf-8"))
            except ValueError:
                continue
            if summary.get("fingerprint") == split_fingerprint:
                return {**summary, "_path": path.as_posix()}
    return None


def _finding(module: str, severity: str, text: str, **extra: Any) -> dict[str, Any]:
    return {"module": module, "severity": severity, "finding": text, **extra}


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------

def run_validation(settings: Settings, config: ProjectConfig, run_id: str, source: str = "home_credit") -> ValidationResult:
    folder = settings.processed_dir / source
    package_dir = folder / "model_package"
    validation = config.validation

    # 6.1 package
    try:
        package = load_package(package_dir, config.model_dev, config.definitions)
    except PackageError as exc:
        raise ValidationFailed(f"package: {exc}") from exc
    result = ValidationResult(run_id=run_id, package=package)
    checks = [package.checks.assign(module="package")]
    result.sections["package"] = {"summary": package.summary(), "checks": package.checks}
    log.info("  validate-model: package %s loaded, %d checks passed", package.fingerprint[:12], len(package.checks))

    # The applicants, once: every model input plus what stability and segments need.
    stab_spec = stab.StabilitySpec.from_config(validation, config.definitions)
    seg_configured = seg.SegmentSpec.from_config(validation)
    con = connect(settings)
    available = [r[0] for r in con.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{(folder / 'features.parquet').as_posix()}')").fetchall()]
    seg_spec, seg_missing = seg_configured.split_by_availability(available)
    columns = list(dict.fromkeys([*package.features, stab_spec.contract_column, *seg_spec.source_columns]))
    frame = load_scoring_frame(con, folder / "features.parquet", folder / "splits.parquet", columns)
    scored = score_with_package(package, frame).join(frame.drop(columns=[ID]))
    train_mask = (scored["split"] == "train").to_numpy()
    # Development artifacts to compare with: the train step's run (from the scorecard payload) and the
    # calibrate step's run (the package's own). They differ when calibrate was rerun alone.
    champion_payload = json.loads((package_dir / "champion.json").read_text(encoding="utf-8"))["scorecard"]
    train_run = settings.artifacts_dir / champion_payload.get("run_id", package.meta["run_id"])
    dev_run = settings.artifacts_dir / package.meta["run_id"]

    # 6.2 population: stop if it does not reconcile
    split_spec = SplitSpec.from_config(config.model_dev)
    tolerances = pop.Tolerances.from_config(validation)
    recount = pop.recount_population(con, folder, split_spec)
    split_summary = find_split_summary(settings.artifacts_dir, package.meta["split_fingerprint"])
    population = pop.reconcile_population(package, recount, split_spec, tolerances,
                                          {k: v for k, v in (split_summary or {}).items() if k != "_path"} or None)
    checks.append(population.checks.assign(module="population"))
    result.sections["population"] = {"recount": recount, "checks": population.checks,
                                     "split_summary_used": (split_summary or {}).get("_path")}
    result.markdown["population"] = pop.population_markdown(population)
    if not population.ok:
        for r in population.failures.itertuples(index=False):
            result.findings.append(_finding("population", "blocking", f"{r.check}: công bố {r.claimed}, đếm lại {r.recounted}"))
        result.checks = pd.concat(checks, ignore_index=True)
        raise ValidationFailed(f"population does not reconcile: {len(population.failures)} check(s) failed")

    # 6.3 discrimination
    disc_spec = disc.DiscriminationSpec.from_config(validation)
    disc_table = disc.recompute_discrimination(scored, disc_spec)
    disc_pub = disc.published_from_train_artifacts(train_run / "train")
    disc_checks = disc.discrimination_checks(disc_table, disc_spec, disc_pub)
    checks.append(disc_checks.assign(module="discrimination"))
    result.sections["discrimination"] = {"table": disc_table, "checks": disc_checks, "published": disc_pub}
    result.markdown["discrimination"] = disc.discrimination_markdown(disc_table, disc_checks)

    # 6.4 calibration
    cal_spec = cal.CalibrationCheckSpec.from_config(validation)
    cal_summary, cal_bands = cal.recompute_calibration(scored, cal_spec)
    cal_pub = cal.published_from_calibrate_artifacts(dev_run / "calibrate")
    cal_checks = cal.calibration_checks(cal_summary, cal_bands, cal_spec, cal_pub)
    checks.append(cal_checks.assign(module="calibration"))
    result.sections["calibration"] = {"summary": cal_summary, "bands": cal_bands, "checks": cal_checks}
    result.markdown["calibration"] = cal.calibration_markdown(cal_summary, cal_bands, cal_checks, cal_spec)
    for r in cal_summary[cal_summary["headline"]].itertuples(index=False):
        if r.bands_flagged:
            result.findings.append(_finding("calibration", "note",
                                            f"{r.model}, {r.split}: {r.bands_flagged} band lệch hệ thống ({r.bands_flagged_list}); "
                                            f"slope {r.slope:.3f} (SE {r.slope_se:.3f}), HL p {r.hl_p_value:.3f}"))

    # 6.5 stability
    stab_table, stab_bins, stab_features = stab.recompute_stability(scored, package, stab_spec)
    stab_pub = stab.published_from_train_artifacts(train_run / "train")
    stab_checks = stab.stability_checks(stab_table, stab_spec, stab_pub)
    checks.append(stab_checks.assign(module="stability"))
    result.sections["stability"] = {"table": stab_table, "features": stab_features, "checks": stab_checks,
                                    "bins": {" | ".join(map(str, k)): v for k, v in stab_bins.items()
                                             if k[3] == "application_test" and k[4] == "all"}}
    result.markdown["stability"] = stab.stability_markdown(stab_table, stab_features, stab_checks, stab_spec, stab_bins)
    for r in stab_table[(stab_table["current"] == "application_test") & (stab_table["light"] != "green")].itertuples(index=False):
        result.findings.append(_finding("stability", "major" if r.light == "red" else "note",
                                        f"PSI {r.light}: {r.model} {r.name} ({r.segment}) so với application_test = {r.psi:.4f}"))

    # 6.6 uncertainty
    unc_spec = unc.UncertaintySpec.from_config(validation)
    intervals, differences = unc.recompute_uncertainty(scored, unc_spec)
    unc_checks = unc.uncertainty_checks(intervals, disc_table)
    checks.append(unc_checks.assign(module="uncertainty"))
    result.sections["uncertainty"] = {"intervals": intervals, "differences": differences, "checks": unc_checks}
    result.markdown["uncertainty"] = unc.uncertainty_markdown(intervals, differences, unc_checks, unc_spec)

    # 6.7 segments
    scored_seg = seg.add_derived_columns(scored, seg_spec, train_mask)
    seg_table = seg.recompute_segments(scored_seg, seg_spec)
    seg_checks = seg.segment_checks(seg_table, scored_seg, seg_spec)
    for missing in seg_missing:
        seg_checks = pd.concat([seg_checks, pd.DataFrame([{
            "check": f"segment {missing.name} computable", "claimed": missing.column, "recounted": "column not in features",
            "source": "validation.yaml", "ok": False}])], ignore_index=True)
        result.findings.append(_finding("segments", "note", f"phân khúc {missing.name} ({missing.column}) không tính được: thiếu cột"))
    checks.append(seg_checks.assign(module="segments"))
    result.sections["segments"] = {"table": seg_table, "checks": seg_checks, "not_computable": [m.name for m in seg_missing]}
    result.markdown["segments"] = seg.segments_markdown(seg_table, seg_checks, seg_spec)
    fair = seg_table[(seg_table["fairness_review"]) & (seg_table["group"] != "(all)")]
    for r in fair.itertuples(index=False):
        why = []
        if r.calibration_flag:
            why.append(f"{r.calibration_flag} (z {r.calibration_z:+.2f})")
        if not np.isnan(r.adverse_impact_ratio) and r.adverse_impact_ratio < 0.8:
            why.append(f"AIR {r.adverse_impact_ratio:.2f}")
        if not np.isnan(r.auc_gap_vs_all) and abs(r.auc_gap_vs_all) > 0.02:
            why.append(f"AUC {r.auc_gap_vs_all:+.3f} so với tổng thể")
        if why:
            result.findings.append(_finding("segments", "fairness", f"{r.segment} = {r.group}, {r.model}: " + "; ".join(why)))

    # 6.8 implementation
    impl_spec = impl.ImplementationSpec.from_config(validation)
    impl_summary, impl_diffs, impl_info = impl.check_implementation(package, frame, impl_spec)
    impl_checks = pd.DataFrame([{"check": f"implementation: {r.quantity}", "claimed": "published", "recounted": f"{r.differences} khác",
                                 "source": "calibrated_scores.parquet", "ok": r.ok} for r in impl_summary.itertuples(index=False)])
    checks.append(impl_checks.assign(module="implementation"))
    result.sections["implementation"] = {"summary": impl_summary, "differences": impl_diffs.head(1000), "info": impl_info}
    result.markdown["implementation"] = impl.implementation_markdown(impl_summary, impl_diffs, impl_info)
    if not impl_info["ok"]:
        result.findings.append(_finding("implementation", "blocking", f"{impl_info['differences']} chênh lệch khi chấm lại"))

    result.checks = pd.concat(checks, ignore_index=True)[["module", "check", "claimed", "recounted", "source", "ok"]]
    for r in result.checks[(~result.checks["ok"]) & (result.checks["module"].isin(["discrimination", "calibration", "stability", "uncertainty"]))].itertuples(index=False):
        result.findings.append(_finding(r.module, "note", f"{r.check}: công bố {r.claimed}, tính lại {r.recounted}"))

    result.provenance = {
        "run_id": run_id, "source": source,
        "package_fingerprint": package.fingerprint, "champion_fingerprint": package.meta["champion_fingerprint"],
        "challenger_fingerprint": package.meta["challenger_fingerprint"], "split_fingerprint": package.meta["split_fingerprint"],
        "package_run_id": package.meta["run_id"], "train_run_id": champion_payload.get("run_id"), "config_hash": config.hash, "seed": unc_spec.seed,
        "bootstrap_resamples": unc_spec.resamples, "implementation_sample": impl_info["sample_size"],
        "rows": {k: int(v) for k, v in scored["split"].fillna("application_test").value_counts().to_dict().items()},
        "checks_total": int(len(result.checks)), "checks_failed": int((~result.checks["ok"]).sum()),
        "findings": len(result.findings),
    }
    return result


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def metrics_payload(result: ValidationResult) -> dict[str, Any]:
    return _jsonable({
        "provenance": result.provenance,
        "checks": result.checks,
        "findings": result.findings,
        "sections": {name: {k: v for k, v in section.items()} for name, section in result.sections.items()},
    })


def report_markdown(result: ValidationResult, settings: Settings) -> str:
    p = result.provenance
    failed = result.checks[~result.checks["ok"]]
    lines = [f"# Báo cáo thẩm định độc lập — run `{p['run_id']}`", "",
             f"Gói model `{p['package_fingerprint'][:12]}` (champion `{p['champion_fingerprint'][:12]}`, challenger "
             f"`{p['challenger_fingerprint'][:12]}`, split `{p['split_fingerprint'][:12]}`), đóng băng ở run `{p['package_run_id']}`. "
             f"Config hash `{p['config_hash'][:12]}`, seed {p['seed']}, bootstrap {p['bootstrap_resamples']:,} lần, "
             f"mẫu đối chiếu triển khai {p['implementation_sample']:,} hồ sơ.", "",
             f"**Kiểm tra:** {p['checks_total']} ({p['checks_failed']} lệch). **Phát hiện:** {p['findings']}. "
             "Kết luận phê duyệt / khắc phục: chặng 7.", "",
             "| Module | Kiểm tra | Lệch |", "|---|---:|---:|"]
    for module, part in result.checks.groupby("module", sort=False):
        lines.append(f"| {module} | {len(part)} | {int((~part['ok']).sum())} |")
    lines += ["", "## Phát hiện", ""]
    if not result.findings:
        lines.append("Không có.")
    for i, f in enumerate(result.findings, 1):
        lines.append(f"{i}. [{f['severity']}] {f['module']}: {f['finding']}")
    if not failed.empty:
        lines += ["", "## Kiểm tra lệch", "", "| Module | Kiểm tra | Công bố | Tính lại | Nguồn |", "|---|---|---|---|---|"]
        for r in failed.itertuples(index=False):
            lines.append(f"| {r.module} | {r.check} | {r.claimed} | {r.recounted} | {r.source} |")
    lines += ["", "## Gói model (6.1)", "", "| Kiểm tra | Kết quả |", "|---|---|"]
    for r in result.package.checks.itertuples(index=False):
        lines.append(f"| {r.check} | {'OK' if r.ok else '**LỆCH**'} |")
    for module in ("population", "discrimination", "calibration", "stability", "uncertainty", "segments", "implementation"):
        if module in result.markdown:
            lines += ["", result.markdown[module].rstrip()]
    return "\n".join(lines) + "\n"


def write_outputs(result: ValidationResult, out_dir: Path, settings: Settings) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = out_dir / "metrics.json"
    report = out_dir / "validation_report.md"
    metrics.write_text(json.dumps(metrics_payload(result), indent=1, ensure_ascii=False), encoding="utf-8")
    report.write_text(report_markdown(result, settings), encoding="utf-8")
    return [metrics, report]
