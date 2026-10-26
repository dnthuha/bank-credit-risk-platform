"""Stage 10: the `report` step - one bundle of everything the dashboard shows.

The dashboard (app/) reads artifacts only and never recomputes a number, so this step
gathers the latest completed output of every stage into `artifacts/<run_id>/report/`
and mirrors that folder to `reports/latest/`:

    index.json            run ids per step, provenance, headline numbers for the overview page
    development/          model card, scorecard table, challenger report, calibration, package.json,
                          applicant_examples.json (a few applicants with points per feature and reason codes)
    validation/           metrics.json + validation_report.md of the validate-model step
    monitoring/           monitoring_metrics.json + report + the monthly / vintage / roll-rate tables
    ews/                  ews_metrics.json + report + watchlist and backtest tables

Which runs: the validate-model run names the package run (calibrate) and the train run it
validated, so development and validation are always the same model. Monitoring and EWS take
the newest successful run of their step. A step that ran in this same pipeline run wins.
A missing module is reported in index.json (`missing`), not invented; the step fails only
when nothing at all is available.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from credit_risk.config import ProjectConfig
from credit_risk.data.db import connect
from credit_risk.settings import Settings

BUNDLE_VERSION = 1

# What each module copies. Files that do not exist are skipped and listed in index.json,
# so the dashboard can say so instead of crashing.
DEVELOPMENT_TRAIN_FILES = ("model_card.md", "scorecard_table.csv", "performance.json", "sign_trail.csv",
                           "challenger_report.md", "challenger_importance.csv", "challenger_performance.json",
                           "challenger_trials.csv")
DEVELOPMENT_CALIBRATE_FILES = ("calibration_report.md", "calibration.json", "package.json")
VALIDATION_FILES = ("metrics.json", "validation_report.md")
MONITORING_FILES = ("monitoring_metrics.json", "monitoring_report.md", "portfolio_monthly.csv", "vintage_curves_year.csv",
                    "vintage_curves.csv", "roll_rates.csv", "alerts.csv", "concentration.csv", "segments_monthly.csv",
                    "checks.csv")
EWS_FILES = ("ews_metrics.json", "ews_report.md", "watchlist.csv", "by_split.csv", "monthly_lists.csv",
             "calibration.csv", "segments.csv", "survival_curves.csv", "importance.csv", "coefficients.csv",
             "cohort_by_split.csv", "trials.csv", "checks.csv")

APPLICANT_EXAMPLES = 12
DESCRIPTIVE_COLUMNS = ("NAME_CONTRACT_TYPE", "CODE_GENDER", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY",
                       "AMT_GOODS_PRICE", "DAYS_BIRTH", "DAYS_EMPLOYED")


class BundleError(ValueError):
    """Nothing to report: no completed step output anywhere in artifacts/."""


@dataclass
class Bundle:
    run_id: str
    out_dir: Path
    runs: dict[str, str | None] = field(default_factory=dict)      # step -> run_id used
    copied: list[Path] = field(default_factory=list)
    missing: dict[str, list[str]] = field(default_factory=dict)    # module -> what was not found
    headline: dict[str, Any] = field(default_factory=dict)
    index: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Finding runs
# ---------------------------------------------------------------------------

def read_manifest(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "run_manifest.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return None


def step_ok(manifest: dict[str, Any] | None, step: str) -> bool:
    return bool(manifest) and any(s.get("step") == step and s.get("status") == "ok" for s in manifest.get("steps", []))


def latest_ok_run(artifacts_dir: Path, step: str, prefer: str | None = None) -> str | None:
    """Newest run (by run id, which starts with the UTC time) whose manifest records `step` as ok.
    `prefer` is this pipeline run: its manifest is not written yet, so it is checked by folder."""
    if prefer and (artifacts_dir / prefer / step).exists():
        return prefer
    if not artifacts_dir.exists():
        return None
    for run_dir in sorted(artifacts_dir.iterdir(), reverse=True):
        if run_dir.is_dir() and step_ok(read_manifest(run_dir), step):
            return run_dir.name
    return None


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------

def _copy(bundle: Bundle, module: str, src_dir: Path | None, names: tuple[str, ...], dst: Path) -> None:
    for name in names:
        src = src_dir / name if src_dir is not None else None
        if src is None or not src.exists():
            bundle.missing.setdefault(module, []).append(name)
            continue
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst / name)
        bundle.copied.append(dst / name)


def build_bundle(settings: Settings, config: ProjectConfig, run_id: str, out_dir: Path,
                 current_steps: tuple[str, ...] = ()) -> Bundle:
    """Collect the latest outputs into `out_dir` and write index.json. Raises BundleError when empty."""
    art = settings.artifacts_dir
    bundle = Bundle(run_id=run_id, out_dir=out_dir)

    def prefer(step: str) -> str | None:
        return run_id if step in current_steps else None

    # 1. validation names the package (calibrate run) and the train run it validated.
    val_run = latest_ok_run(art, "validate-model", prefer("validate-model"))
    val_metrics: dict[str, Any] | None = None
    train_run = cal_run = None
    if val_run and (art / val_run / "validate-model" / "metrics.json").exists():
        val_metrics = _read_json(art / val_run / "validate-model" / "metrics.json")
        prov = val_metrics.get("provenance", {})
        cal_run, train_run = prov.get("package_run_id"), prov.get("train_run_id")
    if cal_run is None:
        cal_run = latest_ok_run(art, "calibrate", prefer("calibrate"))
    if train_run is None:
        train_run = latest_ok_run(art, "train", prefer("train"))
    mon_run = latest_ok_run(art, "monitor", prefer("monitor"))
    ews_run = latest_ok_run(art, "ews", prefer("ews"))
    bundle.runs = {"train": train_run, "calibrate": cal_run, "validate-model": val_run, "monitor": mon_run, "ews": ews_run}
    if not any(bundle.runs.values()):
        raise BundleError(f"no completed train / calibrate / validate-model / monitor / ews output under {art}")

    # 2. copy
    _copy(bundle, "development", art / train_run / "train" if train_run else None, DEVELOPMENT_TRAIN_FILES,
          out_dir / "development")
    _copy(bundle, "development", art / cal_run / "calibrate" if cal_run else None, DEVELOPMENT_CALIBRATE_FILES,
          out_dir / "development")
    _copy(bundle, "validation", art / val_run / "validate-model" if val_run else None, VALIDATION_FILES,
          out_dir / "validation")
    _copy(bundle, "monitoring", art / mon_run / "monitor" if mon_run else None, MONITORING_FILES, out_dir / "monitoring")
    _copy(bundle, "ews", art / ews_run / "ews" if ews_run else None, EWS_FILES, out_dir / "ews")

    # 3. applicant examples from the frozen package (development module)
    examples: dict[str, Any] | None = None
    package_dir = settings.processed_dir / "home_credit" / "model_package"
    if package_dir.exists() and (settings.processed_dir / "home_credit" / "features.parquet").exists():
        examples = applicant_examples(settings, config, package_dir)
        path = out_dir / "development" / "applicant_examples.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(examples, indent=1, ensure_ascii=False, default=_default), encoding="utf-8")
        bundle.copied.append(path)
    else:
        bundle.missing.setdefault("development", []).append("applicant_examples.json")

    # 4. headline numbers + index
    bundle.headline = headline(out_dir, val_metrics, examples)
    bundle.index = {
        "bundle_version": BUNDLE_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_id": run_id,
        "config_hash": config.hash,
        "runs": bundle.runs,
        "run_manifests": {step: _manifest_summary(art / rid) for step, rid in bundle.runs.items() if rid},
        "missing": bundle.missing,
        "files": sorted(p.relative_to(out_dir).as_posix() for p in bundle.copied),
        "headline": bundle.headline,
    }
    index = out_dir / "index.json"
    index.write_text(json.dumps(bundle.index, indent=1, ensure_ascii=False, default=_default), encoding="utf-8")
    bundle.copied.append(index)
    return bundle


def _manifest_summary(run_dir: Path) -> dict[str, Any]:
    m = read_manifest(run_dir) or {}
    return {k: m.get(k) for k in ("status", "started_at", "finished_at", "config_hash", "git", "python", "packages")}


def mirror_latest(bundle: Bundle, reports_root: Path) -> Path:
    """Replace reports/latest with this bundle; the dashboard opens that folder by default."""
    latest = reports_root / "latest"
    if latest.exists():
        shutil.rmtree(latest)
    shutil.copytree(bundle.out_dir, latest)
    return latest


# ---------------------------------------------------------------------------
# Applicant examples: "why did this applicant lose points?"
# ---------------------------------------------------------------------------

def applicant_examples(settings: Settings, config: ProjectConfig, package_dir: Path,
                       n: int = APPLICANT_EXAMPLES) -> dict[str, Any]:
    """A deterministic spread of test-split applicants, riskiest first, each with the scorecard's
    points per feature (value, bin, points), reason codes of both models and the calibrated PDs."""
    from credit_risk.reporting.challenger_report import example_rows
    from credit_risk.validation.package import ID, load_package, load_scoring_frame, score_with_package

    package = load_package(package_dir, config.model_dev, config.definitions)
    folder = settings.processed_dir / "home_credit"
    con = connect(settings)
    available = {r[0] for r in con.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{(folder / 'features.parquet').as_posix()}')").fetchall()}
    descriptive = [c for c in DESCRIPTIVE_COLUMNS if c in available]
    columns = list(dict.fromkeys([*package.features, *descriptive]))
    full = load_scoring_frame(con, folder / "features.parquet", folder / "splits.parquet", columns)
    con.close()
    frame = full[full["split"] == "test"].reset_index(drop=True)
    if frame.empty:
        frame = full[full["split"].notna()].reset_index(drop=True)
    scored = score_with_package(package, frame)
    chosen = example_rows(scored["pd_champion"], min(n, len(frame)))
    card = package.champion
    sub = frame.loc[chosen]
    points = card.points_frame(sub)
    codes = card.reason_codes(points)
    contrib = package.challenger.contributions(sub)
    challenger_codes = package.challenger.reason_codes(contrib)
    best = {f: max(card.points[f].values()) for f in card.features}

    applicants = []
    for k, idx in enumerate(chosen):
        row, pts = sub.loc[idx], points.loc[idx]
        breakdown = []
        for f in card.features:
            label = str(card.binnings[f].assign(pd.Series([row[f]]))[0])
            value = row[f]
            breakdown.append({"feature": f, "value": None if pd.isna(value) else value, "bin": label,
                              "points": int(pts[f]), "best_points": int(best[f]), "lost": int(best[f] - pts[f])})
        applicants.append({
            ID: int(row[ID]), "split": row["split"], "target": None if pd.isna(row["TARGET"]) else int(row["TARGET"]),
            "score": int(scored.loc[idx, "score"]), "base_points": card.base_points,
            "pd_champion": float(scored.loc[idx, "pd_champion"]), "pd_challenger": float(scored.loc[idx, "pd_challenger"]),
            "reason_codes": codes[k], "reason_codes_challenger": challenger_codes[k],
            "descriptive": {c: (None if pd.isna(row[c]) else row[c]) for c in descriptive},
            "points": breakdown,
        })
    return {
        "package_fingerprint": package.fingerprint, "split": "test" if (frame["split"] == "test").all() else "mixed",
        "base_points": card.base_points, "factor": card.spec.factor, "offset": card.spec.offset,
        "features": card.features, "applicants": applicants,
    }


# ---------------------------------------------------------------------------
# Headline numbers for the overview page
# ---------------------------------------------------------------------------

def _rows(section: dict[str, Any] | None, key: str) -> pd.DataFrame:
    rows = (section or {}).get(key) or []
    return pd.DataFrame(rows)


def headline(out_dir: Path, val_metrics: dict[str, Any] | None, examples: dict[str, Any] | None) -> dict[str, Any]:
    h: dict[str, Any] = {}
    dev = out_dir / "development"
    if (dev / "performance.json").exists():
        perf = _read_json(dev / "performance.json")
        h["development"] = {
            "champion_features": len(examples["features"]) if examples else None,
            "base_points": examples["base_points"] if examples else None,
            "gini_validation_champion": _dig(perf, "performance", "validation", "points", "gini"),
        }
        if (dev / "challenger_performance.json").exists():
            ch = _read_json(dev / "challenger_performance.json")
            h["development"]["gini_validation_challenger"] = (_dig(ch, "performance", "validation", "gini")
                                                              or _dig(ch, "validation", "gini"))
    if val_metrics:
        p = val_metrics.get("provenance", {})
        sample = p.get("headline_sample", "test")
        disc = _rows(val_metrics.get("sections", {}).get("discrimination"), "table")
        unc = _rows(val_metrics.get("sections", {}).get("uncertainty"), "intervals")
        models = {}
        for model in ("champion", "challenger"):
            d = disc[(disc["model"] == model) & (disc["split"] == sample)] if len(disc) else disc
            u = (unc[(unc["model"] == model) & (unc["split"] == sample) & (unc["metric"] == "gini")]
                 if len(unc) else unc)
            models[model] = {
                "gini": float(d["gini"].iloc[0]) if len(d) else None,
                "gini_ci": [float(u["ci_low"].iloc[0]), float(u["ci_high"].iloc[0])] if len(u) else None,
                "ks": float(d["ks"].iloc[0]) if len(d) else None,
                "verdict": _dig(val_metrics, "conclusion", "verdicts", model),
            }
        h["validation"] = {
            "sample": sample, "checks_total": p.get("checks_total"), "checks_failed": p.get("checks_failed"),
            "findings": p.get("findings"), "package_fingerprint": p.get("package_fingerprint"),
            "models": models, "recommendation": _dig(val_metrics, "conclusion", "recommendation"),
        }
    mon = out_dir / "monitoring" / "monitoring_metrics.json"
    if mon.exists():
        m = _read_json(mon)
        p = m.get("provenance", {})
        monthly = pd.DataFrame(m.get("sections", {}).get("portfolio_monthly") or [])
        row: dict[str, Any] = {}
        if len(monthly) and "period" in monthly:
            monthly["period"] = monthly["period"].astype(str).str[:7]
            last = monthly[monthly["period"] == p.get("reporting_month")]
            row = last.iloc[0].to_dict() if len(last) else {}
        h["monitoring"] = {
            "reporting_month": p.get("reporting_month"), "status": m.get("status"), "alerts": p.get("alerts"),
            "panel": p.get("panel"), "checks_total": p.get("checks_total"), "checks_failed": p.get("checks_failed"),
            "dpd90plus_rate_count": row.get("dpd90plus_rate_count"), "dpd90plus_rate_upb": row.get("dpd90plus_rate_upb"),
            "red_alerts": [a for a in m.get("alerts", []) if a.get("light") == "red"],
        }
    ews = out_dir / "ews" / "ews_metrics.json"
    if ews.exists():
        e = _read_json(ews)
        p = e.get("provenance", {})
        summary = _dig(e, "sections", "watchlist_summary") or {}
        h["ews"] = {
            "scoring_month": p.get("scoring_month"), "selected_model": p.get("selected_model"),
            "horizon_months": p.get("horizon_months"), "out_of_time": p.get("out_of_time"), "splits": p.get("splits"),
            "watchlist": {**(p.get("watchlist") or {}),
                          **{k: v for k, v in summary.items() if k in ("capacity_share", "listed_share", "mean_pd", "upb")}},
            "checks_total": p.get("checks_total"), "checks_failed": p.get("checks_failed"),
        }
    return h


def _dig(obj: Any, *keys: str) -> Any:
    for k in keys:
        if not isinstance(obj, dict) or k not in obj:
            return None
        obj = obj[k]
    return obj


def _default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return obj.as_posix()
    raise TypeError(f"not JSON serialisable: {type(obj)}")
