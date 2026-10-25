"""Stage 8: portfolio monitoring, end to end.

One call builds the panel and runs every module on it, in order:

    8.0 panel          monitoring_panel.parquet; its checks failing stops the run
    8.1 portfolio      monthly book, delinquency, defaults, payoffs by count and UPB
    8.2 vintage        static-pool cumulative default and payoff curves
    8.3 roll           roll / cure rates and migration matrices
    8.4 concentration  segment shares and rates, HHI, mix PSI
    8.5 alerts         ews.yaml rules on the tables above -> lights and a status

and writes, under artifacts/<run_id>/monitor/:

    monitoring_metrics.json   provenance, status, alerts, checks, limitations and
                              the tables a reader needs (all months, curves by
                              year, migration, concentration)
    monitoring_report.md      status and alerts first, then scope and limitations,
                              then every module
    *.csv                     the full tables, for the dashboard (Stage 10)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from credit_risk.config import ProjectConfig
from credit_risk.data.db import connect
from credit_risk.monitoring import alerts as al
from credit_risk.monitoring import concentration as cn
from credit_risk.monitoring import portfolio as pf
from credit_risk.monitoring import roll as rl
from credit_risk.monitoring import vintage as vt
from credit_risk.monitoring.panel import PanelSpec, build_panel, panel_checks
from credit_risk.settings import Settings

MODULES = ("panel", "portfolio", "vintage", "roll", "concentration")


class MonitoringFailed(RuntimeError):
    """The panel does not reconcile with its sources: no figure built on it can be trusted."""


@dataclass
class MonitoringResult:
    run_id: str
    month: pd.Timestamp
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    checks: pd.DataFrame = field(default_factory=pd.DataFrame)
    alerts: pd.DataFrame = field(default_factory=pd.DataFrame)
    status: str = "green"
    markdown: dict[str, str] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    limitations: list[str] = field(default_factory=list)
    panel_path: Path | None = None


def run_monitoring(settings: Settings, config: ProjectConfig, run_id: str) -> MonitoringResult:
    folder = settings.processed_dir / "freddie_mac"
    if not (folder / "performance.parquet").exists():
        raise FileNotFoundError("performance.parquet missing; run ingest for freddie_mac first")
    ews, definitions = config.ews, config.definitions
    mon = ews["monitoring"]
    spec = PanelSpec.from_config(ews, definitions)
    pf_spec = pf.PortfolioSpec.from_config(ews, definitions)
    alert_spec = al.AlertSpec.from_config(ews, definitions)
    con = connect(settings)

    # 8.0 panel
    panel_path = build_panel(con, folder, spec)
    p_checks, summary = panel_checks(con, folder, spec)
    if not p_checks["ok"].all():
        raise MonitoringFailed("panel checks failed: " + "; ".join(p_checks.loc[~p_checks["ok"], "check"]))
    checks = [p_checks.assign(module="panel")]

    # 8.1 portfolio
    monthly = pf.portfolio_monthly(con, folder, pf_spec)
    month = pf.reporting_month(monthly, pf_spec)
    result = MonitoringResult(run_id=run_id, month=month, panel_path=panel_path)
    pf_checks = pf.portfolio_checks(con, folder, monthly, pf_spec, month)
    checks.append(pf_checks.assign(module="portfolio"))
    result.markdown["portfolio"] = pf.portfolio_markdown(monthly, month, pf_checks)

    # 8.2 vintage
    loans = vt.loan_outcomes(con, folder)
    reach = float(mon["vintage_curve"].get("min_reach_share", 0.99))
    curves = vt.vintage_curves(loans, "vintage", reach)
    curves_year = vt.vintage_curves(loans, "vintage_year", reach)
    vt_checks = vt.vintage_checks(con, folder, loans, curves, "vintage")
    checks.append(vt_checks.assign(module="vintage"))
    result.markdown["vintage"] = vt.vintage_markdown(curves_year, loans, vt_checks)

    # 8.3 roll
    counts = rl.transition_counts(con, folder)
    rates = rl.roll_rates(counts)
    rl_checks = rl.roll_checks(con, folder, counts, month, int(mon.get("reference_sample_loans", 20000)), int(ews.get("seed", 0)))
    checks.append(rl_checks.assign(module="roll"))
    result.markdown["roll"] = rl.roll_markdown(counts, rates, monthly.set_index("period")["regime"], month, rl_checks)

    # 8.4 concentration
    seg_table = cn.segment_table(con, folder, list(mon["concentration_dimensions"]), month, alert_spec.trailing)
    view = cn.reporting_view(seg_table, month)
    conc = cn.concentration_summary(seg_table, month, float(definitions["metric_thresholds"]["psi"]["zero_bin_epsilon"]))
    cn_checks = cn.concentration_checks(seg_table, monthly, month)
    checks.append(cn_checks.assign(module="concentration"))
    result.markdown["concentration"] = cn.concentration_markdown(view, conc, month, cn_checks, alert_spec.psi_stable,
                                                                 alert_spec.psi_significant, cn.group_order(spec.bands))

    # 8.5 alerts
    result.alerts = pd.DataFrame(al.metric_alerts(monthly, rates, month, alert_spec) + al.segment_alerts(view, alert_spec)
                                 + al.mix_alerts(conc, alert_spec) + al.vintage_alerts(curves_year, alert_spec))
    result.status = al.overall_status(result.alerts)

    result.tables = {"portfolio_monthly": monthly, "vintage_curves": curves, "vintage_curves_year": curves_year,
                     "transition_counts": counts, "roll_rates": rates, "segments_monthly": seg_table,
                     "segments_reporting": view, "concentration": conc}
    result.checks = pd.concat(checks, ignore_index=True)[["module", "check", "claimed", "recounted", "source", "ok"]]
    result.limitations = list(alert_spec.limitations)
    lights = result.alerts["light"].value_counts()
    result.provenance = {
        "run_id": run_id, "config_hash": config.hash, "reporting_month": f"{month:%Y-%m}", "seed": int(ews.get("seed", 0)),
        "panel": summary, "checks_total": int(len(result.checks)), "checks_failed": int((~result.checks["ok"]).sum()),
        "status": result.status, "alerts": {k: int(lights.get(k, 0)) for k in ("red", "amber", "green", "excluded")},
    }
    return result


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def metrics_payload(result: MonitoringResult) -> dict[str, Any]:
    month = result.month
    counts = result.tables["transition_counts"]
    migration = {"reporting_month": rl.matrix(counts[counts["period"] == month]).reset_index()}
    regimes = result.tables["portfolio_monthly"].set_index("period")["regime"]
    for regime, part in counts.assign(regime=counts["period"].map(regimes)).groupby("regime", sort=False):
        migration[regime] = rl.matrix(part).reset_index()
    curves = result.tables["vintage_curves_year"]
    return {
        "provenance": result.provenance,
        "status": result.status,
        "alerts": _records(result.alerts),
        "checks": _records(result.checks),
        "limitations": result.limitations,
        "sections": {
            "portfolio_monthly": _records(result.tables["portfolio_monthly"]),
            "roll_rates": _records(result.tables["roll_rates"]),
            "vintage_checkpoints": {m: _records(vt.checkpoint_table(curves, "vintage_year", m).reset_index())
                                    for m in ("cum_default_rate_count", "cum_default_rate_upb", "cum_payoff_rate_count")},
            "migration": {k: _records(v) for k, v in migration.items()},
            "segments_reporting": _records(result.tables["segments_reporting"]),
            "concentration": _records(result.tables["concentration"]),
        },
    }


def report_markdown(result: MonitoringResult) -> str:
    p = result.provenance
    lines = [f"# Báo cáo giám sát danh mục — run `{p['run_id']}`", "",
             f"Freddie Mac sample, tháng báo cáo **{p['reporting_month']}**. Panel {p['panel']['rows']:,} dòng, "
             f"{p['panel']['loans']:,} khoản, {p['panel']['first_month']} → {p['panel']['last_month']}. "
             f"Config hash `{p['config_hash'][:12]}`, seed {p['seed']}.", "",
             f"**Kiểm tra:** {p['checks_total']} ({p['checks_failed']} lệch). **Trạng thái:** {al.TEXT[result.status]}.", "",
             al.alerts_markdown(result.alerts, result.status).rstrip(), "", "## Phạm vi và giới hạn", ""]
    lines += [f"- {x}" for x in result.limitations] or ["Không ghi nhận."]
    lines += ["", "## Kiểm tra theo module", "", "| Module | Kiểm tra | Lệch |", "|---|---:|---:|"]
    for module, part in result.checks.groupby("module", sort=False):
        lines.append(f"| {module} | {len(part)} | {int((~part['ok']).sum())} |")
    failed = result.checks[~result.checks["ok"]]
    if not failed.empty:
        lines += ["", "| Module | Kiểm tra lệch | Công bố | Tính lại | Nguồn |", "|---|---|---|---|---|"]
        for r in failed.itertuples(index=False):
            lines.append(f"| {r.module} | {r.check} | {r.claimed} | {r.recounted} | {r.source} |")
    for module in ("portfolio", "vintage", "roll", "concentration"):
        lines += ["", result.markdown[module].rstrip()]
    return "\n".join(lines) + "\n"


def write_outputs(result: MonitoringResult, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in ("portfolio_monthly", "vintage_curves", "vintage_curves_year", "transition_counts", "roll_rates",
                 "segments_monthly", "concentration"):
        path = out_dir / f"{name}.csv"
        result.tables[name].to_csv(path, index=False)
        paths.append(path)
    for name, frame in (("checks", result.checks), ("alerts", result.alerts)):
        path = out_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        paths.append(path)
    metrics = out_dir / "monitoring_metrics.json"
    metrics.write_text(json.dumps(metrics_payload(result), indent=1, ensure_ascii=False, default=_default), encoding="utf-8")
    report = out_dir / "monitoring_report.md"
    report.write_text(report_markdown(result), encoding="utf-8")
    return [metrics, report, *paths]


def _default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    raise TypeError(f"not JSON serialisable: {type(obj)}")
