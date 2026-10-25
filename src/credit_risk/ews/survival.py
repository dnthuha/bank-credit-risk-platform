"""Stage 9.4: descriptive survival on months on book, with prepayment as a competing risk.

Decision 2026-10-25: Kaplan-Meier and Aalen-Johansen written here in numpy, no
covariates. One row per loan from the monitoring panel:

    entry   first MOB with data (0 or 1 for most loans; later for the 8.7% of
            loans the panel shows only from a later age: delayed entry)
    time    MOB of the first default (90+ or liquidation), else of the payoff,
            else the last MOB seen (censored: still active or other exit)
    cause   1 default, 2 payoff, 0 censored

A loan is at risk at MOB m when entry <= m <= time. A left-truncated loan that
is already bad in its first month is left out: its first default happened
before the panel shows it. With n_m at risk, d_m defaults and p_m payoffs:

    hazard_default(m) = d_m / n_m
    km_default(m)     = 1 - prod_{k<=m} (1 - d_k / n_k)     payoff treated as censored:
                                                            default in a world without prepayment
    S(m)              = prod_{k<=m} (1 - (d_k + p_k) / n_k) still active, neither event
    cif_default(m)    = sum_{k<=m} S(k-1) d_k / n_k         Aalen-Johansen: share that defaults
    cif_payoff(m)     = sum_{k<=m} S(k-1) p_k / n_k         with payoff competing

so S + cif_default + cif_payoff = 1 and km_default >= cif_default at every m.
A curve stops at the first MOB whose risk set is below `min_at_risk`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from credit_risk.data.db import sql_str
from credit_risk.monitoring.panel import PANEL_FILE

CENSORED, DEFAULT, PAYOFF = 0, 1, 2
STATIC_POOL_TOLERANCE = 0.02


@dataclass(frozen=True)
class SurvivalSpec:
    groups: list[str]
    checkpoints: list[int]
    min_at_risk: int

    @classmethod
    def from_config(cls, ews: dict) -> "SurvivalSpec":
        cfg = ews["ews"]["survival"]
        groups = list(cfg["groups"])
        allowed = {"vintage_year", "fico_band", "ltv_band", "dti_band", "channel", "property_state", "harp"}
        if set(groups) - allowed:
            raise ValueError(f"ews.survival.groups: unknown {sorted(set(groups) - allowed)}; allowed {sorted(allowed)}")
        if int(cfg["min_at_risk"]) < 1:
            raise ValueError("ews.survival.min_at_risk must be positive")
        return cls(groups=groups, checkpoints=sorted(int(m) for m in cfg["checkpoints"]), min_at_risk=int(cfg["min_at_risk"]))


def loan_times(con: duckdb.DuckDBPyConnection, folder: Path, groups: list[str]) -> pd.DataFrame:
    """One row per loan: entry, time, cause, left_truncated, bad_at_entry and the group columns."""
    panel = sql_str(folder / PANEL_FILE)
    extra = "".join(f", any_value({g}) AS {g}" for g in groups)
    frame = con.execute(f"""
        WITH loans AS (
            SELECT loan_id, min(mob) AS entry, bool_or(left_truncated) AS left_truncated,
                   arg_min(is_bad, period) AS bad_at_entry,
                   min(mob) FILTER (WHERE first_bad) AS default_mob,
                   min(mob) FILTER (WHERE exit_type = 'payoff') AS payoff_mob,
                   max(mob) AS last_mob{extra}
            FROM read_parquet({panel}) GROUP BY loan_id)
        SELECT *, coalesce(default_mob, payoff_mob, last_mob) AS time,
               CASE WHEN default_mob IS NOT NULL THEN {DEFAULT} WHEN payoff_mob IS NOT NULL THEN {PAYOFF} ELSE {CENSORED} END AS cause
        FROM loans
    """).df()
    return frame.sort_values("loan_id").reset_index(drop=True)


def usable(loans: pd.DataFrame) -> pd.Series:
    """Loans that enter the risk sets: not already bad at a late first month, time not before entry."""
    return ~(loans["left_truncated"] & loans["bad_at_entry"]) & (loans["time"] >= loans["entry"])


def curves(entry, time, cause, min_at_risk: int = 1) -> pd.DataFrame:
    """Risk sets, hazards, Kaplan-Meier default and Aalen-Johansen cumulative incidence by MOB."""
    entry, time, cause = (np.asarray(a, dtype=np.int64) for a in (entry, time, cause))
    if len(time) == 0:
        return pd.DataFrame(columns=["mob", "at_risk", "defaults", "payoffs", "censored", "hazard_default",
                                     "hazard_payoff", "km_default", "survival", "cif_default", "cif_payoff"])
    if (time < entry).any() or (entry < 0).any():
        raise ValueError("survival needs 0 <= entry <= time")
    size = int(time.max()) + 1
    entering = np.bincount(entry, minlength=size)
    leaving = np.bincount(time, minlength=size)
    at_risk = np.cumsum(entering) - np.r_[0, np.cumsum(leaving)[:-1]]
    d = np.bincount(time[cause == DEFAULT], minlength=size)
    p = np.bincount(time[cause == PAYOFF], minlength=size)
    c = np.bincount(time[cause == CENSORED], minlength=size)
    with np.errstate(divide="ignore", invalid="ignore"):
        hd = np.where(at_risk > 0, d / at_risk, 0.0)
        hp = np.where(at_risk > 0, p / at_risk, 0.0)
    km = 1 - np.cumprod(1 - hd)
    surv = np.cumprod(1 - hd - hp)
    before = np.r_[1.0, surv[:-1]]
    out = pd.DataFrame({"mob": np.arange(size), "at_risk": at_risk, "defaults": d, "payoffs": p, "censored": c,
                        "hazard_default": hd, "hazard_payoff": hp, "km_default": km, "survival": surv,
                        "cif_default": np.cumsum(before * hd), "cif_payoff": np.cumsum(before * hp)})
    first_mob = int(entry.min())
    out = out[out["mob"] >= first_mob]
    small = out.index[out["at_risk"] < min_at_risk]
    return (out.loc[: small[0] - 1] if len(small) else out).reset_index(drop=True)


def survival_tables(loans: pd.DataFrame, spec: SurvivalSpec) -> pd.DataFrame:
    """Long table: dimension, group, then the curve columns; `all` first."""
    keep = loans[usable(loans)]
    parts = [curves(keep["entry"], keep["time"], keep["cause"], spec.min_at_risk).assign(dimension="all", group="all")]
    for dim in spec.groups:
        for level, part in keep.groupby(dim, sort=True):
            parts.append(curves(part["entry"], part["time"], part["cause"], spec.min_at_risk).assign(dimension=dim, group=str(level)))
    out = pd.concat(parts, ignore_index=True)
    return out[["dimension", "group"] + [c for c in out.columns if c not in ("dimension", "group")]]


def checkpoint_table(tables: pd.DataFrame, dimension: str, column: str, checkpoints: list[int]) -> pd.DataFrame:
    part = tables[(tables["dimension"] == dimension) & tables["mob"].isin(checkpoints)]
    return part.pivot(index="group", columns="mob", values=column).reindex(columns=checkpoints)


def survival_checks(loans: pd.DataFrame, tables: pd.DataFrame, static_pool_curves: pd.DataFrame | None,
                    checkpoints: list[int]) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []

    def add(check, claimed, recounted, source, ok=None):
        rows.append({"check": check, "claimed": claimed, "recounted": recounted, "source": source,
                     "ok": bool(claimed == recounted) if ok is None else bool(ok)})

    keep = usable(loans)
    overall = tables[tables["dimension"] == "all"]
    add("each loan in one cause", len(loans), int(loans["cause"].isin([CENSORED, DEFAULT, PAYOFF]).sum()), PANEL_FILE)
    add("defaults at MOB = loans with a first default (usable loans)",
        int((loans.loc[keep, "cause"] == DEFAULT).sum()),
        int(curves(loans.loc[keep, "entry"], loans.loc[keep, "time"], loans.loc[keep, "cause"])["defaults"].sum()), PANEL_FILE)
    identity = (tables["survival"] + tables["cif_default"] + tables["cif_payoff"] - 1).abs().max()
    add("S + CIF default + CIF payoff = 1 at every MOB", True, bool(identity < 1e-9), "survival.py")
    add("Kaplan-Meier default >= CIF default at every MOB", True,
        bool((tables["km_default"] >= tables["cif_default"] - 1e-12).all()), "survival.py")
    add("curves are non-decreasing", True,
        bool(all((g[["km_default", "cif_default", "cif_payoff"]].diff().fillna(0) >= -1e-12).all().all()
                 for _, g in tables.groupby(["dimension", "group"]))), "survival.py")

    # Static pool cross-check: on loans seen from the start, cif_default by vintage year against the Stage 8 curve.
    compare = pd.DataFrame()
    if static_pool_curves is not None and not static_pool_curves.empty:
        pool = loans[keep & ~loans["left_truncated"]]
        recs = []
        for year, part in pool.groupby("vintage_year"):
            c = curves(part["entry"], part["time"], part["cause"]).set_index("mob")
            stage8 = static_pool_curves[static_pool_curves["vintage_year"] == year].set_index("mob")["cum_default_rate_count"]
            for m in checkpoints:
                if m in c.index and m in stage8.index:
                    recs.append({"vintage_year": int(year), "mob": m, "cif_default": float(c.loc[m, "cif_default"]),
                                 "static_pool": float(stage8.loc[m])})
        compare = pd.DataFrame(recs)
        if not compare.empty:
            compare["difference"] = compare["cif_default"] - compare["static_pool"]
            # Censoring only adds to the estimate: other exits and the <= 1% of the pool too young to
            # reach the MOB (min_reach_share) are counted as non-defaults by the static pool.
            rel = (compare["difference"] / compare["static_pool"].where(compare["static_pool"] > 0)).fillna(0)
            add(f"static pool: CIF default >= Stage 8 curve, within {STATIC_POOL_TOLERANCE:.0%} relative "
                f"({len(compare)} vintage x MOB)", True,
                bool((compare["difference"] >= -1e-12).all() and (rel <= STATIC_POOL_TOLERANCE).all()), "vintage.py")
    summary = {
        "loans": int(len(loans)), "usable_loans": int(keep.sum()),
        "left_truncated_loans": int(loans["left_truncated"].sum()),
        "left_truncated_bad_at_entry": int((loans["left_truncated"] & loans["bad_at_entry"]).sum()),
        "time_before_entry": int((loans["time"] < loans["entry"]).sum()),
        "causes": {name: int((loans.loc[keep, "cause"] == code).sum())
                   for name, code in (("default", DEFAULT), ("payoff", PAYOFF), ("censored", CENSORED))},
        "last_mob": int(overall["mob"].max()) if not overall.empty else None,
    }
    return pd.DataFrame(rows), {"summary": summary, "static_pool_comparison": compare}
