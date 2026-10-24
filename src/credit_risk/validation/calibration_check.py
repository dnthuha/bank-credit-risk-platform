"""Stage 6.4: calibration checked by the validator.

Discrimination says whether the ranking is right; calibration says whether
the PD *numbers* are right. Both models were Platt-calibrated on the
calibration split in Stage 5; here their calibrated PDs are checked again,
from the rescored package, on every split with a target. The untouched
test split is the headline.

What is computed, all in numpy (no sklearn)
    mean PD vs bad rate     calibration in the large;
    Brier, log loss         overall accuracy of the probabilities;
    slope and intercept     logit P(y) = a + b logit(PD), by Newton-Raphson
                            maximum likelihood, with standard errors from
                            the observed information, so the report can say
                            whether b differs from 1 and a from 0 beyond
                            sampling noise (ideal b = 1, a = 0; b < 1 means
                            PDs too extreme, a > 0 means PDs too low overall);
    intercept in the large  a with b fixed at 1;
    reliability table       equal-count PD bands: observed bads against the
                            sum of PDs (expected), the ratio, and a z-score
                            (observed - expected) / sqrt(sum PD (1 - PD)).
                            A band with |z| above the configured limit is
                            flagged as systematic under- or over-prediction;
    Hosmer-Lemeshow         chi-square over the bands, with its p-value.

Published figures (the development run's calibration.json: validation-split
metrics and reliability table after calibration) are compared with the
recomputed values when the artifacts are on this machine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

from credit_risk.validation.metrics import _binary_target_and_score, brier

MODELS = ("champion", "challenger")
SPLITS_WITH_TARGET = ("train", "validation", "calibration", "test")
EPS = 1e-12


class CalibrationCheckError(ValueError):
    pass


@dataclass(frozen=True)
class CalibrationCheckSpec:
    n_bins: int = 10
    headline_samples: tuple[str, ...] = ("test",)
    band_z_limit: float = 2.0                # |z| above this flags a band
    abs_tolerance_metric: float = 1e-6       # recomputed vs published mean PD / bad rate / Brier
    abs_tolerance_fit: float = 1e-4          # recomputed vs published slope / intercept. The validator's
                                             # Newton fit converges to 1e-12; development used lbfgs at
                                             # its default tolerance, so a slope off by ~1e-3 is a finding.

    @classmethod
    def from_config(cls, validation: dict[str, Any]) -> CalibrationCheckSpec:
        raw = validation.get("calibration") or {}
        n_bins = int(raw.get("n_bins", 10))
        if n_bins < 2:
            raise CalibrationCheckError("calibration.n_bins must be at least 2")
        samples = tuple(str(s) for s in raw.get("samples", ("test",)))
        unknown = set(samples) - set(SPLITS_WITH_TARGET)
        if unknown:
            raise CalibrationCheckError(f"calibration.samples: unknown split {sorted(unknown)}")
        return cls(n_bins=n_bins, headline_samples=samples,
                   band_z_limit=float(raw.get("band_z_limit", 2.0)),
                   abs_tolerance_metric=float(raw.get("abs_tolerance_metric", 1e-6)),
                   abs_tolerance_fit=float(raw.get("abs_tolerance_fit", 1e-4)))


# ---------------------------------------------------------------------------
# Own implementations
# ---------------------------------------------------------------------------

def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def log_loss(y_true, pd_hat) -> float:
    y, p = _binary_target_and_score(y_true, pd_hat)
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def logistic_recalibration(y_true, pd_hat, max_iter: int = 100, tol: float = 1e-12) -> dict[str, float]:
    """Maximum-likelihood a, b in logit P(y) = a + b logit(PD), by Newton-Raphson, with standard errors.

    Also the intercept in the large (a with b fixed at 1). Starting from (0, 1), the
    identity, so a well-calibrated model converges in a few steps.
    """
    y, p = _binary_target_and_score(y_true, pd_hat)
    x = _logit(p)
    X = np.column_stack([np.ones_like(x), x])
    beta = np.array([0.0, 1.0])
    for _ in range(max_iter):
        q = _sigmoid(X @ beta)
        w = q * (1 - q)
        gradient = X.T @ (y - q)
        hessian = (X * w[:, None]).T @ X
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError as exc:
            raise CalibrationCheckError("recalibration fit is singular (constant PD?)") from exc
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            break
    else:
        raise CalibrationCheckError("recalibration fit did not converge")
    q = _sigmoid(X @ beta)
    covariance = np.linalg.inv((X * (q * (1 - q))[:, None]).T @ X)
    se = np.sqrt(np.diag(covariance))

    a = 0.0   # intercept in the large: one parameter, offset x
    for _ in range(max_iter):
        q1 = _sigmoid(a + x)
        step = np.sum(y - q1) / np.sum(q1 * (1 - q1))
        a += step
        if abs(step) < tol:
            break
    return {
        "intercept": float(beta[0]), "intercept_se": float(se[0]),
        "slope": float(beta[1]), "slope_se": float(se[1]),
        "slope_z_vs_1": float((beta[1] - 1) / se[1]),
        "intercept_z_vs_0": float(beta[0] / se[0]),
        "intercept_in_the_large": float(a),
    }


def reliability_bands(y_true, pd_hat, n_bins: int, z_limit: float) -> pd.DataFrame:
    """Equal-count PD bands, lowest PD first, with observed vs expected bads and a z-score per band.

    Bands are formed on the rank of PD (ties broken by order), as the development report did,
    so the two tables line up band for band.
    """
    y, p = _binary_target_and_score(y_true, pd_hat)
    frame = pd.DataFrame({"pd": p, "bad": y})
    frame["band"] = pd.qcut(frame["pd"].rank(method="first"), n_bins, labels=False) + 1
    grouped = frame.groupby("band")
    out = grouped.agg(n=("bad", "size"), pd_min=("pd", "min"), pd_max=("pd", "max"),
                      mean_pd=("pd", "mean"), bad_rate=("bad", "mean"), observed=("bad", "sum"))
    out["expected"] = grouped["pd"].sum()
    out["variance"] = grouped["pd"].apply(lambda s: float((s * (1 - s)).sum()))
    out["ratio"] = out["bad_rate"] / out["mean_pd"]
    out["z"] = (out["observed"] - out["expected"]) / np.sqrt(out["variance"])
    out["flag"] = np.select([out["z"] > z_limit, out["z"] < -z_limit], ["PD thấp (thiếu)", "PD cao (thừa)"], "")
    out["observed"] = out["observed"].astype(int)
    return out.reset_index()


def hosmer_lemeshow(bands: pd.DataFrame) -> dict[str, float]:
    """Chi-square of observed vs expected bads (and goods) over the bands; df = bands - 2."""
    o, e, n = bands["observed"].to_numpy(float), bands["expected"].to_numpy(float), bands["n"].to_numpy(float)
    statistic = float(np.sum((o - e) ** 2 / (e * (1 - e / n))))
    df = max(int(len(bands) - 2), 1)
    return {"hl_statistic": statistic, "hl_df": df, "hl_p_value": float(chi2.sf(statistic, df))}


# ---------------------------------------------------------------------------
# Recompute for both models on every split
# ---------------------------------------------------------------------------

def recompute_calibration(scored: pd.DataFrame, spec: CalibrationCheckSpec) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(summary, bands): one summary row per (model, split); one bands row per (model, split, band).

    `scored` holds TARGET, split, pd_champion and pd_challenger (from score_with_package joined to
    the applicants). Rows without a split (application_test) have no target and are skipped.
    """
    for column in ("TARGET", "split", *(f"pd_{m}" for m in MODELS)):
        if column not in scored.columns:
            raise CalibrationCheckError(f"scored frame lacks '{column}'")
    summary, bands = [], []
    for model in MODELS:
        for split in SPLITS_WITH_TARGET:
            part = scored[scored["split"] == split]
            if part.empty:
                continue
            y = part["TARGET"].to_numpy(dtype=float)
            if np.isnan(y).any():
                raise CalibrationCheckError(f"split {split} has applicants without TARGET")
            y = y.astype(int)
            p = part[f"pd_{model}"].to_numpy(dtype=float)
            fit = logistic_recalibration(y, p)
            table = reliability_bands(y, p, spec.n_bins, spec.band_z_limit)
            hl = hosmer_lemeshow(table)
            flagged = table[table["flag"] != ""]
            summary.append({
                "model": model, "split": split, "headline": split in spec.headline_samples,
                "n": int(len(y)), "bads": int(y.sum()),
                "mean_pd": float(p.mean()), "bad_rate": float(y.mean()),
                "ratio_bad_rate_to_pd": float(y.mean() / p.mean()),
                "brier": brier(y, p), "log_loss": log_loss(y, p),
                **fit, **hl,
                "bands_flagged": int(len(flagged)),
                "bands_flagged_list": ", ".join(f"{int(b)}:{f}" for b, f in zip(flagged["band"], flagged["flag"])),
            })
            bands.append(table.assign(model=model, split=split))
    if not summary:
        raise CalibrationCheckError("no split with a target to evaluate")
    return pd.DataFrame(summary), pd.concat(bands, ignore_index=True)[
        ["model", "split", "band", "n", "pd_min", "pd_max", "mean_pd", "bad_rate", "observed", "expected", "ratio", "z", "flag"]]


# ---------------------------------------------------------------------------
# Checks against what development published
# ---------------------------------------------------------------------------

def calibration_checks(summary: pd.DataFrame, bands: pd.DataFrame, spec: CalibrationCheckSpec,
                       published: dict[str, dict[str, Any]] | None = None) -> pd.DataFrame:
    """Internal consistency on every row and, when given, published vs recomputed.

    `published` is {model: {"after": {split: {mean_pd, bad_rate, brier, slope, intercept}},
    "reliability": [band rows on validation]}}, as calibration.json holds it.
    """
    rows = []

    def add(check, claimed, recounted, source, tolerance):
        ok = claimed is not None and recounted is not None and abs(float(claimed) - float(recounted)) <= tolerance
        rows.append({"check": check, "claimed": claimed, "recounted": recounted, "source": source, "ok": bool(ok)})

    for r in summary.itertuples(index=False):
        part = bands[(bands["model"] == r.model) & (bands["split"] == r.split)]
        add(f"bands cover every applicant ({r.model}, {r.split})", r.n, int(part["n"].sum()), "recount", 0)
        add(f"bands observed bads sum to the split's bads ({r.model}, {r.split})", r.bads, int(part["observed"].sum()),
            "recount", 0)
        add(f"expected bads = mean PD x n ({r.model}, {r.split})", r.mean_pd * r.n, float(part["expected"].sum()),
            "recount", 1e-6 * r.n)
    for model, payload in (published or {}).items():
        for split, metrics in (payload.get("after") or {}).items():
            ours = summary[(summary["model"] == model) & (summary["split"] == split)]
            if ours.empty:
                continue
            row = ours.iloc[0]
            for metric in ("mean_pd", "bad_rate", "brier"):
                if metric in metrics:
                    add(f"published {metric} ({model}, {split})", metrics[metric], float(row[metric]),
                        "development run", spec.abs_tolerance_metric)
            # Development reports the slope of the two-parameter fit (at its solver's default
            # tolerance) and, as "intercept", the intercept in the large (slope fixed at 1).
            if "slope" in metrics:
                add(f"published slope ({model}, {split})", metrics["slope"], float(row["slope"]),
                    "development run", spec.abs_tolerance_fit)
            if "intercept" in metrics:
                add(f"published intercept in the large ({model}, {split})", metrics["intercept"],
                    float(row["intercept_in_the_large"]), "development run", spec.abs_tolerance_fit)
        reliability = payload.get("reliability") or []
        ours = bands[(bands["model"] == model) & (bands["split"] == "validation")]
        if reliability and len(reliability) == len(ours):
            for claimed, (_, mine) in zip(reliability, ours.iterrows()):
                add(f"published reliability band {int(claimed['band'])} bad rate ({model}, validation)",
                    claimed["bad_rate"], float(mine["bad_rate"]), "development run", spec.abs_tolerance_metric)
                add(f"published reliability band {int(claimed['band'])} mean PD ({model}, validation)",
                    claimed["mean_pd"], float(mine["mean_pd"]), "development run", spec.abs_tolerance_metric)
    return pd.DataFrame(rows)


def published_from_calibrate_artifacts(calibrate_dir) -> dict[str, dict[str, Any]]:
    """The development run's calibration.json, reduced to what the checks compare."""
    import json
    from pathlib import Path

    path = Path(calibrate_dir) / "calibration.json"
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {model: {"after": raw[model].get("after", {}), "reliability": raw[model].get("reliability", [])}
            for model in MODELS if model in raw}


def calibration_markdown(summary: pd.DataFrame, bands: pd.DataFrame, checks: pd.DataFrame,
                         spec: CalibrationCheckSpec) -> str:
    lines = ["## Hiệu chỉnh (tính lại)", "",
             "| Mô hình | Tập | Bad rate | PD TB | Tỷ số | Brier | Slope (SE) | Intercept (SE) | Intercept lớn | HL p | Band lệch |",
             "|---|---|---:|---:|---:|---:|---|---|---:|---:|---|"]
    for r in summary.itertuples(index=False):
        mark = "**" if r.headline else ""
        lines.append(f"| {r.model} | {mark}{r.split}{mark} | {r.bad_rate:.4%} | {r.mean_pd:.4%} | {r.ratio_bad_rate_to_pd:.3f} | "
                     f"{r.brier:.5f} | {r.slope:.3f} ({r.slope_se:.3f}) | {r.intercept:+.3f} ({r.intercept_se:.3f}) | "
                     f"{r.intercept_in_the_large:+.3f} | {r.hl_p_value:.3f} | {r.bands_flagged_list or '-'} |")
    for r in summary[summary["headline"]].itertuples(index=False):
        part = bands[(bands["model"] == r.model) & (bands["split"] == r.split)]
        lines += ["", f"### {r.model}, {r.split}: {spec.n_bins} band theo PD", "",
                  "| Band | PD từ - đến | Hồ sơ | Bad quan sát | Bad kỳ vọng | Bad rate | PD TB | Tỷ số | z | Cờ |",
                  "|---:|---|---:|---:|---:|---:|---:|---:|---:|---|"]
        for b in part.itertuples(index=False):
            lines.append(f"| {int(b.band)} | {b.pd_min:.4f} - {b.pd_max:.4f} | {b.n:,} | {b.observed:,} | {b.expected:,.1f} | "
                         f"{b.bad_rate:.4%} | {b.mean_pd:.4%} | {b.ratio:.3f} | {b.z:+.2f} | {b.flag} |")
    failed = checks[~checks["ok"]]
    lines += ["", f"Kiểm tra: {len(checks)}; lệch: {len(failed)}. Band bị cờ khi |z| > {spec.band_z_limit:g}."]
    if not failed.empty:
        lines += ["", "| Kiểm tra | Công bố | Tính lại | Nguồn |", "|---|---:|---:|---|"]
        for r in failed.itertuples(index=False):
            lines.append(f"| {r.check} | {float(r.claimed):.6f} | {float(r.recounted):.6f} | {r.source} |")
    return "\n".join(lines) + "\n"
