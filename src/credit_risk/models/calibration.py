"""Stage 5 - PD calibration on the calibration split, then a frozen model package.

Both models rank well; calibration makes their PD mean what it says: among
applicants given PD 5%, about 5% should turn bad.

Input of every calibrator: the uncalibrated log-odds of bad of the model
(scorecard: from its score through the scaling; challenger: logit of its PD).

Methods
    none      the model's own PD, as a baseline;
    platt     PD = sigmoid(a + b * log-odds): two parameters, keeps the ranking;
    isotonic  a non-decreasing step function of the log-odds, fitted freely:
              fixes any shape of miscalibration but can overfit, and ties
              applicants it maps to one step (the ranking coarsens).

Selection
    `cv_folds`-fold stratified cross-validation inside the calibration split:
    each method is fitted on the other folds and scored (Brier) on the held-out
    fold. "none" is scored as the baseline only. Among the configured methods,
    those whose out-of-fold Brier is within `selection_tolerance_brier` of the
    best are tied, and the simplest wins (platt before isotonic). The winner is
    refitted on the whole calibration split.

Nothing here reads the test split; train and validation are only scored.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from credit_risk.lineage import round_floats
from credit_risk.validation.metrics import auc

SIMPLICITY = ("none", "platt", "isotonic")


class CalibrationError(ValueError):
    pass


@dataclass(frozen=True)
class CalibrationSpec:
    methods: tuple[str, ...] = ("platt", "isotonic")
    cv_folds: int = 5
    selection_tolerance_brier: float = 1e-4
    n_bins: int = 10
    pd_floor: float = 1e-6   # calibrated PD is kept in [pd_floor, 1 - pd_floor]
    seed: int = 42

    @classmethod
    def from_config(cls, model_dev: dict[str, Any]) -> CalibrationSpec:
        raw = model_dev.get("calibration") or {}
        spec = cls(
            methods=tuple(raw.get("methods", ("platt", "isotonic")) or ()),
            cv_folds=int(raw.get("cv_folds", 5)),
            selection_tolerance_brier=float(raw.get("selection_tolerance_brier", 1e-4)),
            n_bins=int(raw.get("n_bins_for_diagnostics", 10)),
            pd_floor=float(raw.get("pd_floor", 1e-6)),
            seed=int(model_dev.get("seed", 42)),
        )
        unknown = set(spec.methods) - {"platt", "isotonic"}
        if unknown or not spec.methods:
            raise CalibrationError(f"calibration.methods must be a subset of [platt, isotonic], got {list(spec.methods)}")
        if spec.cv_folds < 2:
            raise CalibrationError("calibration.cv_folds must be at least 2")
        if spec.selection_tolerance_brier < 0:
            raise CalibrationError("calibration.selection_tolerance_brier must not be negative")
        if not 0 < spec.pd_floor < 0.01:
            raise CalibrationError("calibration.pd_floor must be a small positive number")
        return spec


def logit(p) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    return np.log(p / (1 - p))


def sigmoid(x) -> np.ndarray:
    return 1 / (1 + np.exp(-np.asarray(x, dtype=float)))


# ---------------------------------------------------------------------------
# Calibrators
# ---------------------------------------------------------------------------

@dataclass
class Calibrator:
    method: str
    params: dict[str, Any] = field(default_factory=dict)
    pd_floor: float = 1e-6

    def fit(self, log_odds, y) -> Calibrator:
        x, target = np.asarray(log_odds, dtype=float), np.asarray(y, dtype=int)
        if self.method == "platt":
            model = LogisticRegression(C=np.inf, max_iter=1000).fit(x.reshape(-1, 1), target)
            self.params = {"a": float(model.intercept_[0]), "b": float(model.coef_[0, 0])}
        elif self.method == "isotonic":
            iso = IsotonicRegression(out_of_bounds="clip", y_min=self.pd_floor, y_max=1 - self.pd_floor).fit(x, target)
            self.params = {"x": [float(v) for v in iso.X_thresholds_], "y": [float(v) for v in iso.y_thresholds_]}
        elif self.method != "none":
            raise CalibrationError(f"unknown calibration method {self.method}")
        return self

    def apply(self, log_odds) -> np.ndarray:
        x = np.asarray(log_odds, dtype=float)
        if self.method == "none":
            p = sigmoid(x)
        elif self.method == "platt":
            p = sigmoid(self.params["a"] + self.params["b"] * x)
        else:  # isotonic: linear interpolation between thresholds, flat beyond them (as sklearn predicts)
            p = np.interp(x, self.params["x"], self.params["y"])
        return np.clip(p, self.pd_floor, 1 - self.pd_floor)

    def to_dict(self) -> dict[str, Any]:
        return {"method": self.method, "params": self.params, "pd_floor": self.pd_floor}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Calibrator:
        return cls(raw["method"], dict(raw["params"]), float(raw["pd_floor"]))


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def log_loss(y, p) -> float:
    y, p = np.asarray(y, dtype=float), np.clip(np.asarray(p, dtype=float), 1e-15, 1 - 1e-15)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def calibration_slope_intercept(y, p) -> dict[str, float]:
    """slope: b in logit P(y) = a + b logit(p) (ideal 1; < 1 means PD too extreme).
    intercept: a in logit P(y) = a + logit(p), slope fixed at 1 - calibration in the large (ideal 0;
    > 0 means PD too low on average). Both by maximum likelihood, no penalty."""
    y = np.asarray(y, dtype=float)
    x = logit(np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12))
    # tol=1e-10: at lbfgs's default tolerance the reported slope was off the maximum-likelihood
    # value by ~1e-3 (Stage 6.4 finding 2). This is a diagnostic; the Platt fit above is unchanged.
    slope = float(LogisticRegression(C=np.inf, max_iter=10000, tol=1e-10).fit(x.reshape(-1, 1), y).coef_[0, 0])
    a = 0.0
    for _ in range(50):  # Newton on the one-parameter model with offset x
        q = sigmoid(a + x)
        step = np.sum(y - q) / np.sum(q * (1 - q))
        a += step
        if abs(step) < 1e-12:
            break
    return {"slope": slope, "intercept": float(a)}


def calibration_metrics(y, p) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return {"mean_pd": float(p.mean()), "bad_rate": float(y.mean()), "brier": float(np.mean((p - y) ** 2)),
            "log_loss": log_loss(y, p), **calibration_slope_intercept(y, p)}


def reliability_table(y, p, n_bins: int = 10) -> pd.DataFrame:
    """Equal-count PD bands (lowest PD first): mean PD against the observed bad rate."""
    frame = pd.DataFrame({"pd": np.asarray(p, dtype=float), "bad": np.asarray(y, dtype=float)})
    frame["band"] = pd.qcut(frame["pd"].rank(method="first"), n_bins, labels=False) + 1
    out = frame.groupby("band").agg(n=("bad", "size"), pd_min=("pd", "min"), pd_max=("pd", "max"),
                                    mean_pd=("pd", "mean"), bad_rate=("bad", "mean"))
    out["ratio"] = out["bad_rate"] / out["mean_pd"]
    return out.reset_index()


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

@dataclass
class Selection:
    calibrator: Calibrator           # the winner, refitted on the whole calibration split
    cv: pd.DataFrame                 # one row per candidate: out-of-fold Brier, log loss, AUC, chosen
    out_of_fold: dict[str, np.ndarray]


def select_calibrator(log_odds, y, spec: CalibrationSpec) -> Selection:
    x, target = np.asarray(log_odds, dtype=float), np.asarray(y, dtype=int)
    if not set(np.unique(target)) == {0, 1}:
        raise CalibrationError("the calibration sample must contain both goods and bads")
    candidates = ["none", *[m for m in SIMPLICITY if m in spec.methods]]
    folds = StratifiedKFold(spec.cv_folds, shuffle=True, random_state=spec.seed)
    oof = {m: np.zeros(len(x)) for m in candidates}
    for fit_idx, held_idx in folds.split(x, target):
        for m in candidates:
            oof[m][held_idx] = Calibrator(m, pd_floor=spec.pd_floor).fit(x[fit_idx], target[fit_idx]).apply(x[held_idx])

    rows = [{"method": m, "brier_oof": float(np.mean((oof[m] - target) ** 2)), "log_loss_oof": log_loss(target, oof[m]),
             "auc_oof": auc(target, oof[m])} for m in candidates]
    cv = pd.DataFrame(rows)
    # "none" is the baseline the table is read against, never a winner: the configured methods are.
    eligible = cv["method"] != "none"
    best = cv.loc[eligible, "brier_oof"].min()
    cv["within_tolerance"] = eligible & (cv["brier_oof"] <= best + spec.selection_tolerance_brier)
    winner = next(m for m in candidates if cv.set_index("method").loc[m, "within_tolerance"])
    cv["chosen"] = cv["method"] == winner
    return Selection(Calibrator(winner, pd_floor=spec.pd_floor).fit(x, target), cv, oof)


# ---------------------------------------------------------------------------
# Frozen package
# ---------------------------------------------------------------------------

def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def package_fingerprint(content: dict[str, Any]) -> str:
    """Hash of what the package scores with: the model fingerprints and the calibrators.

    Deliberately not the file bytes: a rerun of the train step writes the same
    models under a new run_id, and that must not look like a new package. The
    files' SHA-256 are recorded next to it for integrity checks.
    """
    return hashlib.sha256(json.dumps(round_floats(content), sort_keys=True).encode("utf-8")).hexdigest()
