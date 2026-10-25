"""Stage 9.2: the 3-month 90+ models, baseline and challenger.

Train sample (decision 2026-10-25): every train event and a seeded random share
`non_event_rate` of the train non-events, each carrying weight 1 / rate, so the
weighted sample has the train event rate and the PDs are not pushed up. The
draw hashes (loan_id, period, seed) in DuckDB, so it does not depend on row order.

  baseline    logistic regression on binned features: every numeric feature is
              cut into weighted train quantiles (a value holding at least half
              the sample, like 999 "never delinquent" or 0 "no deferral", gets a
              bin of its own; a feature with few values gets one bin per value),
              missing has its own bin, and categoricals keep the levels with at
              least `categorical_min_share` of the weighted sample (rarer -> other).
              One-hot, L2.
  challenger  LightGBM on the raw features (NaN native, categoricals on the same
              kept levels), a small grid, early stopping on validation average
              precision; the trial with the best validation selection metric wins.

The model used for the watchlist is the one with the higher selection metric on
validation (baseline on a tie). Validation is used for early stopping and for
that choice; stress test and out-of-time are never seen before evaluation.
Reason codes: TreeSHAP for LightGBM; for the logistic model, a feature's
contribution is its bin coefficient minus the weighted train mean of that
feature's coefficients, so both say how far a feature pushes a loan above an
average loan, in log-odds.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression

from credit_risk.ews.metrics import average_precision
from credit_risk.validation.metrics import auc

OTHER, MISSING = "other", "missing"
FEW_VALUES = 12           # a numeric feature with at most this many distinct values gets one bin per value
QUANTILES = np.linspace(0.1, 0.9, 9)
MODAL_SHARE = 0.5


@dataclass(frozen=True)
class ModelSpec:
    numeric: list[str]
    categorical: list[str]
    non_event_rate: float
    categorical_min_share: float
    baseline: dict[str, Any]
    challenger: dict[str, Any]
    selection_metric: str
    seed: int
    num_threads: int

    @classmethod
    def from_config(cls, ews: dict, num_threads: int = 4) -> "ModelSpec":
        cfg = ews["ews"]
        rate = float(cfg["train_sample"]["non_event_rate"])
        if not 0 < rate <= 1:
            raise ValueError(f"ews.train_sample.non_event_rate must be in (0, 1], got {rate}")
        models = cfg["models"]
        if models["selection_metric"] not in ("pr_auc", "roc_auc"):
            raise ValueError("ews.models.selection_metric must be pr_auc or roc_auc")
        if not models["challenger"].get("grid"):
            raise ValueError("ews.models.challenger.grid must list at least one configuration")
        return cls(numeric=list(cfg["features"]["numeric"]), categorical=list(cfg["features"]["categorical"]),
                   non_event_rate=rate, categorical_min_share=float(cfg["features"].get("categorical_min_share", 0.001)),
                   baseline=dict(models["baseline"]), challenger=dict(models["challenger"]),
                   selection_metric=str(models["selection_metric"]), seed=int(ews.get("seed", 0)), num_threads=num_threads)

    @property
    def features(self) -> list[str]:
        return self.numeric + self.categorical


def train_sample_where(rate: float, seed: int) -> str:
    """SQL condition: every train event, and a seeded share `rate` of the train non-events."""
    threshold = int(round(rate * 1_000_000))
    return (f"split = 'train' AND (y = 1 OR hash(loan_id || '|' || CAST(period AS VARCHAR) || '|{seed}') "
            f"% 1000000 < {threshold})")


def weights(y: np.ndarray, rate: float) -> np.ndarray:
    return np.where(np.asarray(y) == 1, 1.0, 1.0 / rate)


def _weighted_quantiles(x: np.ndarray, w: np.ndarray, qs: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    x, w = x[order], w[order]
    cum = np.cumsum(w) / w.sum()
    return x[np.minimum(np.searchsorted(cum, qs, side="left"), len(x) - 1)]


# ---------------------------------------------------------------------------
# Encoding shared by both models
# ---------------------------------------------------------------------------

@dataclass
class Encoder:
    numeric: list[str]
    categorical: list[str]
    levels: dict[str, list[str]] = field(default_factory=dict)        # kept categorical levels (sorted)
    bins: dict[str, dict[str, Any]] = field(default_factory=dict)     # numeric binning for the baseline
    columns: list[str] = field(default_factory=list)                 # one-hot column names, in order

    @classmethod
    def fit(cls, train: pd.DataFrame, w: np.ndarray, numeric: list[str], categorical: list[str], min_share: float) -> "Encoder":
        enc = cls(numeric=list(numeric), categorical=list(categorical))
        total = w.sum()
        for c in categorical:
            text = enc._text(train[c])
            share = pd.Series(w).groupby(text.to_numpy()).sum() / total
            enc.levels[c] = sorted(str(k) for k, v in share.items() if v >= min_share and k != MISSING)
        for f in numeric:
            x = pd.to_numeric(train[f], errors="coerce").to_numpy(dtype=float)
            ok = ~np.isnan(x)
            values = np.unique(x[ok])
            if len(values) <= FEW_VALUES:
                enc.bins[f] = {"kind": "values", "values": values.tolist()}
                continue
            share = pd.Series(w[ok]).groupby(x[ok]).sum() / w[ok].sum()
            modal = float(share.idxmax()) if share.max() >= MODAL_SHARE else None
            rest = ok & (x != modal) if modal is not None else ok
            edges = np.unique(_weighted_quantiles(x[rest], w[rest], QUANTILES)) if rest.any() else np.array([])
            enc.bins[f] = {"kind": "quantile", "edges": edges.tolist(), "modal": modal}
        enc.columns = [col for f in numeric + categorical for col in enc._labels(f)]
        return enc

    @staticmethod
    def _text(s: pd.Series) -> pd.Series:
        return s.astype("string").fillna(MISSING).astype(object)

    def category(self, c: str, s: pd.Series) -> pd.Series:
        text = self._text(s)
        return text.where(text.isin(self.levels[c]) | text.eq(MISSING), OTHER)

    def _labels(self, f: str) -> list[str]:
        if f in self.levels:
            return [f"{f}={v}" for v in self.levels[f] + [OTHER, MISSING]]
        b = self.bins[f]
        if b["kind"] == "values":
            names = [repr(v) for v in b["values"]] + [OTHER]
        else:
            names = ([f"=={b['modal']!r}"] if b["modal"] is not None else []) + [f"bin{i}" for i in range(len(b["edges"]) + 1)]
        return [f"{f}:{n}" for n in names + [MISSING]]

    def _codes(self, f: str, s: pd.Series) -> np.ndarray:
        """Position of each row's column within the feature's labels (see `_labels`)."""
        if f in self.levels:
            cats = self.levels[f] + [OTHER, MISSING]
            return pd.Categorical(self.category(f, s), categories=cats).codes.astype(np.int64)
        x = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
        b = self.bins[f]
        missing = np.isnan(x)
        if b["kind"] == "values":
            values = np.asarray(b["values"], dtype=float)
            pos = np.minimum(np.searchsorted(values, x), max(len(values) - 1, 0))
            hit = (values[pos] == x) if len(values) else np.zeros(len(x), dtype=bool)
            codes = np.where(hit, pos, len(values))
            n_other = len(values) + 1
        else:
            shift = 1 if b["modal"] is not None else 0
            codes = shift + np.searchsorted(np.asarray(b["edges"], dtype=float), x, side="right")
            if shift:
                codes = np.where(x == b["modal"], 0, codes)
            n_other = shift + len(b["edges"]) + 1
        return np.where(missing, n_other, codes).astype(np.int64)

    def one_hot(self, frame: pd.DataFrame) -> sparse.csr_matrix:
        n, offset, cols = len(frame), 0, []
        for f in self.numeric + self.categorical:
            cols.append(offset + self._codes(f, frame[f]))
            offset += len(self._labels(f))
        j = np.column_stack(cols).ravel()
        r = np.repeat(np.arange(n), len(cols))
        return sparse.csr_matrix((np.ones(len(r)), (r, j)), shape=(n, len(self.columns)))

    def feature_columns(self, f: str) -> slice:
        start = 0
        for g in self.numeric + self.categorical:
            size = len(self._labels(g))
            if g == f:
                return slice(start, start + size)
            start += size
        raise KeyError(f)

    def lgb_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = {f: pd.to_numeric(frame[f], errors="coerce").to_numpy(dtype=float) for f in self.numeric}
        for c in self.categorical:
            out[c] = pd.Categorical(self.category(c, frame[c]), categories=self.levels[c] + [OTHER, MISSING])
        return pd.DataFrame(out, index=frame.index)

    def to_dict(self) -> dict[str, Any]:
        return {"numeric": self.numeric, "categorical": self.categorical, "levels": self.levels, "bins": self.bins}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

@dataclass
class Baseline:
    encoder: Encoder
    model: LogisticRegression
    reference: dict[str, float]          # weighted train mean of each feature's coefficient

    name = "baseline"

    def predict_pd(self, frame: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self.encoder.one_hot(frame))[:, 1]

    def contributions(self, frame: pd.DataFrame) -> pd.DataFrame:
        X = self.encoder.one_hot(frame)
        coef = self.model.coef_[0]
        out = {}
        for f in self.encoder.numeric + self.encoder.categorical:
            cols = self.encoder.feature_columns(f)
            out[f] = np.asarray(X[:, cols] @ coef[cols]).ravel() - self.reference[f]
        return pd.DataFrame(out, index=frame.index)

    def fingerprint(self) -> str:
        body = {"encoder": self.encoder.to_dict(), "coef": np.round(self.model.coef_[0], 10).tolist(),
                "intercept": round(float(self.model.intercept_[0]), 10)}
        return hashlib.sha256(json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest()

    def coefficients(self) -> pd.DataFrame:
        return pd.DataFrame({"column": self.encoder.columns, "coef": self.model.coef_[0]})


@dataclass
class Challenger:
    encoder: Encoder
    booster: lgb.Booster
    best_iteration: int
    params: dict[str, Any]
    trials: pd.DataFrame

    name = "challenger"

    def predict_pd(self, frame: pd.DataFrame) -> np.ndarray:
        return self.booster.predict(self.encoder.lgb_frame(frame), num_iteration=self.best_iteration)

    def contributions(self, frame: pd.DataFrame) -> pd.DataFrame:
        raw = self.booster.predict(self.encoder.lgb_frame(frame), num_iteration=self.best_iteration, pred_contrib=True)
        return pd.DataFrame(raw[:, :-1], columns=self.encoder.numeric + self.encoder.categorical, index=frame.index)

    def fingerprint(self) -> str:
        text = self.booster.model_to_string(num_iteration=self.best_iteration)
        trees = text[text.index("Tree=0"): text.index("end of trees")]
        body = json.dumps({"encoder": self.encoder.to_dict(), "trees": trees}, sort_keys=True)
        return hashlib.sha256(body.encode("utf-8")).hexdigest()

    def importance(self) -> pd.DataFrame:
        gain = self.booster.feature_importance("gain", iteration=self.best_iteration)
        names = self.booster.feature_name()
        frame = pd.DataFrame({"feature": names, "gain": gain})
        frame["gain_share"] = frame["gain"] / frame["gain"].sum()
        return frame.sort_values(["gain", "feature"], ascending=[False, True]).reset_index(drop=True)


def fit_baseline(train: pd.DataFrame, spec: ModelSpec, encoder: Encoder) -> Baseline:
    y = train["y"].to_numpy(dtype=int)
    w = weights(y, spec.non_event_rate)
    model = LogisticRegression(C=float(spec.baseline.get("C", 1.0)), max_iter=int(spec.baseline.get("max_iter", 2000)))
    X = encoder.one_hot(train)
    model.fit(X, y, sample_weight=w)
    coef = model.coef_[0]
    reference = {}
    for f in encoder.numeric + encoder.categorical:
        cols = encoder.feature_columns(f)
        reference[f] = float(np.average(np.asarray(X[:, cols] @ coef[cols]).ravel(), weights=w))
    return Baseline(encoder, model, reference)


def fit_challenger(train: pd.DataFrame, validation: pd.DataFrame, spec: ModelSpec, encoder: Encoder) -> Challenger:
    y_train, y_valid = train["y"].to_numpy(dtype=int), validation["y"].to_numpy(dtype=int)
    X_train, X_valid = encoder.lgb_frame(train), encoder.lgb_frame(validation)
    cfg = spec.challenger
    base = {"objective": "binary", "metric": "average_precision", "seed": spec.seed, "deterministic": True,
            "force_col_wise": True, "num_threads": spec.num_threads, "verbose": -1, **cfg.get("common", {})}
    log, fitted = [], {}
    for i, trial in enumerate(cfg["grid"], 1):
        params = {**base, **trial}
        dtrain = lgb.Dataset(X_train, y_train, weight=weights(y_train, spec.non_event_rate),
                             categorical_feature=encoder.categorical or "auto", free_raw_data=False)
        dvalid = lgb.Dataset(X_valid, y_valid, reference=dtrain)
        booster = lgb.train(params, dtrain, int(cfg.get("n_estimators", 2000)), valid_sets=[dvalid], valid_names=["validation"],
                            callbacks=[lgb.early_stopping(int(cfg.get("early_stopping_rounds", 100)), verbose=False)])
        it = booster.best_iteration or booster.current_iteration()
        p = booster.predict(X_valid, num_iteration=it)
        score = average_precision(y_valid, p) if spec.selection_metric == "pr_auc" else auc(y_valid, p)
        log.append({"trial": i, **trial, "best_iteration": it, f"validation_{spec.selection_metric}": score})
        fitted[i] = (booster, it, params)
    trials = pd.DataFrame(log)
    best = int(trials.sort_values([f"validation_{spec.selection_metric}", "trial"], ascending=[False, True])["trial"].iloc[0])
    booster, it, params = fitted[best]
    trials["selected"] = trials["trial"] == best
    return Challenger(encoder, booster, it, params, trials)

