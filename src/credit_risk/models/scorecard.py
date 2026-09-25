"""Stage 3 - the champion: a WoE logistic regression scaled to integer points.

Fit (3.1)
    A logistic regression of TARGET on the WoE of the shortlisted features,
    fitted on the train split only. WoE = ln(%Good / %Bad), so a bin richer in
    goods has a higher WoE and every coefficient on P(bad) must be NEGATIVE. A
    positive coefficient reverses what the bins say on their own (usually a
    correlated partner soaking up the signal), and a scorecard whose riskier bin
    earns more points cannot be explained to an applicant.

    Sign stability: each round refits `bootstrap_resamples` times on stratified
    bootstrap resamples of train. A feature whose coefficient is negative in
    fewer than `min_sign_share` of them is unstable. The least stable feature
    (ties: the larger coefficient, then the name) is dropped and the model is
    refitted, until every feature is stable. Every drop is kept as an audit trail.

Points (3.2)
    Score = Offset + Factor * ln(odds), odds = Good:Bad,
    Factor = PDO / ln 2, Offset = BaseScore - Factor * ln(BaseOdds).
    With ln(odds) = -(b0 + sum b_i * WoE_i):
        base points       = Offset - Factor * b0
        points of a bin   = -Factor * b_i * WoE_bin
    Both are rounded to integers: the table a credit officer reads is the model,
    and the score is the plain sum of its integers. The intercept stays a
    separate line (base points) rather than being spread over the features.

Reason codes (3.3)
    For every feature, the points an applicant lost against the best bin of that
    feature. The `reason_codes` features with the largest loss (> 0) are the
    reasons, largest first; ties go to the feature name.

A value the binning never saw (an unseen category) scores WoE 0, hence 0 points.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from credit_risk.features.binning import FeatureBinning


class ScorecardError(ValueError):
    pass


@dataclass(frozen=True)
class ScorecardSpec:
    base_score: float
    base_odds: float
    pdo: float
    regularization: str = "l2"
    C: float = 1.0
    bootstrap_resamples: int = 50
    min_sign_share: float = 0.95
    reason_codes: int = 4
    seed: int = 42
    require_stable_sign: bool = True   # false: fit once, report the sign shares, drop nothing

    @classmethod
    def from_config(cls, model_dev: dict[str, Any], definitions: dict[str, Any]) -> ScorecardSpec:
        raw = model_dev.get("champion") or {}
        stability = raw.get("sign_stability") or {}
        scaling = definitions["scorecard_scaling"]
        spec = cls(
            base_score=float(scaling["base_score"]),
            base_odds=float(scaling["base_odds"]),
            pdo=float(scaling["pdo"]),
            regularization=str(raw.get("regularization", "l2")),
            C=float(raw.get("C", 1.0)),
            bootstrap_resamples=int(stability.get("bootstrap_resamples", 50)),
            min_sign_share=float(stability.get("min_share", 0.95)),
            reason_codes=int((model_dev.get("explainability") or {}).get("reason_codes_per_applicant", 4)),
            seed=int(model_dev.get("seed", 42)),
            require_stable_sign=bool((model_dev.get("feature_selection") or {}).get(
                "require_stable_coefficient_sign", True)),
        )
        if spec.regularization != "l2":
            raise ScorecardError("champion.regularization: only l2 is supported")
        if spec.C <= 0:
            raise ScorecardError("champion.C must be positive")
        if spec.pdo <= 0 or spec.base_odds <= 0:
            raise ScorecardError("scorecard_scaling: pdo and base_odds must be positive")
        if spec.bootstrap_resamples < 1:
            raise ScorecardError("champion.sign_stability.bootstrap_resamples must be at least 1")
        if not 0.5 < spec.min_sign_share <= 1:
            raise ScorecardError("champion.sign_stability.min_share must be in (0.5, 1]")
        if spec.reason_codes < 1:
            raise ScorecardError("explainability.reason_codes_per_applicant must be at least 1")
        return spec

    @property
    def factor(self) -> float:
        return self.pdo / math.log(2)

    @property
    def offset(self) -> float:
        return self.base_score - self.factor * math.log(self.base_odds)


# ---------------------------------------------------------------------------
# 3.1 Fit with sign stability
# ---------------------------------------------------------------------------

def _logit(X: np.ndarray, y: np.ndarray, spec: ScorecardSpec) -> LogisticRegression:
    return LogisticRegression(C=spec.C, max_iter=1000).fit(X, y)


def _negative_share(X: np.ndarray, y: np.ndarray, spec: ScorecardSpec) -> np.ndarray:
    """Share of stratified bootstrap refits in which each coefficient is negative.

    Stratified (goods and bads resampled separately) so no resample loses a class.
    The generator is re-seeded every round: a round's verdict depends only on
    the features it holds, not on how many rounds came before.
    """
    rng = np.random.default_rng(spec.seed)
    bads, goods = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    negative = np.zeros(X.shape[1])
    for _ in range(spec.bootstrap_resamples):
        idx = np.r_[rng.choice(bads, bads.size), rng.choice(goods, goods.size)]
        negative += _logit(X[idx], y[idx], spec).coef_[0] < 0
    return negative / spec.bootstrap_resamples


@dataclass
class FitResult:
    features: list[str]
    intercept: float
    coefficients: dict[str, float]
    sign_share: dict[str, float]      # final model: share of resamples with a negative coefficient
    trail: pd.DataFrame               # one row per dropped feature, in drop order
    initial: tuple[float, dict[str, float]] = (0.0, {})  # round 1, every input feature: (intercept, coefficients)


def fit_sign_stable(woe: pd.DataFrame, y, spec: ScorecardSpec) -> FitResult:
    """Fit on the WoE columns of `woe` (train rows only), dropping unstable features one at a time."""
    target = np.asarray(y)
    if not set(np.unique(target)) == {0, 1}:
        raise ScorecardError("y must contain both 0 (good) and 1 (bad)")
    kept = list(woe.columns)
    dropped = []
    initial = None
    while kept:
        X = woe[kept].to_numpy(dtype=float)
        model = _logit(X, target, spec)
        coef = model.coef_[0]
        if initial is None:
            initial = (float(model.intercept_[0]), {f: float(c) for f, c in zip(kept, coef)})
        share = _negative_share(X, target, spec)
        unstable = [i for i in range(len(kept)) if share[i] < spec.min_sign_share]
        if not unstable or not spec.require_stable_sign:
            return FitResult(
                features=kept,
                intercept=float(model.intercept_[0]),
                coefficients={f: float(c) for f, c in zip(kept, coef)},
                sign_share={f: float(s) for f, s in zip(kept, share)},
                trail=pd.DataFrame(dropped, columns=["round", "feature", "coefficient", "negative_share",
                                                     "features_before", "reason"]),
                initial=initial,
            )
        worst = min(unstable, key=lambda i: (share[i], -coef[i], kept[i]))
        reason = ("positive coefficient: reverses its own bins" if coef[worst] > 0
                  else "negative coefficient but unstable")
        dropped.append((len(dropped) + 1, kept[worst], float(coef[worst]), float(share[worst]), len(kept),
                        f"{reason} (negative in {share[worst]:.0%} of {spec.bootstrap_resamples} "
                        f"resamples, needs {spec.min_sign_share:.0%})"))
        kept.pop(worst)
    raise ScorecardError("every feature was dropped for an unstable coefficient sign")


# ---------------------------------------------------------------------------
# 3.2 Points and the scorecard object
# ---------------------------------------------------------------------------

@dataclass
class Scorecard:
    features: list[str]
    binnings: dict[str, FeatureBinning]
    intercept: float
    coefficients: dict[str, float]
    spec: ScorecardSpec
    base_points: int = 0
    points: dict[str, dict[str, int]] = field(default_factory=dict)   # feature -> bin label -> points

    @classmethod
    def build(cls, fit: FitResult, binnings: dict[str, FeatureBinning], spec: ScorecardSpec) -> Scorecard:
        card = cls(fit.features, {f: binnings[f] for f in fit.features}, fit.intercept, fit.coefficients, spec)
        card.base_points = int(round(spec.offset - spec.factor * fit.intercept))
        for f in fit.features:
            b = fit.coefficients[f]
            card.points[f] = {row["label"]: int(round(-spec.factor * b * row["woe"])) for row in binnings[f].bins}
        return card

    # -- scoring ---------------------------------------------------------
    def points_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Points per feature for every row; a bin label the table does not know scores 0."""
        out = {}
        for f in self.features:
            lookup = self.points[f]
            out[f] = np.array([lookup.get(label, 0) for label in self.binnings[f].assign(frame[f])], dtype=int)
        return pd.DataFrame(out, index=frame.index)

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        return self.base_points + self.points_frame(frame).sum(axis=1).to_numpy(dtype=int)

    def exact_log_odds_bad(self, frame: pd.DataFrame) -> np.ndarray:
        """b0 + sum b_i * WoE_i, before any rounding: the fitted model itself."""
        woe = np.column_stack([self.binnings[f].transform(frame[f]) for f in self.features])
        return self.intercept + woe @ np.array([self.coefficients[f] for f in self.features])

    def pd_from_score(self, score) -> np.ndarray:
        """Uncalibrated PD implied by the scaling: odds(good) = exp((score - Offset) / Factor)."""
        s = np.asarray(score, dtype=float)
        return 1.0 / (1.0 + np.exp((s - self.spec.offset) / self.spec.factor))

    # -- 3.3 reason codes --------------------------------------------------
    def reason_codes(self, points: pd.DataFrame) -> list[list[str]]:
        """Up to `spec.reason_codes` features per row, by points lost against the feature's best bin."""
        best = np.array([max(self.points[f].values()) for f in self.features])
        lost = best - points[self.features].to_numpy()
        # Order: most points lost first, ties by feature name (stable sort on a name-sorted order).
        by_name = np.argsort(self.features, kind="stable")
        order = by_name[np.argsort(-lost[:, by_name], axis=1, kind="stable")]
        codes = []
        for row_lost, row_order in zip(lost, order[:, : self.spec.reason_codes]):
            codes.append([self.features[i] for i in row_order if row_lost[i] > 0])
        return codes

    # -- persistence -----------------------------------------------------
    def table(self) -> pd.DataFrame:
        """The scorecard as a credit officer reads it: one row per feature x bin."""
        rows = [{"feature": "(base points)", "bin": "", "woe": None, "coefficient": None,
                 "points": self.base_points, "train_share": None, "train_bad_rate": None}]
        for f in self.features:
            for row in self.binnings[f].bins:
                rows.append({"feature": f, "bin": row["label"], "woe": row["woe"],
                             "coefficient": self.coefficients[f], "points": self.points[f][row["label"]],
                             "train_share": row["share"], "train_bad_rate": row["bad_rate"]})
        return pd.DataFrame(rows)

    def to_dict(self) -> dict[str, Any]:
        return {
            "features": self.features,
            "intercept": self.intercept,
            "coefficients": self.coefficients,
            "base_points": self.base_points,
            "points": self.points,
            "factor": self.spec.factor,
            "offset": self.spec.offset,
            "binnings": {f: self.binnings[f].to_dict() for f in self.features},
        }

    def fingerprint(self) -> str:
        body = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(body.encode("utf-8")).hexdigest()

    @classmethod
    def from_payload(cls, payload: dict[str, Any], spec: ScorecardSpec) -> Scorecard:
        return cls(
            features=list(payload["features"]),
            binnings={f: FeatureBinning.from_dict(raw) for f, raw in payload["binnings"].items()},
            intercept=float(payload["intercept"]),
            coefficients={f: float(c) for f, c in payload["coefficients"].items()},
            spec=spec,
            base_points=int(payload["base_points"]),
            points={f: {label: int(p) for label, p in bins.items()} for f, bins in payload["points"].items()},
        )


def scorecard_payload(card: Scorecard, fit: FitResult, meta: dict[str, Any]) -> dict[str, Any]:
    return {
        **meta,
        "spec": {**card.spec.__dict__, "factor": card.spec.factor, "offset": card.spec.offset},
        "fingerprint": card.fingerprint(),
        "sign_share": fit.sign_share,
        "dropped_for_sign": fit.trail.to_dict(orient="records"),
        **card.to_dict(),
    }
