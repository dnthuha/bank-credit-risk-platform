"""Stage 4 - the challenger: LightGBM on the raw applicant-level features.

Inputs
    Every feature of the Stage 2.3 table (the availability matrix already kept
    out anything not known at application), as raw values: LightGBM routes
    missing values itself and splits categoricals natively. The categories are
    frozen from train; a category train never showed becomes missing.

Search
    `max_trials` configurations drawn without replacement from a small grid
    (seeded, so the same trials every run), each boosted until the validation
    AUC stops improving for `early_stopping_rounds`. Every trial is logged.
    Selection: the trials within `selection_tolerance` validation AUC of the best
    are treated as tied - a gap that small is sampling noise - and among them the
    one with the smallest train - validation AUC gap (the least overfit) wins.
    Validation therefore picks both the number of trees and the configuration:
    its AUC is optimistic for the challenger. The fair comparison with the
    champion is on test, in stage 6.

Explanations
    TreeSHAP contributions from LightGBM itself (`pred_contrib`), in log-odds of
    bad: base value + one contribution per feature = the raw score. Reason codes
    are the `reason_codes` features that pushed the risk up the most (> 0).

Deterministic: `deterministic`, column-wise histograms and a fixed thread count,
so the same data and configuration give the same trees on any machine.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass, field
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd


class ChallengerError(ValueError):
    pass


@dataclass(frozen=True)
class ChallengerSpec:
    search_space: dict[str, list]
    max_trials: int = 30
    early_stopping_rounds: int = 100
    num_boost_round: int = 5000
    learning_rate: float = 0.05
    num_threads: int = 4
    selection_tolerance: float = 0.001   # validation AUC closer than this to the best counts as a tie
    reason_codes: int = 4
    shap_examples: int = 5
    seed: int = 42

    @classmethod
    def from_config(cls, model_dev: dict[str, Any]) -> ChallengerSpec:
        raw = model_dev.get("challenger") or {}
        explain = model_dev.get("explainability") or {}
        space = {str(k): list(v) for k, v in (raw.get("search_space") or {}).items()}
        spec = cls(
            search_space=space,
            max_trials=int(raw.get("max_trials", 30)),
            early_stopping_rounds=int(raw.get("early_stopping_rounds", 100)),
            num_boost_round=int(raw.get("num_boost_round", 5000)),
            learning_rate=float(raw.get("learning_rate", 0.05)),
            num_threads=int(raw.get("num_threads", 4)),
            selection_tolerance=float(raw.get("selection_tolerance_auc", 0.001)),
            reason_codes=int(explain.get("reason_codes_per_applicant", 4)),
            shap_examples=int(explain.get("shap_waterfall_examples", 5)),
            seed=int(model_dev.get("seed", 42)),
        )
        if raw.get("model", "lightgbm") != "lightgbm":
            raise ChallengerError("challenger.model: only lightgbm is supported")
        if not space or any(not values for values in space.values()):
            raise ChallengerError("challenger.search_space needs at least one value per parameter")
        if spec.max_trials < 1 or spec.early_stopping_rounds < 1 or spec.num_threads < 1:
            raise ChallengerError("challenger: max_trials, early_stopping_rounds and num_threads must be positive")
        if spec.selection_tolerance < 0:
            raise ChallengerError("challenger.selection_tolerance_auc must not be negative")
        return spec

    def trials(self) -> list[dict[str, Any]]:
        """The configurations to try: a seeded sample of the grid, in a fixed order."""
        names = sorted(self.search_space)
        grid = [dict(zip(names, values)) for values in itertools.product(*(self.search_space[n] for n in names))]
        rng = np.random.default_rng(self.seed)
        picked = rng.choice(len(grid), size=min(self.max_trials, len(grid)), replace=False)
        return [grid[i] for i in picked]

    def base_params(self) -> dict[str, Any]:
        return {
            "objective": "binary",
            "metric": "auc",
            "learning_rate": self.learning_rate,
            "seed": self.seed,
            "deterministic": True,
            "force_col_wise": True,
            "num_threads": self.num_threads,
            "verbose": -1,
        }


def frozen_categories(train: pd.DataFrame, categorical: list[str]) -> dict[str, list[str]]:
    """The categories of each categorical feature, as train showed them (sorted, so order never varies)."""
    return {c: sorted(train[c].dropna().astype(str).unique()) for c in categorical}


def design_matrix(frame: pd.DataFrame, features: list[str], categories: dict[str, list[str]]) -> pd.DataFrame:
    """Raw features in a fixed column order; categoricals on the train categories, unseen -> missing."""
    columns = {}
    for f in features:
        if f in categories:
            text = frame[f].astype("object").map(lambda v: None if pd.isna(v) else str(v))
            known = text.where(text.isin(categories[f]))
            columns[f] = pd.Categorical(known, categories=categories[f])
        else:
            columns[f] = pd.to_numeric(frame[f], errors="raise").astype(float).to_numpy()
    return pd.DataFrame(columns, index=frame.index)


@dataclass
class Challenger:
    features: list[str]
    categories: dict[str, list[str]]
    params: dict[str, Any]
    best_iteration: int
    booster: lgb.Booster
    spec: ChallengerSpec
    trials: pd.DataFrame = field(default_factory=pd.DataFrame)

    def matrix(self, frame: pd.DataFrame) -> pd.DataFrame:
        return design_matrix(frame, self.features, self.categories)

    def predict_pd(self, frame: pd.DataFrame) -> np.ndarray:
        return self.booster.predict(self.matrix(frame), num_iteration=self.best_iteration)

    def contributions(self, frame: pd.DataFrame) -> pd.DataFrame:
        """TreeSHAP in log-odds of bad: one column per feature plus `base_value`; rows sum to the raw score."""
        raw = self.booster.predict(self.matrix(frame), num_iteration=self.best_iteration, pred_contrib=True)
        out = pd.DataFrame(raw[:, :-1], columns=self.features, index=frame.index)
        out["base_value"] = raw[:, -1]
        return out

    def reason_codes(self, contributions: pd.DataFrame) -> list[list[str]]:
        """Up to `spec.reason_codes` features per row with the largest positive push towards bad."""
        values = contributions[self.features].to_numpy()
        by_name = np.argsort(self.features, kind="stable")
        order = by_name[np.argsort(-values[:, by_name], axis=1, kind="stable")]
        return [[self.features[i] for i in row_order if row[i] > 0]
                for row, row_order in zip(values, order[:, : self.spec.reason_codes])]

    def model_string(self) -> str:
        return self.booster.model_to_string(num_iteration=self.best_iteration)

    def fingerprint(self) -> str:
        """Hash of the trees and of how inputs are prepared. Only the tree section of the model string
        counts: the parameter dump after it lists machine-level settings that do not change a prediction."""
        text = self.model_string()
        trees = text[text.index("Tree=0"): text.index("end of trees")]
        body = json.dumps({"features": self.features, "categories": self.categories, "trees": trees},
                          sort_keys=True)
        return hashlib.sha256(body.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "features": self.features,
            "categories": self.categories,
            "params": self.params,
            "best_iteration": self.best_iteration,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any], model_string: str, spec: ChallengerSpec) -> Challenger:
        return cls(
            features=list(payload["features"]),
            categories={k: list(v) for k, v in payload["categories"].items()},
            params=dict(payload["params"]),
            best_iteration=int(payload["best_iteration"]),
            booster=lgb.Booster(model_str=model_string),
            spec=spec,
        )


def fit_challenger(train: pd.DataFrame, validation: pd.DataFrame, features: list[str], categorical: list[str],
                   target: str, spec: ChallengerSpec) -> Challenger:
    """Run the search: fit on train, early-stop and select on validation."""
    for name, part in (("train", train), ("validation", validation)):
        if not set(np.unique(part[target])) == {0, 1}:
            raise ChallengerError(f"{name} must contain both goods and bads")
    categories = frozen_categories(train, categorical)
    X_train = design_matrix(train, features, categories)
    X_valid = design_matrix(validation, features, categories)
    y_train, y_valid = train[target].to_numpy(dtype=int), validation[target].to_numpy(dtype=int)

    log, boosters = [], {}
    for i, trial in enumerate(spec.trials(), 1):
        params = {**spec.base_params(), **trial}
        # Datasets are rebuilt per trial: min_child_samples etc. are fixed at construction.
        dtrain = lgb.Dataset(X_train, y_train, categorical_feature=categorical or "auto", free_raw_data=False)
        dvalid = lgb.Dataset(X_valid, y_valid, reference=dtrain)
        evals: dict[str, Any] = {}
        booster = lgb.train(params, dtrain, spec.num_boost_round, valid_sets=[dtrain, dvalid],
                            valid_names=["train", "validation"],
                            callbacks=[lgb.early_stopping(spec.early_stopping_rounds, verbose=False),
                                       lgb.record_evaluation(evals)])
        it = booster.best_iteration
        row = {"trial": i, **trial, "best_iteration": it,
               "auc_train": evals["train"]["auc"][it - 1], "auc_validation": evals["validation"]["auc"][it - 1]}
        log.append(row)
        boosters[i] = (booster, params)

    trials = pd.DataFrame(log)
    trials["gap"] = trials["auc_train"] - trials["auc_validation"]
    # Differences in validation AUC below the tolerance are noise: among the trials that close to
    # the best, take the one that overfits least (smallest train - validation gap; ties: earlier).
    trials["within_tolerance"] = trials["auc_validation"] >= trials["auc_validation"].max() - spec.selection_tolerance
    chosen = trials[trials["within_tolerance"]].sort_values(["gap", "trial"]).iloc[0]
    trials["selected"] = trials["trial"] == chosen["trial"]
    booster, params = boosters[int(chosen["trial"])]
    return Challenger(features, categories, params, int(chosen["best_iteration"]), booster, spec, trials)
