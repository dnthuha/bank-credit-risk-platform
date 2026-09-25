"""Stage 4: the challenger search is seeded and logged, inputs are prepared the
same way at fit and at scoring, and SHAP contributions add up to the prediction."""

import numpy as np
import pandas as pd
import pytest

from credit_risk.models.challenger import (
    Challenger,
    ChallengerError,
    ChallengerSpec,
    design_matrix,
    fit_challenger,
)

MODEL_DEV = {
    "seed": 42,
    "challenger": {
        "model": "lightgbm", "early_stopping_rounds": 20, "max_trials": 3, "num_boost_round": 300,
        "learning_rate": 0.1, "num_threads": 2,
        "search_space": {"num_leaves": [7, 15], "min_child_samples": [20, 50], "lambda_l2": [0.0, 1.0]},
    },
    "explainability": {"reason_codes_per_applicant": 2, "shap_waterfall_examples": 3},
}


def spec(**challenger):
    return ChallengerSpec.from_config({**MODEL_DEV, "challenger": {**MODEL_DEV["challenger"], **challenger}})


def sample(n, seed):
    rng = np.random.default_rng(seed)
    a, b = rng.normal(size=(2, n))
    grade = rng.choice(["A", "B", "C"], n, p=[0.5, 0.3, 0.2])
    logit = -2.0 + 0.8 * a - 0.5 * b + np.select([grade == "C", grade == "B"], [0.8, 0.3], 0.0)
    a[rng.random(n) < 0.1] = np.nan
    return pd.DataFrame({"a": a, "b": b, "grade": grade, "noise": rng.normal(size=n),
                         "TARGET": (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)})


FEATURES, CATEGORICAL = ["a", "b", "grade", "noise"], ["grade"]


@pytest.fixture(scope="module")
def fitted():
    train, validation = sample(6000, 0), sample(2000, 1)
    return fit_challenger(train, validation, FEATURES, CATEGORICAL, "TARGET", spec()), train, validation


def test_trials_are_a_seeded_sample_of_the_grid():
    s = spec()
    trials = s.trials()
    assert trials == s.trials()
    assert len(trials) == 3
    assert len({tuple(sorted(t.items())) for t in trials}) == 3
    assert len(spec(max_trials=100).trials()) == 8  # never more than the grid holds


def test_every_trial_is_logged_and_the_least_overfit_near_tie_wins(fitted):
    challenger, _, _ = fitted
    trials = challenger.trials
    assert len(trials) == 3
    assert trials["selected"].sum() == 1
    chosen = trials.loc[trials["selected"]].iloc[0]
    assert chosen["auc_validation"] >= trials["auc_validation"].max() - challenger.spec.selection_tolerance
    near = trials[trials["within_tolerance"]]
    assert chosen["gap"] == near["gap"].min()
    assert challenger.best_iteration == chosen["best_iteration"]


def test_without_tolerance_the_best_validation_auc_wins():
    train, validation = sample(6000, 0), sample(2000, 1)
    challenger = fit_challenger(train, validation, FEATURES, CATEGORICAL, "TARGET",
                                spec(selection_tolerance_auc=0.0))
    chosen = challenger.trials.loc[challenger.trials["selected"]].iloc[0]
    assert chosen["auc_validation"] == challenger.trials["auc_validation"].max()


def test_a_wide_tolerance_picks_the_least_overfit_trial():
    train, validation = sample(6000, 0), sample(2000, 1)
    challenger = fit_challenger(train, validation, FEATURES, CATEGORICAL, "TARGET",
                                spec(selection_tolerance_auc=1.0))
    chosen = challenger.trials.loc[challenger.trials["selected"]].iloc[0]
    assert chosen["gap"] == challenger.trials["gap"].min()


def test_categories_come_from_train_and_unseen_ones_become_missing(fitted):
    challenger, train, _ = fitted
    assert challenger.categories == {"grade": ["A", "B", "C"]}
    new = train.head(4).copy()
    new["grade"] = ["A", "Z", None, "C"]
    X = design_matrix(new, challenger.features, challenger.categories)
    assert list(X.columns) == FEATURES
    assert X["grade"].isna().tolist() == [False, True, True, False]
    assert np.isfinite(challenger.predict_pd(new)).all()


def test_shap_contributions_add_up_to_the_prediction(fitted):
    challenger, _, validation = fitted
    contributions = challenger.contributions(validation)
    log_odds = contributions[challenger.features].sum(axis=1) + contributions["base_value"]
    assert np.allclose(1 / (1 + np.exp(-log_odds)), challenger.predict_pd(validation))


def test_reason_codes_are_the_largest_positive_contributions(fitted):
    challenger, _, validation = fitted
    contributions = challenger.contributions(validation)
    codes = challenger.reason_codes(contributions)
    for i in range(0, len(validation), 157):
        row = contributions.iloc[i][challenger.features]
        expected = sorted((f for f in row.index if row[f] > 0), key=lambda f: (-row[f], f))[:2]
        assert codes[i] == expected
    safest = contributions[challenger.features].max(axis=1).idxmin()
    if contributions.loc[safest, challenger.features].max() <= 0:
        assert codes[contributions.index.get_loc(safest)] == []


def test_the_fit_is_deterministic(fitted):
    challenger, train, validation = fitted
    again = fit_challenger(train, validation, FEATURES, CATEGORICAL, "TARGET", spec())
    assert again.fingerprint() == challenger.fingerprint()
    pd.testing.assert_frame_equal(again.trials, challenger.trials)


def test_the_saved_model_scores_exactly_like_the_fitted_one(fitted):
    challenger, _, validation = fitted
    again = Challenger.from_payload(challenger.to_dict(), challenger.model_string(), challenger.spec)
    assert again.fingerprint() == challenger.fingerprint()
    assert np.array_equal(again.predict_pd(validation), challenger.predict_pd(validation))


def test_a_sample_without_bads_is_rejected():
    train, validation = sample(3000, 0), sample(1000, 1)
    validation["TARGET"] = 0
    with pytest.raises(ChallengerError, match="validation"):
        fit_challenger(train, validation, FEATURES, CATEGORICAL, "TARGET", spec())


@pytest.mark.parametrize(
    "override, message",
    [
        ({"model": "xgboost"}, "only lightgbm"),
        ({"search_space": {}}, "search_space"),
        ({"search_space": {"num_leaves": []}}, "search_space"),
        ({"max_trials": 0}, "must be positive"),
        ({"selection_tolerance_auc": -0.1}, "must not be negative"),
    ],
)
def test_bad_challenger_config_is_rejected(override, message):
    with pytest.raises(ChallengerError, match=message):
        spec(**override)
