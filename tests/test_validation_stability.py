"""Stage 6.5: PSI of scores, PDs and the most influential features, reference = train.

Unit checks of the categorical PSI and the traffic light, then the real chain on the
synthetic build: same-period splits are stable, application_test is measured, published
PSI figures are reproduced, and a shifted feature is caught.
"""

import json

import numpy as np
import pandas as pd
import pytest

from credit_risk.data.db import connect
from credit_risk.validation.package import load_package, load_scoring_frame, score_with_package
from credit_risk.validation.stability import (
    OTHER,
    StabilityError,
    StabilitySpec,
    categorical_psi,
    influential_features,
    published_from_train_artifacts,
    recompute_stability,
    stability_checks,
    stability_markdown,
)


def test_categorical_psi_is_zero_for_the_same_shares_and_pools_unseen_values():
    ref = pd.Series(["a"] * 60 + ["b"] * 40)
    value, table = categorical_psi(ref, ref, epsilon=1e-4)
    assert value == 0 and list(table["bin"]) == ["a", "b", OTHER]
    cur = pd.Series(["a"] * 30 + ["b"] * 30 + ["zzz"] * 40 + [None] * 0)
    value, table = categorical_psi(ref, cur, epsilon=1e-4)
    assert value > 0.25
    assert table.loc[table["bin"] == OTHER, "current_count"].iloc[0] == 40
    # Missing values are their own category, seen or not in the reference.
    value_missing, table_missing = categorical_psi(ref, pd.Series(["a"] * 50 + [None] * 50), epsilon=1e-4)
    assert value_missing > 0 and OTHER in set(table_missing["bin"])


def test_the_traffic_light_follows_the_definitions_thresholds():
    spec = StabilitySpec.from_config({"stability": {"psi_n_bins": 10, "top_features_for_psi": 3}},
                                     {"metric_thresholds": {"psi": {"stable_below": 0.1, "significant_above": 0.25}}})
    assert (spec.light(0.05), spec.light(0.1), spec.light(0.2), spec.light(0.25)) == ("green", "amber", "amber", "red")
    assert spec.top_features == 3
    with pytest.raises(StabilityError):
        StabilitySpec.from_config({}, {"metric_thresholds": {"psi": {"stable_below": 0.3, "significant_above": 0.25}}})
    with pytest.raises(StabilityError):
        StabilitySpec.from_config({"stability": {"psi_n_bins": 1}}, {})


# ---------------------------------------------------------------------------
# The real chain on the synthetic build
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def chain(stage6_build):
    settings, config = stage6_build
    folder = settings.processed_dir / "home_credit"
    package = load_package(folder / "model_package", config.model_dev, config.definitions)
    spec = StabilitySpec.from_config(config.validation, config.definitions)
    frame = load_scoring_frame(connect(settings), folder / "features.parquet", folder / "splits.parquet",
                               package.features + [spec.contract_column])
    scored = score_with_package(package, frame).join(frame.drop(columns=["SK_ID_CURR"]))
    published = published_from_train_artifacts(settings.artifacts_dir / package.meta["run_id"] / "train")
    return package, spec, scored, published


def test_influential_features_are_ranked_per_model_from_the_package_alone(chain):
    package, spec, _, _ = chain
    features = influential_features(package, spec.top_features)
    assert set(features["model"]) == {"champion", "challenger"}
    assert (features.groupby("model").size() <= spec.top_features).all()
    for _, part in features.groupby("model"):
        assert list(part["rank"]) == list(range(1, len(part) + 1))
        assert part["value"].is_monotonic_decreasing
    assert set(features.loc[features["model"] == "champion", "feature"]) <= set(package.champion.features)
    assert set(features.loc[features["model"] == "challenger", "feature"]) <= set(package.challenger.features)


def test_same_period_splits_are_stable_and_published_psi_is_reproduced(chain):
    package, spec, scored, published = chain
    table, bins, features = recompute_stability(scored, package, spec)
    checks = stability_checks(table, spec, published)
    assert checks["ok"].all(), checks[~checks["ok"]].to_string()
    assert set(published) == {"champion", "challenger"}
    assert (checks["source"] == "development run").sum() >= 2            # at least "all" for each model
    same_period = table[table["current"] != "application_test"]
    assert (same_period["light"] == "green").all()
    assert set(table["current"]) >= {"test", "validation", "calibration", "application_test"}
    outputs = table[(table["kind"] == "output") & (table["segment"] == "all")]
    assert len(outputs) == 2 * 4
    assert len(bins) == len(table)
    text = stability_markdown(table, features, checks, spec, bins)
    assert "lệch: 0" in text and "application_test" in text


def test_a_shifted_influential_feature_turns_red_only_where_it_shifted(chain):
    package, spec, scored, _ = chain
    feature = influential_features(package, spec.top_features).iloc[0]["feature"]
    shifted = scored.copy()
    new_population = shifted["population"] == "test"
    shifted.loc[new_population, feature] = shifted.loc[new_population, feature] * 3 + 1
    table, _, _ = recompute_stability(shifted, package, spec)
    row = table[(table["kind"] == "feature") & (table["name"] == feature) & (table["current"] == "application_test")]
    assert (row["light"] == "red").all()
    unchanged = table[(table["kind"] == "feature") & (table["name"] == feature) & (table["current"] != "application_test")]
    assert (unchanged["light"] == "green").all()
    checks = stability_checks(table, spec)
    assert checks["ok"].all()                                            # same-period sanity still holds


def test_a_shifted_score_fails_the_same_period_sanity_check(chain):
    package, spec, scored, _ = chain
    shifted = scored.copy()
    shifted.loc[shifted["split"] == "test", "score"] += 40
    table, _, _ = recompute_stability(shifted, package, spec)
    checks = stability_checks(table, spec)
    failed = checks.loc[~checks["ok"], "check"]
    assert len(failed) >= 1 and failed.str.contains("champion score vs test").all()


def test_missing_inputs_are_refused(chain):
    package, spec, scored, _ = chain
    with pytest.raises(StabilityError, match="lacks"):
        recompute_stability(scored.drop(columns=["score"]), package, spec)
    with pytest.raises(StabilityError, match="no train rows"):
        recompute_stability(scored[scored["split"] != "train"], package, spec)
