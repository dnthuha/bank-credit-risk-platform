"""Stage 2.1: the population split is exact, disjoint, stratified and reproducible."""

import numpy as np
import pandas as pd
import pytest

from credit_risk.features.split import (
    SPLIT_COLUMN,
    SplitError,
    SplitSpec,
    assign_splits,
    fingerprint,
    split_summary,
)

MODEL_DEV = {
    "seed": 42,
    "split": {
        "method": "stratified_random",
        "id_column": "SK_ID_CURR",
        "stratify_on": "TARGET",
        "fractions": {"train": 0.60, "validation": 0.10, "calibration": 0.15, "test": 0.15},
    },
}


@pytest.fixture
def spec():
    return SplitSpec.from_config(MODEL_DEV)


def population(n=1000, bad_rate=0.08, seed=0):
    rng = np.random.default_rng(seed)
    n_bad = int(round(n * bad_rate))
    target = np.zeros(n, dtype=int)
    target[rng.choice(n, n_bad, replace=False)] = 1
    return pd.DataFrame({"SK_ID_CURR": np.arange(100001, 100001 + n), "TARGET": target})


def test_spec_comes_from_the_config(spec):
    assert spec.id_column == "SK_ID_CURR" and spec.stratify_on == "TARGET" and spec.seed == 42
    assert spec.names == ["train", "validation", "calibration", "test"]


@pytest.mark.parametrize(
    "broken, message",
    [
        ({"method": "random"}, "split.method"),
        ({"fractions": {"train": 0.6, "test": 0.3}}, "sum to 1"),
        ({"fractions": {"train": 1.2, "test": -0.2}}, "> 0"),
        ({"id_column": None}, "id_column"),
    ],
)
def test_config_problems_are_rejected(broken, message):
    config = {**MODEL_DEV, "split": {**MODEL_DEV["split"], **broken}}
    with pytest.raises(SplitError, match=message):
        SplitSpec.from_config(config)


def test_every_id_lands_in_exactly_one_split(spec):
    pop = population()
    splits = assign_splits(pop, spec)
    assert len(splits) == len(pop)
    assert set(splits["SK_ID_CURR"]) == set(pop["SK_ID_CURR"])
    assert not splits["SK_ID_CURR"].duplicated().any()
    assert set(splits[SPLIT_COLUMN]) == set(spec.names)


def test_split_sizes_match_the_configured_fractions(spec):
    splits = assign_splits(population(n=1000), spec)
    sizes = splits[SPLIT_COLUMN].value_counts().to_dict()
    # 1000 rows, two strata: rounding can move at most one row per stratum per split.
    for name, share in spec.fractions.items():
        assert abs(sizes[name] - 1000 * share) <= 2
    assert sum(sizes.values()) == 1000


def test_bad_rate_is_preserved_in_every_split(spec):
    pop = population(n=5000, bad_rate=0.08)
    merged = assign_splits(pop, spec).merge(pop, on="SK_ID_CURR")
    overall = merged["TARGET"].mean()
    for name in spec.names:
        rate = merged.loc[merged[SPLIT_COLUMN] == name, "TARGET"].mean()
        assert abs(rate - overall) < 0.005, f"{name}: {rate} vs {overall}"


def test_assignment_is_reproducible_and_order_independent(spec):
    pop = population()
    first = assign_splits(pop, spec)
    again = assign_splits(pop, spec)
    shuffled = assign_splits(pop.sample(frac=1, random_state=7), spec)

    assert fingerprint(first, spec) == fingerprint(again, spec) == fingerprint(shuffled, spec)
    pd.testing.assert_frame_equal(first, again)
    pd.testing.assert_frame_equal(first, shuffled)


def test_another_seed_gives_another_assignment(spec):
    pop = population()
    other = SplitSpec.from_config({**MODEL_DEV, "seed": 7})
    assert fingerprint(assign_splits(pop, spec), spec) != fingerprint(assign_splits(pop, other), other)


def test_population_problems_are_rejected(spec):
    pop = population(n=100)
    with pytest.raises(SplitError, match="not unique"):
        assign_splits(pd.concat([pop, pop.head(1)]), spec)
    with pytest.raises(SplitError, match="columns missing"):
        assign_splits(pop.drop(columns=["TARGET"]), spec)
    with pytest.raises(SplitError, match="empty"):
        assign_splits(pop.head(0), spec)
    nulls = pop.copy()
    nulls.loc[0, "TARGET"] = None
    with pytest.raises(SplitError, match="TARGET contains nulls"):
        assign_splits(nulls, spec)


def test_small_population_still_assigns_every_row(spec):
    pop = population(n=40, bad_rate=0.2)
    splits = assign_splits(pop, spec)
    assert len(splits) == 40 and not splits[SPLIT_COLUMN].isna().any()


def test_summary_reports_sizes_bad_rates_and_fingerprint(spec):
    pop = population(n=2000)
    splits = assign_splits(pop, spec)
    summary = split_summary(splits, pop, spec)

    assert summary["rows"] == 2000 and summary["seed"] == 42
    assert summary["fingerprint"] == fingerprint(splits, spec)
    assert [row["split"] for row in summary["splits"]] == spec.names
    assert sum(row["rows"] for row in summary["splits"]) == 2000
    for row in summary["splits"]:
        assert abs(row["share"] - row["target_share"]) < 0.01
        assert abs(row["bad_rate"] - summary["bad_rate_overall"]) < 0.01
