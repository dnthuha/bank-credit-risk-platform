"""Stage 2.4: bins are learnt on train only, respect the configured constraints,
score every possible value, and come out identical on every run."""

import json
import math

import numpy as np
import pandas as pd
import pytest

from credit_risk.features.binning import (
    MISSING,
    RARE,
    BinningError,
    BinningSpec,
    FeatureBinning,
    binning_frame,
    binning_payload,
    fit_all,
    fit_categorical,
    fit_numeric,
    load_binnings,
)
from credit_risk.features.woe import woe_iv_table

MODEL_DEV = {"binning": {
    "initial_quantile_bins": 20, "max_bins": 8, "min_bin_share": 0.05,
    "prefer_monotonic_trend": True, "missing_as_own_bin": True, "rare_category_share": 0.01,
}}
DEFINITIONS = {"metric_thresholds": {"woe": {"zero_count_adjustment": 0.5}}}


@pytest.fixture
def spec():
    return BinningSpec.from_config(MODEL_DEV, DEFINITIONS)


def risky_numeric(n=20000, seed=0, missing_share=0.1):
    """Bad rate rises with x, plus noise and a missing block with its own bad rate."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    p = 1 / (1 + np.exp(-(-2.5 + 0.8 * x)))
    y = (rng.random(n) < p).astype(int)
    missing = rng.random(n) < missing_share
    x[missing] = np.nan
    y[missing] = (rng.random(missing.sum()) < 0.30).astype(int)
    return x, y


def non_missing(binning):
    return [r for r in binning.bins if r["label"] != MISSING]


# --- constraints --------------------------------------------------------------

def test_numeric_bins_respect_every_constraint(spec):
    x, y = risky_numeric()
    b = fit_numeric("x", x, y, spec)
    rows = non_missing(b)
    assert 2 <= len(rows) <= spec.max_bins
    assert all(r["share"] >= spec.min_bin_share for r in rows)
    rates = [r["bad_rate"] for r in rows]
    assert b.trend == "increasing" and all(np.diff(rates) >= 0)
    assert sum(r["n"] for r in b.bins) == len(x)


def test_higher_bad_rate_means_lower_woe(spec):
    x, y = risky_numeric()
    rows = non_missing(fit_numeric("x", x, y, spec))
    woes = [r["woe"] for r in rows]
    assert all(np.diff(woes) <= 0)  # bad rate increases, so WoE (ln %good / %bad) decreases


def test_missing_values_keep_their_own_bin(spec):
    x, y = risky_numeric(missing_share=0.1)
    b = fit_numeric("x", x, y, spec)
    missing = next(r for r in b.bins if r["label"] == MISSING)
    assert missing["n"] == int(np.isnan(x).sum())
    assert missing["bad_rate"] == pytest.approx(y[np.isnan(x)].mean())


def test_iv_matches_the_reference_woe_implementation(spec):
    x, y = risky_numeric()
    b = fit_numeric("x", x, y, spec)
    reference = woe_iv_table(b.assign(x), y, spec.zero_count_adjustment)
    assert b.iv == pytest.approx(reference.iv)
    ref_woe = reference.mapping()
    for row in b.bins:
        if row["n"]:
            assert row["woe"] == pytest.approx(ref_woe[row["label"]])


def test_prebin_iv_is_recorded_and_never_below_the_final_iv(spec):
    x, y = risky_numeric()
    b = fit_numeric("x", x, y, spec)
    assert b.prebin_iv >= b.iv > 0


# --- non-monotonic shapes -----------------------------------------------------

def inverted_u(n=30000, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, n)
    p = 0.02 + 0.2 * (1 - x ** 2)  # risk peaks in the middle
    return x, (rng.random(n) < p).astype(int)


def test_a_costly_monotonic_fit_is_flagged_for_review(spec):
    x, y = inverted_u()
    b = fit_numeric("x", x, y, spec)
    assert any("review the shape" in note for note in b.notes)


def test_an_allowed_exception_keeps_its_shape_and_its_rationale():
    spec = BinningSpec.from_config(
        {"binning": {**MODEL_DEV["binning"], "allow_non_monotonic": {"x": "mid-size loans are riskiest"}}},
        DEFINITIONS,
    )
    x, y = inverted_u()
    monotonic = fit_numeric("x", x, y, BinningSpec.from_config(MODEL_DEV, DEFINITIONS))
    shaped = fit_numeric("x", x, y, spec)
    assert shaped.trend == "non_monotonic"
    assert shaped.iv > 2 * monotonic.iv
    assert any("mid-size loans are riskiest" in note for note in shaped.notes)


def test_an_exception_without_a_rationale_is_rejected():
    with pytest.raises(BinningError, match="business rationale"):
        BinningSpec.from_config({"binning": {**MODEL_DEV["binning"], "allow_non_monotonic": {"x": " "}}},
                                DEFINITIONS)


# --- categorical --------------------------------------------------------------

def categorical_sample(n=20000, seed=2):
    rng = np.random.default_rng(seed)
    cats = rng.choice(["A", "B", "C", "D", "tiny1", "tiny2"], n, p=[0.4, 0.3, 0.2, 0.094, 0.003, 0.003])
    base = {"A": 0.04, "B": 0.08, "C": 0.12, "D": 0.20, "tiny1": 0.5, "tiny2": 0.5}
    y = (rng.random(n) < np.vectorize(base.get)(cats)).astype(int)
    return pd.Series(cats, dtype="object"), y


def test_rare_categories_are_pooled_and_unseen_ones_follow_them(spec):
    s, y = categorical_sample()
    b = fit_categorical("c", s, y, spec)
    rare_group = next(g for g in b.groups if RARE in g)
    assert {"tiny1", "tiny2"} <= set(rare_group)
    # A category never seen in train scores like the rare pool it would have joined.
    rare_label = b.assign(pd.Series(["tiny1"], dtype="object"))[0]
    assert b.assign(pd.Series(["brand_new"], dtype="object"))[0] == rare_label


def test_categorical_labels_lead_with_the_largest_category(spec):
    s, y = categorical_sample()
    b = fit_categorical("c", s, y, spec)
    for label, group in zip([r["label"] for r in non_missing(b)], b.groups):
        named = [c for c in group if c not in (RARE, "tiny1", "tiny2")]
        if named:
            biggest = max(named, key=lambda c: (s == c).sum())
            assert label.split("] ", 1)[1].startswith(biggest)


def test_categorical_bins_are_ordered_by_bad_rate(spec):
    s, y = categorical_sample()
    rates = [r["bad_rate"] for r in non_missing(fit_categorical("c", s, y, spec))]
    assert rates == sorted(rates)


# --- scoring ------------------------------------------------------------------

def test_values_outside_the_train_range_land_in_the_edge_bins(spec):
    x, y = risky_numeric()
    b = fit_numeric("x", x, y, spec)
    rows = non_missing(b)
    labels = b.assign(np.array([-1e9, 1e9]))
    assert list(labels) == [rows[0]["label"], rows[-1]["label"]]


def test_a_missing_value_never_seen_in_train_scores_zero(spec):
    x, y = risky_numeric(missing_share=0.0)
    b = fit_numeric("x", x, y, spec)
    assert b.transform(np.array([np.nan]))[0] == 0.0


def test_right_closed_edges(spec):
    x, y = risky_numeric()
    b = fit_numeric("x", x, y, spec)
    edge = b.edges[0]
    first, second = non_missing(b)[0]["label"], non_missing(b)[1]["label"]
    assert list(b.assign(np.array([edge, np.nextafter(edge, math.inf)]))) == [first, second]


def test_degenerate_features_do_not_break_the_fit(spec):
    y = np.array([0, 1] * 500)
    constant = fit_numeric("k", np.ones(1000), y, spec)
    assert len(non_missing(constant)) == 1 and constant.iv == 0.0
    empty = fit_numeric("e", np.full(1000, np.nan), y, spec)
    assert [r["label"] for r in empty.bins] == [MISSING] and empty.iv == 0.0


# --- train only, reproducible, persistable -------------------------------------

def frame_with_splits(n=6000, seed=3):
    x, y = risky_numeric(n=n, seed=seed)
    s, _ = categorical_sample(n=n, seed=seed)
    split = np.where(np.arange(n) % 10 < 6, "train", "other")
    return pd.DataFrame({"x": x, "c": s, "TARGET": y, "split": split})


def test_rows_outside_train_cannot_change_the_bins(spec):
    frame = frame_with_splits()
    train = frame[frame["split"] == "train"]
    first = binning_payload(fit_all(train, ["x", "c"], "TARGET", spec), spec, {})

    tampered = frame.copy()
    other = tampered["split"] != "train"
    tampered.loc[other, "x"] = 1e6
    tampered.loc[other, "TARGET"] = 1
    again = binning_payload(fit_all(tampered[tampered["split"] == "train"], ["x", "c"], "TARGET", spec), spec, {})
    assert first["fingerprint"] == again["fingerprint"]


def test_the_same_train_rows_always_give_the_same_table(spec):
    frame = frame_with_splits()
    train = frame[frame["split"] == "train"]
    shuffled = train.sample(frac=1, random_state=11)
    a = binning_payload(fit_all(train, ["x", "c"], "TARGET", spec), spec, {})
    b = binning_payload(fit_all(shuffled, ["x", "c"], "TARGET", spec), spec, {})
    assert a["fingerprint"] == b["fingerprint"]


def test_a_saved_table_scores_exactly_like_the_fitted_one(spec):
    frame = frame_with_splits()
    fitted = fit_all(frame, ["x", "c"], "TARGET", spec)
    payload = json.loads(json.dumps(binning_payload(fitted, spec, {}), default=str))
    reloaded = load_binnings(payload)
    for name in ["x", "c"]:
        np.testing.assert_array_equal(fitted[name].transform(frame[name]), reloaded[name].transform(frame[name]))


def test_target_problems_are_rejected(spec):
    frame = frame_with_splits()
    frame.loc[0, "TARGET"] = None
    with pytest.raises(BinningError, match="nulls"):
        fit_all(frame, ["x"], "TARGET", spec)


def test_review_table_has_one_row_per_bin(spec):
    frame = frame_with_splits()
    fitted = fit_all(frame, ["x", "c"], "TARGET", spec)
    table = binning_frame(fitted)
    assert len(table) == sum(len(b.bins) for b in fitted.values())
    assert {"feature", "label", "n", "bad_rate", "woe", "iv_contribution", "feature_iv"} <= set(table.columns)
    assert isinstance(FeatureBinning.from_dict(fitted["x"].to_dict()), FeatureBinning)
