"""Stage 2.5: train bins are applied, never refitted, and the report flags features
that do not hold up out of sample or whose distribution moves."""

import copy

import numpy as np
import pandas as pd
import pytest

from credit_risk.features.binning import BinningSpec, fit_all
from credit_risk.features.iv_report import ReportThresholds, iv_report, iv_report_markdown

MODEL_DEV = {
    "binning": {"initial_quantile_bins": 20, "max_bins": 8, "min_bin_share": 0.05,
                "prefer_monotonic_trend": True, "missing_as_own_bin": True, "rare_category_share": 0.01},
    "feature_selection": {"min_iv_retention_validation": 0.5, "min_bad_rate_rank_corr": 0.8},
}
DEFINITIONS = {"metric_thresholds": {
    "psi": {"stable_below": 0.10, "significant_above": 0.25, "zero_bin_epsilon": 1e-4},
    "woe": {"zero_count_adjustment": 0.5},
    "iv": {"min_useful": 0.02, "leakage_suspect_above": 0.50},
}}


@pytest.fixture
def th():
    return ReportThresholds.from_config(DEFINITIONS, MODEL_DEV)


def sample(n, seed, slope=0.6, shift=0.0):
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=n) + shift
    y = (rng.random(n) < 1 / (1 + np.exp(-(-2.4 + slope * (signal - shift))))).astype(int)
    return pd.DataFrame({
        "signal": signal,
        "noise": rng.normal(size=n),
        "segment": rng.choice(["a", "b", "c"], n),
        "TARGET": y,
    })


@pytest.fixture
def fitted():
    spec = BinningSpec.from_config(MODEL_DEV, DEFINITIONS)
    train = sample(30000, seed=0)
    return fit_all(train, ["signal", "noise", "segment"], "TARGET", spec)


def by_feature(table):
    return table.set_index("feature")


def test_a_real_signal_holds_up_out_of_sample(fitted, th):
    table, _ = iv_report(fitted, sample(10000, seed=1), sample(10000, seed=2).drop(columns="TARGET"), "TARGET", th)
    row = by_feature(table).loc["signal"]
    assert row.iv_retention > 0.7
    assert row.bad_rate_rank_corr > 0.9
    assert row.psi_validation < 0.02 and row.psi_current < 0.02
    assert row.review_flags == ""


def test_an_implausibly_strong_feature_is_flagged_as_possible_leakage(th):
    spec = BinningSpec.from_config(MODEL_DEV, DEFINITIONS)
    strong = fit_all(sample(30000, seed=0, slope=2.0), ["signal"], "TARGET", spec)
    table, _ = iv_report(strong, sample(10000, seed=1, slope=2.0), sample(10000, seed=2), "TARGET", th)
    row = by_feature(table).loc["signal"]
    assert row.iv_train > th.leakage_suspect
    assert row.iv_band == "suspicious" and "leakage_suspect" in row.review_flags


def test_noise_is_flagged_weak(fitted, th):
    table, _ = iv_report(fitted, sample(10000, seed=1), sample(10000, seed=2), "TARGET", th)
    assert "weak" in by_feature(table).loc["noise", "review_flags"]


def test_a_relationship_that_reverses_is_caught(fitted, th):
    reversed_validation = sample(10000, seed=1, slope=-0.6)
    table, _ = iv_report(fitted, reversed_validation, sample(10000, seed=2), "TARGET", th)
    flags = by_feature(table).loc["signal", "review_flags"]
    assert "trend_not_confirmed" in flags


def test_a_moved_current_population_is_flagged_without_labels(fitted, th):
    moved = sample(10000, seed=2, shift=1.5).drop(columns="TARGET")
    table, _ = iv_report(fitted, sample(10000, seed=1), moved, "TARGET", th)
    row = by_feature(table).loc["signal"]
    assert row.psi_current >= th.psi_significant
    assert "shift_vs_current" in row.review_flags


def test_a_category_unseen_in_train_is_counted_not_dropped(fitted, th):
    current = sample(10000, seed=2)
    current.loc[:999, "segment"] = "brand_new"
    table, bins = iv_report(fitted, sample(10000, seed=1), current, "TARGET", th)
    segment = bins[bins.feature == "segment"]
    assert segment["share_current"].sum() == pytest.approx(1.0)
    assert by_feature(table).loc["segment", "psi_current"] > 0.05


def test_the_report_never_refits_the_bins(fitted, th):
    before = copy.deepcopy({k: v.to_dict() for k, v in fitted.items()})
    iv_report(fitted, sample(10000, seed=1, slope=-0.6), sample(10000, seed=2, shift=2.0), "TARGET", th)
    assert {k: v.to_dict() for k, v in fitted.items()} == before


def test_bin_rows_line_up_with_the_train_table(fitted, th):
    _, bins = iv_report(fitted, sample(10000, seed=1), sample(10000, seed=2), "TARGET", th)
    for name, binning in fitted.items():
        rows = bins[bins.feature == name]
        assert list(rows["label"])[: len(binning.bins)] == [r["label"] for r in binning.bins]
        assert rows["share_train"].sum() == pytest.approx(1.0)
        assert rows["share_validation"].sum() == pytest.approx(1.0)


def test_validation_labels_are_required(fitted, th):
    validation = sample(1000, seed=1).astype({"TARGET": "float"})
    validation.loc[0, "TARGET"] = np.nan
    with pytest.raises(ValueError, match="nulls"):
        iv_report(fitted, validation, sample(1000, seed=2), "TARGET", th)


def test_markdown_states_thresholds_and_counts(fitted, th):
    table, _ = iv_report(fitted, sample(10000, seed=1), sample(10000, seed=2), "TARGET", th)
    text = iv_report_markdown(table, th, {"rows_train": 30000, "rows_validation": 10000, "rows_current": 10000})
    assert "PSI stable < 0.1" in text and "Calibration and test splits are not used" in text
    assert "`signal`" in text
