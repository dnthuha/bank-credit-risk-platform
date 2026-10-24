"""Stage 6.6: stratified, paired, seeded bootstrap intervals for AUC / Gini / KS.

The count-based metric equals the engine's rank-based one; intervals are
reproducible from the seed, contain the estimate, narrow with sample size and
cover the truth at about the nominal rate; the paired difference between two
models is tighter than two independent intervals would suggest.
"""

import numpy as np
import pandas as pd
import pytest

from credit_risk.data.db import connect
from credit_risk.validation.discrimination import DiscriminationSpec, recompute_discrimination
from credit_risk.validation.metrics import auc, ks
from credit_risk.validation.package import load_package, load_scoring_frame, score_with_package
from credit_risk.validation.uncertainty import (
    UncertaintyError,
    UncertaintySpec,
    auc_ks_from_counts,
    bootstrap_split,
    point_estimates,
    recompute_uncertainty,
    uncertainty_checks,
    uncertainty_markdown,
)


def simulated(n, seed, strength=1.0):
    rng = np.random.default_rng(seed)
    y = (rng.uniform(size=n) < 0.08).astype(int)
    risk = rng.normal(size=n) + strength * y
    return y, risk


def frame_of(y, risk_champion, risk_challenger, split="test"):
    # champion risk is minus the score; challenger risk is the PD itself
    return pd.DataFrame({"TARGET": y, "split": split, "score": -risk_champion, "pd_challenger": 1 / (1 + np.exp(-risk_challenger))})


@pytest.mark.parametrize("seed", range(3))
def test_count_based_metrics_equal_the_rank_based_ones_with_ties(seed):
    y, risk = simulated(4000, seed)
    risk = np.round(risk, 1)
    est = point_estimates(y, risk)
    assert est["auc"] == pytest.approx(auc(y, risk), abs=1e-12)
    assert est["ks"] == pytest.approx(ks(y, risk).statistic, abs=1e-12)
    assert est["gini"] == pytest.approx(2 * est["auc"] - 1)


def test_a_resample_without_bads_is_refused():
    with pytest.raises(UncertaintyError):
        auc_ks_from_counts(np.array([0, 0]), np.array([3, 4]))


def test_the_same_seed_gives_the_same_intervals_and_a_different_seed_does_not():
    y, risk = simulated(3000, 7)
    frame = frame_of(y, risk, risk + 0.1 * np.random.default_rng(1).normal(size=3000))
    spec = UncertaintySpec(resamples=200, samples=("test",), seed=11)
    first, d1 = recompute_uncertainty(frame, spec)
    second, d2 = recompute_uncertainty(frame, spec)
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(d1, d2)
    third, _ = recompute_uncertainty(frame, UncertaintySpec(resamples=200, samples=("test",), seed=12))
    assert not np.allclose(first["ci_low"], third["ci_low"])


def test_stratified_resamples_keep_the_number_of_bads_and_intervals_contain_the_estimate():
    y, risk = simulated(3000, 3)
    rng = np.random.default_rng(0)
    result = bootstrap_split(y, {"m": risk}, UncertaintySpec(resamples=100, samples=("test",)), rng)
    assert result.bads == y.sum()
    draws = result.draws["m"]["auc"]
    assert draws.min() <= result.estimates["m"]["auc"] <= draws.max()
    intervals, _ = recompute_uncertainty(frame_of(y, risk, risk), UncertaintySpec(resamples=100, samples=("test",)))
    assert ((intervals["ci_low"] <= intervals["estimate"]) & (intervals["estimate"] <= intervals["ci_high"])).all()
    assert uncertainty_checks(intervals)["ok"].all()


def test_intervals_narrow_with_sample_size_and_cover_the_truth_at_about_the_nominal_rate():
    spec = UncertaintySpec(resamples=200, samples=("test",), seed=5)
    small, _ = recompute_uncertainty(frame_of(*simulated(2000, 1)[::1], simulated(2000, 1)[1]), spec)
    large, _ = recompute_uncertainty(frame_of(*simulated(32000, 1)[::1], simulated(32000, 1)[1]), spec)
    ratio = (large["half_width"] / small["half_width"]).mean()
    assert 0.15 < ratio < 0.4                     # ~ sqrt(2000 / 32000) = 0.25
    # Coverage: the "truth" is the AUC on a very large sample from the same process.
    y_big, r_big = simulated(400000, 99)
    truth = point_estimates(y_big, r_big)["auc"]
    covered = 0
    for seed in range(40):
        y, r = simulated(3000, 100 + seed)
        intervals, _ = recompute_uncertainty(frame_of(y, r, r), UncertaintySpec(resamples=150, samples=("test",), seed=seed))
        row = intervals[(intervals["model"] == "champion") & (intervals["metric"] == "auc")].iloc[0]
        covered += row["ci_low"] <= truth <= row["ci_high"]
    assert covered >= 32                          # 95% nominal; 40 trials, allow sampling noise


def test_the_paired_difference_is_tighter_than_the_marginal_intervals():
    y, risk = simulated(5000, 21)
    almost_same = risk + 0.05 * np.random.default_rng(2).normal(size=5000)
    intervals, differences = recompute_uncertainty(frame_of(y, risk, almost_same),
                                                   UncertaintySpec(resamples=300, samples=("test",), seed=3))
    paired = differences[(differences["paired"]) & (differences["metric"] == "auc")].iloc[0]
    marginal = intervals[intervals["metric"] == "auc"]["half_width"].max()
    assert (paired["ci_high"] - paired["ci_low"]) / 2 < marginal / 2
    assert not paired["excludes_zero"]


def test_the_spec_guardrails():
    with pytest.raises(UncertaintyError):
        UncertaintySpec.from_config({"uncertainty": {"bootstrap_resamples": 1}})
    with pytest.raises(UncertaintyError):
        UncertaintySpec.from_config({"uncertainty": {"method": "bca"}})
    with pytest.raises(UncertaintyError):
        UncertaintySpec.from_config({"uncertainty": {"confidence_level": 1.0}})
    spec = UncertaintySpec.from_config({"seed": 7, "uncertainty": {"bootstrap_resamples": 50, "confidence_level": 0.9}})
    assert (spec.seed, spec.resamples, spec.alpha) == (7, 50, pytest.approx(0.1))
    with pytest.raises(UncertaintyError, match="no rows"):
        recompute_uncertainty(frame_of(*simulated(100, 0)[::1], simulated(100, 0)[1], split="train"), spec)


# ---------------------------------------------------------------------------
# The real chain on the synthetic build
# ---------------------------------------------------------------------------

def test_on_the_synthetic_package_the_estimates_equal_stage_6_3_and_every_interval_holds(stage6_build):
    settings, config = stage6_build
    folder = settings.processed_dir / "home_credit"
    package = load_package(folder / "model_package", config.model_dev, config.definitions)
    frame = load_scoring_frame(connect(settings), folder / "features.parquet", folder / "splits.parquet", package.features)
    scored = score_with_package(package, frame).join(frame[["TARGET", "split"]])
    spec = UncertaintySpec.from_config(config.validation)
    intervals, differences = recompute_uncertainty(scored, UncertaintySpec(resamples=200, seed=spec.seed, samples=spec.samples))
    discrimination = recompute_discrimination(scored, DiscriminationSpec.from_config(config.validation))
    checks = uncertainty_checks(intervals, discrimination)
    assert checks["ok"].all(), checks[~checks["ok"]].to_string()
    assert (checks["source"] == "discrimination").sum() == len(intervals)
    assert set(differences["comparison"]) == {"challenger - champion", "validation - test"}
    text = uncertainty_markdown(intervals, differences, checks, spec)
    assert "ghép đôi" in text and "lệch: 0" in text
