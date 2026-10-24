"""Stage 6.4: calibration recomputed by the validator.

Own Newton fit against a known distortion and against sklearn at tight tolerance;
reliability bands and their z-scores on a constructed example; then the real chain
on the synthetic build against the development run's calibration.json.
"""

import json

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from credit_risk.data.db import connect
from credit_risk.validation.calibration_check import (
    CalibrationCheckError,
    CalibrationCheckSpec,
    calibration_checks,
    calibration_markdown,
    hosmer_lemeshow,
    log_loss,
    logistic_recalibration,
    published_from_calibrate_artifacts,
    recompute_calibration,
    reliability_bands,
)
from credit_risk.validation.package import load_package, load_scoring_frame, score_with_package


def simulated(seed=0, n=40000, slope=1.0, shift=0.0):
    """True PD p; the model reports a distorted PD: logit(reported) = (logit(p) - shift) / slope,
    so recalibrating the reported PD recovers slope and intercept = shift."""
    rng = np.random.default_rng(seed)
    x = rng.normal(-2.5, 1.0, n)
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-x))).astype(int)
    reported = 1 / (1 + np.exp(-(x - shift) / slope))
    return y, reported


def test_the_newton_fit_recovers_a_known_distortion_and_is_the_true_optimum():
    y, p = simulated(slope=0.7, shift=0.4)
    fit = logistic_recalibration(y, p)
    assert fit["slope"] == pytest.approx(0.7, abs=0.03)
    assert fit["intercept"] == pytest.approx(0.4, abs=0.08)
    assert abs(fit["slope_z_vs_1"]) > 5                      # clearly not 1
    x = np.log(p / (1 - p))
    tight = LogisticRegression(C=np.inf, max_iter=10000, tol=1e-10).fit(x.reshape(-1, 1), y)
    assert fit["slope"] == pytest.approx(tight.coef_[0, 0], abs=1e-6)
    assert fit["intercept"] == pytest.approx(tight.intercept_[0], abs=1e-6)


def test_a_well_calibrated_model_has_slope_one_intercept_zero_and_no_flagged_band():
    y, p = simulated(seed=1)
    fit = logistic_recalibration(y, p)
    assert fit["slope"] == pytest.approx(1.0, abs=3 * fit["slope_se"])
    assert fit["intercept"] == pytest.approx(0.0, abs=3 * fit["intercept_se"])
    assert fit["intercept_in_the_large"] == pytest.approx(0.0, abs=0.1)
    bands = reliability_bands(y, p, 10, z_limit=3.0)
    assert len(bands) == 10 and bands["n"].sum() == len(y)
    assert (bands["flag"] == "").all()
    assert hosmer_lemeshow(bands)["hl_p_value"] > 0.001


def test_reliability_bands_flag_systematic_under_and_over_prediction():
    rng = np.random.default_rng(2)
    p = np.sort(rng.uniform(0.01, 0.3, 20000))
    y = (rng.uniform(size=20000) < p).astype(int)
    # The lowest band under-predicts (more bads than its PD says), the top band over-predicts.
    y[:2000] = (rng.uniform(size=2000) < 4 * p[:2000]).astype(int)
    y[-2000:] = 0
    bands = reliability_bands(y, p, 10, z_limit=2.0)
    assert bands.loc[0, "flag"] == "PD thấp (thiếu)" and bands.loc[0, "z"] > 2
    assert bands.loc[9, "flag"] == "PD cao (thừa)" and bands.loc[9, "z"] < -2
    assert bands["observed"].sum() == y.sum()
    assert bands["expected"].sum() == pytest.approx(p.sum())


def test_log_loss_and_the_spec_guardrails():
    assert log_loss([0, 1], [0.5, 0.5]) == pytest.approx(np.log(2))
    with pytest.raises(CalibrationCheckError):
        CalibrationCheckSpec.from_config({"calibration": {"n_bins": 1}})
    with pytest.raises(CalibrationCheckError):
        CalibrationCheckSpec.from_config({"calibration": {"samples": ["oot"]}})
    spec = CalibrationCheckSpec.from_config({"calibration": {"n_bins": 5, "samples": ["test", "validation"]}})
    assert spec.n_bins == 5 and spec.headline_samples == ("test", "validation")


def test_a_constant_pd_cannot_be_recalibrated():
    with pytest.raises(CalibrationCheckError, match="singular"):
        logistic_recalibration(np.array([0, 1, 0, 1]), np.array([0.1, 0.1, 0.1, 0.1]))


# ---------------------------------------------------------------------------
# The real chain on the synthetic build
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def recomputed(stage6_build):
    settings, config = stage6_build
    folder = settings.processed_dir / "home_credit"
    package = load_package(folder / "model_package", config.model_dev, config.definitions)
    frame = load_scoring_frame(connect(settings), folder / "features.parquet", folder / "splits.parquet", package.features)
    scored = score_with_package(package, frame).join(frame[["TARGET", "split"]])
    spec = CalibrationCheckSpec.from_config(config.validation)
    summary, bands = recompute_calibration(scored, spec)
    published = published_from_calibrate_artifacts(settings.artifacts_dir / package.meta["run_id"] / "calibrate")
    return summary, bands, calibration_checks(summary, bands, spec, published), published, spec


def test_the_recount_reproduces_the_development_figures(recomputed):
    summary, bands, checks, published, spec = recomputed
    assert set(published) == {"champion", "challenger"}
    assert set(summary["split"]) == {"train", "validation", "calibration", "test"}
    assert summary.loc[summary["headline"], "split"].eq("test").all()
    failed = checks[~checks["ok"]]
    # Everything agrees except, possibly, the slope: development's solver stopped at its default
    # tolerance, the validator's Newton fit at 1e-12. That gap is a finding, never larger than 1e-2.
    assert failed["check"].str.startswith("published slope").all(), failed.to_string()
    assert (abs(failed["claimed"].astype(float) - failed["recounted"].astype(float)) < 1e-2).all()
    assert checks["check"].str.contains("intercept in the large").sum() == 2
    assert checks.loc[checks["check"].str.contains("intercept in the large"), "ok"].all()
    assert checks["check"].str.contains("reliability band").sum() == 2 * 2 * spec.n_bins


def test_the_calibration_split_is_calibrated_in_the_large_by_construction(recomputed):
    summary, *_ = recomputed
    on_calibration = summary[summary["split"] == "calibration"]
    assert (on_calibration["ratio_bad_rate_to_pd"] - 1).abs().max() < 1e-3
    assert (on_calibration["intercept_in_the_large"].abs() < 1e-2).all()


def test_an_edited_published_figure_is_flagged_and_reported(recomputed):
    summary, bands, _, published, spec = recomputed
    edited = json.loads(json.dumps(published))
    edited["champion"]["after"]["validation"]["brier"] += 0.001
    edited["champion"]["reliability"][0]["bad_rate"] += 0.05
    checks = calibration_checks(summary, bands, spec, edited)
    failed = set(checks.loc[~checks["ok"], "check"])
    assert {"published brier (champion, validation)",
            "published reliability band 1 bad rate (champion, validation)"} <= failed
    text = calibration_markdown(summary, bands, checks, spec)
    assert "published brier (champion, validation)" in text and "### champion, test" in text
