"""Stage 5: calibrators fix the level of PD without touching (Platt) the ranking,
the method is chosen out of fold, and the diagnostics mean what they say."""

import numpy as np
import pytest

from credit_risk.models.calibration import (
    CalibrationError,
    CalibrationSpec,
    Calibrator,
    calibration_slope_intercept,
    reliability_table,
    select_calibrator,
    sigmoid,
)

MODEL_DEV = {"seed": 42, "calibration": {"methods": ["platt", "isotonic"], "n_bins_for_diagnostics": 10,
                                         "cv_folds": 5, "selection_tolerance_brier": 1e-4}}


def spec(**overrides):
    return CalibrationSpec.from_config({**MODEL_DEV, "calibration": {**MODEL_DEV["calibration"], **overrides}})


def sample(n=40000, seed=0, a=-2.5, b=1.0):
    """True log-odds z; the model reports `reported` = (z - a_shift) / b, i.e. PD that is off in level and spread."""
    rng = np.random.default_rng(seed)
    z = a + rng.normal(scale=1.0, size=n)
    y = (rng.random(n) < sigmoid(z)).astype(int)
    return z, y


def test_platt_undoes_a_known_distortion():
    z, y = sample()
    reported = (z - 0.4) / 0.7  # too low on average, too spread out
    calibrator = Calibrator("platt").fit(reported, y)
    assert calibrator.params["b"] == pytest.approx(0.7, abs=0.05)
    assert calibrator.params["a"] == pytest.approx(0.4, abs=0.1)
    before = calibration_slope_intercept(y, sigmoid(reported))
    after = calibration_slope_intercept(y, calibrator.apply(reported))
    assert before["slope"] == pytest.approx(0.7, abs=0.05)
    assert after["slope"] == pytest.approx(1.0, abs=0.05)
    assert abs(after["intercept"]) < 0.05 < abs(before["intercept"])


def test_platt_keeps_the_ranking_and_isotonic_never_reverses_it():
    z, y = sample()
    platt = Calibrator("platt").fit(z, y).apply(z)
    isotonic = Calibrator("isotonic").fit(z, y).apply(z)
    order = np.argsort(z)
    assert (np.diff(platt[order]) > 0).all()
    assert (np.diff(isotonic[order]) >= 0).all()
    assert len(np.unique(isotonic)) < len(np.unique(platt))  # isotonic ties applicants


def test_isotonic_fixes_a_shape_platt_cannot():
    z, y = sample(seed=1)
    reported = np.where(z > -2.5, z + 1.0, z)  # a jump in the middle: not an S-curve distortion
    platt = Calibrator("platt").fit(reported, y)
    isotonic = Calibrator("isotonic").fit(reported, y)
    held_z, held_y = sample(seed=2)
    held = np.where(held_z > -2.5, held_z + 1.0, held_z)
    brier = lambda p: np.mean((p - held_y) ** 2)
    assert brier(isotonic.apply(held)) < brier(platt.apply(held))


def test_the_calibrator_round_trips_and_stays_inside_the_floor():
    z, y = sample()
    for method in ("none", "platt", "isotonic"):
        calibrator = Calibrator(method, pd_floor=1e-6).fit(z, y)
        again = Calibrator.from_dict(calibrator.to_dict())
        extreme = np.array([-50.0, 0.0, 50.0])
        assert np.array_equal(again.apply(z), calibrator.apply(z))
        p = again.apply(extreme)
        assert (p >= 1e-6).all() and (p <= 1 - 1e-6).all()


def test_selection_prefers_platt_when_isotonic_is_only_marginally_better():
    z, y = sample()
    selection = select_calibrator(z, y, spec())
    cv = selection.cv.set_index("method")
    assert list(cv.index) == ["none", "platt", "isotonic"]
    assert not cv.loc["none", "within_tolerance"]  # the baseline never wins
    assert selection.calibrator.method == "platt"


def test_selection_takes_isotonic_when_it_is_clearly_better():
    z, y = sample(seed=1)
    reported = np.where(z > -2.5, z + 1.0, z)
    assert select_calibrator(reported, y, spec()).calibrator.method == "isotonic"


def test_selection_is_deterministic():
    z, y = sample()
    first, second = select_calibrator(z, y, spec()), select_calibrator(z, y, spec())
    assert first.calibrator.to_dict() == second.calibrator.to_dict()
    assert first.cv.equals(second.cv)


def test_diagnostics_of_a_perfect_model():
    z, y = sample(n=200000)
    p = sigmoid(z)
    si = calibration_slope_intercept(y, p)
    assert si["slope"] == pytest.approx(1.0, abs=0.03)
    assert si["intercept"] == pytest.approx(0.0, abs=0.03)
    table = reliability_table(y, p, 10)
    assert len(table) == 10 and table["n"].sum() == len(y)
    assert (table["mean_pd"].diff().dropna() > 0).all()
    assert np.allclose(table["ratio"], 1.0, atol=0.15)


@pytest.mark.parametrize(
    "override, message",
    [
        ({"methods": ["beta"]}, "subset"),
        ({"methods": []}, "subset"),
        ({"cv_folds": 1}, "at least 2"),
        ({"selection_tolerance_brier": -1}, "must not be negative"),
    ],
)
def test_bad_calibration_config_is_rejected(override, message):
    with pytest.raises(CalibrationError, match=message):
        spec(**override)
