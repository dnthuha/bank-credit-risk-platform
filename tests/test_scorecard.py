"""Stage 3: the scorecard keeps only features whose coefficient sign is right and
stable, scales to the configured points, and explains every score."""

import math

import numpy as np
import pandas as pd
import pytest

from credit_risk.features.binning import BinningSpec, fit_all
from credit_risk.models.scorecard import (
    Scorecard,
    ScorecardError,
    ScorecardSpec,
    fit_sign_stable,
    scorecard_payload,
)

BINNING = {"binning": {"initial_quantile_bins": 20, "max_bins": 8, "min_bin_share": 0.05,
                       "prefer_monotonic_trend": True, "missing_as_own_bin": True, "rare_category_share": 0.01}}
DEFINITIONS = {
    "scorecard_scaling": {"base_score": 600, "base_odds": 50, "pdo": 20},
    "metric_thresholds": {"woe": {"zero_count_adjustment": 0.5}},
}
MODEL_DEV = {**BINNING, "seed": 42, "champion": {"regularization": "l2", "C": 1.0,
                                                 "sign_stability": {"bootstrap_resamples": 20, "min_share": 0.95}},
             "explainability": {"reason_codes_per_applicant": 2}}


def spec(**overrides):
    config = {**MODEL_DEV, "champion": {**MODEL_DEV["champion"], **overrides.pop("champion", {})}, **overrides}
    return ScorecardSpec.from_config(config, DEFINITIONS)


def sample(n=20000, seed=0):
    """a and b carry the signal. `suppressor` rises with risk on its own (it follows a) but,
    given a, lowers it: its WoE coefficient comes out positive. `twin` is a noisy copy of b."""
    rng = np.random.default_rng(seed)
    a, b, noise = rng.normal(size=(3, n))
    suppressor = 0.9 * a + math.sqrt(1 - 0.81) * rng.normal(size=n)
    logit = -2.3 + 0.9 * a + 0.6 * b - 0.5 * suppressor
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame({
        "a": a,
        "b": b,
        "suppressor": suppressor,
        "twin": b + rng.normal(scale=0.05, size=n),
        "grade": np.where(b > 0.5, "A", np.where(b > -0.5, "B", "C")),
        "TARGET": y,
    })


def fitted(features=("a", "b", "suppressor", "grade"), s=None, data=None):
    s = s or spec()
    data = sample() if data is None else data
    binnings = fit_all(data, list(features), "TARGET", BinningSpec.from_config(BINNING, DEFINITIONS))
    woe = pd.DataFrame({f: binnings[f].transform(data[f]) for f in features})
    fit = fit_sign_stable(woe, data["TARGET"], s)
    return Scorecard.build(fit, binnings, s), fit, data


# ---------------------------------------------------------------------------
# 3.1 sign stability
# ---------------------------------------------------------------------------

def test_a_feature_whose_coefficient_reverses_its_bins_is_dropped():
    card, fit, _ = fitted()
    assert "suppressor" not in card.features
    row = fit.trail.set_index("feature").loc["suppressor"]
    assert row["coefficient"] > 0
    assert row["reason"].startswith("positive coefficient")
    assert fit.initial[1]["suppressor"] > 0  # it really was positive with every feature in


def test_every_kept_coefficient_is_negative_and_stable():
    card, fit, _ = fitted(features=("a", "b", "suppressor", "twin", "grade"))
    assert all(c < 0 for c in card.coefficients.values())
    assert all(share >= 0.95 for share in fit.sign_share.values())
    # b and its twin carry the same information: at most one of them can hold a stable sign.
    assert not {"b", "twin"} <= set(card.features)


def test_the_fit_is_deterministic():
    first, fit1, _ = fitted()
    second, fit2, _ = fitted()
    assert first.fingerprint() == second.fingerprint()
    pd.testing.assert_frame_equal(fit1.trail, fit2.trail)


def test_without_the_sign_requirement_nothing_is_dropped():
    s = spec(feature_selection={"require_stable_coefficient_sign": False})
    card, fit, _ = fitted(s=s)
    assert fit.trail.empty
    assert "suppressor" in card.features


def test_dropping_every_feature_is_an_error():
    # A constant has one bin, WoE 0 and a zero coefficient: never negative, so never stable.
    rng = np.random.default_rng(3)
    data = pd.DataFrame({"x": np.ones(5000), "TARGET": (rng.random(5000) < 0.1).astype(int)})
    with pytest.raises(ScorecardError, match="every feature was dropped"):
        fitted(features=("x",), data=data)


# ---------------------------------------------------------------------------
# 3.2 points
# ---------------------------------------------------------------------------

def test_scaling_follows_definitions():
    s = spec()
    assert s.factor == pytest.approx(20 / math.log(2))
    assert s.offset == pytest.approx(600 - 20 / math.log(2) * math.log(50))
    card, _, _ = fitted()
    # 600 points means odds 50:1; every 20 points more doubles the odds.
    assert card.pd_from_score([600])[0] == pytest.approx(1 / 51)
    assert card.pd_from_score([620])[0] == pytest.approx(1 / 101)


def test_the_score_is_the_integer_sum_of_the_table_and_tracks_the_exact_model():
    card, _, data = fitted()
    points = card.points_frame(data)
    score = card.score(data)
    assert score.dtype.kind == "i"
    assert np.array_equal(score, card.base_points + points.sum(axis=1).to_numpy())
    exact = card.spec.offset - card.spec.factor * card.exact_log_odds_bad(data)
    # Each of the (features + base) integers is off by at most one half.
    assert np.abs(score - exact).max() <= 0.5 * (len(card.features) + 1)


def test_a_riskier_bin_never_earns_more_points():
    card, _, _ = fitted()
    for f in card.features:
        rows = [r for r in card.binnings[f].bins if r["n"] > 0]
        by_bad_rate = sorted(rows, key=lambda r: r["bad_rate"])
        pts = [card.points[f][r["label"]] for r in by_bad_rate]
        assert pts == sorted(pts, reverse=True), f


def test_an_unseen_category_scores_zero_points():
    card, _, data = fitted(features=("a", "grade"))
    assert "grade" in card.features
    new = data.head(3).copy()
    new["grade"] = "Z"
    assert (card.points_frame(new)["grade"] == 0).all()


def test_the_payload_round_trips_to_the_same_scores():
    card, fit, data = fitted()
    payload = scorecard_payload(card, fit, {"run_id": "x"})
    again = Scorecard.from_payload(payload, card.spec)
    assert again.fingerprint() == payload["fingerprint"] == card.fingerprint()
    assert np.array_equal(again.score(data), card.score(data))


# ---------------------------------------------------------------------------
# 3.3 reason codes
# ---------------------------------------------------------------------------

def test_reason_codes_are_the_features_that_lost_the_most_points():
    card, _, data = fitted()
    points = card.points_frame(data)
    codes = card.reason_codes(points)
    best = {f: max(card.points[f].values()) for f in card.features}
    for i in range(0, len(data), 997):
        lost = {f: best[f] - points.iloc[i][f] for f in card.features}
        expected = sorted((f for f in lost if lost[f] > 0), key=lambda f: (-lost[f], f))[:2]
        assert codes[i] == expected


def test_an_applicant_in_the_best_bin_everywhere_has_no_reason_code():
    card, _, data = fitted()
    points = card.points_frame(data)
    best = {f: max(card.points[f].values()) for f in card.features}
    perfect = points.apply(lambda r: all(r[f] == best[f] for f in card.features), axis=1)
    assert perfect.any()
    assert all(codes == [] for codes, p in zip(card.reason_codes(points), perfect) if p)


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "champion, message",
    [
        ({"regularization": "l1"}, "only l2"),
        ({"C": 0}, "C must be positive"),
        ({"sign_stability": {"bootstrap_resamples": 0}}, "at least 1"),
        ({"sign_stability": {"min_share": 0.5}}, "min_share"),
    ],
)
def test_bad_champion_config_is_rejected(champion, message):
    with pytest.raises(ScorecardError, match=message):
        spec(champion=champion)
