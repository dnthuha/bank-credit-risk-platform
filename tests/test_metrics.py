"""AUC, Gini, KS, Brier, PSI, WoE / IV and roll rates.

Each metric is checked twice: against a small example computed by hand, and
against an independent implementation (sklearn / scipy) on random data.
"""

import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score

from credit_risk.features.woe import MISSING_BIN, woe_iv_table, woe_transform
from credit_risk.monitoring.buckets import BUCKET_ORDER
from credit_risk.monitoring.transitions import transition_matrix
from credit_risk.validation.metrics import auc, brier, gini, ks, psi, psi_from_proportions

# Hand example: bads score 0.35 and 0.8, goods score 0.1 and 0.4.
# Pairs (bad, good): (0.35>0.1) (0.35<0.4) (0.8>0.1) (0.8>0.4) -> 3 of 4 -> AUC 0.75.
Y_HAND = np.array([0, 0, 1, 1])
S_HAND = np.array([0.1, 0.4, 0.35, 0.8])


@pytest.fixture(scope="module")
def random_scores():
    rng = np.random.default_rng(42)
    y = (rng.random(5000) < 0.08).astype(int)
    score = rng.normal(loc=y * 0.8, scale=1.0)
    return y, score


class TestAUC:
    def test_hand_example(self):
        assert auc(Y_HAND, S_HAND) == pytest.approx(0.75)
        assert gini(Y_HAND, S_HAND) == pytest.approx(0.5)

    def test_ties_count_half(self):
        assert auc([0, 1], [0.5, 0.5]) == pytest.approx(0.5)

    def test_perfect_and_inverted_ranking(self):
        assert auc([0, 0, 1, 1], [1, 2, 3, 4]) == 1.0
        assert auc([0, 0, 1, 1], [4, 3, 2, 1]) == 0.0

    def test_matches_sklearn(self, random_scores):
        y, score = random_scores
        assert auc(y, score) == pytest.approx(roc_auc_score(y, score), abs=1e-12)

    def test_gini_is_exactly_two_auc_minus_one(self, random_scores):
        y, score = random_scores
        assert gini(y, score) == pytest.approx(2 * auc(y, score) - 1, abs=1e-15)

    @pytest.mark.parametrize(
        "y, score",
        [([0, 0, 0], [1, 2, 3]), ([0, 2], [1, 2]), ([0, 1], [np.nan, 1.0]), ([0, 1], [1.0])],
    )
    def test_rejects_invalid_input(self, y, score):
        with pytest.raises(ValueError):
            auc(y, score)


class TestKS:
    def test_hand_example(self):
        # sorted: 0.1(g) 0.35(b) 0.4(g) 0.8(b); cum_good .5 .5 1 1; cum_bad 0 .5 .5 1
        result = ks(Y_HAND, S_HAND)
        assert result.statistic == pytest.approx(0.5)
        assert result.threshold == pytest.approx(0.1)

    def test_matches_scipy_two_sample_statistic(self, random_scores):
        y, score = random_scores
        expected = ks_2samp(score[y == 1], score[y == 0]).statistic
        assert ks(y, score).statistic == pytest.approx(expected, abs=1e-12)

    def test_ties_move_together(self):
        # All scores equal: the distributions can never be separated.
        assert ks([0, 1, 0, 1], [0.3, 0.3, 0.3, 0.3]).statistic == 0.0


def test_brier_hand_example():
    # ((0.1-0)^2 + (0.4-0)^2 + (0.35-1)^2 + (0.8-1)^2) / 4
    expected = (0.01 + 0.16 + 0.4225 + 0.04) / 4
    assert brier(Y_HAND, S_HAND) == pytest.approx(expected)


class TestPSI:
    def test_hand_example_from_proportions(self):
        # (0.6-0.5)ln(1.2) + (0.4-0.5)ln(0.8)
        expected = 0.1 * math.log(1.2) + (-0.1) * math.log(0.8)
        assert psi_from_proportions([0.5, 0.5], [0.6, 0.4], epsilon=1e-4) == pytest.approx(expected)

    def test_zero_share_uses_epsilon(self):
        eps = 1e-4
        expected = (eps - 0.5) * math.log(eps / 0.5) + (1.0 - 0.5) * math.log(1.0 / 0.5)
        assert psi_from_proportions([0.5, 0.5], [0.0, 1.0], epsilon=eps) == pytest.approx(expected)

    def test_identical_samples_give_zero(self):
        x = np.random.default_rng(1).normal(size=2000)
        assert psi(x, x.copy(), n_bins=10).value == pytest.approx(0.0, abs=1e-12)

    def test_shift_is_detected_with_reference_edges(self):
        rng = np.random.default_rng(2)
        reference = rng.normal(size=20000)
        shifted = rng.normal(loc=0.5, size=20000)
        result = psi(reference, shifted, n_bins=10)
        assert result.value > 0.10
        # Re-binning on the current sample hides the shift; the edges must come from reference.
        rebinned = psi(shifted, shifted, n_bins=10)
        assert rebinned.value == pytest.approx(0.0, abs=1e-12)
        assert np.allclose(result.edges, psi(reference, reference, n_bins=10).edges)

    def test_reference_bins_are_deciles(self):
        reference = np.arange(1000, dtype=float)
        table = psi(reference, reference, n_bins=10).table
        assert len(table) == 10
        assert np.allclose(table["reference_pct"], 0.1)

    def test_values_outside_reference_range_land_in_end_bins(self):
        reference = np.linspace(0, 1, 1000)
        current = np.r_[np.full(100, -5.0), np.full(100, 5.0)]
        table = psi(reference, current, n_bins=10).table
        assert table["current_count"].iloc[0] == 100
        assert table["current_count"].iloc[-1] == 100

    def test_missing_values_get_their_own_bin(self):
        reference = np.r_[np.linspace(0, 1, 90), np.full(10, np.nan)]
        current = np.r_[np.linspace(0, 1, 50), np.full(50, np.nan)]
        table = psi(reference, current, n_bins=5).table
        missing = table[table["bin"] == "missing"].iloc[0]
        assert missing["reference_pct"] == pytest.approx(0.10)
        assert missing["current_pct"] == pytest.approx(0.50)


class TestWoE:
    def test_hand_example(self):
        # A: 40 good / 10 bad, B: 60 good / 40 bad; totals 100 good / 50 bad.
        bins = ["A"] * 50 + ["B"] * 100
        y = [0] * 40 + [1] * 10 + [0] * 60 + [1] * 40
        result = woe_iv_table(bins, y)
        table = result.table.set_index("bin")
        assert table.loc["A", "woe"] == pytest.approx(math.log(0.4 / 0.2))
        assert table.loc["B", "woe"] == pytest.approx(math.log(0.6 / 0.8))
        expected_iv = (0.4 - 0.2) * math.log(2) + (0.6 - 0.8) * math.log(0.75)
        assert result.iv == pytest.approx(expected_iv)
        assert not result.zero_count_adjusted

    def test_positive_woe_means_more_goods(self):
        result = woe_iv_table(["low_risk"] * 10 + ["high_risk"] * 10, [0] * 9 + [1] + [0] * 3 + [1] * 7)
        table = result.table.set_index("bin")
        assert table.loc["low_risk", "woe"] > 0 > table.loc["high_risk", "woe"]

    def test_zero_count_adjustment_applies_to_every_bin(self):
        # A: 10 good / 0 bad, B: 10 good / 10 bad -> +0.5 everywhere.
        bins = ["A"] * 10 + ["B"] * 20
        y = [0] * 10 + [0] * 10 + [1] * 10
        result = woe_iv_table(bins, y, zero_count_adjustment=0.5)
        table = result.table.set_index("bin")
        assert result.zero_count_adjusted
        # pct_good A = 10.5/21 = 0.5, pct_bad A = 0.5/11 -> WoE = ln(11)
        assert table.loc["A", "woe"] == pytest.approx(math.log(11))
        assert np.isfinite(table["woe"]).all()

    def test_missing_is_its_own_bin(self):
        result = woe_iv_table(["A", None, "A", None], [0, 1, 1, 0])
        assert MISSING_BIN in set(result.table["bin"])

    def test_transform_uses_fitted_table(self):
        fitted = woe_iv_table(["A", "A", "B", "B"], [0, 1, 1, 1], zero_count_adjustment=0.5)
        mapped = woe_transform(["B", "A"], fitted)
        mapping = fitted.mapping()
        assert mapped.tolist() == [mapping["B"], mapping["A"]]

    def test_transform_rejects_unseen_bins(self):
        fitted = woe_iv_table(["A", "B"], [0, 1], zero_count_adjustment=0.5)
        with pytest.raises(ValueError, match="C"):
            woe_transform(["A", "C"], fitted)


class TestRollRate:
    @staticmethod
    def panel(rows):
        return pd.DataFrame(rows, columns=["loan_id", "period", "bucket"])

    def test_hand_example(self):
        panel = self.panel([
            ("L1", "2019-01-01", "current"), ("L1", "2019-02-01", "dpd30"), ("L1", "2019-03-01", "dpd60"),
            ("L2", "2019-01-01", "current"), ("L2", "2019-02-01", "current"), ("L2", "2019-03-01", "dpd30"),
            ("L3", "2019-01-01", "dpd30"), ("L3", "2019-02-01", "current"),
        ])
        counts = transition_matrix(panel, states=BUCKET_ORDER, normalize=False)
        # current -> current 1, current -> dpd30 2; dpd30 -> dpd60 1, dpd30 -> current 1 (cure)
        assert counts.loc["current", "current"] == 1
        assert counts.loc["current", "dpd30"] == 2
        assert counts.loc["dpd30", "dpd60"] == 1
        assert counts.loc["dpd30", "current"] == 1

        rates = transition_matrix(panel, states=BUCKET_ORDER)
        assert rates.loc["current", "dpd30"] == pytest.approx(2 / 3)
        assert rates.loc["dpd30", "current"] == pytest.approx(0.5)
        observed_rows = rates.dropna(how="all")
        assert np.allclose(observed_rows.sum(axis=1), 1.0)
        assert rates.loc["dpd90plus"].isna().all()  # no observations: NaN, not zero

    def test_reporting_gap_breaks_the_pair(self):
        panel = self.panel([("L1", "2019-01-01", "current"), ("L1", "2019-03-01", "dpd60")])
        assert transition_matrix(panel, states=BUCKET_ORDER, normalize=False).to_numpy().sum() == 0

    def test_rejects_duplicated_loan_months(self):
        panel = self.panel([("L1", "2019-01-01", "current"), ("L1", "2019-01-01", "dpd30")])
        with pytest.raises(ValueError, match="duplicated"):
            transition_matrix(panel)
