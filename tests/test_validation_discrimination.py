"""Stage 6.3: AUC / Gini / KS recomputed by the validator.

Hand examples first, then the two AUC implementations against each other and
against sklearn on random data with ties, then the real chain: the rescored
package on the synthetic build must reproduce the development run's figures.
"""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from credit_risk.data.db import connect
from credit_risk.validation.discrimination import (
    DiscriminationError,
    DiscriminationSpec,
    auc_by_pairs,
    discrimination_checks,
    discrimination_markdown,
    ks_location,
    published_from_train_artifacts,
    recompute_discrimination,
)
from credit_risk.validation.metrics import auc
from credit_risk.validation.package import load_package, load_scoring_frame, score_with_package

# Hand example: bads at risk 0.35 and 0.8, goods at 0.1 and 0.4 -> 3 of 4 pairs concordant.
Y = np.array([0, 1, 0, 1])
RISK = np.array([0.1, 0.35, 0.4, 0.8])


def test_pair_counting_auc_matches_the_hand_example_and_counts_ties():
    value, tied = auc_by_pairs(Y, RISK)
    assert value == pytest.approx(0.75) and tied == 0
    # One good moves onto a bad's risk: that pair is tied and counts one half.
    value, tied = auc_by_pairs(Y, np.array([0.1, 0.35, 0.35, 0.8]))
    assert value == pytest.approx((3 + 0.5) / 4) and tied == pytest.approx(0.25)


@pytest.mark.parametrize("seed", range(5))
def test_the_two_aucs_agree_with_each_other_and_with_sklearn_under_heavy_ties(seed):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, 5000)
    risk = np.round(rng.normal(size=5000) + 0.6 * y, 1)      # 1-decimal risk: many ties
    rank_based, (pairs, tied) = auc(y, risk), auc_by_pairs(y, risk)
    assert abs(rank_based - pairs) < 1e-12
    assert rank_based == pytest.approx(roc_auc_score(y, risk), abs=1e-12)
    assert tied > 0.01


def test_ks_location_reports_where_the_gap_is_and_the_shares_there():
    loc = ks_location(Y, RISK)
    # Cumulative at risk <= 0.1: goods 1/2, bads 0 -> gap 0.5; at 0.4: goods 1, bads 1/2 -> 0.5.
    # The first maximum wins, so the threshold is 0.1.
    assert loc["ks"] == pytest.approx(0.5)
    assert loc["ks_threshold_risk"] == 0.1
    assert loc["share_at_or_below"] == 0.25 and loc["goods_at_or_below"] == 0.5 and loc["bads_at_or_below"] == 0


def test_a_perfect_and_a_random_model_give_the_textbook_values():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    spec = DiscriminationSpec()
    perfect = pd.DataFrame({"TARGET": y, "split": "test", "score": -y.astype(float), "pd_challenger": y.astype(float)})
    out = recompute_discrimination(perfect, spec)
    assert (out["auc"] == 1).all() and (out["gini"] == 1).all() and (out["ks"] == 1).all()
    constant = perfect.assign(score=600.0, pd_challenger=0.08)
    out = recompute_discrimination(constant, spec)
    assert (out["auc"] == 0.5).all() and (out["gini"] == 0).all() and (out["tied_pair_share"] == 1).all()


def test_the_spec_rejects_unknown_metrics_and_samples():
    with pytest.raises(DiscriminationError):
        DiscriminationSpec.from_config({"discrimination": {"metrics": ["auc", "lift"]}})
    with pytest.raises(DiscriminationError):
        DiscriminationSpec.from_config({"discrimination": {"samples": ["oot"]}})
    spec = DiscriminationSpec.from_config({"discrimination": {"metrics": ["auc"], "samples": ["test"]}})
    assert spec.metrics == ("auc",) and spec.headline_samples == ("test",)


def test_a_split_without_target_is_refused():
    frame = pd.DataFrame({"TARGET": [np.nan, 1.0], "split": "test", "score": [1.0, 2.0], "pd_challenger": [0.1, 0.2]})
    with pytest.raises(DiscriminationError, match="without TARGET"):
        recompute_discrimination(frame, DiscriminationSpec())


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
    spec = DiscriminationSpec.from_config(config.validation)
    table = recompute_discrimination(scored, spec)
    published = published_from_train_artifacts(settings.artifacts_dir / package.meta["run_id"] / "train")
    return table, discrimination_checks(table, spec, published), published


def test_the_recount_reproduces_the_development_figures_exactly(recomputed):
    table, checks, published = recomputed
    assert set(published) == {"champion", "challenger"}
    assert {"train", "validation"} <= set(published["champion"])
    assert checks["ok"].all(), checks[~checks["ok"]].to_string()
    assert (checks["source"] == "development run").sum() >= 2 * 2 * 4   # 2 models x 2 splits x (3 metrics + identity)
    assert set(table["split"]) == {"train", "validation", "calibration", "test"}
    assert table.loc[table["headline"], "split"].isin({"train", "test"}).all()


def test_a_published_figure_that_was_edited_is_flagged(recomputed):
    table, _, published = recomputed
    edited = json.loads(json.dumps(published))
    edited["champion"]["validation"]["auc"] += 0.01
    checks = discrimination_checks(table, DiscriminationSpec(), edited)
    failed = set(checks.loc[~checks["ok"], "check"])
    assert failed == {"published auc (champion, validation)", "published gini = 2*auc - 1 (champion, validation)"}
    assert "lệch: 2" in discrimination_markdown(table, checks)


def test_the_champion_ks_is_located_in_score_points(recomputed):
    table, _, _ = recomputed
    champion = table[table["model"] == "champion"]
    assert champion["ks_cut"].str.startswith("score >= ").all()
    assert (champion["ks_at"] == champion["ks_at"].round()).all()
    challenger = table[table["model"] == "challenger"]
    assert challenger["ks_cut"].str.startswith("pd <= ").all()
    assert challenger["ks_at"].between(0, 1).all()
