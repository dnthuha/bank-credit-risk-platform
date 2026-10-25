"""Stage 9.2-9.3: encoder, the two models, reason codes and the watchlist on synthetic data."""

import copy
import math

import numpy as np
import pandas as pd
import pytest
from scipy.special import logit

from credit_risk.ews import models as md
from credit_risk.ews import watchlist as wl


def synthetic(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    bucket = rng.choice([0, 1, 2], n, p=[0.94, 0.04, 0.02])
    fico = rng.integers(600, 820, n).astype(float)
    fico[rng.random(n) < 0.01] = np.nan
    since = np.where(rng.random(n) < 0.8, 999, rng.integers(0, 60, n))
    state = rng.choice(["CA", "TX", "FL", "ZZ"], n, p=[0.5, 0.3, 0.1995, 0.0005])
    logit_p = -6 + 2.0 * bucket - 0.01 * (np.nan_to_num(fico, nan=700) - 700) + 1.0 * (since < 999) + 0.5 * (state == "FL")
    y = (rng.random(n) < 1 / (1 + np.exp(-logit_p))).astype(int)
    months = pd.date_range("2018-01-01", periods=6, freq="MS")
    return pd.DataFrame({"loan_id": [f"L{i:05d}" for i in range(n)], "period": months[np.arange(n) % 6], "y": y,
                         "bucket_num": bucket, "fico": fico, "months_since_delinquent": since, "property_state": state,
                         "bucket_at_t": np.array(["current", "dpd30", "dpd60"])[bucket], "fico_band": "x", "ltv_band": "y",
                         "vintage_year": 2017, "upb": 100000.0})


@pytest.fixture
def spec(project_config):
    ews = copy.deepcopy(project_config.ews)
    ews["ews"]["features"]["numeric"] = ["bucket_num", "fico", "months_since_delinquent"]
    ews["ews"]["features"]["categorical"] = ["property_state"]
    ews["ews"]["features"]["categorical_min_share"] = 0.01
    ews["ews"]["train_sample"]["non_event_rate"] = 0.5
    ews["ews"]["models"]["challenger"]["n_estimators"] = 200
    ews["ews"]["models"]["challenger"]["early_stopping_rounds"] = 20
    ews["ews"]["models"]["challenger"]["grid"] = [{"num_leaves": 4, "learning_rate": 0.1, "min_child_samples": 20},
                                                  {"num_leaves": 8, "learning_rate": 0.1, "min_child_samples": 20}]
    return md.ModelSpec.from_config(ews, num_threads=2)


@pytest.fixture
def fitted(spec):
    data = synthetic()
    train, valid = data.iloc[:4000].reset_index(drop=True), data.iloc[4000:].reset_index(drop=True)
    w = md.weights(train["y"].to_numpy(), spec.non_event_rate)
    enc = md.Encoder.fit(train, w, spec.numeric, spec.categorical, spec.categorical_min_share)
    return train, valid, enc, md.fit_baseline(train, spec, enc), md.fit_challenger(train, valid, spec, enc)


def test_encoder_bins(fitted):
    train, _, enc, _, _ = fitted
    assert enc.bins["bucket_num"] == {"kind": "values", "values": [0.0, 1.0, 2.0]}
    assert enc.bins["months_since_delinquent"]["modal"] == 999.0          # never delinquent: a bin of its own
    assert enc.bins["fico"]["kind"] == "quantile" and enc.bins["fico"]["modal"] is None
    assert enc.levels["property_state"] == ["CA", "FL", "TX"]               # ZZ is rarer than 1% -> other
    probe = pd.DataFrame({"bucket_num": [0, 2, 7, np.nan], "fico": [np.nan, 650, 800, 700],
                          "months_since_delinquent": [999, 3, 999, 40], "property_state": ["ZZ", None, "CA", "NV"]})
    X = enc.one_hot(probe).toarray()
    assert (X.sum(axis=1) == 4).all()                                       # one column per feature per row
    cols = np.array(enc.columns)
    assert list(cols[X[0] == 1]) == ["bucket_num:0.0", "fico:missing", "months_since_delinquent:==999.0", "property_state=other"]
    assert "bucket_num:other" in cols[X[2] == 1] and "property_state=missing" in cols[X[1] == 1]
    assert "bucket_num:missing" in cols[X[3] == 1]


def test_baseline_contributions_rebuild_the_log_odds(fitted):
    _, valid, enc, baseline, _ = fitted
    part = valid.head(50)
    contrib = baseline.contributions(part)
    rebuilt = baseline.model.intercept_[0] + (contrib + pd.Series(baseline.reference)).sum(axis=1)
    assert np.allclose(rebuilt, logit(baseline.predict_pd(part)))


def test_weighted_baseline_keeps_the_event_rate(fitted, spec):
    train, _, _, baseline, _ = fitted
    w = md.weights(train["y"].to_numpy(), spec.non_event_rate)
    assert np.average(baseline.predict_pd(train), weights=w) == pytest.approx(np.average(train["y"], weights=w), rel=0.05)


def test_challenger_selects_the_best_trial_and_is_reproducible(fitted, spec):
    train, valid, enc, _, challenger = fitted
    t = challenger.trials
    assert t["selected"].sum() == 1
    assert t.loc[t["selected"], "validation_pr_auc"].iloc[0] == t["validation_pr_auc"].max()
    again = md.fit_challenger(train, valid, spec, enc)
    assert again.fingerprint() == challenger.fingerprint()
    contrib = challenger.contributions(valid.head(20))
    assert list(contrib.columns) == spec.features
    assert challenger.importance()["gain_share"].sum() == pytest.approx(1.0)


def test_train_sample_condition_and_weights():
    where = md.train_sample_where(0.05, 7)
    assert "split = 'train'" in where and "y = 1" in where and "< 50000" in where and "|7'" in where
    assert md.weights(np.array([1, 0, 0]), 0.05).tolist() == [1.0, 20.0, 20.0]


def test_config_errors(project_config):
    ews = copy.deepcopy(project_config.ews)
    ews["ews"]["train_sample"]["non_event_rate"] = 0
    with pytest.raises(ValueError, match="non_event_rate"):
        md.ModelSpec.from_config(ews)
    ews = copy.deepcopy(project_config.ews)
    ews["ews"]["models"]["selection_metric"] = "brier"
    with pytest.raises(ValueError, match="selection_metric"):
        md.ModelSpec.from_config(ews)


@pytest.mark.parametrize("which", ["baseline", "challenger"])
def test_watchlist_takes_the_capacity_with_reasons(fitted, which):
    _, valid, _, baseline, challenger = fitted
    model = baseline if which == "baseline" else challenger
    month = valid[valid["period"] == valid["period"].max()].reset_index(drop=True)
    watch = wl.build_watchlist(month, model, 0.05, 3)
    assert len(watch) == math.ceil(0.05 * len(month))
    assert watch["rank"].tolist() == list(range(1, len(watch) + 1))
    assert (np.diff(watch["pd"]) <= 0).all()
    assert (watch["reason_1"] != "").all()
    for k in (1, 2, 3):
        named = watch[f"reason_{k}"] != ""
        assert (watch.loc[named, f"push_{k}"] > 0).all()
    assert (watch["push_1"] >= watch["push_2"].fillna(-np.inf)).all()
    first = watch.iloc[0]
    assert first["reason_1"].split("=")[0] in model.encoder.numeric + model.encoder.categorical
    summary = wl.watchlist_summary(watch, len(month), 0.05)
    assert summary["listed"] == len(watch) and summary["expected_events"] == pytest.approx(watch["pd"].sum())
