"""Stage 9.2: rare-event metrics and the monthly list."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from credit_risk.ews import metrics as mt


def test_average_precision_matches_sklearn_with_and_without_ties():
    rng = np.random.default_rng(0)
    y = (rng.random(5000) < 0.02).astype(int)
    s = rng.random(5000) + y * 0.5
    assert mt.average_precision(y, s) == pytest.approx(average_precision_score(y, s), abs=1e-12)
    tied = np.round(s, 1)                                       # many ties
    assert mt.average_precision(y, tied) == pytest.approx(average_precision_score(y, tied), abs=1e-12)
    assert mt.average_precision([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8]) == 1.0
    with pytest.raises(ValueError):
        mt.average_precision([0, 0], [0.1, 0.2])


def frame(months=3, per_month=40, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for m in range(months):
        for i in range(per_month):
            rows.append({"period": pd.Timestamp(2020, 1 + m, 1), "loan_id": f"L{i:03d}", "pd": rng.random(), "y": int(rng.random() < 0.1)})
    return pd.DataFrame(rows)


def test_monthly_list_takes_the_top_share_of_each_month():
    f = frame()
    flags = mt.monthly_flags(f, "pd", 0.05)
    assert f[flags].groupby("period").size().tolist() == [2, 2, 2]     # ceil(0.05 * 40)
    for _, g in f.assign(flag=flags).groupby("period"):
        assert g.loc[g["flag"], "pd"].min() >= g.loc[~g["flag"], "pd"].max()


def test_ties_are_broken_by_key_and_order_does_not_matter():
    f = pd.DataFrame({"period": pd.Timestamp("2020-01-01"), "loan_id": ["B", "A", "C", "D"], "pd": [0.5, 0.5, 0.1, 0.1], "y": 0})
    assert f.loc[mt.monthly_flags(f, "pd", 0.25), "loan_id"].tolist() == ["A"]
    shuffled = f.sample(frac=1, random_state=3)
    assert shuffled.loc[mt.monthly_flags(shuffled, "pd", 0.25), "loan_id"].tolist() == ["A"]


def test_list_performance_and_the_monthly_table():
    out = mt.list_performance([1, 0, 1, 0, 0, 0, 0, 0, 0, 1], [True, True, False, False, False, False, False, False, False, False])
    assert out == {"listed": 2, "caught": 1, "capture": pytest.approx(1 / 3), "precision": 0.5, "lift": pytest.approx(0.5 / 0.3)}
    f = frame()
    table = mt.monthly_list_table(f, "pd", 0.1)
    assert table["loans"].tolist() == [40, 40, 40] and table["listed"].tolist() == [4, 4, 4]
    assert table["events"].sum() == f["y"].sum()


def test_evaluate_reads_the_metric_names():
    f = frame()
    out = mt.evaluate(f, "pd", ["roc_auc", "pr_auc", "brier", "capture_top_5pct", "precision_top_10pct", "lift_top_5pct"])
    assert out["rows"] == 120 and out["events"] == f["y"].sum()
    flags = mt.monthly_flags(f, "pd", 0.05)
    assert out["capture_top_5pct"] == pytest.approx(f.loc[flags, "y"].sum() / f["y"].sum())
    with pytest.raises(ValueError, match="unknown EWS metric"):
        mt.evaluate(f, "pd", ["gini_top"])


def test_calibration_table_by_decile():
    f = frame(months=1, per_month=1000)
    table = mt.calibration_table(f, "pd")
    assert table["rows"].sum() == 1000 and (table["rows"] == 100).all()
    assert table["mean_pd"].is_monotonic_increasing
