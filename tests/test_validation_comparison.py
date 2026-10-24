"""Stage 7.2: champion against challenger - agreement, swap set at the reject rule, equal-volume
comparison and the side-by-side table built from the engine's own tables."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from credit_risk.validation.comparison import (
    ComparisonError,
    ComparisonSpec,
    agreement,
    at_equal_volume,
    comparison_checks,
    comparison_markdown,
    head_to_head,
    side_by_side,
    spearman,
    swap_set,
)
from credit_risk.validation.engine import run_validation


def frame(n=4000, seed=0, same=False):
    rng = np.random.default_rng(seed)
    risk = rng.normal(size=n)
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-(risk - 2.4)))).astype(int)
    pd_c = 1 / (1 + np.exp(-(risk - 2.4)))
    pd_g = pd_c if same else 1 / (1 + np.exp(-(risk + 0.4 * rng.normal(size=n) - 2.4)))
    return pd.DataFrame({"TARGET": y, "split": "test", "score": np.round(600 - 30 * risk), "pd_champion": pd_c, "pd_challenger": pd_g})


def test_own_spearman_matches_scipy_with_ties():
    rng = np.random.default_rng(1)
    a, b = np.round(rng.normal(size=500), 1), rng.normal(size=500)
    assert spearman(a, b) == pytest.approx(spearmanr(a, b).statistic, abs=1e-12)
    with pytest.raises(ComparisonError):
        spearman(np.ones(10), b[:10])


def test_identical_models_agree_completely_and_swap_nothing():
    f = frame(same=True)
    a = agreement(f["score"], f["pd_challenger"], f["TARGET"], 0.10)
    # The score is a rounded, monotone transform of the PD: ranks agree up to ties.
    assert a["spearman"] > 0.999 and a["top_overlap"] > 0.95
    swap = swap_set(f["TARGET"], f["pd_champion"], f["pd_challenger"], 0.10).set_index("cell")
    assert swap.loc["champion only rejects (challenger accepts)", "n"] == 0
    assert swap.loc["challenger only rejects (champion accepts)", "n"] == 0
    assert swap["n"].sum() == len(f)


def test_swap_cells_partition_the_sample_and_equal_volume_rejects_exactly_as_many():
    f = frame()
    swap = swap_set(f["TARGET"], f["pd_champion"], f["pd_challenger"], 0.10)
    assert swap["n"].sum() == len(f) and swap["bads"].sum() == f["TARGET"].sum()
    assert swap["share"].sum() == pytest.approx(1.0)
    k = int((f["pd_champion"] > 0.10).sum())
    ev = at_equal_volume(f["TARGET"], f["pd_champion"], f["pd_challenger"], 0.10)
    assert ev["rejected"] == k and ev["reject_rate"] == pytest.approx(k / len(f))
    # Rejecting with the true PD at the same volume cannot approve a worse book than the noisy one.
    own = at_equal_volume(f["TARGET"], f["pd_challenger"], f["pd_champion"], 0.10)
    assert own["bad_rate_approved"] <= at_equal_volume(f["TARGET"], f["pd_challenger"], f["pd_challenger"], 0.10)["bad_rate_approved"] + 0.01


def test_head_to_head_runs_the_configured_samples_and_checks_add_up():
    f = pd.concat([frame(seed=2).assign(split="validation"), frame(seed=3)], ignore_index=True)
    spec = ComparisonSpec()
    h2h = head_to_head(f, spec)
    assert set(h2h["agreement"]) == {"validation", "test"}
    checks = comparison_checks(h2h, spec, int((f["split"] == "test").sum()), None)
    assert checks["ok"].all() and len(checks) == 3
    published = {k: v for k, v in h2h["agreement"]["validation"].items() if k in ("spearman", "top_overlap")}
    assert comparison_checks(h2h, spec, 4000, published)["ok"].all()
    published["spearman"] += 0.01
    assert not comparison_checks(h2h, spec, 4000, published)["ok"].all()
    with pytest.raises(ComparisonError, match="no rows"):
        head_to_head(f[f["split"] == "validation"], spec)


def test_the_spec_reads_the_reject_rule_and_headline_sample_from_the_project_config():
    spec = ComparisonSpec.from_config({"segment_options": {"reject_rule": {"pd_above": 0.2}}, "conclusion_rules": {"headline_sample": "validation"}})
    assert spec.reject_pd_above == 0.2 and spec.sample == "validation"
    with pytest.raises(ComparisonError):
        ComparisonSpec.from_config({"comparison": {"top_share": 1.5}})


def test_on_the_synthetic_build_the_side_by_side_table_agrees_with_the_modules(stage6_build):
    settings, config = stage6_build
    result = run_validation(settings, config, "cmp-run")
    s = result.sections
    table = s["comparison"]["table"].set_index("model")
    disc = s["discrimination"]["table"]
    for model in ("champion", "challenger"):
        d = disc[(disc["model"] == model) & (disc["split"] == "test")].iloc[0]
        assert table.loc[model, "auc"] == d["auc"] and table.loc[model, "gini"] == d["gini"]
        c = s["calibration"]["summary"]
        assert table.loc[model, "slope"] == c[(c["model"] == model) & (c["split"] == "test")].iloc[0]["slope"]
    assert s["comparison"]["checks"]["ok"].all(), s["comparison"]["checks"].to_string()
    assert (s["comparison"]["checks"]["source"] == "development run").sum() == 4
    text = result.markdown["comparison"]
    assert "Hoán đổi quyết định" in text and "cùng số từ chối" in text
    assert "comparison" in {c for c in result.checks["module"]}
