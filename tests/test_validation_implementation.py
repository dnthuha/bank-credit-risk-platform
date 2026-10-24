"""Stage 6.8: rescoring a seeded sample from the package reproduces the published outputs,
and any altered published value is caught with its applicant id."""

import numpy as np
import pandas as pd
import pytest

from credit_risk.data.db import connect
from credit_risk.validation.implementation import (
    ImplementationError,
    ImplementationSpec,
    check_implementation,
    draw_sample,
    implementation_markdown,
)
from credit_risk.validation.package import LoadedPackage, load_package, load_scoring_frame


@pytest.fixture(scope="module")
def chain(stage6_build):
    settings, config = stage6_build
    folder = settings.processed_dir / "home_credit"
    package = load_package(folder / "model_package", config.model_dev, config.definitions)
    frame = load_scoring_frame(connect(settings), folder / "features.parquet", folder / "splits.parquet", package.features)
    return package, frame, ImplementationSpec.from_config(config.validation)


def with_published(package: LoadedPackage, published: pd.DataFrame) -> LoadedPackage:
    return LoadedPackage(**{**package.__dict__, "published": published})


def test_the_sample_is_seeded_stratified_and_without_replacement(chain):
    package, _, spec = chain
    ids = draw_sample(package.published, spec)
    again = draw_sample(package.published, spec)
    assert np.array_equal(ids, again) and len(ids) == len(set(ids))
    assert len(ids) == min(spec.sample_size, len(package.published))
    strata = package.published.set_index("SK_ID_CURR").loc[ids, "split"].fillna("application_test")
    assert set(strata) == set(package.published["split"].fillna("application_test"))
    other = draw_sample(package.published, ImplementationSpec(sample_size=spec.sample_size, seed=spec.seed + 1))
    assert not np.array_equal(ids, other)


def test_the_published_outputs_are_reproduced_exactly(chain):
    package, frame, spec = chain
    summary, differences, info = check_implementation(package, frame, spec)
    assert info["ok"] and differences.empty
    assert set(summary["quantity"]) >= {"score", "pd_champion", "pd_challenger", "champion_reason_1", "challenger_reason_1"}
    assert (summary.loc[summary["tolerance"] != "exact", "max_abs_diff"] == 0).all()
    text = implementation_markdown(summary, differences, info)
    assert "khớp hoàn toàn" in text and "LỆCH" not in text


def test_an_altered_score_pd_or_reason_code_is_caught_with_its_id(chain):
    package, frame, spec = chain
    published = package.published.copy()
    ids = draw_sample(published, spec)
    victim = published.index[published["SK_ID_CURR"] == ids[0]][0]
    published.loc[victim, "score"] += 1
    published.loc[victim, "pd_champion"] += 1e-6
    published.loc[victim, "pd_challenger"] += 5e-10            # inside the tolerance: not a difference
    published.loc[victim, "challenger_reason_1"] = "NOT_A_FEATURE"
    summary, differences, info = check_implementation(with_published(package, published), frame, spec)
    assert not info["ok"]
    flagged = summary.set_index("quantity")
    assert flagged.loc["score", "differences"] == 1 and flagged.loc["pd_champion", "differences"] == 1
    assert flagged.loc["pd_challenger", "differences"] == 0 and flagged.loc["pd_challenger", "max_abs_diff"] <= 5e-10
    assert flagged.loc["challenger_reason_1", "differences"] == 1
    assert set(differences["SK_ID_CURR"]) == {ids[0]}
    assert set(differences["quantity"]) == {"score", "pd_champion", "challenger_reason_1"}
    assert "LỆCH" in implementation_markdown(summary, differences, info)


def test_a_sampled_applicant_missing_from_the_frame_is_an_error(chain):
    package, frame, spec = chain
    ids = draw_sample(package.published, spec)
    with pytest.raises(ImplementationError, match="every sampled applicant"):
        check_implementation(package, frame[frame["SK_ID_CURR"] != ids[0]], spec)


def test_the_spec_guardrails():
    with pytest.raises(ImplementationError):
        ImplementationSpec.from_config({"implementation_check": {"sample_size": 0}})
    spec = ImplementationSpec.from_config({"seed": 3, "implementation_check": {"sample_size": 10, "abs_tolerance_pd": 1e-6}})
    assert (spec.sample_size, spec.abs_tolerance_pd, spec.seed) == (10, 1e-6, 3)
