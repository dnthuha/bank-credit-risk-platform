"""Stage 6.1: the validator's package loader refuses anything altered or inconsistent,
and rescoring from the package alone reproduces the published outputs exactly.

A real package is built once per module with the pipeline on synthetic data
(the Stage 5 chain), then each test works on a copy of it.
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import REPO_ROOT, Settings, quick_config_dir, write_home_credit_raw
from credit_risk.config import load_config
from credit_risk.data.db import connect
from credit_risk.pipeline import run_pipeline
from credit_risk.validation.package import (
    PackageError,
    load_package,
    load_scoring_frame,
    score_with_package,
)

STEPS = ["ingest", "validate-data", "features", "train", "calibrate"]


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("stage61")
    settings = Settings(repo_root=REPO_ROOT, config_dir=quick_config_dir(root / "configs"),
                        data_dir=root / "data", artifacts_dir=root / "artifacts",
                        duckdb_memory_limit="512MB", duckdb_threads=2)
    write_home_credit_raw(settings, n=20000, signal=True)
    assert run_pipeline(steps=STEPS, sources=["home_credit"], settings=settings) == 0
    return settings, load_config(settings.config_dir)


@pytest.fixture
def package_dir(built, tmp_path):
    settings, _ = built
    source = settings.processed_dir / "home_credit" / "model_package"
    return Path(shutil.copytree(source, tmp_path / "model_package"))


def load(built, folder):
    _, config = built
    return load_package(folder, config.model_dev, config.definitions)


def test_a_clean_package_passes_every_check(built, package_dir):
    package = load(built, package_dir)
    assert package.checks["ok"].all()
    meta = json.loads((package_dir / "package.json").read_text(encoding="utf-8"))
    assert package.fingerprint == meta["fingerprint"]
    assert package.champion.fingerprint() == meta["champion_fingerprint"]
    assert package.challenger.fingerprint() == meta["challenger_fingerprint"]
    assert set(package.champion.features) <= set(package.features)
    assert package.summary()["published_rows"] == len(package.published)


def test_one_changed_byte_in_a_model_file_fails_the_load(built, package_dir):
    path = package_dir / "champion.json"
    text = path.read_text(encoding="utf-8")
    # A whitespace change: the JSON still parses to the same model, but the bytes differ.
    path.write_text(text + "\n", encoding="utf-8")
    with pytest.raises(PackageError, match="sha256 champion.json"):
        load(built, package_dir)


def test_a_changed_coefficient_with_a_patched_hash_fails_on_the_fingerprint(built, package_dir):
    """An attacker who recomputes the file hash still cannot pass: the model is rebuilt and
    its fingerprint recomputed from the loaded coefficients."""
    from credit_risk.models.calibration import sha256_file

    champion_path = package_dir / "champion.json"
    champion = json.loads(champion_path.read_text(encoding="utf-8"))
    feature = champion["scorecard"]["features"][0]
    champion["scorecard"]["coefficients"][feature] *= 1.5
    champion_path.write_text(json.dumps(champion, default=str), encoding="utf-8")
    meta_path = package_dir / "package.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["files"]["champion.json"] = sha256_file(champion_path)
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(PackageError, match="champion fingerprint recomputed"):
        load(built, package_dir)


def test_a_missing_file_fails_the_load(built, package_dir):
    (package_dir / "challenger_model.txt").unlink()
    with pytest.raises(PackageError, match="missing"):
        load(built, package_dir)


def test_a_package_built_under_other_scaling_is_refused(built, package_dir):
    _, config = built
    definitions = json.loads(json.dumps(config.definitions))
    definitions["scorecard_scaling"]["pdo"] = float(definitions["scorecard_scaling"]["pdo"]) * 2
    with pytest.raises(PackageError, match="scaling"):
        load_package(package_dir, config.model_dev, definitions)


def test_rescoring_from_the_package_reproduces_the_published_outputs(built, package_dir):
    settings, _ = built
    package = load(built, package_dir)
    folder = settings.processed_dir / "home_credit"
    frame = load_scoring_frame(connect(settings), folder / "features.parquet", folder / "splits.parquet",
                               package.features)
    assert frame["SK_ID_CURR"].is_monotonic_increasing
    assert frame["split"].isna().sum() == (frame["population"] == "test").sum()
    scored = score_with_package(package, frame)
    merged = scored.merge(package.published, on="SK_ID_CURR", suffixes=("", "_published"))
    assert len(merged) == len(package.published) == len(scored)
    assert np.array_equal(merged["score"], merged["score_published"])
    assert np.array_equal(merged["pd_champion"], merged["pd_champion_published"])
    assert np.array_equal(merged["pd_challenger"], merged["pd_challenger_published"])
    # The published PD is the calibrator applied to the model's own log-odds, nothing else.
    assert np.allclose(package.champion_calibrator.apply(scored["log_odds_champion"]), scored["pd_champion"])


def test_rescoring_refuses_a_frame_without_a_model_input(built, package_dir):
    package = load(built, package_dir)
    frame = pd.DataFrame({"SK_ID_CURR": [1, 2]})
    with pytest.raises(PackageError, match="lacks model inputs"):
        score_with_package(package, frame)


def test_the_scoring_frame_refuses_an_unknown_column(built, package_dir):
    settings, _ = built
    folder = settings.processed_dir / "home_credit"
    with pytest.raises(PackageError, match="NOT_A_COLUMN"):
        load_scoring_frame(connect(settings), folder / "features.parquet", folder / "splits.parquet",
                           ["NOT_A_COLUMN"])
