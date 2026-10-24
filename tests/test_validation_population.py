"""Stage 6.2: the population recount agrees with an honest package and catches
a changed claim, a dropped applicant, a duplicated applicant and an altered split."""

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from conftest import REPO_ROOT, Settings, quick_config_dir, write_home_credit_raw
from credit_risk.config import load_config
from credit_risk.data.db import connect
from credit_risk.features.split import SplitSpec
from credit_risk.pipeline import run_pipeline
from credit_risk.validation.package import load_package
from credit_risk.validation.population import (
    PopulationError,
    Tolerances,
    population_markdown,
    recount_population,
    reconcile_population,
)

STEPS = ["ingest", "validate-data", "features", "train", "calibrate"]


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("stage62")
    settings = Settings(repo_root=REPO_ROOT, config_dir=quick_config_dir(root / "configs"),
                        data_dir=root / "data", artifacts_dir=root / "artifacts",
                        duckdb_memory_limit="512MB", duckdb_threads=2)
    write_home_credit_raw(settings, n=20000, signal=True)
    assert run_pipeline(steps=STEPS, sources=["home_credit"], settings=settings) == 0
    return settings, load_config(settings.config_dir)


@pytest.fixture
def workspace(built, tmp_path):
    """A private copy of the processed folder, so a test can alter data without touching the others."""
    settings, config = built
    folder = Path(shutil.copytree(settings.processed_dir / "home_credit", tmp_path / "home_credit"))
    return settings, config, folder


def reconcile(settings, config, folder, summary=None):
    package = load_package(folder / "model_package", config.model_dev, config.definitions)
    spec = SplitSpec.from_config(config.model_dev)
    recount = recount_population(connect(settings), folder, spec)
    return reconcile_population(package, recount, spec, Tolerances.from_config(config.validation), summary)


def latest_split_summary(settings):
    runs = sorted(settings.artifacts_dir.iterdir())
    for run in reversed(runs):
        path = run / "features" / "splits_summary.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    raise AssertionError("no splits_summary.json in the artifacts")


def test_an_honest_package_reconciles_with_zero_tolerance(workspace):
    settings, config, folder = workspace
    result = reconcile(settings, config, folder, latest_split_summary(settings))
    assert result.ok, result.failures.to_string()
    assert result.tolerances.rows == 0
    checks = set(result.checks["check"])
    assert {"rows_train (scorecard)", "published rows, total", "split fingerprint (splits_summary)"} <= checks
    assert all(c.startswith("exclusion:") and v == 0 for c, v in
               result.checks[result.checks["source"].str.startswith("model card")][["check", "recounted"]].itertuples(index=False))
    text = population_markdown(result)
    assert "khớp" in text and "LỆCH" not in text


def test_a_changed_rows_claim_is_a_failed_check(workspace):
    settings, config, folder = workspace
    path = folder / "model_package" / "champion.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["scorecard"]["rows_train"] += 1
    path.write_text(json.dumps(raw, default=str), encoding="utf-8")
    from credit_risk.models.calibration import sha256_file
    meta_path = folder / "model_package" / "package.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["files"]["champion.json"] = sha256_file(path)   # rows_train is not part of the fingerprint
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    result = reconcile(settings, config, folder)
    assert not result.ok
    assert list(result.failures["check"]) == ["rows_train (scorecard)"]
    assert "LỆCH" in population_markdown(result)


def test_a_dropped_applicant_shows_up_as_an_exclusion_and_a_count_mismatch(workspace):
    settings, config, folder = workspace
    path = folder / "application_train.parquet"
    frame = pd.read_parquet(path)
    frame.iloc[1:].to_parquet(path, index=False)

    result = reconcile(settings, config, folder)
    failed = set(result.failures["check"])
    assert "exclusion: split_ids_not_in_application_train" in failed
    assert "exclusion: feature_rows_not_in_any_application_file" in failed
    assert "published rows, total" in failed


def test_a_duplicated_applicant_is_caught(workspace):
    settings, config, folder = workspace
    path = folder / "application_train.parquet"
    frame = pd.read_parquet(path)
    pd.concat([frame, frame.iloc[:1]]).to_parquet(path, index=False)

    result = reconcile(settings, config, folder)
    assert "exclusion: duplicate_ids_application_train" in set(result.failures["check"])


def test_a_moved_applicant_changes_the_split_fingerprint_and_the_counts(workspace):
    settings, config, folder = workspace
    path = folder / "splits.parquet"
    splits = pd.read_parquet(path)
    first_train = splits.index[splits["split"] == "train"][0]
    splits.loc[first_train, "split"] = "test"
    splits.to_parquet(path, index=False)

    result = reconcile(settings, config, folder)
    failed = set(result.failures["check"])
    assert {"split fingerprint (package.json)", "rows_train (scorecard)", "published rows, split train",
            "published rows, split test"} <= failed


def test_a_summary_from_another_run_is_flagged(workspace):
    settings, config, folder = workspace
    summary = latest_split_summary(settings)
    summary["splits"][0]["bad_rate"] += 0.01
    result = reconcile(settings, config, folder, summary)
    assert list(result.failures["check"]) == [f"bad rate of split {summary['splits'][0]['split']} (splits_summary)"]


def test_tolerances_come_from_config_and_must_be_non_negative():
    tol = Tolerances.from_config({"population_reconciliation": {"abs_tolerance_rows": 2, "abs_tolerance_bad_rate": 0.001}})
    assert (tol.rows, tol.bad_rate) == (2, 0.001)
    with pytest.raises(PopulationError):
        Tolerances.from_config({"population_reconciliation": {"abs_tolerance_rows": -1}})


def test_a_missing_processed_file_is_an_error(workspace):
    settings, config, folder = workspace
    (folder / "splits.parquet").unlink()
    with pytest.raises(PopulationError, match="splits.parquet"):
        recount_population(connect(settings), folder, SplitSpec.from_config(config.model_dev))
