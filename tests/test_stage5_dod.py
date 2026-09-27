"""Stage 5 Definition of Done, end to end through the pipeline: calibrators are
learnt on the calibration split only, the frozen package is reproducible and
self-consistent, and its published PDs are exactly what the package gives."""

import json

import numpy as np
import pandas as pd

from credit_risk.models.calibration import Calibrator, logit, sha256_file
from credit_risk.pipeline import run_pipeline
from conftest import write_home_credit_raw
from test_stage2_dod import edit_rows

STEPS = ["ingest", "validate-data", "features", "train", "calibrate"]


def package(settings):
    assert run_pipeline(steps=STEPS, sources=["home_credit"], settings=settings) == 0
    folder = settings.processed_dir / "home_credit" / "model_package"
    return json.loads((folder / "package.json").read_text(encoding="utf-8")), folder


def split_of(settings, name):
    splits = pd.read_parquet(settings.processed_dir / "home_credit" / "splits.parquet")
    return set(splits.loc[splits["split"] == name, "SK_ID_CURR"])


def test_the_same_data_always_gives_the_same_package(settings):
    write_home_credit_raw(settings, n=20000, signal=True)
    first, _ = package(settings)
    again, _ = package(settings)
    assert again["fingerprint"] == first["fingerprint"]
    assert again["files"]["calibrated_scores.parquet"] == first["files"]["calibrated_scores.parquet"]


def test_test_rows_cannot_move_the_package(settings):
    raw = write_home_credit_raw(settings, n=20000, signal=True)
    before, _ = package(settings)
    wrecked = {"EXT_SOURCE_2": 0.999, "EXT_SOURCE_3": 0.001, "AMT_CREDIT": 9.9e9}
    assert edit_rows(raw / "application_train.csv", split_of(settings, "test"), wrecked) > 0
    assert package(settings)[0]["fingerprint"] == before["fingerprint"]


def test_calibration_rows_do_move_the_calibrators(settings):
    """Control for the test above, and proof the calibrators are learnt where they should be:
    the models stay the same (calibration rows never reach train), the calibrators change."""
    raw = write_home_credit_raw(settings, n=20000, signal=True)
    before, _ = package(settings)
    edited = sorted(split_of(settings, "calibration"))[:1500]
    assert edit_rows(raw / "application_train.csv", edited, {"EXT_SOURCE_2": 0.999, "EXT_SOURCE_3": 0.001}) == 1500
    after, _ = package(settings)
    assert after["champion_fingerprint"] == before["champion_fingerprint"]
    assert after["challenger_fingerprint"] == before["challenger_fingerprint"]
    assert after["fingerprint"] != before["fingerprint"]


def test_the_package_is_self_consistent(settings):
    write_home_credit_raw(settings, n=20000, signal=True)
    meta, folder = package(settings)
    for name, digest in meta["files"].items():
        assert sha256_file(folder / name) == digest, name

    champion = json.loads((folder / "champion.json").read_text(encoding="utf-8"))
    challenger = json.loads((folder / "challenger.json").read_text(encoding="utf-8"))
    assert champion["scorecard"]["fingerprint"] == meta["champion_fingerprint"]
    assert challenger["challenger"]["fingerprint"] == meta["challenger_fingerprint"]
    assert meta["fitted_on"] == "calibration"

    # Published PDs are the package's calibrators applied to the models' own outputs.
    scores = pd.read_parquet(folder / "calibrated_scores.parquet")
    raw = pd.read_parquet(settings.processed_dir / "home_credit" / "challenger_scores.parquet")
    merged = scores.merge(raw[["SK_ID_CURR", "pd_challenger"]], on="SK_ID_CURR", suffixes=("", "_raw"))
    card = champion["scorecard"]
    champion_x = -(merged["score"].to_numpy(dtype=float) - card["offset"]) / card["factor"]
    assert np.allclose(Calibrator.from_dict(champion["calibration"]).apply(champion_x), merged["pd_champion"])
    challenger_x = logit(np.clip(merged["pd_challenger_raw"].to_numpy(dtype=float), 1e-15, 1 - 1e-15))
    assert np.allclose(Calibrator.from_dict(challenger["calibration"]).apply(challenger_x), merged["pd_challenger"])


def test_platt_keeps_the_scorecard_ranking(settings):
    write_home_credit_raw(settings, n=20000, signal=True)
    meta, folder = package(settings)
    scores = pd.read_parquet(folder / "calibrated_scores.parquet")
    if meta["calibration"]["champion"] == "platt":
        ordered = scores.sort_values(["score", "SK_ID_CURR"])
        assert (ordered["pd_champion"].diff().dropna() <= 0).all()  # higher score, lower PD


def test_the_manifest_and_report_tie_the_package_to_its_run(settings):
    write_home_credit_raw(settings, n=20000, signal=True)
    meta, _ = package(settings)
    run_dir = settings.artifacts_dir / meta["run_id"]
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    step = next(s for s in manifest["steps"] if s["step"] == "calibrate")
    assert step["status"] == "ok" and meta["fingerprint"][:12] in step["detail"]
    report = (run_dir / "calibrate" / "calibration_report.md").read_text(encoding="utf-8")
    for must_say in ("không được đọc", "Reliability", "Model package đã đóng băng", meta["fingerprint"][:12]):
        assert must_say in report
