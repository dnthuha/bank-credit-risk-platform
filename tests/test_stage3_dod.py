"""Stage 3 Definition of Done, end to end through the pipeline: the scorecard is
reproducible, learnt on train only, and its published scores are exactly what
the frozen scorecard.json gives.

Each test runs the real command sequence (ingest -> validate-data -> features ->
train) on a fixture whose TARGET follows EXT_SOURCE_2 / EXT_SOURCE_3.
"""

import json

import numpy as np
import pandas as pd

from credit_risk.models.scorecard import Scorecard, ScorecardSpec
from credit_risk.pipeline import run_pipeline
from conftest import write_home_credit_raw
from test_stage2_dod import edit_rows, split_ids

STEPS = ["ingest", "validate-data", "features", "train"]


def run(settings):
    assert run_pipeline(steps=STEPS, sources=["home_credit"], settings=settings) == 0
    folder = settings.processed_dir / "home_credit"
    return json.loads((folder / "scorecard.json").read_text(encoding="utf-8"))


def test_the_same_data_always_gives_the_same_scorecard(settings):
    write_home_credit_raw(settings, n=20000, signal=True)
    first = run(settings)
    assert first["features"], "the fixture must yield a non-empty scorecard"
    assert run(settings)["fingerprint"] == first["fingerprint"]


def reserved_ids(settings):
    splits = pd.read_parquet(settings.processed_dir / "home_credit" / "splits.parquet")
    return set(splits.loc[splits["split"].isin(["calibration", "test"]), "SK_ID_CURR"])


def test_calibration_and_test_rows_cannot_move_the_scorecard(settings):
    """Validation and application_test legitimately shape the shortlist (Stage 2.5 checks);
    calibration and test must stay untouched until stages 5 and 6."""
    raw = write_home_credit_raw(settings, n=20000, signal=True)
    before = run(settings)
    outside = reserved_ids(settings)
    wrecked = {"EXT_SOURCE_2": 0.999, "EXT_SOURCE_3": 0.001, "AMT_CREDIT": 9.9e9}
    assert edit_rows(raw / "application_train.csv", outside, wrecked) > 0
    assert run(settings)["fingerprint"] == before["fingerprint"]


def test_the_same_edit_on_train_rows_does_move_the_scorecard(settings):
    """Control for the test above."""
    raw = write_home_credit_raw(settings, n=20000, signal=True)
    before = run(settings)
    train = sorted(split_ids(settings, train=True))[:300]
    assert edit_rows(raw / "application_train.csv", train, {"EXT_SOURCE_2": 0.999, "EXT_SOURCE_3": 0.001}) == 300
    assert run(settings)["fingerprint"] != before["fingerprint"]


def test_published_scores_are_what_the_frozen_scorecard_gives(settings, project_config):
    write_home_credit_raw(settings, n=20000, signal=True)
    payload = run(settings)
    folder = settings.processed_dir / "home_credit"
    card = Scorecard.from_payload(payload, ScorecardSpec.from_config(project_config.model_dev,
                                                                       project_config.definitions))
    features = pd.read_parquet(folder / "features.parquet")
    scores = pd.read_parquet(folder / "scores.parquet")

    # Every applicant, train population and application_test alike, has a score.
    assert set(scores["SK_ID_CURR"]) == set(features["SK_ID_CURR"])
    merged = features.merge(scores, on="SK_ID_CURR")
    assert np.array_equal(card.score(merged), merged["score"].to_numpy())
    assert np.allclose(card.pd_from_score(merged["score"]), merged["pd_uncalibrated"])
    reasons = [[r for r in row if pd.notna(r)]
               for row in merged[[f"reason_{i}" for i in range(1, card.spec.reason_codes + 1)]].itertuples(index=False)]
    assert reasons == card.reason_codes(card.points_frame(merged))


def test_the_manifest_and_model_card_tie_the_scorecard_to_its_inputs(settings):
    write_home_credit_raw(settings, n=20000, signal=True)
    payload = run(settings)
    folder = settings.processed_dir / "home_credit"
    binning = json.loads((folder / "binning.json").read_text(encoding="utf-8"))
    shortlist = json.loads((folder / "shortlist.json").read_text(encoding="utf-8"))
    assert payload["fitted_on"] == "train"
    assert payload["binning_fingerprint"] == binning["fingerprint"]
    assert payload["shortlist_fingerprint"] == shortlist["fingerprint"]
    assert set(payload["features"]) <= set(shortlist["selected"])

    run_dir = settings.artifacts_dir / payload["run_id"]
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    train_step = next(s for s in manifest["steps"] if s["step"] == "train")
    assert train_step["status"] == "ok"
    assert payload["fingerprint"][:12] in train_step["detail"]

    card = (run_dir / "train" / "model_card.md").read_text(encoding="utf-8")
    for must_say in ("Không có out-of-time thật", "application_test", "PROJECT_SCOPE #7", "chưa calibrate"):
        assert must_say in card


def test_train_stops_when_the_shortlist_no_longer_matches_the_bins(settings):
    write_home_credit_raw(settings, n=20000, signal=True)
    run(settings)
    path = settings.processed_dir / "home_credit" / "shortlist.json"
    shortlist = json.loads(path.read_text(encoding="utf-8"))
    shortlist["binning_fingerprint"] = "0" * 64
    path.write_text(json.dumps(shortlist), encoding="utf-8")
    assert run_pipeline(steps=["train"], sources=["home_credit"], settings=settings) == 1
