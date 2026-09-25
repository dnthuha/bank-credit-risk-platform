"""Stage 4 Definition of Done, end to end through the pipeline: the challenger is
reproducible, never sees the calibration or test rows, and its published PDs
and reason codes are exactly what the saved model gives."""

import json

import numpy as np
import pandas as pd

from credit_risk.models.challenger import Challenger, ChallengerSpec
from conftest import write_home_credit_raw
from test_stage2_dod import edit_rows, split_ids
from test_stage3_dod import reserved_ids, run


def challenger_payload(settings):
    run(settings)
    folder = settings.processed_dir / "home_credit"
    return json.loads((folder / "challenger.json").read_text(encoding="utf-8"))


def test_the_same_data_always_gives_the_same_challenger(settings):
    write_home_credit_raw(settings, n=20000, signal=True)
    first = challenger_payload(settings)
    assert challenger_payload(settings)["fingerprint"] == first["fingerprint"]


def test_calibration_and_test_rows_cannot_move_the_challenger(settings):
    raw = write_home_credit_raw(settings, n=20000, signal=True)
    before = challenger_payload(settings)
    wrecked = {"EXT_SOURCE_2": 0.999, "EXT_SOURCE_3": 0.001, "AMT_CREDIT": 9.9e9}
    assert edit_rows(raw / "application_train.csv", reserved_ids(settings), wrecked) > 0
    assert challenger_payload(settings)["fingerprint"] == before["fingerprint"]


def test_the_same_edit_on_train_rows_does_move_the_challenger(settings):
    """Control for the test above."""
    raw = write_home_credit_raw(settings, n=20000, signal=True)
    before = challenger_payload(settings)
    train = sorted(split_ids(settings, train=True))[:300]
    assert edit_rows(raw / "application_train.csv", train, {"EXT_SOURCE_2": 0.999, "EXT_SOURCE_3": 0.001}) == 300
    assert challenger_payload(settings)["fingerprint"] != before["fingerprint"]


def test_published_pds_are_what_the_saved_model_gives(settings):
    write_home_credit_raw(settings, n=20000, signal=True)
    payload = challenger_payload(settings)
    folder = settings.processed_dir / "home_credit"
    model_dev = json.loads(json.dumps(payload["spec"]))
    spec = ChallengerSpec(**model_dev)
    challenger = Challenger.from_payload(payload, (folder / "challenger_model.txt").read_text(encoding="utf-8"), spec)
    assert challenger.fingerprint() == payload["fingerprint"]

    features = pd.read_parquet(folder / "features.parquet")
    scores = pd.read_parquet(folder / "challenger_scores.parquet")
    assert set(scores["SK_ID_CURR"]) == set(features["SK_ID_CURR"])
    merged = features.merge(scores, on="SK_ID_CURR")
    assert np.allclose(challenger.predict_pd(merged), merged["pd_challenger"], rtol=0, atol=1e-12)
    reasons = [[r for r in row if pd.notna(r)]
               for row in merged[[f"reason_{i}" for i in range(1, spec.reason_codes + 1)]].itertuples(index=False)]
    assert reasons == challenger.reason_codes(challenger.contributions(merged))


def test_the_manifest_and_report_tie_the_challenger_to_its_inputs(settings):
    write_home_credit_raw(settings, n=20000, signal=True)
    payload = challenger_payload(settings)
    folder = settings.processed_dir / "home_credit"
    scorecard = json.loads((folder / "scorecard.json").read_text(encoding="utf-8"))
    binning = json.loads((folder / "binning.json").read_text(encoding="utf-8"))
    assert payload["fitted_on"] == "train" and payload["early_stopped_on"] == "validation"
    assert payload["binning_fingerprint"] == binning["fingerprint"]
    assert payload["champion_fingerprint"] == scorecard["fingerprint"]
    assert set(payload["features"]) == set(binning["features"])
    assert len(payload["trials"]) == 3 and sum(t["selected"] for t in payload["trials"]) == 1

    run_dir = settings.artifacts_dir / payload["run_id"]
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    train_step = next(s for s in manifest["steps"] if s["step"] == "train")
    assert payload["fingerprint"][:12] in train_step["detail"]
    assert scorecard["fingerprint"][:12] in train_step["detail"]
    assert manifest["packages"]["lightgbm"] is not None

    report = (run_dir / "train" / "challenger_report.md").read_text(encoding="utf-8")
    for must_say in ("lạc quan cho challenger", "PROJECT_SCOPE #7", "chưa calibrate", "Không ràng buộc đơn điệu"):
        assert must_say in report
