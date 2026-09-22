"""Stage 2 Definition of Done, end to end through the pipeline:
"the bin table is reproducible and learnt on train only".

Each test runs the real command sequence (ingest -> validate-data -> features)
on the fixture data, so it covers the split, the feature build, the binning,
the IV report and the shortlist together.
"""

import json

import pandas as pd

from credit_risk.pipeline import run_pipeline
from conftest import write_home_credit_raw

STEPS = ["ingest", "validate-data", "features"]


def fingerprints(settings):
    assert run_pipeline(steps=STEPS, sources=["home_credit"], settings=settings) == 0
    folder = settings.processed_dir / "home_credit"
    binning = json.loads((folder / "binning.json").read_text(encoding="utf-8"))
    shortlist = json.loads((folder / "shortlist.json").read_text(encoding="utf-8"))
    return binning["fingerprint"], shortlist["fingerprint"]


def split_ids(settings, train: bool):
    splits = pd.read_parquet(settings.processed_dir / "home_credit" / "splits.parquet")
    return set(splits.loc[(splits["split"] == "train") == train, "SK_ID_CURR"])


def edit_rows(path, ids, changes):
    # round_trip: pandas' default float parser can be off by one ulp, which would
    # silently change the untouched train rows and fail this test for the wrong reason.
    frame = pd.read_csv(path, float_precision="round_trip")
    rows = frame["SK_ID_CURR"].isin(ids)
    for column, value in changes.items():
        frame.loc[rows, column] = value
    frame.to_csv(path, index=False)
    return int(rows.sum())


def test_the_same_data_always_gives_the_same_bins_and_shortlist(settings):
    write_home_credit_raw(settings)
    first = fingerprints(settings)
    assert fingerprints(settings) == first


def test_rows_outside_train_cannot_move_the_bins(settings):
    raw = write_home_credit_raw(settings)
    before, _ = fingerprints(settings)
    outside = split_ids(settings, train=False)

    # Wreck every non-train applicant - validation, calibration, test - and the
    # whole unlabelled application_test population. Targets are left alone so
    # the stratified split itself cannot change.
    wrecked = {"AMT_CREDIT": 9.9e9, "AMT_ANNUITY": 1.0, "EXT_SOURCE_2": 0.999, "DAYS_BIRTH": -7000}
    assert edit_rows(raw / "application_train.csv", outside, wrecked) > 0
    test_ids = set(pd.read_csv(raw / "application_test.csv", float_precision="round_trip")["SK_ID_CURR"])
    assert edit_rows(raw / "application_test.csv", test_ids, {"AMT_CREDIT": 9.9e9, "DAYS_BIRTH": -7000}) > 0

    after, _ = fingerprints(settings)
    assert after == before


def test_the_same_edit_on_train_rows_does_move_the_bins(settings):
    """Control for the test above: the check is sensitive, the train rows really are what is learnt."""
    raw = write_home_credit_raw(settings)
    before, _ = fingerprints(settings)
    train = sorted(split_ids(settings, train=True))[:5]
    assert edit_rows(raw / "application_train.csv", train, {"AMT_CREDIT": 9.9e9, "EXT_SOURCE_2": 0.999}) == 5

    after, _ = fingerprints(settings)
    assert after != before


def test_the_run_manifest_ties_the_bins_to_the_split_and_the_code(settings):
    write_home_credit_raw(settings)
    fingerprints(settings)
    folder = settings.processed_dir / "home_credit"
    binning = json.loads((folder / "binning.json").read_text(encoding="utf-8"))
    shortlist = json.loads((folder / "shortlist.json").read_text(encoding="utf-8"))
    manifest_path = next(settings.artifacts_dir.glob(f"{binning['run_id']}/run_manifest.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert binning["fitted_on"] == "train"
    assert shortlist["binning_fingerprint"] == binning["fingerprint"]
    assert manifest["config_hash"].startswith(binning["run_id"].split("-")[-1])
    assert "commit" in manifest["git"]
    features_step = next(s for s in manifest["steps"] if s["step"] == "features")
    assert binning["fingerprint"][:12] in features_step["detail"]
