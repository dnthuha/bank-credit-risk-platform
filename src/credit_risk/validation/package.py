"""Stage 6.1: load the frozen model package as the validator, not as its author.

The validator trusts nothing in the package until it has checked it:

    integrity     every file listed in package.json exists and has the SHA-256
                  the manifest recorded (a changed byte fails the load);
    identity      the fingerprints of the scorecard, the challenger and the
                  package itself are recomputed from the loaded objects and
                  must equal what package.json and the model files claim;
    rescoring     `score_with_package` recomputes score and calibrated PD
                  for any applicant frame from the models and calibrators
                  only, never from `calibrated_scores.parquet`. Stage 6.8
                  compares the two row by row.

`load_scoring_frame` reads the applicants with the columns the two models
and the configured segments need, joined to their split. It reads
features.parquet and splits.parquet exactly as Stage 3-5 did, so a
difference in rescoring is a difference in the package, not in the data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from credit_risk.data.db import sql_ident, sql_str
from credit_risk.models.calibration import Calibrator, logit, package_fingerprint, sha256_file
from credit_risk.models.challenger import Challenger, ChallengerSpec
from credit_risk.models.scorecard import Scorecard, ScorecardSpec

PACKAGE_FILES = ("package.json", "champion.json", "challenger.json", "challenger_model.txt",
                 "calibrated_scores.parquet")
ID = "SK_ID_CURR"
BASE_COLUMNS = (ID, "population", "TARGET")


class PackageError(ValueError):
    """The package is incomplete, altered, or not what it says it is."""


@dataclass(frozen=True)
class LoadedPackage:
    folder: Path
    meta: dict[str, Any]
    champion: Scorecard
    champion_calibrator: Calibrator
    challenger: Challenger
    challenger_calibrator: Calibrator
    published: pd.DataFrame          # calibrated_scores.parquet, as published by Stage 5
    checks: pd.DataFrame             # one row per integrity / identity check, all passed

    @property
    def fingerprint(self) -> str:
        return self.meta["fingerprint"]

    @property
    def features(self) -> list[str]:
        """Every input column either model reads, in a stable order."""
        return list(dict.fromkeys([*self.champion.features, *self.challenger.features]))

    def summary(self) -> dict[str, Any]:
        return {
            "fingerprint": self.fingerprint,
            "champion_fingerprint": self.meta["champion_fingerprint"],
            "challenger_fingerprint": self.meta["challenger_fingerprint"],
            "split_fingerprint": self.meta["split_fingerprint"],
            "run_id": self.meta["run_id"],
            "frozen_at": self.meta["frozen_at"],
            "calibration": dict(self.meta["calibration"]),
            "champion_features": len(self.champion.features),
            "challenger_features": len(self.challenger.features),
            "published_rows": int(len(self.published)),
            "checks": int(len(self.checks)),
        }


# ---------------------------------------------------------------------------
# Loading with integrity and identity checks
# ---------------------------------------------------------------------------

def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PackageError(f"{path.name}: cannot be read as JSON ({exc})") from exc


def load_package(folder: Path, model_dev: dict[str, Any], definitions: dict[str, Any]) -> LoadedPackage:
    """Load and verify the package in `folder`. Raises PackageError on the first failed check.

    `model_dev` and `definitions` are the project configuration: the scorecard scaling
    (points per PDO) and the challenger's input handling come from there, and the loader
    checks that the package was built under the same scaling it is now read with.
    """
    folder = Path(folder)
    missing = [name for name in PACKAGE_FILES if not (folder / name).exists()]
    if missing:
        raise PackageError(f"{folder}: missing {missing}")
    meta = _read_json(folder / "package.json")
    for key in ("fingerprint", "champion_fingerprint", "challenger_fingerprint", "split_fingerprint",
                "calibration", "files", "run_id", "frozen_at"):
        if key not in meta:
            raise PackageError(f"package.json lacks '{key}'")

    checks: list[dict[str, Any]] = []

    def check(name: str, expected: Any, actual: Any) -> None:
        checks.append({"check": name, "expected": str(expected), "actual": str(actual), "ok": expected == actual})
        if expected != actual:
            raise PackageError(f"{name}: expected {expected}, got {actual}")

    # 1. Integrity: the bytes on disk are the bytes the manifest froze.
    listed = set(meta["files"])
    expected_listed = set(PACKAGE_FILES) - {"package.json"}
    check("files listed in package.json", sorted(expected_listed), sorted(listed))
    for name in PACKAGE_FILES[1:]:
        check(f"sha256 {name}", meta["files"][name], sha256_file(folder / name))

    # 2. Identity: rebuild each model and recompute what the files claim to be.
    champion_raw = _read_json(folder / "champion.json")
    challenger_raw = _read_json(folder / "challenger.json")
    for raw, name, key in ((champion_raw, "champion.json", "scorecard"), (challenger_raw, "challenger.json", "challenger")):
        if key not in raw or "calibration" not in raw:
            raise PackageError(f"{name} must hold '{key}' and 'calibration'")

    try:
        scorecard_spec = ScorecardSpec.from_config(model_dev, definitions)
        challenger_spec = ChallengerSpec.from_config(model_dev)
    except ValueError as exc:
        raise PackageError(f"configuration: {exc}") from exc
    payload = champion_raw["scorecard"]
    check("scorecard scaling factor vs definitions", round(scorecard_spec.factor, 8), round(float(payload["factor"]), 8))
    check("scorecard scaling offset vs definitions", round(scorecard_spec.offset, 8), round(float(payload["offset"]), 8))
    try:
        champion = Scorecard.from_payload(payload, scorecard_spec)
        challenger = Challenger.from_payload(
            challenger_raw["challenger"], (folder / "challenger_model.txt").read_text(encoding="utf-8"),
            challenger_spec)
        calibrators = {name: Calibrator.from_dict(raw["calibration"])
                       for name, raw in (("champion", champion_raw), ("challenger", challenger_raw))}
    except (KeyError, TypeError, ValueError) as exc:
        raise PackageError(f"model files cannot be rebuilt: {exc}") from exc

    check("champion fingerprint recomputed", payload["fingerprint"], champion.fingerprint())
    check("champion fingerprint vs package.json", meta["champion_fingerprint"], champion.fingerprint())
    check("challenger fingerprint recomputed", challenger_raw["challenger"]["fingerprint"], challenger.fingerprint())
    check("challenger fingerprint vs package.json", meta["challenger_fingerprint"], challenger.fingerprint())
    check("challenger trained next to this champion", meta["champion_fingerprint"],
          challenger_raw["challenger"].get("champion_fingerprint"))
    check("calibration methods vs package.json", dict(meta["calibration"]),
          {name: c.method for name, c in calibrators.items()})
    content = {"champion": champion.fingerprint(), "challenger": challenger.fingerprint(),
               "calibration": {n: c.to_dict() for n, c in calibrators.items()}}
    check("package fingerprint recomputed", meta["fingerprint"], package_fingerprint(content))

    published = pd.read_parquet(folder / "calibrated_scores.parquet")
    for column in (ID, "population", "split", "score", "pd_champion", "pd_challenger"):
        if column not in published.columns:
            raise PackageError(f"calibrated_scores.parquet lacks '{column}'")
    check("published ids are unique", 0, int(published[ID].duplicated().sum()))

    return LoadedPackage(
        folder=folder, meta=meta, champion=champion, champion_calibrator=calibrators["champion"],
        challenger=challenger, challenger_calibrator=calibrators["challenger"],
        published=published, checks=pd.DataFrame(checks),
    )


# ---------------------------------------------------------------------------
# Applicants and independent rescoring
# ---------------------------------------------------------------------------

def load_scoring_frame(con: duckdb.DuckDBPyConnection, features_path: Path, splits_path: Path,
                       columns: list[str]) -> pd.DataFrame:
    """Applicants with id, population, TARGET, the requested columns and their split, ordered by id.

    application_test rows have no split (NULL). A requested column that features.parquet
    does not hold is an error: the validator must not silently score without an input.
    """
    available = {row[0] for row in con.execute(
        f"DESCRIBE SELECT * FROM read_parquet({sql_str(features_path)})").fetchall()}
    wanted = list(dict.fromkeys([*BASE_COLUMNS, *columns]))
    missing = [c for c in wanted if c not in available]
    if missing:
        raise PackageError(f"features.parquet lacks {missing}")
    return con.execute(
        f"SELECT {', '.join('f.' + sql_ident(c) for c in wanted)}, s.split "
        f"FROM read_parquet({sql_str(features_path)}) f "
        f"LEFT JOIN read_parquet({sql_str(splits_path)}) s USING ({ID}) ORDER BY {ID}"
    ).df()


def score_with_package(package: LoadedPackage, frame: pd.DataFrame) -> pd.DataFrame:
    """Score, log-odds and calibrated PD of both models for `frame`, from the package only.

    Mirrors how Stage 5 published its PDs: the scorecard's log-odds come from the integer
    score through the scaling (not from the unrounded fit), the challenger's from the logit
    of its raw PD. Anything else would compare a different model to the published one.
    """
    missing = [c for c in package.features if c not in frame.columns]
    if missing:
        raise PackageError(f"frame lacks model inputs {missing}")
    card = package.champion
    score = card.score(frame)
    champion_x = -(score.astype(float) - card.spec.offset) / card.spec.factor
    pd_raw = package.challenger.predict_pd(frame)
    challenger_x = logit(np.clip(pd_raw, 1e-15, 1 - 1e-15))
    return pd.DataFrame({
        ID: frame[ID].to_numpy(),
        "score": score,
        "log_odds_champion": champion_x,
        "pd_champion": package.champion_calibrator.apply(champion_x),
        "pd_challenger_raw": pd_raw,
        "log_odds_challenger": challenger_x,
        "pd_challenger": package.challenger_calibrator.apply(challenger_x),
    }, index=frame.index)
