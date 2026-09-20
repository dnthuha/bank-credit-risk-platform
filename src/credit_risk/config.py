"""Load the YAML configs, fingerprint them, and sanity-check business definitions."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

CONFIG_FILES = ("definitions", "model_dev", "validation", "ews", "feature_availability")


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class ProjectConfig:
    definitions: dict[str, Any]
    model_dev: dict[str, Any]
    validation: dict[str, Any]
    ews: dict[str, Any]
    feature_availability: dict[str, Any]
    contracts: dict[str, dict[str, Any]]
    hash: str


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Missing config file: {path}")
    with path.open(encoding="utf-8") as fh:
        content = yaml.safe_load(fh)
    if not isinstance(content, dict):
        raise ConfigError(f"Config file must contain a mapping: {path}")
    return content


def load_config(config_dir: Path) -> ProjectConfig:
    parts = {name: _read_yaml(config_dir / f"{name}.yaml") for name in CONFIG_FILES}
    contracts = {
        path.stem: _read_yaml(path)
        for path in sorted((config_dir / "data_contracts").glob("*.yaml"))
    }
    # Any change to any definition, threshold or contract changes the hash,
    # so every artifact can be traced back to the exact configuration.
    canonical = json.dumps({**parts, "data_contracts": contracts}, sort_keys=True, default=str)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return ProjectConfig(**parts, contracts=contracts, hash=digest)


def check_definitions(definitions: dict[str, Any]) -> list[str]:
    """Return a list of problems; an empty list means the definitions are coherent."""
    problems: list[str] = []

    buckets = definitions.get("delinquency_buckets") or []
    if not buckets:
        problems.append("delinquency_buckets is empty")
    else:
        if buckets[0]["dpd_min"] != 0:
            problems.append("first delinquency bucket must start at 0 DPD")
        for prev, nxt in zip(buckets, buckets[1:]):
            if prev["dpd_max"] is None or nxt["dpd_min"] != prev["dpd_max"] + 1:
                problems.append(
                    f"buckets {prev['name']} and {nxt['name']} are not contiguous"
                )
        if buckets[-1]["dpd_max"] is not None:
            problems.append("last delinquency bucket must be open-ended (dpd_max: null)")
        if buckets[-1]["dpd_min"] != 90:
            problems.append("last bucket must start at 90 DPD to match the 90+ default definition")

    windows = definitions.get("windows", {})
    for key in ("observation_months", "performance_months"):
        value = windows.get(key)
        if not isinstance(value, int) or value <= 0:
            problems.append(f"windows.{key} must be a positive integer")

    psi = definitions.get("metric_thresholds", {}).get("psi", {})
    if not psi.get("stable_below", 0) < psi.get("significant_above", 0):
        problems.append("psi.stable_below must be lower than psi.significant_above")
    if not 0 < psi.get("zero_bin_epsilon", 0) < 0.01:
        problems.append("psi.zero_bin_epsilon must be a small positive number")

    covid = definitions.get("covid_period", {})
    if str(covid.get("start")) > str(covid.get("end")):
        problems.append("covid_period.start must not be after covid_period.end")
    if covid.get("treatment") not in {"exclude", "stress_segment", "oot"}:
        problems.append("covid_period.treatment must be exclude | stress_segment | oot")

    zb = definitions.get("mortgage", {}).get("zero_balance_codes", {})
    groups = [set(zb.get(k, [])) for k in ("voluntary_payoff", "liquidation", "other_exit")]
    if any(a & b for i, a in enumerate(groups) for b in groups[i + 1 :]):
        problems.append("a zero balance code appears in more than one group")

    return problems
