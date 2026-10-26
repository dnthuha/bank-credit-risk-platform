"""Shared loaders for the dashboard: find a report bundle, read its files, format numbers.

The app only reads what the `report` pipeline step wrote (reports/latest by default, or any
artifacts/<run_id>/report folder). It never recomputes a metric.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = REPO_ROOT / "reports" / "latest"

LIGHT_COLOURS = {"green": "#2e7d32", "amber": "#ed8f00", "red": "#c62828", "excluded": "#9e9e9e"}
VERDICT_COLOURS = {"pass": "#2e7d32", "note": "#ed8f00", "remediate": "#c62828"}
MODEL_COLOURS = {"champion": "#1f5f8b", "challenger": "#d9480f", "baseline": "#1f5f8b"}


# ---------------------------------------------------------------------------
# Bundle discovery
# ---------------------------------------------------------------------------

def candidate_bundles() -> list[Path]:
    """reports/latest first, then every artifacts/<run>/report, newest first."""
    found = []
    if (DEFAULT_BUNDLE / "index.json").exists():
        found.append(DEFAULT_BUNDLE)
    artifacts = Path(os.environ.get("CREDIT_RISK_ARTIFACTS_DIR", REPO_ROOT / "artifacts"))
    if artifacts.exists():
        for run_dir in sorted(artifacts.iterdir(), reverse=True):
            if (run_dir / "report" / "index.json").exists():
                found.append(run_dir / "report")
    return found


def resolve_bundle() -> Path | None:
    """Sidebar control + ?bundle= query parameter + CREDIT_RISK_BUNDLE_DIR; None when nothing is found."""
    options = candidate_bundles()
    override = st.query_params.get("bundle") or os.environ.get("CREDIT_RISK_BUNDLE_DIR")
    if override and (Path(override) / "index.json").exists():
        options = [Path(override), *[p for p in options if p != Path(override)]]   # the override is the default
    if not options:
        return None
    labels = {p: ("reports/latest" if p == DEFAULT_BUNDLE else f"{p.parent.name}/report") for p in options}
    with st.sidebar:
        chosen = st.selectbox("Bundle dữ liệu", options, format_func=lambda p: labels[p], key="bundle_choice")
    return chosen


# ---------------------------------------------------------------------------
# Cached readers (keyed on path + mtime so a re-run of `report` is picked up)
# ---------------------------------------------------------------------------

def _stamp(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return -1.0


@st.cache_data(show_spinner=False)
def _read_json(path: str, stamp: float) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def _read_csv(path: str, stamp: float) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def _read_text(path: str, stamp: float) -> str:
    return Path(path).read_text(encoding="utf-8")


class Bundle:
    """Lazy, cached access to one report bundle."""

    def __init__(self, folder: Path):
        self.folder = Path(folder)
        self.index = _read_json(str(self.folder / "index.json"), _stamp(self.folder / "index.json"))

    def has(self, rel: str) -> bool:
        return (self.folder / rel).exists()

    def json(self, rel: str) -> dict[str, Any] | None:
        path = self.folder / rel
        return _read_json(str(path), _stamp(path)) if path.exists() else None

    def csv(self, rel: str) -> pd.DataFrame | None:
        path = self.folder / rel
        return _read_csv(str(path), _stamp(path)) if path.exists() else None

    def text(self, rel: str) -> str | None:
        path = self.folder / rel
        return _read_text(str(path), _stamp(path)) if path.exists() else None

    @property
    def headline(self) -> dict[str, Any]:
        return self.index.get("headline", {})

    @property
    def runs(self) -> dict[str, str | None]:
        return self.index.get("runs", {})

    def missing(self, module: str) -> list[str]:
        return self.index.get("missing", {}).get(module, [])


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def pct(x: Any, nd: int = 1) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "—"
    return f"{float(x) * 100:.{nd}f}%"


def num(x: Any, nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "—"
    if isinstance(x, (int,)) or (isinstance(x, float) and float(x).is_integer() and abs(x) >= 1000):
        return f"{int(x):,}"
    return f"{float(x):.{nd}f}"


def ci(pair: Any, nd: int = 3) -> str:
    if not pair or len(pair) != 2:
        return ""
    return f"[{pair[0]:.{nd}f}, {pair[1]:.{nd}f}]"


def short(fingerprint: str | None, n: int = 12) -> str:
    return (fingerprint or "")[:n]


def verdict_label(code: str | None) -> str:
    return {"pass": "Đạt", "note": "Đạt có lưu ý", "remediate": "Phải khắc phục"}.get(code or "", code or "—")


def light_label(light: str | None) -> str:
    return {"green": "Xanh", "amber": "Vàng", "red": "Đỏ", "excluded": "Loại"}.get(light or "", light or "—")


def badge(text: str, colour: str) -> str:
    return (f'<span style="background:{colour};color:white;padding:2px 10px;border-radius:12px;'
            f'font-weight:600;font-size:0.9em">{text}</span>')


def section_missing(module: str, bundle: Bundle, what: str) -> bool:
    """Show a notice and return True when the module's main file is not in the bundle."""
    if bundle.has(what):
        return False
    st.warning(f"Bundle này không có kết quả của module **{module}** (`{what}`). "
               "Chạy `python -m credit_risk run` rồi `--steps report` để bổ sung.")
    return True


def markdown_section(text: str, heading: str) -> str:
    """The body of one `## heading` section of a markdown report (until the next `## `)."""
    lines, keep, out = text.splitlines(), False, []
    for line in lines:
        if line.startswith("## "):
            if keep:
                break
            keep = line[3:].strip().startswith(heading)
            continue
        if keep:
            out.append(line)
    return "\n".join(out).strip()


def model_scale() -> alt.Scale:
    return alt.Scale(domain=list(MODEL_COLOURS), range=list(MODEL_COLOURS.values()))


def light_scale() -> alt.Scale:
    return alt.Scale(domain=list(LIGHT_COLOURS), range=list(LIGHT_COLOURS.values()))
