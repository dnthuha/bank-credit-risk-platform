"""Command line entry point: `python -m credit_risk ...` or `credit-risk ...`."""

from __future__ import annotations

import argparse
import logging
import sys

from credit_risk.config import ConfigError, check_definitions, load_config
from credit_risk.pipeline import SOURCES, STEPS, run_pipeline
from credit_risk.settings import load_settings


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="credit-risk", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run pipeline steps (default: all nine, in order)")
    run.add_argument("--steps", nargs="+", choices=list(STEPS), metavar="STEP",
                     help=f"subset of steps: {', '.join(STEPS)}")
    run.add_argument("--source", nargs="+", choices=SOURCES, default=list(SOURCES),
                     help="data sources for ingest / validate-data")

    sub.add_parser("check-config", help="load configs, print the config hash, check definitions")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    args = _build_parser().parse_args(argv)

    if args.command == "check-config":
        settings = load_settings()
        try:
            config = load_config(settings.config_dir)
        except ConfigError as exc:
            print(f"config error: {exc}")
            return 1
        problems = check_definitions(config.definitions)
        print(f"config hash: {config.hash}")
        for problem in problems:
            print(f"  problem: {problem}")
        print("definitions: OK" if not problems else f"definitions: {len(problems)} problem(s)")
        return 0 if not problems else 1

    try:
        return run_pipeline(steps=args.steps, sources=args.source)
    except (ConfigError, ValueError) as exc:
        logging.getLogger("credit_risk").error("%s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
