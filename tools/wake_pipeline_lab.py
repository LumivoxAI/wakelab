"""Run the local wake-pipeline diagnostic workbench."""

from __future__ import annotations

import argparse
from pathlib import Path
from collections.abc import Sequence

DEFAULT_PROFILE = Path(__file__).with_name("profile.json")


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE, help="initial server-local profile JSON")
    parser.add_argument("--port", type=int, default=8080, help="local HTTP port (default: 8080)")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = create_parser().parse_args(argv)
    from wake_pipeline_support.app import run_app  # type: ignore[import-not-found]

    run_app(args)


if __name__ in {"__main__", "__mp_main__"}:
    main()
