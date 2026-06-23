#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from spoofing_detection.lob.negative_controls import add_time_shift_placebo, add_wrong_side_placebo


def build_report(*, events_path: Path, output_path: Path, shift_events: int) -> None:
    events = pl.read_parquet(events_path)
    time_shift = add_time_shift_placebo(events, shift_events=shift_events)
    wrong_side = add_wrong_side_placebo(events)
    lines = [
        "# Spoofing Negative-Control Report",
        "",
        "This report defines placebo controls that should be scored in a later calibration run.",
        "",
        f"Real candidate events: {events.height}",
        f"time_shift placebo events: {time_shift.height}",
        f"wrong_side placebo events: {wrong_side.height}",
        "",
        "## Interpretation",
        "",
        "A production detector should score real candidate events higher than time-shifted or wrong-side placebo events. If placebo scores are comparable, the model may be detecting generic quote refresh rather than spoofing-like conditional cancellation.",
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build spoofing negative-control report.")
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shift-events", type=int, default=50)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    build_report(events_path=args.events, output_path=args.output, shift_events=args.shift_events)
    print(args.output)


if __name__ == "__main__":
    main()
