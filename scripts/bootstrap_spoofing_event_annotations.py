#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from spoofing_detection.lob.annotations import ANNOTATION_COLUMNS, validate_annotations


def build_empty_annotations(events: pl.DataFrame, *, reviewer: str) -> pl.DataFrame:
    ids = sorted(events.get_column("review_event_id").cast(pl.Utf8).unique().to_list())
    now = datetime.now(timezone.utc).isoformat()
    return pl.DataFrame(
        {
            "review_event_id": ids,
            "analyst_label": ["unclear_needs_more_context"] * len(ids),
            "confidence": [0.0] * len(ids),
            "benign_explanation": [""] * len(ids),
            "notes": [""] * len(ids),
            "reviewer": [reviewer] * len(ids),
            "reviewed_at_utc": [now] * len(ids),
        }
    ).select(ANNOTATION_COLUMNS)


def merge_existing_annotations(events: pl.DataFrame, existing: pl.DataFrame, *, reviewer: str) -> pl.DataFrame:
    base = build_empty_annotations(events, reviewer=reviewer)
    existing = validate_annotations(existing)
    existing_ids = set(existing.get_column("review_event_id").to_list())
    new_rows = base.filter(~pl.col("review_event_id").is_in(existing_ids))
    return pl.concat([existing, new_rows], how="vertical").sort("review_event_id")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Initialize or update event annotation CSV for spoofing dashboard events.")
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reviewer", default="analyst")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    events = pl.read_parquet(args.events)
    if args.output.exists():
        annotations = merge_existing_annotations(events, pl.read_csv(args.output), reviewer=args.reviewer)
    else:
        annotations = build_empty_annotations(events, reviewer=args.reviewer)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    annotations.write_csv(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
