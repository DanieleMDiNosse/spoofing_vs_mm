#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import polars as pl


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-generate local LLM reviews for matched spoofing-like events.")
    parser.add_argument("--review-dir", type=Path, required=True)
    parser.add_argument("--parameter-grid-root", type=Path, default=None)
    parser.add_argument("--prompt", type=Path, default=Path("prompts/spoofing_surveillance_analyst.md"))
    parser.add_argument("--model", default="gemma4-hermes:latest")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout-seconds", type=int, default=550)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _run(command: list[str], *, dry_run: bool) -> None:
    printable = " ".join(command)
    if dry_run:
        print(f"DRY RUN: {printable}")
        return
    subprocess.run(command, check=True)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    events = pl.read_parquet(args.review_dir / "matched_spoofing_events.parquet").sort("review_event_id")
    if args.limit is not None:
        events = events.head(args.limit)
    for row in events.iter_rows(named=True):
        event_id = str(row["review_event_id"])
        output_dir = args.review_dir / "llm_reviews" / event_id
        response_path = output_dir / "response.md"
        if response_path.exists() and not args.overwrite:
            print(f"SKIP {event_id}: {response_path} exists")
            continue
        dossier_cmd = [
            sys.executable,
            "scripts/build_spoofing_event_dossier.py",
            "--review-dir",
            str(args.review_dir),
            "--event-id",
            event_id,
            "--output-dir",
            str(output_dir),
        ]
        if args.parameter_grid_root is not None:
            dossier_cmd.extend(["--parameter-grid-root", str(args.parameter_grid_root)])
        analyze_cmd = [
            sys.executable,
            "scripts/analyze_spoofing_event_with_llm.py",
            "--dossier",
            str(output_dir / "dossier.md"),
            "--prompt",
            str(args.prompt),
            "--output-dir",
            str(output_dir),
            "--model",
            args.model,
            "--timeout-seconds",
            str(args.timeout_seconds),
        ]
        print(f"EVENT {event_id}")
        _run(dossier_cmd, dry_run=args.dry_run)
        _run(analyze_cmd, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
