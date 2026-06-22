#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from spoofing_detection.lob.client_identity_audit import audit_missing_client_trading_capacity


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit missing client ids against order trading capacity.")
    parser.add_argument("inputs", nargs="+", type=Path, help="Raw parquet files to audit")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional output JSON path")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    results = {}
    for path in args.inputs:
        df = pl.read_parquet(path)
        results[str(path)] = audit_missing_client_trading_capacity(df)
    payload = json.dumps(results, indent=2, sort_keys=True)
    print(payload)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(payload)


if __name__ == "__main__":
    main()
