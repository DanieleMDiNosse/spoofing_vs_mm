#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from spoofing_detection.lob.audit import audit_paths, write_audit_report


def _expand_inputs(inputs: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for input_path in inputs:
        if input_path.is_dir():
            paths.extend(sorted(input_path.glob("*.parquet")))
            paths.extend(sorted(input_path.glob("*.csv")))
        else:
            paths.append(input_path)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit LOB input schemas and provisional enum coverage.")
    parser.add_argument("inputs", nargs="+", type=Path, help="Input .parquet/.csv files or directories")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for schema_audit.md/json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_paths = _expand_inputs(args.inputs)
    if not input_paths:
        raise SystemExit("no .parquet or .csv inputs found")
    audit = audit_paths(input_paths)
    paths = write_audit_report(audit, args.output_dir)
    print(f"files_audited: {audit['files_audited']}")
    print(f"total_rows: {audit['aggregate']['row_count']}")
    print(f"schema_audit_json: {paths['json']}")
    print(f"schema_audit_markdown: {paths['markdown']}")


if __name__ == "__main__":
    main()
