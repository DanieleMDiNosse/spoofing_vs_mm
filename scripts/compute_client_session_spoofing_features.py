#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from spoofing_detection.lob.client_session_features import compute_client_session_features


def compute_and_write(*, input_path: Path, output_dir: Path, msci_threshold: float) -> dict[str, Path]:
    executions = pl.read_parquet(input_path)
    features = compute_client_session_features(executions, msci_threshold=msci_threshold)
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / "client_session_features.parquet"
    csv_path = output_dir / "client_session_features.csv"
    metadata_path = output_dir / "metadata.json"
    features.write_parquet(parquet_path)
    features.write_csv(csv_path)
    metadata_path.write_text(json.dumps({"created_at_utc": datetime.now(timezone.utc).isoformat(), "input_path": str(input_path), "msci_threshold": msci_threshold, "rows": features.height}, indent=2, sort_keys=True))
    return {"parquet": parquet_path, "csv": csv_path, "metadata": metadata_path}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute client-session spoofing surveillance features.")
    parser.add_argument("--execution-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--msci-threshold", type=float, default=0.5)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    outputs = compute_and_write(input_path=args.execution_metrics, output_dir=args.output_dir, msci_threshold=args.msci_threshold)
    print(outputs["parquet"])


if __name__ == "__main__":
    main()
