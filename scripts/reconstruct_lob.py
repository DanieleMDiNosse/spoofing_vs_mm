#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from spoofing_detection.lob.config import LOBConfig
from spoofing_detection.lob.io import reconstruct_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconstruct visible event-sourced LOB panel.")
    parser.add_argument("input", type=Path, help="Input .parquet or .csv file")
    parser.add_argument("output_dir", type=Path, help="Output directory")
    parser.add_argument("--top-n", type=int, default=10, help="Number of bid/ask levels to emit")
    parser.add_argument(
        "--snapshot-mode",
        choices=["none", "every_event_for_sample", "issue_rows_only", "end_of_partition"],
        default="end_of_partition",
        help="Granularity for debug active-order and price-level depth snapshots",
    )
    parser.add_argument("--max-rows", type=int, default=None, help="Optional smoke-test row limit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = LOBConfig(top_n=args.top_n, snapshot_mode=args.snapshot_mode)
    paths = reconstruct_file(args.input, args.output_dir, config=config, max_rows=args.max_rows)
    print(f"panel: {paths.panel_path}")
    print(f"normalized_events: {paths.normalized_path}")
    print(f"agent_event_state_panel: {paths.agent_panel_path}")
    print(f"active_order_snapshots: {paths.active_orders_path}")
    print(f"price_level_depth_snapshots: {paths.price_level_depth_path}")
    print(f"metadata: {paths.metadata_path}")
    print(f"validation_report: {paths.validation_path}")


if __name__ == "__main__":
    main()
