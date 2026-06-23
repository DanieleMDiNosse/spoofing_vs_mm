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

from spoofing_detection.lob.annotations import validate_annotations
from spoofing_detection.lob.calibration import build_threshold_table


def _markdown_table(df: pl.DataFrame) -> str:
    if df.is_empty():
        return ""
    columns = df.columns
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in df.iter_rows(named=True):
        rows.append("| " + " | ".join("" if row[col] is None else str(row[col]) for col in columns) + " |")
    return "\n".join(rows)


def build_report(*, scores_path: Path, annotations_path: Path, output_dir: Path) -> dict[str, Path]:
    scores = pl.read_parquet(scores_path)
    annotations = validate_annotations(pl.read_csv(annotations_path))
    table = build_threshold_table(scores, annotations, score_column="MSCI", thresholds=[0.0, 0.1, 0.25, 0.5, 0.75, 1.0])
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "threshold_calibration.csv"
    markdown_path = output_dir / "threshold_calibration.md"
    table.write_csv(csv_path)
    markdown_path.write_text("\n".join(["# Threshold Calibration", "", "This report summarizes alert workload and positive-label concentration by MSCI threshold.", "", _markdown_table(table), "", "Interpret precision_proxy cautiously when labels are sparse or exploratory.", ""]))
    return {"csv": csv_path, "markdown": markdown_path}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build spoofing score threshold calibration report.")
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    outputs = build_report(scores_path=args.scores, annotations_path=args.annotations, output_dir=args.output_dir)
    print(outputs["markdown"])


if __name__ == "__main__":
    main()
