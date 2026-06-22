#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from spoofing_detection.lob.quote_diagnostics import (
    compute_quote_diagnostics,
    collapse_session_reload_rows,
    read_panel,
    write_interactive_quote_html,
    write_metadata,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot all-event event-time best bid/ask diagnostics from a reconstructed LOB panel."
    )
    parser.add_argument("--panel", type=Path, required=True, help="Input lob_event_state_panel.parquet")
    parser.add_argument("--output-html", type=Path, required=True, help="Output interactive .html path")
    parser.add_argument("--metadata", type=Path, default=None, help="Optional metadata .json path")
    parser.add_argument(
        "--title",
        default="Best bid / best ask event-time diagnostics",
        help="Plot title",
    )
    parser.add_argument(
        "--session-reload-mode",
        choices=("all", "collapse"),
        default="all",
        help=(
            "How to handle session_reload snapshot replay rows. 'all' plots every raw panel row; "
            "'collapse' keeps only the completed final reload row per day plus all live events."
        ),
    )
    parser.add_argument(
        "--best-levels",
        type=int,
        default=1,
        help="Number of post-event bid/ask price levels to plot, e.g. 2 plots best and second-best levels.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    panel = read_panel(args.panel)
    input_row_count = panel.height
    if args.session_reload_mode == "collapse":
        panel = collapse_session_reload_rows(panel)
    diagnostics = compute_quote_diagnostics(panel)
    write_interactive_quote_html(
        diagnostics,
        args.output_html,
        title=args.title,
        source_label=str(args.panel),
        best_levels=args.best_levels,
    )
    if args.metadata is not None:
        write_metadata(
            output_path=args.metadata,
            panel_path=args.panel,
            diagnostics=diagnostics,
            html_path=args.output_html,
            command=sys.argv,
            session_reload_mode=args.session_reload_mode,
            input_row_count=input_row_count,
            best_levels=args.best_levels,
        )
    print(f"html: {args.output_html}")
    if args.metadata is not None:
        print(f"metadata: {args.metadata}")
    print(f"rows_plotted: {diagnostics.height}")
    print(f"session_reload_mode: {args.session_reload_mode}")
    print(f"best_levels: {args.best_levels}")


if __name__ == "__main__":
    main()
