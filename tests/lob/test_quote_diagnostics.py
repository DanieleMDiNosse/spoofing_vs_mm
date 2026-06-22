from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from spoofing_detection.lob.quote_diagnostics import (
    compute_quote_diagnostics,
    collapse_session_reload_rows,
    validate_quote_panel_columns,
    write_interactive_quote_html,
    write_metadata,
)


def test_compute_quote_diagnostics_adds_mid_spread_relative_spread_and_imbalance():
    panel = pl.DataFrame(
        {
            "sort_index": [1, 2, 3, 4],
            "post_best_bid": [100.0, 100.5, None, 101.0],
            "post_best_ask": [101.0, 101.5, 102.0, None],
            "post_bid_level_1_visible_qty": [10.0, 20.0, 5.0, 0.0],
            "post_ask_level_1_visible_qty": [30.0, 20.0, 5.0, 10.0],
        }
    )

    out = compute_quote_diagnostics(panel)

    assert out.select("sort_index").to_series().to_list() == [1, 2, 3, 4]
    assert out.select("mid_price").to_series().to_list()[:2] == [100.5, 101.0]
    assert out.select("spread").to_series().to_list()[:2] == [1.0, 1.0]
    assert out.select("relative_spread_bps").to_series().to_list()[:2] == pytest.approx(
        [1.0 / 100.5 * 10_000, 1.0 / 101.0 * 10_000]
    )
    assert out.select("top_of_book_imbalance").to_series().to_list()[:2] == pytest.approx(
        [10.0 / 40.0, 20.0 / 40.0]
    )
    assert out.select("valid_touch").to_series().to_list() == [True, True, False, False]


def test_compute_quote_diagnostics_preserves_all_event_rows_and_order():
    panel = pl.DataFrame(
        {
            "sort_index": [3, 1, 2],
            "post_best_bid": [99.0, 98.0, 98.5],
            "post_best_ask": [100.0, 99.0, 99.5],
        }
    )

    out = compute_quote_diagnostics(panel)

    assert out.height == panel.height
    # Function should not silently reorder rows; event order should be supplied by reconstruction.
    assert out.select("sort_index").to_series().to_list() == [3, 1, 2]


def test_validate_quote_panel_columns_rejects_missing_required_columns():
    panel = pl.DataFrame({"sort_index": [1], "post_best_bid": [100.0]})

    with pytest.raises(ValueError, match="post_best_ask"):
        validate_quote_panel_columns(panel)


def test_collapse_session_reload_rows_keeps_only_completed_daily_reload_snapshot():
    panel = pl.DataFrame(
        {
            "sort_index": [1, 2, 3, 4, 5, 6, 7],
            "TRADEDATE": [
                "2024-06-03",
                "2024-06-03",
                "2024-06-03",
                "2024-06-03",
                "2024-06-04",
                "2024-06-04",
                "2024-06-04",
            ],
            "event_class": [
                "session_reload",
                "session_reload",
                "session_reload",
                "new_order",
                "session_reload",
                "session_reload",
                "cancel",
            ],
            "post_best_bid": [None, 99.0, 99.5, 99.5, None, 100.0, 100.0],
            "post_best_ask": [101.0, 101.0, 100.5, 100.5, 102.0, 101.0, 101.5],
        }
    )

    out = collapse_session_reload_rows(panel)

    assert out.select("sort_index").to_series().to_list() == [3, 4, 6, 7]
    assert out.height == 4


def test_write_interactive_quote_html_creates_expected_file(tmp_path: Path):
    panel = pl.DataFrame(
        {
            "sort_index": [1, 2, 3],
            "TRADEDATE": ["2024-06-03", "2024-06-03", "2024-06-03"],
            "event_class": ["session_reload", "new_order", "cancel"],
            "event_order_type_label": ["limit", "limit", "limit"],
            "event_side": ["bid", "ask", "ask"],
            "post_best_bid": [0.0286, 0.0287, 0.0287],
            "post_best_ask": [0.0292, 0.0292, 0.0293],
            "post_bid_level_1_visible_qty": [100.0, 200.0, 200.0],
            "post_ask_level_1_visible_qty": [300.0, 300.0, 150.0],
            "lob_issue_flags": [None, "marketable_order_not_resting", None],
            "normalization_issue_flags": [None, None, None],
        }
    )
    diagnostics = compute_quote_diagnostics(panel)
    html_path = tmp_path / "quotes.html"

    write_interactive_quote_html(
        diagnostics,
        html_path,
        title="Risanamento best bid/ask event-time diagnostics",
        source_label="test-panel.parquet",
    )

    html = html_path.read_text()
    assert "Risanamento best bid/ask event-time diagnostics" in html
    assert "Best bid" in html
    assert "Best ask" in html
    assert "Spread" in html
    assert "Relative spread" in html
    assert "Top-of-book imbalance" in html


def test_write_interactive_quote_html_can_plot_two_best_price_levels(tmp_path: Path):
    panel = pl.DataFrame(
        {
            "sort_index": [1, 2, 3],
            "post_best_bid": [100.0, 100.5, 100.5],
            "post_best_ask": [101.0, 101.5, 101.5],
            "post_bid_level_2_price": [99.5, 100.0, 100.0],
            "post_ask_level_2_price": [101.5, 102.0, 102.0],
            "post_bid_level_1_visible_qty": [10.0, 20.0, 20.0],
            "post_ask_level_1_visible_qty": [30.0, 40.0, 40.0],
            "post_bid_level_2_visible_qty": [50.0, 60.0, 60.0],
            "post_ask_level_2_visible_qty": [70.0, 80.0, 80.0],
        }
    )
    diagnostics = compute_quote_diagnostics(panel)
    html_path = tmp_path / "quotes_two_levels.html"

    write_interactive_quote_html(
        diagnostics,
        html_path,
        title="Two-level quote diagnostics",
        source_label="test-panel.parquet",
        best_levels=2,
    )

    html = html_path.read_text()
    assert "Best levels plotted: 2" in html
    assert "Bid L2" in html
    assert "Ask L2" in html
    assert "Bid L2 visible qty" in html
    assert "Ask L2 visible qty" in html


def test_write_metadata_records_raw_panel_rows_and_reload_mode(tmp_path: Path):
    panel = pl.DataFrame(
        {
            "sort_index": [1, 2],
            "post_best_bid": [100.0, 100.5],
            "post_best_ask": [101.0, 101.5],
        }
    )
    diagnostics = compute_quote_diagnostics(panel)
    metadata_path = tmp_path / "metadata.json"

    write_metadata(
        output_path=metadata_path,
        panel_path="panel.parquet",
        diagnostics=diagnostics,
        html_path="plot.html",
        command=["plot_best_quotes.py"],
        session_reload_mode="collapse",
        input_row_count=7,
        best_levels=2,
    )

    payload = json.loads(metadata_path.read_text())
    assert payload["row_count"] == 2
    assert payload["input_row_count"] == 7
    assert payload["all_raw_panel_rows"] is False
    assert payload["all_events"] is False
    assert payload["all_live_events"] is True
    assert payload["session_reload_mode"] == "collapse"
    assert payload["best_levels"] == 2
