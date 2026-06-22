from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.verify_lob_reconstruction import (  # noqa: E402
    Problem,
    WarningDetail,
    build_warning_details,
    check_marketable_lifecycle_flags,
    check_row_counts,
    check_spread_flags,
    check_stop_limit_visibility,
    check_top_n_depth,
    render_markdown_report,
)


def test_check_row_counts_records_mismatch_problem():
    problems = check_row_counts(
        input_rows=3,
        panel_rows=2,
        normalized_rows=3,
        agent_rows=6,
    )

    assert len(problems) == 1
    assert problems[0].severity == "error"
    assert "panel_rows" in problems[0].problem
    assert problems[0].count == 1
    assert "rerun reconstruction" in problems[0].possible_fix.lower()


def test_check_spread_flags_records_crossed_book_problem():
    panel = pl.DataFrame(
        {
            "sort_index": [1, 2],
            "book_crossed_post_flag": [False, True],
            "book_locked_post_flag": [False, False],
            "book_crossed_pre_flag": [False, False],
            "book_locked_pre_flag": [False, False],
            "post_best_bid": [100.0, 101.0],
            "post_best_ask": [101.0, 100.5],
            "pre_best_bid": [None, 100.0],
            "pre_best_ask": [None, 101.0],
        }
    )

    problems = check_spread_flags(panel)

    assert any(p.problem == "Crossed post-book state" for p in problems)
    crossed = next(p for p in problems if p.problem == "Crossed post-book state")
    assert crossed.count == 1
    assert crossed.example_sort_index == 2
    assert "marketable" in crossed.possible_fix.lower()


def test_top_n_depth_records_bad_bid_sort_order():
    panel = pl.DataFrame(
        {
            "sort_index": [10],
            "post_bid_level_1_price": [99.0],
            "post_bid_level_2_price": [100.0],
            "post_bid_level_1_visible_qty": [5.0],
            "post_bid_level_2_visible_qty": [3.0],
        }
    )

    problems = check_top_n_depth(panel, top_n=2)

    assert len(problems) == 1
    assert problems[0].severity == "error"
    assert "bid price levels" in problems[0].problem
    assert problems[0].example_sort_index == 10


def test_marketable_order_not_resting_rows_must_not_lock_or_cross_post_book():
    panel = pl.DataFrame(
        {
            "sort_index": [1, 2],
            "lob_issue_flags": ["marketable_order_not_resting", "marketable_order_not_resting"],
            "book_crossed_post_flag": [False, True],
            "book_locked_post_flag": [False, False],
        }
    )

    problems = check_marketable_lifecycle_flags(panel)

    assert len(problems) == 1
    assert problems[0].severity == "error"
    assert problems[0].example_sort_index == 2
    assert "deferred residual" in problems[0].possible_fix.lower()


def test_stop_limit_rows_should_not_have_best_price_equal_to_own_price_when_clean_book_would_not():
    panel = pl.DataFrame(
        {
            "sort_index": [1],
            "event_order_type_label": ["stop_limit_or_stop_limit_on_quote"],
            "event_side": ["ask"],
            "event_price": [99.0],
            "pre_best_bid": [100.0],
            "pre_best_ask": [101.0],
            "post_best_bid": [100.0],
            "post_best_ask": [99.0],
        }
    )

    problems = check_stop_limit_visibility(panel)

    assert len(problems) == 1
    assert problems[0].severity == "error"
    assert problems[0].example_sort_index == 1
    assert "stop-limit" in problems[0].possible_fix.lower()


def test_build_warning_details_explains_warnings_with_examples():
    panel = pl.DataFrame(
        {
            "sort_index": [1, 2],
            "TRADEDATE": ["2024-01-02", "2024-01-02"],
            "event_class": ["new_order", "modify_order"],
            "event_order_type_label": ["limit", "limit"],
            "ORDERID": ["B1", "A1"],
            "event_side": ["bid", "ask"],
            "event_price": [100.0, 101.0],
            "event_leaves_qty": [10.0, 5.0],
            "event_displayed_qty": [10.0, 5.0],
            "pre_best_bid": [99.0, 99.0],
            "pre_best_ask": [100.0, 101.0],
            "post_best_bid": [99.0, 99.0],
            "post_best_ask": [100.0, 101.0],
            "book_crossed_post_flag": [False, False],
            "book_locked_post_flag": [False, False],
            "lob_issue_flags": ["marketable_order_not_resting", "modify_for_unseen_order"],
        }
    )
    normalized = pl.DataFrame(
        {
            "sort_index": [3],
            "TRADEDATE": ["2024-01-02"],
            "event_class": ["new_order"],
            "event_order_type_label": ["limit"],
            "ORDERID": ["C1"],
            "side_label": ["bid"],
            "ORDERPX": [99.0],
            "LEAVESQTY": [1.0],
            "DISPLAYEDQTY": [1.0],
            "normalization_issue_flags": ["missing_client_original_id"],
        }
    )

    details = build_warning_details(panel, normalized, examples_per_warning=1)

    marketable = next(detail for detail in details if detail.flag == "marketable_order_not_resting")
    assert marketable.status == "handled_visible_book_caveat"
    assert "not inserted" in marketable.explanation
    assert marketable.examples[0]["sort_index"] == 1
    assert marketable.observed_evidence["post_locked_or_crossed_rows"] == 0

    missing_client = next(detail for detail in details if detail.flag == "missing_client_original_id")
    assert missing_client.status == "data_limitation_client_level"
    assert missing_client.examples[0]["sort_index"] == 3


def test_render_markdown_report_contains_problem_inventory():
    markdown = render_markdown_report(
        source_file="data/example.parquet",
        output_dir="outputs/example",
        summary={"input_rows": 1, "panel_rows": 1},
        problems=[
            Problem(
                severity="warning",
                problem="Example limitation",
                evidence="Synthetic evidence",
                count=1,
                example_sort_index=10,
                possible_fix="Document or fix it.",
                status="open",
            )
        ],
        warning_details=[
            WarningDetail(
                flag="marketable_order_not_resting",
                source="lob_issue_flags",
                status="handled_visible_book_caveat",
                explanation="Marketable order rows are not inserted into visible resting depth.",
                recommended_action="Sample-check examples before final claims.",
                observed_evidence={"post_locked_or_crossed_rows": 0},
                examples=[{"sort_index": 1, "event_class": "new_order", "event_price": 100.0}],
            )
        ],
    )

    assert "# LOB Reconstruction Verification Report" in markdown
    assert "hard_error_categories" in markdown
    assert "hard_error_rows" not in markdown
    assert "## Problem inventory" in markdown
    assert "## Warning details and examples" in markdown
    assert "### `marketable_order_not_resting`" in markdown
    assert "post_locked_or_crossed_rows" in markdown
    assert "Example limitation" in markdown
    assert "Document or fix it." in markdown
