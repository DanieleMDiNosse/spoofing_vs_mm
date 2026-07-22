from __future__ import annotations

import math
from datetime import datetime

import polars as pl
import pytest

from spoofing_detection.lob.spoofing_metrics import (
    _build_execution_cancel_candidates,
    attach_cancel_anchored_reversion,
)


def _execution() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "partition_id": "P",
                "client_id": "C1",
                "execution_cluster_id": "EC1",
                "execution_side": "ask",
                "deceptive_side": "bid",
                "cluster_first_sort_index": 10,
                "cluster_last_sort_index": 10,
                "cluster_end_ts": datetime(2024, 1, 2, 9, 30, 5),
                "candidate_deceptive_order_ids_pre": "B1;B2",
            }
        ]
    )


def test_withdrawal_candidate_window_includes_two_seconds_and_excludes_later_cancels():
    cancellations = pl.DataFrame(
        [
            {
                "partition_id": "P",
                "client_id": "C1",
                "side": "bid",
                "ORDERID": "B1",
                "sort_index": 20,
                "event_ts": datetime(2024, 1, 2, 9, 30, 7),
                "visible_qty_pre_cancel": 10.0,
            },
            {
                "partition_id": "P",
                "client_id": "C1",
                "side": "bid",
                "ORDERID": "B2",
                "sort_index": 21,
                "event_ts": datetime(2024, 1, 2, 9, 30, 7, 1_000),
                "visible_qty_pre_cancel": 30.0,
            },
        ]
    )

    candidates = _build_execution_cancel_candidates(
        _execution(),
        cancellations,
        window_seconds=2.0,
    )

    assert candidates["candidate_order_id"].to_list() == ["B1"]
    assert candidates.item(0, "cancel_event_ts") == datetime(2024, 1, 2, 9, 30, 7)
    assert candidates.item(0, "assigned_flag") is True


def test_cancel_anchored_reversion_uses_each_actual_cancel_and_quantity_delay_weights():
    states = pl.DataFrame(
        {
            "partition_id": ["P"] * 7,
            "client_id": ["C1"] * 7,
            "sort_index": [10, 19, 20, 21, 22, 23, 24],
            "event_ts": [
                datetime(2024, 1, 2, 9, 30, 5),
                datetime(2024, 1, 2, 9, 30, 9, 900_000),
                datetime(2024, 1, 2, 9, 30, 10),
                datetime(2024, 1, 2, 9, 30, 10, 400_000),
                datetime(2024, 1, 2, 9, 30, 10, 500_000),
                datetime(2024, 1, 2, 9, 30, 11, 900_000),
                datetime(2024, 1, 2, 9, 30, 12, 600_000),
            ],
            "DWI": [0.0] * 7,
            "L_bid_topN": [1.0] * 7,
            "L_ask_topN": [1.0] * 7,
            "market_mid": [101.0, 100.30, 100.20, 100.20, 100.10, 99.90, 99.90],
            "market_microprice": [101.1, 100.35, 100.25, 100.25, 100.15, 99.95, 99.95],
        }
    )
    assigned = pl.DataFrame(
        [
            {
                "partition_id": "P",
                "client_id": "C1",
                "candidate_order_id": "B1",
                "ORDERID": "B1",
                "execution_cluster_id": "EC1",
                "execution_side": "ask",
                "cluster_end_ts": datetime(2024, 1, 2, 9, 30, 5),
                "cancel_sort_index": 20,
                "cancel_event_ts": datetime(2024, 1, 2, 9, 30, 10),
                "cancel_visible_qty": 10.0,
                "assigned_flag": True,
            },
            {
                "partition_id": "P",
                "client_id": "C1",
                "candidate_order_id": "B2",
                "ORDERID": "B2",
                "execution_cluster_id": "EC1",
                "execution_side": "ask",
                "cluster_end_ts": datetime(2024, 1, 2, 9, 30, 5),
                "cancel_sort_index": 22,
                "cancel_event_ts": datetime(2024, 1, 2, 9, 30, 10, 500_000),
                "cancel_visible_qty": 30.0,
                "assigned_flag": True,
            },
        ]
    )

    execution_out, candidate_out = attach_cancel_anchored_reversion(
        _execution(),
        states,
        assigned,
        reversion_horizon_seconds=2.0,
        withdrawal_decay_seconds=10.0,
    )

    by_order = {row["candidate_order_id"]: row for row in candidate_out.to_dicts()}
    assert by_order["B1"]["cancel_mid_pre"] == pytest.approx(100.30)
    assert by_order["B1"]["cancel_mid_post_horizon"] == pytest.approx(99.90)
    assert by_order["B1"]["post_cancel_mid_reversion"] == pytest.approx(0.40)
    assert by_order["B1"]["cancel_reversion_target_ts"] == datetime(2024, 1, 2, 9, 30, 12)
    assert by_order["B1"]["cancel_pre_state_sort_index"] == 19
    assert by_order["B1"]["cancel_post_state_sort_index"] == 23

    assert by_order["B2"]["cancel_mid_pre"] == pytest.approx(100.20)
    assert by_order["B2"]["cancel_mid_post_horizon"] == pytest.approx(99.90)
    assert by_order["B2"]["post_cancel_mid_reversion"] == pytest.approx(0.30)

    weight_b1 = 10.0 * math.exp(-5.0 / 10.0)
    weight_b2 = 30.0 * math.exp(-5.5 / 10.0)
    expected = (weight_b1 * 0.40 + weight_b2 * 0.30) / (weight_b1 + weight_b2)
    assert execution_out.item(0, "post_cancel_mid_reversion") == pytest.approx(expected)
    assert execution_out.item(0, "cancel_reversion_observation_count") == 2
    assert execution_out.item(0, "first_matched_cancel_ts") == datetime(2024, 1, 2, 9, 30, 10)
    assert execution_out.item(0, "last_matched_cancel_ts") == datetime(2024, 1, 2, 9, 30, 10, 500_000)


def test_cancel_anchored_reversion_keeps_unobserved_horizon_null():
    states = pl.DataFrame(
        {
            "partition_id": ["P", "P", "P"],
            "client_id": ["C1", "C1", "C1"],
            "sort_index": [10, 19, 20],
            "event_ts": [
                datetime(2024, 1, 2, 9, 30, 5),
                datetime(2024, 1, 2, 9, 30, 9, 900_000),
                datetime(2024, 1, 2, 9, 30, 10),
            ],
            "DWI": [0.0, 0.0, 0.0],
            "L_bid_topN": [1.0, 1.0, 1.0],
            "L_ask_topN": [1.0, 1.0, 1.0],
            "market_mid": [101.0, 100.30, 100.20],
            "market_microprice": [101.1, 100.35, 100.25],
        }
    )
    assigned = pl.DataFrame(
        [
            {
                "partition_id": "P",
                "client_id": "C1",
                "candidate_order_id": "B1",
                "ORDERID": "B1",
                "execution_cluster_id": "EC1",
                "execution_side": "ask",
                "cluster_end_ts": datetime(2024, 1, 2, 9, 30, 5),
                "cancel_sort_index": 20,
                "cancel_event_ts": datetime(2024, 1, 2, 9, 30, 10),
                "cancel_visible_qty": 10.0,
                "assigned_flag": True,
            }
        ]
    )

    execution_out, candidate_out = attach_cancel_anchored_reversion(
        _execution(),
        states,
        assigned,
        reversion_horizon_seconds=2.0,
    )

    assert candidate_out.item(0, "has_cancel_reversion_state") is False
    assert candidate_out.item(0, "post_cancel_mid_reversion") is None
    assert execution_out.item(0, "post_cancel_mid_reversion") is None
    assert execution_out.item(0, "cancel_reversion_observation_count") == 0


def test_cancel_reversion_coverage_handles_timestamps_nonmonotonic_in_sort_order():
    base = datetime(2024, 1, 2, 9, 30)
    states = pl.DataFrame(
        {
            "partition_id": ["P"] * 4,
            "client_id": ["C1"] * 4,
            "sort_index": [19, 20, 21, 22],
            "event_ts": [
                base.replace(second=9, microsecond=900_000),
                base.replace(second=10),
                base.replace(second=13),
                base.replace(second=11, microsecond=500_000),
            ],
            "DWI": [0.0] * 4,
            "L_bid_topN": [1.0] * 4,
            "L_ask_topN": [1.0] * 4,
            "market_mid": [100.4, 100.3, 99.5, 100.0],
            "market_microprice": [100.45, 100.35, 99.55, 100.05],
        }
    )
    assigned = pl.DataFrame(
        [
            {
                "partition_id": "P",
                "client_id": "C1",
                "candidate_order_id": "B1",
                "ORDERID": "B1",
                "execution_cluster_id": "EC1",
                "execution_side": "ask",
                "cluster_end_ts": base.replace(second=5),
                "cancel_sort_index": 20,
                "cancel_event_ts": base.replace(second=10),
                "cancel_visible_qty": 10.0,
                "assigned_flag": True,
            }
        ]
    )

    execution_out, candidate_out = attach_cancel_anchored_reversion(
        _execution(),
        states,
        assigned,
        reversion_horizon_seconds=2.0,
    )

    assert candidate_out.item(0, "has_cancel_reversion_state") is True
    assert candidate_out.item(0, "cancel_post_state_sort_index") == 22
    assert candidate_out.item(0, "cancel_mid_post_horizon") == pytest.approx(100.0)
    assert candidate_out.item(0, "post_cancel_mid_reversion") == pytest.approx(0.4)
    assert execution_out.item(0, "cancel_reversion_observation_count") == 1
