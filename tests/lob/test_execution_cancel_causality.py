from datetime import datetime

import polars as pl

from spoofing_detection.lob.spoofing_metrics import _build_execution_cancel_candidates


def test_cancel_candidate_must_follow_cluster_in_canonical_sort_order():
    executions = pl.DataFrame(
        [
            {
                "partition_id": "P",
                "actor_key": "client_original:C1",
                "deceptive_side": "bid",
                "candidate_deceptive_order_ids_pre": "BD",
                "execution_cluster_id": "EC000000020-000000022",
                "cluster_first_sort_index": 20,
                "cluster_last_sort_index": 22,
                "cluster_end_ts": datetime(2024, 1, 2, 9, 30),
            }
        ]
    )
    cancellations = pl.DataFrame(
        [
            {
                "partition_id": "P",
                "actor_key": "client_original:C1",
                "side": "bid",
                "ORDERID": "BD",
                "sort_index": 10,
                "event_ts": datetime(2024, 1, 2, 9, 30, 0, 500_000),
                "visible_qty_pre_cancel": 50.0,
            }
        ]
    )

    candidates = _build_execution_cancel_candidates(
        executions,
        cancellations,
        window_seconds=1.0,
    )

    assert candidates.is_empty()
    assert candidates.columns == [
        "partition_id",
        "cancel_sort_index",
        "candidate_order_id",
        "execution_cluster_id",
        "actor_key",
        "actor_id",
        "identity_level",
        "identity_source",
        "identity_fallback_flag",
        "execution_anchor_mode",
        "execution_side",
        "deceptive_side",
        "cluster_end_ts",
        "cluster_first_sort_index",
        "cluster_last_sort_index",
        "cancel_event_ts",
        "cancel_visible_qty",
        "ORDERID",
        "event_ts",
        "visible_qty_pre_cancel",
        "assigned_flag",
        "assignment_rule",
        "competing_cluster_count",
        "cancel_reversion_target_ts",
        "cancel_pre_state_sort_index",
        "cancel_post_state_sort_index",
        "cancel_mid_pre",
        "cancel_mid_post_horizon",
        "cancel_microprice_pre",
        "cancel_microprice_post_horizon",
        "post_cancel_mid_reversion",
        "post_cancel_microprice_reversion",
        "cancel_reversion_weight",
        "has_cancel_reversion_state",
    ]
    assert candidates.schema["cancel_sort_index"] == pl.Int64
    assert candidates.schema["cancel_event_ts"] == pl.Datetime("us")
    assert candidates.schema["assigned_flag"] == pl.Boolean


def test_cancel_candidate_allows_equal_timestamp_with_later_sort_index():
    timestamp = datetime(2024, 1, 2, 9, 30)
    executions = pl.DataFrame(
        [
            {
                "partition_id": "P",
                "actor_key": "client_original:C1",
                "deceptive_side": "bid",
                "candidate_deceptive_order_ids_pre": "BD",
                "execution_cluster_id": "EC000000020-000000022",
                "cluster_first_sort_index": 20,
                "cluster_last_sort_index": 22,
                "cluster_end_ts": timestamp,
            }
        ]
    )
    cancellations = pl.DataFrame(
        [
            {
                "partition_id": "P",
                "actor_key": "client_original:C1",
                "side": "bid",
                "ORDERID": "BD",
                "sort_index": 23,
                "event_ts": timestamp,
                "visible_qty_pre_cancel": 50.0,
            }
        ]
    )

    candidates = _build_execution_cancel_candidates(
        executions,
        cancellations,
        window_seconds=1.0,
    )

    assert candidates.height == 1
    assert candidates.row(0, named=True)["assigned_flag"] is True
