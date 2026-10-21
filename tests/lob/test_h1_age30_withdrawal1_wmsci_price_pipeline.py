from datetime import datetime

import polars as pl
import pytest

from scripts import derive_h1_age30_withdrawal1_wmsci_price_pipeline as pipeline


def _executions() -> pl.DataFrame:
    rows = []
    for cluster_id, first, last, p, q, fpm, rev in (
        ("older", 10, 11, 10.0, 5.0, 1.0, 2.0),
        ("later", 20, 21, 10.0, 5.0, 1.0, 2.0),
        ("missing-price", 30, 31, 10.0, 5.0, None, None),
    ):
        rows.append(
            {
                "partition_id": "P",
                "execution_cluster_id": cluster_id,
                "cluster_first_sort_index": first,
                "cluster_last_sort_index": last,
                "cluster_end_fill_anchor_ns": 1_000_000_000,
                "candidate_deceptive_visible_qty_pre": p,
                "execution_quantity": q,
                "execution_anchor_mode": "passive",
                "execution_side": "ask",
                "deceptive_side": "bid",
                "actor_key": "client_original:C1",
                "actor_id": "C1",
                "identity_level": "client_original",
                "identity_source": "NMSC_ORIGINALCLIENTIDSHORTCODE",
                "identity_fallback_flag": False,
                "cluster_start_ts": datetime(2024, 1, 2, 9, 30),
                "cluster_end_ts": datetime(2024, 1, 2, 9, 30),
                "cluster_start_replay_time_ns": 1_000_000_000,
                "cluster_end_replay_time_ns": 1_000_000_000,
                "favorable_mid_move_pre_fill": fpm,
                "post_cancel_mid_reversion": rev,
            }
        )
    return pl.DataFrame(rows)


def _links() -> pl.DataFrame:
    """Two competing candidates share a physical cancellation; only later wins."""
    timestamp = datetime(2024, 1, 2, 9, 30)
    common = {
        "partition_id": "P",
        "candidate_order_id": "D",
        "actor_key": "client_original:C1",
        "actor_id": "C1",
        "identity_level": "client_original",
        "identity_source": "NMSC_ORIGINALCLIENTIDSHORTCODE",
        "identity_fallback_flag": False,
        "execution_anchor_mode": "passive",
        "execution_side": "ask",
        "deceptive_side": "bid",
        "cancel_event_ts": timestamp,
        "cancellation_removal_ns": 1_999_999_999,
        "cancel_visible_qty": 10.0,
        "candidate_visible_qty_pre": 10.0,
        "attributed_cancel_visible_qty": 10.0,
        "ORDERID": "D",
        "event_ts": timestamp,
        "visible_qty_pre_cancel": 10.0,
    }
    rows = []
    for cluster_id, first, last in (("older", 10, 11), ("later", 20, 21)):
        rows.append({**common, "execution_cluster_id": cluster_id, "cluster_first_sort_index": first, "cluster_last_sort_index": last, "cluster_end_fill_anchor_ns": 1_000_000_000, "cancel_sort_index": 40})
    # Exactly 1 second must be excluded; stale/missing cached delay must not matter.
    rows.append({**common, "execution_cluster_id": "missing-price", "cluster_first_sort_index": 30, "cluster_last_sort_index": 31, "cluster_end_fill_anchor_ns": 1_000_000_000, "cancel_sort_index": 50, "cancellation_removal_ns": 2_000_000_000, "cancel_economic_delay_ns": None})
    return pl.DataFrame(rows)


def test_one_second_filter_is_half_open_and_reassigns_each_physical_cancel_once():
    assigned, audit = pipeline.filter_and_reassign_one_second_links(_links())

    assert audit == {"candidate_link_rows": 3, "eligible_link_rows": 2, "eligible_physical_cancellations": 1, "assigned_link_rows": 1}
    assert assigned.filter(pl.col("assigned_flag")).get_column("execution_cluster_id").to_list() == ["later"]
    assert assigned.get_column("assignment_rule").unique().to_list() == ["latest_prior_cluster_sort_index"]


def test_recomputed_quantity_score_uses_execution_denominator_and_zero_without_assignment():
    events, _ = pipeline.recompute_event_quantities_and_score(_executions(), _links().head(2))
    scored = events.sort("execution_cluster_id")

    assert scored.filter(pl.col("execution_cluster_id") == "later").get_column("WMSCI_event").item() == pytest.approx(
        pipeline.quantity_only_wmsci(10.0, 5.0, 10.0)
    )
    assert scored.filter(pl.col("execution_cluster_id") == "older").get_column("WMSCI_event").item() == 0.0
    assert scored.filter(pl.col("execution_cluster_id") == "later").get_column("withdrawal_to_execution_ratio").item() == 2.0


def test_sequential_population_uses_deterministic_ties_and_excludes_missing_prices():
    frame = pl.DataFrame(
        {
            "execution_cluster_id": ["z", "a", "b", "c", "missing"],
            "WMSCI_event": [9.0, 8.0, 8.0, 1.0, 8.0],
            "withdrawal_to_execution_ratio": [2.0, 2.0, 2.0, 2.0, 2.0],
            "favorable_mid_move_pre_fill": [1.0, 1.0, -1.0, 1.0, None],
            "post_cancel_mid_reversion": [1.0, 1.0, 1.0, 1.0, None],
            "post_cancel_mid_reversion_complete": [True, True, True, True, False],
        }
    )

    selected, counts = pipeline.sequential_population(frame, fraction=0.80)

    assert selected.get_column("execution_cluster_id").to_list() == ["z", "a"]
    assert counts == {
        "all_executions": 5,
        "finite_W_executions": 5,
        "positive_W_executions": 5,
        "top_W_exact_k": 4,
        "top_W_finite_U_over_Q": 4,
        "top_W_after_U_over_Q_gt_1": 4,
        "top_W_after_U_over_Q_gt_1_FPM_observed": 3,
        "top_W_after_U_over_Q_gt_1_FPM_gt_0": 2,
        "top_W_after_FPM_gt_0_REV_observed": 2,
        "final_WMSCI_to_FPM_to_REV_executions": 2,
    }


def test_sequential_population_is_reproducible_with_ties():
    frame = pl.DataFrame(
        {
            "execution_cluster_id": ["b", "a", "c"],
            "WMSCI_event": [1.0, 1.0, 1.0],
            "withdrawal_to_execution_ratio": [2.0, 2.0, 2.0],
            "favorable_mid_move_pre_fill": [1.0, 1.0, 1.0],
            "post_cancel_mid_reversion": [1.0, 1.0, 1.0],
            "post_cancel_mid_reversion_complete": [True, True, True],
        }
    )

    first = pipeline.sequential_population(frame, fraction=2 / 3)
    second = pipeline.sequential_population(frame, fraction=2 / 3)

    assert first[0].to_dicts() == second[0].to_dicts()
    assert first[0].get_column("execution_cluster_id").to_list() == ["a", "b"]
    assert first[1] == second[1]


def test_partial_missing_assigned_positive_quantity_cancel_blocks_reversion_gate():
    """An available-only mean is descriptive, never a strict REV observation."""
    events = pl.DataFrame(
        {
            "execution_cluster_id": ["complete", "partial"],
            "post_cancel_mid_reversion": [2.0, 2.0],
        }
    )
    candidates = pl.DataFrame(
        {
            "execution_cluster_id": ["complete", "partial", "partial"],
            "assigned_flag": [True, True, True],
            "attributed_cancel_visible_qty": [3.0, 2.0, 2.0],
            "post_cancel_mid_reversion": [2.0, 2.0, None],
            "has_cancel_reversion_state": [True, True, False],
        }
    )

    screened = pipeline.enforce_complete_finite_reversion(events, candidates).sort("execution_cluster_id")

    assert screened.get_column("post_cancel_mid_reversion_available_only").to_list() == [2.0, 2.0]
    assert screened.get_column("post_cancel_mid_reversion_complete").to_list() == [True, False]
    assert screened.get_column("post_cancel_mid_reversion").to_list() == [2.0, None]
    assert screened.get_column("assigned_positive_qty_cancel_reversion_missing_count").to_list() == [0, 1]


def test_sequential_population_rejects_nan_at_every_numeric_gate():
    frame = pl.DataFrame(
        {
            "execution_cluster_id": ["nan-w", "nan-u", "nan-fpm", "nan-rev", "good"],
            "WMSCI_event": [float("nan"), 5.0, 4.0, 3.0, 2.0],
            "withdrawal_to_execution_ratio": [2.0, float("nan"), 2.0, 2.0, 2.0],
            "favorable_mid_move_pre_fill": [1.0, 1.0, float("nan"), 1.0, 1.0],
            "post_cancel_mid_reversion": [1.0, 1.0, 1.0, float("nan"), 1.0],
            "post_cancel_mid_reversion_complete": [True, True, True, False, True],
        }
    )

    selected, counts = pipeline.sequential_population(frame, fraction=1.0)

    assert selected.get_column("execution_cluster_id").to_list() == ["good"]
    assert counts["positive_W_executions"] == 4
    assert counts["top_W_after_U_over_Q_gt_1"] == 3
    assert counts["top_W_after_U_over_Q_gt_1_FPM_gt_0"] == 2
    assert counts["final_WMSCI_to_FPM_to_REV_executions"] == 1


def test_step_membership_uses_full_frame_for_gate_columns():
    """Membership flags must use retained score/price columns after ID projection."""
    frame = pl.DataFrame(
        {
            "execution_cluster_id": ["z", "a", "b"],
            "WMSCI_event": [4.0, 3.0, 2.0],
            "withdrawal_to_execution_ratio": [2.0, 2.0, 1.0],
            "favorable_mid_move_pre_fill": [1.0, 1.0, 1.0],
            "post_cancel_mid_reversion_complete": [True, True, True],
            "post_cancel_mid_reversion": [1.0, -1.0, 1.0],
        }
    )

    membership = pipeline._step_membership(frame, fraction=2 / 3, instrument="X")

    assert membership.get_column("execution_cluster_id").to_list() == ["a", "b", "z"]
    assert membership.filter(pl.col("execution_cluster_id") == "z").select("selected_final").item() is True
    assert membership.filter(pl.col("execution_cluster_id") == "a").select("selected_final").item() is False


def test_retained_reversion_sufficient_statistics_match_core_reference():
    """A validated 2s retained cancellation observation equals a fresh core lookup."""
    from spoofing_detection.lob.spoofing_metrics import attach_cancel_anchored_reversion

    executions = _executions().head(1)
    candidates = _links().head(1).with_columns(pl.lit(True).alias("assigned_flag"))
    states = pl.DataFrame(
        {
            "partition_id": ["P", "P", "P"],
            "actor_key": ["client_original:C1"] * 3,
            "sort_index": [39, 40, 41],
            "replay_time_ns": [1_900_000_000, 1_999_999_999, 3_999_999_999],
            "market_mid": [100.0, 101.0, 102.0],
            "market_microprice": [100.0, 101.0, 102.0],
        }
    )
    expected_events, retained = attach_cancel_anchored_reversion(
        executions, states, candidates, reversion_horizon_seconds=2.0, use_v3_clocks=True
    )

    actual_events, actual_candidates = pipeline.reuse_retained_cancel_anchored_reversion(
        executions,
        retained.with_columns(pl.lit(True).alias("_source_assigned_flag")),
    )

    assert actual_events.to_dicts() == expected_events.to_dicts()
    assert actual_candidates.to_dicts() == retained.to_dicts()
