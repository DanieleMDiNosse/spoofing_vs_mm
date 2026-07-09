from __future__ import annotations

import math
from datetime import datetime

import polars as pl
import pytest

from spoofing_detection.lob.models import ActiveOrder
from spoofing_detection.lob.spoofing_metrics import (
    attach_sci_window_metrics,
    choose_event_timestamp,
    compute_client_metric_time_series,
    compute_client_top_n_exposures,
    compute_exploratory_metrics,
    compute_mcps_scores,
    depth_kernel_weights,
    infer_tick_size_from_best_quotes,
    shifted_depth_distance_ticks,
)


def order(order_id, side, price, qty, client):
    return ActiveOrder(
        order_id=order_id,
        side=side,
        price=price,
        leaves_qty=qty,
        displayed_qty=qty,
        order_qty=qty,
        order_priority=order_id,
        order_type_code=2,
        order_type_label="limit",
        time_in_force_code=0,
        firm_id="F1",
        client_original_id=client,
        first_seen_sort_index=1,
        last_update_sort_index=1,
        last_event_class="new_order",
    )


def raw_event(
    seq,
    event_type,
    order_id,
    side,
    price,
    qty,
    displayed,
    client,
    *,
    trade_time=None,
    bookout=None,
    last_shares=None,
    aggressive="N",
):
    timestamp = bookout or f"2024-01-02 09:30:{seq:02d}"
    return {
        "TRADEDATE": "2024-01-02",
        "MIC": "XMIL",
        "MARKETCODE": "MTA",
        "SYMBOLINDEX": 123,
        "EMM (*)": 1,
        "SEQUENCETIME": timestamp,
        "BOOKIN": timestamp,
        "BOOKOUTTIME": timestamp,
        "TRADETIME": trade_time,
        "HDR_APPLKEYSEQUENCENUMBER": seq,
        "HDR_HWMSEQUENCENUMBER": seq,
        "HDR_OFFSETID": seq,
        "ROW_NUMBER": seq,
        "ORDEREVENTTYPE (*)": event_type,
        "ORDERID": order_id,
        "ORDERPRIORITY": str(seq),
        "ORDERSIDE (*)": side,
        "ORDERPX": price,
        "ORDERQTY": qty,
        "DISPLAYEDQTY": displayed,
        "LEAVESQTY": qty,
        "LASTSHARES": last_shares,
        "ORDERTYPE (*)": 2,
        "TIMEINFORCE (*)": 0,
        "FIRMID": "F1",
        "NMSC_ORIGINALCLIENTIDSHORTCODE": client,
        "ORDER_TRADINGCAPACITY (*)": 3 if client is not None else 1,
        "PASSIVEORDER": "Y" if event_type == 3 and aggressive != "Y" else None,
        "AGGRESSIVEORDER": aggressive if event_type == 3 else None,
    }


def test_infer_tick_size_from_best_bid_and_ask_changes():
    panel = pl.DataFrame(
        {
            "post_best_bid": [100.00, 100.01, 100.03, 100.03],
            "post_best_ask": [100.05, 100.06, 100.08, 100.09],
        }
    )

    assert infer_tick_size_from_best_quotes(panel) == pytest.approx(0.01)


def test_infer_tick_size_rejects_flat_quotes():
    panel = pl.DataFrame(
        {
            "post_best_bid": [100.00, 100.00],
            "post_best_ask": [100.05, 100.05],
        }
    )

    with pytest.raises(ValueError, match="tick size"):
        infer_tick_size_from_best_quotes(panel)


def test_shifted_depth_distance_gives_level_one_positive_distance():
    assert shifted_depth_distance_ticks("bid", price=100.0, best_price=100.0, tick_size=0.1) == pytest.approx(1.0)
    assert shifted_depth_distance_ticks("bid", price=99.9, best_price=100.0, tick_size=0.1) == pytest.approx(2.0)
    assert shifted_depth_distance_ticks("ask", price=100.2, best_price=100.2, tick_size=0.1) == pytest.approx(1.0)
    assert shifted_depth_distance_ticks("ask", price=100.3, best_price=100.2, tick_size=0.1) == pytest.approx(2.0)


def test_depth_kernel_weights_are_normalized_and_positive():
    weights = depth_kernel_weights([1.0, 2.0, 3.0], kappa=1.0, lambda_=0.5)

    assert sum(weights) == pytest.approx(1.0)
    assert all(weight > 0 for weight in weights)


def test_compute_client_top_n_exposures_uses_paper_aligned_dwi():
    active = {
        "B1": order("B1", "bid", 100.0, 10.0, "C1"),
        "B2": order("B2", "bid", 99.9, 20.0, "C1"),
        "B3": order("B3", "bid", 99.9, 30.0, "C2"),
        "A1": order("A1", "ask", 100.2, 5.0, "C1"),
        "A2": order("A2", "ask", 100.3, 15.0, "C2"),
        "P1": order("P1", "ask", 100.4, 100.0, None),
    }

    rows = compute_client_top_n_exposures(
        active,
        top_n=2,
        tick_size=0.1,
        kappa=1.0,
        lambda_=0.5,
        partition_id="P",
        sort_index=10,
        event_ts=None,
    )
    by_client = {row["client_id"]: row for row in rows}

    c1 = by_client["C1"]
    weights = depth_kernel_weights([1.0, 2.0], kappa=1.0, lambda_=0.5)
    expected_l_bid = weights[0] * 1.0 + weights[1] * 0.4
    expected_l_ask = weights[0] * 1.0 + weights[1] * 0.0
    expected_dwi = (expected_l_ask - expected_l_bid) / (expected_l_ask + expected_l_bid)

    assert c1["lambda_"] == pytest.approx(0.5)
    assert c1["bid_level_1_depth_distance_ticks"] == pytest.approx(1.0)
    assert c1["bid_level_2_depth_distance_ticks"] == pytest.approx(2.0)
    assert c1["bid_level_1_client_relative_depth"] == pytest.approx(1.0)
    assert c1["bid_level_2_client_relative_depth"] == pytest.approx(0.4)
    assert c1["L_bid_topN"] == pytest.approx(expected_l_bid)
    assert c1["L_ask_topN"] == pytest.approx(expected_l_ask)
    assert c1["DWI"] == pytest.approx(expected_dwi)
    assert "imbalance" not in c1
    assert "weighted_bid_fraction_topN" not in c1
    assert "P1" not in by_client


def test_compute_client_top_n_exposures_can_use_empirical_rank_weights():
    active = {
        "B1": order("B1", "bid", 100.0, 10.0, "C1"),
        "B2": order("B2", "bid", 99.9, 20.0, "C1"),
        "A1": order("A1", "ask", 100.2, 5.0, "C1"),
        "A2": order("A2", "ask", 100.3, 15.0, "C2"),
    }
    empirical_weights = {"bid": {1: 0.25, 2: 0.75}, "ask": {1: 0.80, 2: 0.20}}

    rows = compute_client_top_n_exposures(
        active,
        top_n=2,
        tick_size=0.1,
        kappa=1.0,
        lambda_=0.5,
        partition_id="P",
        sort_index=10,
        event_ts=None,
        empirical_kernel_weights=empirical_weights,
    )

    c1 = {row["client_id"]: row for row in rows}["C1"]
    assert c1["bid_level_1_kernel_weight"] == pytest.approx(0.25)
    assert c1["bid_level_2_kernel_weight"] == pytest.approx(0.75)
    assert c1["ask_level_1_kernel_weight"] == pytest.approx(0.80)
    assert c1["L_bid_topN"] == pytest.approx(0.25 + 0.75)
    assert c1["L_ask_topN"] == pytest.approx(0.80)


def test_compute_client_top_n_exposures_can_filter_to_clients_of_interest():
    active = {
        "B1": order("B1", "bid", 100.0, 10.0, "C1"),
        "B2": order("B2", "bid", 99.9, 20.0, "C2"),
        "A1": order("A1", "ask", 100.2, 5.0, "C1"),
        "A2": order("A2", "ask", 100.3, 15.0, "C3"),
    }

    rows = compute_client_top_n_exposures(
        active,
        top_n=2,
        tick_size=0.1,
        kappa=1.0,
        lambda_=0.5,
        partition_id="P",
        sort_index=10,
        event_ts=None,
        client_ids={"C1", "C3"},
    )

    assert [row["client_id"] for row in rows] == ["C1", "C3"]


def test_compute_client_top_n_exposures_filters_numeric_client_ids_as_strings():
    active = {
        "A1": order("A1", "ask", 100.2, 5.0, 17295),
        "A2": order("A2", "ask", 100.3, 15.0, 999),
    }

    rows = compute_client_top_n_exposures(
        active,
        top_n=2,
        tick_size=0.1,
        kappa=1.0,
        lambda_=0.5,
        partition_id="P",
        sort_index=10,
        event_ts=None,
        client_ids={"17295"},
    )

    assert [row["client_id"] for row in rows] == ["17295"]


def test_compute_exploratory_metrics_can_emit_compact_state_for_selected_clients_only():
    df = pl.DataFrame(
        [
            raw_event(1, 1, "B0", 1, 100.0, 100, 100, "C2"),
            raw_event(2, 1, "BD", 1, 99.9, 50, 50, "C1"),
            raw_event(3, 1, "A1", 2, 100.2, 5, 5, "C1"),
            raw_event(4, 1, "A0", 2, 100.3, 100, 100, "C2"),
            raw_event(5, 3, "A1", 2, 100.2, 0, 0, "C1", last_shares=5),
        ]
    )

    result = compute_exploratory_metrics(
        df,
        top_n=2,
        tick_size=0.1,
        kappa=1.0,
        lambda_=0.5,
        window_seconds=1.0,
        include_level_columns=False,
        state_client_ids={"C1"},
    )

    assert set(result.state_time_series["client_id"].to_list()) == {"C1"}
    assert "bid_level_1_price" not in result.state_time_series.columns
    assert result.execution_metrics.height == 1


def test_choose_event_timestamp_prefers_trade_time_then_book_fields():
    event = {
        "TRADETIME": datetime(2024, 1, 2, 9, 30, 1),
        "BOOKOUTTIME": "2024-01-02 09:30:02",
        "BOOKIN": "2024-01-02 09:30:03",
        "SEQUENCETIME": "2024-01-02 09:30:04",
    }

    assert choose_event_timestamp(event) == datetime(2024, 1, 2, 9, 30, 1)


def test_choose_event_timestamp_falls_back_to_bookout():
    event = {
        "TRADETIME": None,
        "BOOKOUTTIME": "2024-01-02 09:30:02",
        "BOOKIN": "2024-01-02 09:30:03",
        "SEQUENCETIME": "2024-01-02 09:30:04",
    }

    assert choose_event_timestamp(event) == datetime(2024, 1, 2, 9, 30, 2)


def test_compute_client_metric_time_series_emits_client_only_top_n_dwi_states():
    df = pl.DataFrame(
        [
            raw_event(1, 1, "B1", 1, 100.0, 10, 10, "C1"),
            raw_event(2, 1, "A1", 2, 101.0, 10, 10, "C2"),
            raw_event(3, 1, "B2", 1, 99.9, 20, 20, "C1"),
            raw_event(4, 1, "A2", 2, 101.1, 20, 20, None),
        ]
    )

    states = compute_client_metric_time_series(df, top_n=2, tick_size=0.1, kappa=1.0, lambda_=0.5)

    assert set(states["client_id"].drop_nulls().to_list()) == {"C1", "C2"}
    c1_latest = states.filter(pl.col("client_id") == "C1").tail(1).to_dicts()[0]
    assert c1_latest["client_bid_qty_topN"] == 30.0
    assert c1_latest["client_ask_qty_topN"] == 0.0
    assert c1_latest["DWI"] == pytest.approx(-1.0)
    assert "imbalance" not in c1_latest


def test_attach_sci_window_metrics_computes_side_collapse_and_msci():
    states = pl.DataFrame(
        {
            "partition_id": ["P", "P", "P"],
            "client_id": ["C1", "C1", "C1"],
            "event_ts": [
                datetime(2024, 1, 2, 9, 30, 8),
                datetime(2024, 1, 2, 9, 30, 9),
                datetime(2024, 1, 2, 9, 30, 11),
            ],
            "DWI": [-0.2, -0.8, -0.1],
            "L_bid_topN": [0.4, 0.9, 0.2],
            "L_ask_topN": [0.2, 0.4, 0.3],
            "sort_index": [8, 9, 11],
            "market_mid": [100.00, 100.10, 100.02],
            "market_microprice": [100.01, 100.13, 100.03],
        }
    )
    executions = pl.DataFrame(
        {
            "partition_id": ["P"],
            "client_id": ["C1"],
            "event_ts": [datetime(2024, 1, 2, 9, 30, 10)],
            "sort_index": [10],
            "execution_side": ["ask"],
            "deceptive_side": ["bid"],
            "event_price": [100.12],
            "candidate_deceptive_first_seen_sort_index_min": [8],
        }
    )

    out = attach_sci_window_metrics(executions, states, window_seconds=1.0, epsilon=1e-12)

    sci = 0.7
    c_bid = (0.9 - 0.2) / 0.9
    c_ask = (0.4 - 0.3) / 0.4
    expected_msci = sci * c_bid * max(c_bid - c_ask, 0.0)
    assert out.item(0, "DWI_pre_window") == pytest.approx(-0.8)
    assert out.item(0, "DWI_post_window") == pytest.approx(-0.1)
    assert out.item(0, "SCI") == pytest.approx(sci)
    assert out.item(0, "L_bid_pre_window") == pytest.approx(0.9)
    assert out.item(0, "L_bid_post_window") == pytest.approx(0.2)
    assert out.item(0, "collapse_bid") == pytest.approx(c_bid)
    assert out.item(0, "collapse_ask") == pytest.approx(c_ask)
    assert out.item(0, "collapse_opposite_side") == pytest.approx(c_bid)
    assert out.item(0, "collapse_same_side") == pytest.approx(c_ask)
    assert out.item(0, "MSCI") == pytest.approx(expected_msci)
    assert out.item(0, "market_mid_posture") == pytest.approx(100.00)
    assert out.item(0, "market_mid_pre_window") == pytest.approx(100.10)
    assert out.item(0, "market_mid_post_window") == pytest.approx(100.02)
    assert out.item(0, "favorable_mid_move_pre_fill") == pytest.approx(0.10)
    assert out.item(0, "favorable_microprice_move_pre_fill") == pytest.approx(0.12)
    assert out.item(0, "post_cancel_mid_reversion") == pytest.approx(0.08)
    assert out.item(0, "execution_price_advantage_vs_posture_mid") == pytest.approx(0.12)
    assert "imbalance_pre_window" not in out.columns


def test_compute_mcps_scores_groups_by_client_and_gamma():
    executions = pl.DataFrame(
        {
            "partition_id": ["P", "P", "P", "P"],
            "client_id": ["C1", "C1", "C1", "C2"],
            "top_n": [3, 3, 3, 3],
            "kappa": [1.0, 1.0, 1.0, 1.0],
            "lambda_": [0.5, 0.5, 0.5, 0.5],
            "MSCI": [0.2, 0.8, None, 0.9],
            "SCI": [0.4, 0.9, None, 1.0],
            "collapse_opposite_side": [0.5, 0.9, None, 0.8],
            "collapse_same_side": [0.1, 0.2, None, 0.1],
            "has_matched_deceptive_cancel_window": [False, True, False, True],
            "has_direct_opposite_cancel_window": [True, True, False, True],
            "candidate_deceptive_order_count_pre": [1, 1, 0, 1],
            "favorable_mid_move_pre_fill": [0.1, -0.1, None, 0.2],
            "favorable_microprice_move_pre_fill": [0.2, 0.0, None, 0.3],
            "post_cancel_mid_reversion": [0.05, None, None, 0.1],
            "execution_price_advantage_vs_posture_mid": [0.01, 0.02, None, 0.03],
        }
    )

    scores = compute_mcps_scores(executions, gamma_grid=[0.5])
    c1 = scores.filter(pl.col("client_id") == "C1").to_dicts()[0]

    assert c1["executions"] == 3
    assert c1["finite_msci_executions"] == 2
    assert c1["msci_above_gamma_count"] == 1
    assert c1["MCPS"] == pytest.approx(1 / 3)
    assert c1["candidate_profile_share"] == pytest.approx(2 / 3)
    assert c1["mean_favorable_mid_move_pre_fill"] == pytest.approx(0.0)
    assert c1["mean_post_cancel_mid_reversion"] == pytest.approx(0.05)


def test_multilevel_metrics_detect_deceptive_profile_collapse_after_execution():
    df = pl.DataFrame(
        [
            raw_event(1, 1, "B0", 1, 100.0, 100, 100, "C2"),
            raw_event(2, 1, "BD", 1, 99.9, 50, 50, "C1"),
            raw_event(3, 1, "A1", 2, 100.2, 5, 5, "C1"),
            raw_event(4, 1, "A0", 2, 100.3, 100, 100, "C2"),
            raw_event(
                5,
                3,
                "A1",
                2,
                100.2,
                0,
                0,
                "C1",
                last_shares=5,
                bookout="2024-01-02 09:30:05.000000",
            ),
            raw_event(
                6,
                4,
                "BD",
                1,
                99.9,
                0,
                0,
                "C1",
                bookout="2024-01-02 09:30:05.500000",
            ),
        ]
    )

    result = compute_exploratory_metrics(
        df, top_n=2, tick_size=0.1, kappa=1.0, lambda_=0.5, window_seconds=1.0
    )

    assert result.execution_metrics.height == 1
    row = result.execution_metrics.to_dicts()[0]
    assert row["client_id"] == "C1"
    assert row["execution_side"] == "ask"
    assert row["deceptive_side"] == "bid"
    assert row["candidate_deceptive_order_count_pre"] == 1
    assert row["candidate_deceptive_visible_qty_pre"] == pytest.approx(50.0)
    assert row["candidate_deceptive_order_ids_pre"] == "BD"
    assert row["candidate_deceptive_mean_depth_distance_ticks_pre"] == pytest.approx(2.0)
    assert row["has_direct_opposite_cancel_window"] is True
    assert row["has_matched_deceptive_cancel_window"] is True
    assert row["matched_deceptive_cancel_count_window"] == 1
    assert row["matched_deceptive_cancel_visible_qty_window"] == pytest.approx(50.0)
    assert row["matched_deceptive_cancel_order_ids_window"] == "BD"
    assert row["matched_deceptive_cancel_fraction_window"] == pytest.approx(1.0)
    assert row["matched_deceptive_cancel_min_delay_seconds"] == pytest.approx(0.5)
    assert row["weighted_net_withdrawal_qty_window"] == pytest.approx(50.0 * math.exp(-0.5 / 10.0))
    assert row["withdrawal_to_fill_ratio"] == pytest.approx(50.0 / 5.0)
    assert row["weighted_withdrawal_to_fill_ratio"] == pytest.approx(50.0 * math.exp(-0.5 / 10.0) / 5.0)
    assert row["WMSCI_event"] > 0
    assert row["smallness_fraction_market_level"] == pytest.approx(1.0)
    assert row["DWI_pre_window"] is not None
    assert row["MSCI"] is not None
    assert result.direct_cancellations.height == 1
    assert result.candidate_deceptive_orders.height == 1
    candidate = result.candidate_deceptive_orders.to_dicts()[0]
    weights = depth_kernel_weights([1.0, 2.0], kappa=1.0, lambda_=0.5)
    assert candidate["execution_sort_index"] == 5
    assert candidate["deceptive_order_id"] == "BD"
    assert candidate["deceptive_order_level"] == 2
    assert candidate["deceptive_order_delta_ticks"] == pytest.approx(1.0)
    assert candidate["deceptive_order_depth_distance_ticks"] == pytest.approx(2.0)
    assert candidate["deceptive_order_visible_qty_pre"] == pytest.approx(50.0)
    assert candidate["deceptive_order_relative_depth_pre"] == pytest.approx(1.0)
    assert candidate["deceptive_order_kernel_weight"] == pytest.approx(weights[1])
    assert candidate["deceptive_order_weighted_liquidity_contribution_pre"] == pytest.approx(weights[1])
    assert candidate["deceptive_order_age_seconds_pre"] == pytest.approx(3.0)
    assert "fake_side" not in row
    assert "candidate_fake_order_ids_pre" not in row


def test_candidate_deceptive_profile_must_be_recent_within_timing_window():
    df = pl.DataFrame(
        [
            raw_event(1, 1, "B_OLD", 1, 99.9, 50, 50, "C1", bookout="2024-01-02 09:30:00"),
            raw_event(2, 1, "B0", 1, 100.0, 100, 100, "C2", bookout="2024-01-02 09:49:58"),
            raw_event(3, 1, "A1", 2, 100.2, 5, 5, "C1", bookout="2024-01-02 09:49:59"),
            raw_event(
                4,
                3,
                "A1",
                2,
                100.2,
                0,
                0,
                "C1",
                last_shares=5,
                bookout="2024-01-02 09:50:00",
            ),
            raw_event(5, 4, "B_OLD", 1, 99.9, 0, 0, "C1", bookout="2024-01-02 09:50:01"),
        ]
    )

    result = compute_exploratory_metrics(
        df,
        top_n=2,
        tick_size=0.1,
        kappa=1.0,
        lambda_=0.5,
        window_seconds=1.0,
        max_deceptive_order_age_seconds=600.0,
    )

    row = result.execution_metrics.to_dicts()[0]
    assert row["candidate_deceptive_order_count_pre"] == 0
    assert row["candidate_deceptive_order_ids_pre"] == ""
    assert result.candidate_deceptive_orders.is_empty()
    assert row["has_direct_opposite_cancel_window"] is True
    assert row["has_matched_deceptive_cancel_window"] is False


def test_broad_opposite_cancel_is_not_a_matched_deceptive_profile_cancel():
    df = pl.DataFrame(
        [
            raw_event(1, 1, "B0", 1, 100.0, 100, 100, "C2"),
            raw_event(2, 1, "BD", 1, 99.9, 50, 50, "C1"),
            raw_event(3, 1, "A1", 2, 100.2, 5, 5, "C1"),
            raw_event(4, 1, "A0", 2, 100.3, 100, 100, "C2"),
            raw_event(
                5,
                3,
                "A1",
                2,
                100.2,
                0,
                0,
                "C1",
                last_shares=5,
                bookout="2024-01-02 09:30:05.000000",
            ),
            raw_event(
                6,
                1,
                "B_LATE",
                1,
                99.8,
                25,
                25,
                "C1",
                bookout="2024-01-02 09:30:05.200000",
            ),
            raw_event(
                7,
                4,
                "B_LATE",
                1,
                99.8,
                0,
                0,
                "C1",
                bookout="2024-01-02 09:30:05.500000",
            ),
        ]
    )

    result = compute_exploratory_metrics(
        df, top_n=3, tick_size=0.1, kappa=1.0, lambda_=0.5, window_seconds=1.0
    )

    row = result.execution_metrics.to_dicts()[0]
    assert row["candidate_deceptive_order_ids_pre"] == "BD"
    assert row["direct_opposite_cancel_order_ids_window"] == "B_LATE"
    assert row["has_direct_opposite_cancel_window"] is True
    assert row["has_matched_deceptive_cancel_window"] is False
    assert row["matched_deceptive_cancel_count_window"] == 0
    assert row["matched_deceptive_cancel_visible_qty_window"] == pytest.approx(0.0)
    assert row["matched_deceptive_cancel_fraction_window"] == pytest.approx(0.0)