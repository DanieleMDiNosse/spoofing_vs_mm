from __future__ import annotations

import math
from datetime import datetime, timedelta

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import spoofing_detection.lob.spoofing_metrics as spoofing_metrics_module
from spoofing_detection.lob.models import ActiveOrder
from spoofing_detection.lob.spoofing_metrics import (
    _collapse,
    _finite_signed_msci,
    assign_cancellations_to_clusters,
    attach_sci_window_metrics,
    choose_event_timestamp,
    compute_actor_top_n_exposures,
    compute_client_metric_time_series,
    compute_client_top_n_exposures,
    compute_exploratory_metrics,
    compute_mcps_scores,
    infer_tick_size_from_best_quotes,
    shifted_depth_distance_ticks,
)


EMPIRICAL_KERNEL_WEIGHTS = {
    "bid": {rank: 1.0 for rank in range(1, 11)},
    "ask": {rank: 1.0 for rank in range(1, 11)},
}



@pytest.mark.parametrize(
    ("sci", "collapse_opposite", "collapse_same", "expected"),
    [
        (0.8, 0.9, 0.9, 0.4),
        (2.0, 1.0, 0.0, 2.0),
        (0.0, 0.0, 1.0, -1.0),
        (None, 0.9, 0.1, None),
        (math.nan, 0.9, 0.1, None),
    ],
)
def test_signed_msci_preserves_evidence_minus_counterevidence_without_clipping(
    sci,
    collapse_opposite,
    collapse_same,
    expected,
):
    result = _finite_signed_msci(sci, collapse_opposite, collapse_same)

    if expected is None:
        assert result is None
    else:
        assert result == pytest.approx(expected)


@pytest.mark.parametrize(
    ("pre", "post", "expected"),
    [
        (0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (1.0, 0.0, 1.0),
        (1.0, 0.25, 0.75),
        (1.0, 2.0, 0.0),
    ],
)
def test_collapse_uses_exact_piecewise_zero_denominator_semantics(pre, post, expected):
    assert _collapse(pre, post) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("pre", "post"),
    [
        (-1.0, 0.0),
        (1.0, -1.0),
        (math.nan, 0.0),
        (1.0, math.inf),
    ],
)
def test_collapse_rejects_values_outside_nonnegative_finite_domain(pre, post):
    assert _collapse(pre, post) is None


def order(order_id, side, price, qty, client, *, firm="F1"):
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
        firm_id=firm,
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
    last_traded_px=None,
    aggressive="N",
    passive=None,
    firm="F1",
    order_type=2,
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
        "LASTTRADEDPX": last_traded_px,
        "ORDERTYPE (*)": order_type,
        "TIMEINFORCE (*)": 0,
        "FIRMID": firm,
        "NMSC_ORIGINALCLIENTIDSHORTCODE": client,
        "ORDER_TRADINGCAPACITY (*)": 3 if client is not None else 1,
        "PASSIVEORDER": (
            passive
            if event_type == 3 and passive is not None
            else ("Y" if event_type == 3 and aggressive != "Y" else None)
        ),
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


def test_actor_exposures_require_empirical_kernel_weights():
    active = {
        "BID": order("BID", "bid", 100.0, 10.0, "C1"),
        "ASK": order("ASK", "ask", 100.1, 10.0, "C1"),
    }

    with pytest.raises(ValueError, match="empirical depth kernel"):
        spoofing_metrics_module.compute_actor_top_n_exposures(
            active,
            top_n=1,
            tick_size=0.1,
            partition_id="P",
            sort_index=10,
            event_ts=None,
        )


@pytest.mark.parametrize(
    ("weights", "message"),
    [
        ({"bid": {1: 1.0}}, "bid and ask"),
        ({"bid": {1: 1.0}, "ask": {1: 1.0}}, "missing ranks: 2"),
        ({"bid": {1: 1.0, 2: -0.1}, "ask": {1: 1.0, 2: 1.0}}, "finite and non-negative"),
    ],
)
def test_actor_exposures_reject_malformed_empirical_kernel(weights, message):
    with pytest.raises(ValueError, match=message):
        compute_actor_top_n_exposures(
            {},
            top_n=2,
            tick_size=0.1,
            partition_id="P",
            sort_index=10,
            event_ts=None,
            empirical_kernel_weights=weights,
        )


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
        empirical_kernel_weights=EMPIRICAL_KERNEL_WEIGHTS,
        partition_id="P",
        sort_index=10,
        event_ts=None,
    )
    by_client = {row["client_id"]: row for row in rows}

    c1 = by_client["C1"]
    weights = [0.5, 0.5]
    expected_l_bid = weights[0] * 1.0 + weights[1] * 0.4
    expected_l_ask = weights[0] * 1.0 + weights[1] * 0.0
    expected_dwi = (expected_l_ask - expected_l_bid) / (expected_l_ask + expected_l_bid)

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
        empirical_kernel_weights=EMPIRICAL_KERNEL_WEIGHTS,
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
        empirical_kernel_weights=EMPIRICAL_KERNEL_WEIGHTS,
        partition_id="P",
        sort_index=10,
        event_ts=None,
        client_ids={"17295"},
    )

    assert [row["client_id"] for row in rows] == ["17295"]


def test_compute_actor_top_n_exposures_keeps_client_and_firm_namespaces_distinct():
    active = {
        "CLIENT": order("CLIENT", "bid", 100.0, 10.0, "F2", firm="F1"),
        "FALLBACK": order("FALLBACK", "ask", 100.2, 5.0, None, firm="F2"),
    }

    rows = compute_actor_top_n_exposures(
        active,
        top_n=1,
        tick_size=0.1,
        empirical_kernel_weights=EMPIRICAL_KERNEL_WEIGHTS,
        partition_id="P",
        sort_index=10,
        event_ts=None,
    )
    by_actor = {row["actor_key"]: row for row in rows}

    assert set(by_actor) == {"client_original:F2", "firm:F2"}
    assert by_actor["client_original:F2"]["identity_level"] == "client_original"
    assert by_actor["client_original:F2"]["identity_fallback_flag"] is False
    assert by_actor["firm:F2"]["identity_level"] == "firm"
    assert by_actor["firm:F2"]["identity_source"] == "FIRMID"
    assert by_actor["firm:F2"]["identity_fallback_flag"] is True
    assert "client_id" not in by_actor["firm:F2"]
    assert by_actor["firm:F2"]["actor_ask_qty_topN"] == pytest.approx(5.0)


def test_compute_actor_top_n_exposures_emits_explicit_zero_profile_by_actor_key():
    rows = compute_actor_top_n_exposures(
        {"CLIENT": order("CLIENT", "bid", 100.0, 10.0, "C1")},
        top_n=1,
        tick_size=0.1,
        empirical_kernel_weights=EMPIRICAL_KERNEL_WEIGHTS,
        partition_id="P",
        sort_index=10,
        event_ts=None,
        actor_keys={"firm:F9"},
        include_zero_actor_keys={"firm:F9"},
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["actor_key"] == "firm:F9"
    assert row["actor_id"] == "F9"
    assert row["identity_level"] == "firm"
    assert row["has_active_top_n_profile"] is False
    assert row["DWI"] == pytest.approx(0.0)


def test_firm_fallback_passive_execution_is_attributed_end_to_end():
    rows = [
        raw_event(1, 1, "B0", 1, 100.0, 100, 100, None, firm="OTHER"),
        raw_event(2, 1, "BD", 1, 99.9, 50, 50, None, firm="157922_3"),
        raw_event(3, 1, "A1", 2, 100.2, 5, 5, None, firm="157922_3"),
        raw_event(4, 1, "A0", 2, 100.3, 100, 100, None, firm="OTHER"),
        raw_event(
            5, 3, "A1", 2, 100.2, 0, 0, None,
            firm="157922_3", last_shares=5,
            bookout="2024-01-02 09:30:05.000000",
        ),
        raw_event(
            6, 4, "BD", 1, 99.9, 0, 0, None,
            firm="157922_3", bookout="2024-01-02 09:30:05.500000",
        ),
    ]

    result = compute_exploratory_metrics(
        pl.DataFrame(rows),
        top_n=2,
        tick_size=0.1,
        empirical_kernel_weights=EMPIRICAL_KERNEL_WEIGHTS,
        window_seconds=1.0,
    )

    assert result.rejected_executions.height == 0
    assert result.rejected_executions.schema["reject_reason"] == pl.String
    execution = result.execution_metrics.row(0, named=True)
    assert execution["actor_key"] == "firm:157922_3"
    assert execution["identity_level"] == "firm"
    assert execution["identity_fallback_flag"] is True
    assert execution["execution_anchor_mode"] == "passive"
    assert execution["has_matched_deceptive_cancel_window"] is True
    cancel_candidate = result.execution_cancel_candidates.row(0, named=True)
    assert cancel_candidate["actor_key"] == "firm:157922_3"
    assert cancel_candidate["identity_level"] == "firm"
    assert cancel_candidate["execution_anchor_mode"] == "passive"
    assert "client_id" not in result.execution_cancel_candidates.columns


def test_aggressive_execution_uses_trade_fields_without_active_order():
    rows = [
        raw_event(1, 1, "B0", 1, 100.0, 100, 100, "OTHER"),
        raw_event(2, 1, "BD", 1, 99.9, 50, 50, "C1"),
        raw_event(3, 1, "A0", 2, 100.3, 100, 100, "OTHER"),
        raw_event(
            4, 3, "AGGRESSOR", 2, None, 0, 0, "C1",
            aggressive="Y", last_shares=5, last_traded_px=100.2,
            order_type=1,
        ),
    ]

    result = compute_exploratory_metrics(
        pl.DataFrame(rows),
        top_n=2,
        tick_size=0.1,
        empirical_kernel_weights=EMPIRICAL_KERNEL_WEIGHTS,
        window_seconds=1.0,
    )

    execution = result.execution_metrics.row(0, named=True)
    assert execution["actor_key"] == "client_original:C1"
    assert execution["execution_anchor_mode"] == "aggressive"
    assert execution["execution_price_source"] == "LASTTRADEDPX"
    assert execution["event_price"] == pytest.approx(100.2)
    assert execution["fill_qty"] == pytest.approx(5.0)
    assert execution["execution_quantity"] == pytest.approx(5.0)
    assert execution["execution_vwap"] == pytest.approx(100.2)
    assert execution["passive_execution_diagnostics_applicable"] is False
    assert execution["aggressive_execution_diagnostics_applicable"] is True
    assert execution["passive_execution_quantity"] is None
    assert execution["aggressive_execution_quantity"] == pytest.approx(5.0)
    assert execution["passive_same_level_market_visible_qty_pre"] is None
    assert execution["passive_same_level_actor_visible_qty_pre"] is None
    assert execution["passive_smallness_fraction_market_level"] is None
    assert execution["passive_smallness_fraction_actor_level"] is None
    assert execution["MSCI_resting_profile"] == execution["MSCI"]
    assert execution["withdrawal_profile_scale_event"] == execution["WMSCI_event"]
    assert execution["WMSCI_passive"] is None
    assert execution["WMSCI_aggressive"] == pytest.approx(0.0)
    assert execution["withdrawal_profile_scale_denominator_mode"] == "aggressive_execution_quantity"
    assert execution["smallness_fraction_market_level"] is None
    assert execution["smallness_fraction_actor_level"] is None
    assert "client_id" not in execution
    assert "same_level_client_visible_qty_pre" not in execution
    assert "smallness_fraction_client_level" not in execution


def test_selected_aggressive_actor_without_resting_orders_gets_zero_state_row():
    rows = [
        raw_event(1, 1, "B0", 1, 100.0, 100, 100, "OTHER"),
        raw_event(2, 1, "A0", 2, 100.3, 100, 100, "OTHER"),
        raw_event(
            3,
            3,
            "AGGRESSOR",
            2,
            None,
            0,
            0,
            "C1",
            aggressive="Y",
            last_shares=5,
            last_traded_px=100.2,
            order_type=1,
        ),
    ]

    for state_actor_keys in (None, {"client_original:C1"}):
        result = compute_exploratory_metrics(
            pl.DataFrame(rows),
            top_n=2,
            tick_size=0.1,
            empirical_kernel_weights=EMPIRICAL_KERNEL_WEIGHTS,
            window_seconds=1.0,
            state_actor_keys=state_actor_keys,
        )

        assert result.execution_metrics.get_column("actor_key").to_list() == [
            "client_original:C1"
        ]
        actor_state = result.state_time_series.filter(
            pl.col("actor_key") == "client_original:C1"
        )
        assert actor_state.height == 1
        state = actor_state.row(0, named=True)
        assert state["sort_index"] == 3
        assert state["has_active_top_n_profile"] is False
        assert state["DWI"] == pytest.approx(0.0)
        if state_actor_keys is not None:
            assert result.state_time_series.get_column("actor_key").unique().to_list() == [
                "client_original:C1"
            ]


def test_execution_actor_state_filter_preserves_all_analytical_artifacts():
    rows = [
        raw_event(1, 1, "B0", 1, 100.0, 100, 100, "OTHER"),
        raw_event(2, 1, "BD", 1, 99.9, 50, 50, "C1"),
        raw_event(3, 1, "A1", 2, 100.2, 5, 5, "C1"),
        raw_event(4, 1, "A2", 2, 100.3, 100, 100, "C2"),
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
    kwargs = {
        "top_n": 2,
        "tick_size": 0.1,
        "empirical_kernel_weights": EMPIRICAL_KERNEL_WEIGHTS,
        "window_seconds": 1.0,
    }

    complete = compute_exploratory_metrics(pl.DataFrame(rows), **kwargs)
    filtered = compute_exploratory_metrics(
        pl.DataFrame(rows),
        state_actor_keys={"client_original:C1"},
        **kwargs,
    )

    assert set(filtered.state_time_series.get_column("actor_key").unique()) == {"client_original:C1"}
    assert filtered.state_time_series.height < complete.state_time_series.height
    for artifact in (
        "execution_metrics",
        "candidate_deceptive_orders",
        "direct_cancellations",
        "rejected_executions",
        "execution_cluster_members",
        "execution_cancel_candidates",
        "spoofing_compatible_events",
    ):
        assert_frame_equal(getattr(filtered, artifact), getattr(complete, artifact))


def test_actor_state_chunking_preserves_schema_order_and_values(monkeypatch):
    rows = [
        raw_event(1, 1, "B0", 1, 100.0, 100, 100, "OTHER"),
        raw_event(2, 1, "BD", 1, 99.9, 50, 50, "C1"),
        raw_event(3, 1, "A1", 2, 100.2, 5, 5, "C1"),
        raw_event(4, 1, "A2", 2, 100.3, 100, 100, "C2"),
    ]
    kwargs = {
        "top_n": 2,
        "tick_size": 0.1,
        "empirical_kernel_weights": EMPIRICAL_KERNEL_WEIGHTS,
        "include_level_columns": False,
    }
    monkeypatch.setattr(spoofing_metrics_module, "_STATE_ROW_CHUNK_SIZE", 1_000_000)
    unchunked = spoofing_metrics_module.compute_actor_metric_time_series(pl.DataFrame(rows), **kwargs)

    monkeypatch.setattr(spoofing_metrics_module, "_STATE_ROW_CHUNK_SIZE", 2)
    chunked = spoofing_metrics_module.compute_actor_metric_time_series(pl.DataFrame(rows), **kwargs)

    assert chunked.n_chunks() > 1
    assert_frame_equal(chunked, unchunked)


def test_empty_execution_artifacts_preserve_actor_anchor_audit_schema():
    result = compute_exploratory_metrics(
        pl.DataFrame([raw_event(1, 1, "B0", 1, 100.0, 100, 100, "C1")]),
        top_n=2,
        tick_size=0.1,
        empirical_kernel_weights=EMPIRICAL_KERNEL_WEIGHTS,
        window_seconds=1.0,
    )

    assert result.execution_metrics.is_empty()
    assert {"actor_key", "identity_level", "execution_anchor_mode"}.issubset(result.execution_metrics.columns)
    assert result.execution_cluster_members.is_empty()
    assert {"actor_key", "identity_level", "execution_anchor_mode"}.issubset(
        result.execution_cluster_members.columns
    )
    assert result.candidate_deceptive_orders.is_empty()
    assert {"actor_key", "identity_level", "execution_anchor_mode"}.issubset(
        result.candidate_deceptive_orders.columns
    )
    assert result.direct_cancellations.is_empty()
    assert {"actor_key", "identity_level"}.issubset(result.direct_cancellations.columns)
    assert result.execution_cancel_candidates.is_empty()
    assert {"actor_key", "identity_level", "execution_anchor_mode"}.issubset(
        result.execution_cancel_candidates.columns
    )
    assert result.spoofing_compatible_events.is_empty()
    assert {"actor_key", "identity_level", "execution_anchor_mode", "spoofing_compatible_sequence"}.issubset(
        result.spoofing_compatible_events.columns
    )
    assert result.rejected_executions.is_empty()
    assert {"actor_key", "identity_level", "execution_anchor_mode", "reject_reason"}.issubset(
        result.rejected_executions.columns
    )


def test_empty_state_artifact_preserves_actor_metric_schema():
    empty_events = pl.DataFrame([raw_event(1, 1, "B0", 1, 100.0, 100, 100, "C1")]).head(0)

    result = compute_exploratory_metrics(
        empty_events,
        top_n=2,
        tick_size=0.1,
        empirical_kernel_weights=EMPIRICAL_KERNEL_WEIGHTS,
        window_seconds=1.0,
    )

    assert result.state_time_series.is_empty()
    assert {
        "actor_key",
        "actor_id",
        "identity_level",
        "identity_source",
        "identity_fallback_flag",
        "DWI",
        "bid_level_1_actor_visible_qty",
        "ask_level_2_actor_visible_qty",
    }.issubset(result.state_time_series.columns)


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
        empirical_kernel_weights=EMPIRICAL_KERNEL_WEIGHTS,
        window_seconds=1.0,
        include_level_columns=False,
        state_client_ids={"C1"},
    )

    assert set(result.state_time_series["actor_key"].to_list()) == {"client_original:C1"}
    assert "client_id" not in result.state_time_series.columns
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

    states = compute_client_metric_time_series(
        df,
        top_n=2,
        tick_size=0.1,
        empirical_kernel_weights=EMPIRICAL_KERNEL_WEIGHTS,
    )

    assert set(states["client_id"].drop_nulls().to_list()) == {"C1", "C2"}
    c1_latest = states.filter(pl.col("client_id") == "C1").tail(1).to_dicts()[0]
    assert c1_latest["client_bid_qty_topN"] == 30.0
    assert c1_latest["client_ask_qty_topN"] == 0.0
    assert c1_latest["DWI"] == pytest.approx(-1.0)
    assert "imbalance" not in c1_latest


def test_attach_sci_window_metrics_computes_side_collapse_and_signed_msci():
    states = pl.DataFrame(
        {
            "partition_id": ["P", "P", "P"],
            "actor_key": ["client_original:C1"] * 3,
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
            "partition_id": ["P", "P"],
            "actor_key": ["client_original:C1"] * 2,
            "event_ts": [datetime(2024, 1, 2, 9, 30, 10)] * 2,
            "sort_index": [10, 10],
            "execution_side": ["ask", "bid"],
            "deceptive_side": ["bid", "ask"],
            "event_price": [100.12, 100.12],
            "candidate_deceptive_first_seen_sort_index_min": [8, 8],
        }
    )

    out = attach_sci_window_metrics(executions, states, window_seconds=1.0)

    sci = 0.7
    c_bid = (0.9 - 0.2) / 0.9
    c_ask = (0.4 - 0.3) / 0.4
    expected_msci = (sci / 2.0) + c_bid - c_ask
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
    assert out.item(1, "collapse_opposite_side") == pytest.approx(c_ask)
    assert out.item(1, "collapse_same_side") == pytest.approx(c_bid)
    assert out.item(1, "MSCI") == pytest.approx((sci / 2.0) + c_ask - c_bid)
    assert out.item(0, "market_mid_posture") == pytest.approx(100.00)
    assert out.item(0, "market_mid_pre_window") == pytest.approx(100.10)
    assert out.item(0, "market_mid_post_window") == pytest.approx(100.02)
    assert out.item(0, "favorable_mid_move_pre_fill") == pytest.approx(0.10)
    assert out.item(0, "favorable_microprice_move_pre_fill") == pytest.approx(0.12)
    assert out.item(0, "post_fill_mid_reversal_from_pre_fill") == pytest.approx(0.08)
    assert out.item(0, "execution_price_advantage_vs_posture_mid") == pytest.approx(0.12)
    assert "imbalance_pre_window" not in out.columns


def test_attach_sci_window_metrics_does_not_impute_missing_post_state_as_zero():
    states = pl.DataFrame(
        {
            "partition_id": ["P", "P"],
            "actor_key": ["client_original:C1"] * 2,
            "event_ts": [
                datetime(2024, 1, 2, 9, 30, 8),
                datetime(2024, 1, 2, 9, 30, 9),
            ],
            "DWI": [-0.2, -0.8],
            "L_bid_topN": [0.4, 0.9],
            "L_ask_topN": [0.2, 0.4],
            "sort_index": [8, 9],
            "market_mid": [100.00, 100.10],
            "market_microprice": [100.01, 100.13],
        }
    )
    executions = pl.DataFrame(
        {
            "partition_id": ["P"],
            "actor_key": ["client_original:C1"],
            "event_ts": [datetime(2024, 1, 2, 9, 30, 10)],
            "sort_index": [10],
            "execution_side": ["ask"],
            "deceptive_side": ["bid"],
            "event_price": [100.12],
            "candidate_deceptive_first_seen_sort_index_min": [8],
        }
    )

    out = attach_sci_window_metrics(executions, states, window_seconds=1.0)

    assert out.item(0, "has_post_window_state") is False
    for column in (
        "DWI_post_window",
        "SCI",
        "L_bid_post_window",
        "L_ask_post_window",
        "collapse_bid",
        "collapse_ask",
        "MSCI",
    ):
        assert out.item(0, column) is None


def test_attach_sci_window_metrics_accepts_post_event_state_at_cluster_last_sort_index():
    states = pl.DataFrame(
        {
            "partition_id": ["P1", "P1"],
            "actor_key": ["client_original:C1"] * 2,
            "sort_index": [9, 10],
            "event_ts": [
                datetime(2024, 1, 1, 12, 0, 8),
                datetime(2024, 1, 1, 12, 0, 10),
            ],
            "DWI": [0.8, 0.0],
            "L_bid_topN": [10.0, 10.0],
            "L_ask_topN": [90.0, 0.0],
        }
    )
    executions = pl.DataFrame(
        {
            "partition_id": ["P1"],
            "actor_key": ["client_original:C1"],
            "sort_index": [10],
            "cluster_last_sort_index": [10],
            "event_ts": [datetime(2024, 1, 1, 12, 0, 10)],
            "execution_side": ["bid"],
            "deceptive_side": ["ask"],
            "fill_qty": [5.0],
            "candidate_deceptive_weighted_liquidity_pre": [20.0],
            "candidate_deceptive_visible_qty_pre": [20.0],
            "candidate_deceptive_first_seen_sort_index_min": [8],
        }
    )

    # Coverage after the target is necessary to carry index 10 forward.
    states = pl.concat([states, states.tail(1).with_columns(
        pl.lit(11, dtype=pl.Int64).alias("sort_index"),
        pl.lit(datetime(2024, 1, 1, 12, 0, 12)).alias("event_ts"),
    )])
    out = attach_sci_window_metrics(executions, states, window_seconds=1.0)

    assert out.item(0, "has_post_window_state") is True
    assert out.item(0, "post_state_sort_index") == 10
    assert out.item(0, "DWI_post_window") == pytest.approx(0.0)
    assert out.item(0, "MSCI") is not None


def test_compute_mcps_scores_groups_by_actor_anchor_and_gamma():
    executions = pl.DataFrame(
        {
            "partition_id": ["P", "P", "P", "P"],
            "actor_key": ["client_original:C1", "client_original:C1", "client_original:C1", "client_original:C2"],
            "execution_anchor_mode": ["passive", "passive", "passive", "aggressive"],
            "top_n": [3, 3, 3, 3],
            "kappa": [1.0, 1.0, 1.0, 1.0],
            "lambda_": [0.5, 0.5, 0.5, 0.5],
            "MSCI_resting_profile": [0.2, 0.8, None, 0.9],
            # Deliberately discordant compatibility alias: consumers must use
            # the canonical resting-profile field when both are present.
            "MSCI": [0.0, 0.0, 0.0, 0.0],
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
    c1 = scores.filter(pl.col("actor_key") == "client_original:C1").to_dicts()[0]

    assert c1["executions"] == 3
    assert c1["finite_msci_executions"] == 2
    assert c1["msci_above_gamma_count"] == 1
    assert c1["MCPS"] == pytest.approx(1 / 3)
    assert c1["MCPS_resting_profile"] == pytest.approx(1 / 3)
    assert c1["median_MSCI_resting_profile"] == pytest.approx(0.5)
    assert c1["candidate_profile_share"] == pytest.approx(2 / 3)
    assert c1["mean_favorable_mid_move_pre_fill"] == pytest.approx(0.0)
    assert c1["mean_post_cancel_mid_reversion"] == pytest.approx(0.05)


def test_compute_mcps_scores_excludes_nonfinite_values_and_rejects_nonfinite_gamma():
    executions = pl.DataFrame(
        {
            "partition_id": ["P", "P", "P"],
            "actor_key": ["client_original:C1"] * 3,
            "execution_anchor_mode": ["passive"] * 3,
            "MSCI": [0.75, float("nan"), float("inf")],
            "SCI": [0.5, float("nan"), float("-inf")],
            "collapse_opposite_side": [0.4, float("nan"), float("inf")],
            "collapse_same_side": [0.1, float("nan"), float("-inf")],
            "favorable_mid_move_pre_fill": [0.2, float("nan"), float("inf")],
        }
    )

    score = compute_mcps_scores(executions, gamma_grid=[0.5]).to_dicts()[0]

    assert score["executions"] == 3
    assert score["finite_msci_executions"] == 1
    assert score["msci_above_gamma_count"] == 1
    assert score["MCPS"] == pytest.approx(1 / 3)
    assert score["median_MSCI"] == pytest.approx(0.75)
    assert score["max_MSCI"] == pytest.approx(0.75)
    assert score["mean_MSCI"] == pytest.approx(0.75)
    assert score["mean_SCI"] == pytest.approx(0.5)
    assert score["mean_favorable_mid_move_pre_fill"] == pytest.approx(0.2)
    assert all(
        value is None or not isinstance(value, float) or math.isfinite(value)
        for value in score.values()
    )

    for invalid_gamma in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="gamma thresholds must be finite"):
            compute_mcps_scores(executions, gamma_grid=[invalid_gamma])


def test_compute_mcps_scores_excludes_unattributable_actor_bucket():
    executions = pl.DataFrame(
        {
            "actor_key": ["", "client_original:C1"],
            "execution_anchor_mode": ["passive", "passive"],
            "MSCI": [0.9, 0.2],
            "SCI": [1.0, 0.3],
            "collapse_opposite_side": [0.9, 0.4],
            "collapse_same_side": [0.0, 0.1],
        }
    )

    scores = compute_mcps_scores(executions, gamma_grid=[0.5])

    assert scores.get_column("actor_key").to_list() == ["client_original:C1"]


def test_compute_mcps_scores_does_not_pool_execution_anchor_branches():
    executions = pl.DataFrame(
        {
            "partition_id": ["P", "P"],
            "actor_key": ["firm:F1", "firm:F1"],
            "execution_anchor_mode": ["passive", "aggressive"],
            "MSCI": [0.9, 0.1],
            "SCI": [1.0, 0.2],
            "collapse_opposite_side": [0.9, 0.2],
            "collapse_same_side": [0.0, 0.0],
        }
    )

    scores = compute_mcps_scores(executions, gamma_grid=[0.5]).sort("execution_anchor_mode")

    assert scores.get_column("execution_anchor_mode").to_list() == ["aggressive", "passive"]
    assert scores.get_column("executions").to_list() == [1, 1]
    assert scores.get_column("MCPS").to_list() == [0.0, 1.0]


def test_compute_mcps_scores_preserves_actor_anchor_schema_when_empty():
    scores = compute_mcps_scores(pl.DataFrame(), gamma_grid=[0.5])

    assert scores.is_empty()
    assert {
        "actor_key",
        "actor_id",
        "identity_level",
        "identity_source",
        "identity_fallback_flag",
        "execution_anchor_mode",
        "gamma",
        "executions",
        "MCPS",
    }.issubset(scores.columns)


def test_sci_post_state_lookup_handles_timestamps_nonmonotonic_in_sort_order():
    base = datetime(2024, 1, 2, 9, 30)
    executions = pl.DataFrame(
        [
            {
                "partition_id": "P",
                "actor_key": "client_original:C1",
                "event_ts": base,
                "cluster_start_ts": base,
                "cluster_end_ts": base,
                "sort_index": 1,
                "cluster_first_sort_index": 1,
                "cluster_last_sort_index": 1,
                "execution_side": "ask",
                "deceptive_side": "bid",
            }
        ]
    )
    states = pl.DataFrame(
        {
            "partition_id": ["P", "P", "P"],
            "actor_key": ["client_original:C1"] * 3,
            "sort_index": [1, 2, 3],
            "event_ts": [
                base + timedelta(seconds=2),
                base + timedelta(seconds=1),
                base + timedelta(seconds=3),
            ],
            "DWI": [2.0, 1.0, 3.0],
            "L_bid_topN": [2.0, 1.0, 3.0],
            "L_ask_topN": [1.0, 1.0, 1.0],
        }
    )

    result = attach_sci_window_metrics(executions, states, window_seconds=2.5)

    assert result.item(0, "post_state_sort_index") == 1
    assert result.item(0, "DWI_post_window") == pytest.approx(2.0)


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

    # The original fixture stopped before the requested observation horizon.
    df = pl.concat([df, pl.DataFrame([raw_event(
        7, 1, "BFUTURE", 1, 99.8, 5, 5, "C1",
        bookout="2024-01-02 09:30:07.000000",
    )])], how="diagonal_relaxed")
    result = compute_exploratory_metrics(
        df,
        top_n=2,
        tick_size=0.1,
        empirical_kernel_weights=EMPIRICAL_KERNEL_WEIGHTS,
        window_seconds=1.0
    )

    assert result.execution_metrics.height == 1
    row = result.execution_metrics.to_dicts()[0]
    assert row["actor_key"] == "client_original:C1"
    assert row["actor_id"] == "C1"
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
    assert row["MSCI_resting_profile"] == row["MSCI"]
    assert row["withdrawal_profile_scale_event"] == row["WMSCI_event"]
    assert row["WMSCI_passive"] == row["WMSCI_event"]
    assert row["WMSCI_aggressive"] is None
    assert row["withdrawal_profile_scale_denominator_mode"] == "passive_execution_quantity"
    assert row["execution_quantity"] == pytest.approx(5.0)
    assert row["execution_vwap"] == pytest.approx(100.2)
    assert row["passive_execution_diagnostics_applicable"] is True
    assert row["aggressive_execution_diagnostics_applicable"] is False
    assert row["passive_execution_quantity"] == pytest.approx(5.0)
    assert row["passive_execution_vwap"] == pytest.approx(100.2)
    assert row["passive_same_level_market_visible_qty_pre"] == pytest.approx(5.0)
    assert row["passive_same_level_actor_visible_qty_pre"] == pytest.approx(5.0)
    assert row["passive_smallness_fraction_market_level"] == pytest.approx(1.0)
    assert row["passive_smallness_fraction_actor_level"] == pytest.approx(1.0)
    assert row["aggressive_execution_quantity"] is None
    assert row["aggressive_execution_vwap"] is None
    assert row["aggressive_child_fill_count"] is None
    assert row["smallness_fraction_market_level"] == pytest.approx(1.0)
    assert row["DWI_pre_window"] is not None
    assert row["has_post_window_state"] is True
    assert row["DWI_post_window"] == pytest.approx(0.0)
    assert row["L_bid_post_window"] == pytest.approx(0.0)
    assert row["L_ask_post_window"] == pytest.approx(0.0)
    assert row["MSCI"] is not None
    assert result.direct_cancellations.height == 1
    assert result.candidate_deceptive_orders.height == 1
    candidate = result.candidate_deceptive_orders.to_dicts()[0]
    weights = [0.5, 0.5]
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


def test_matched_withdrawal_is_capped_at_pre_execution_candidate_quantity():
    df = pl.DataFrame(
        [
            raw_event(1, 1, "B0", 1, 100.0, 100, 100, "C2"),
            raw_event(2, 1, "BD", 1, 99.9, 50, 50, "C1"),
            raw_event(3, 1, "A1", 2, 100.2, 5, 5, "C1"),
            raw_event(4, 1, "A0", 2, 100.3, 100, 100, "C2"),
            raw_event(
                5, 3, "A1", 2, 100.2, 0, 0, "C1",
                last_shares=5,
                bookout="2024-01-02 09:30:05.000000",
            ),
            raw_event(
                6, 2, "BD", 1, 99.9, 60, 60, "C1",
                bookout="2024-01-02 09:30:05.250000",
            ),
            raw_event(
                7, 4, "BD", 1, 99.9, 0, 0, "C1",
                bookout="2024-01-02 09:30:05.500000",
            ),
        ]
    )

    result = compute_exploratory_metrics(
        df,
        top_n=2,
        tick_size=0.1,
        empirical_kernel_weights=EMPIRICAL_KERNEL_WEIGHTS,
        window_seconds=1.0,
    )

    row = result.execution_metrics.to_dicts()[0]
    candidate = result.execution_cancel_candidates.filter(pl.col("assigned_flag")).to_dicts()[0]
    assert row["candidate_deceptive_visible_qty_pre"] == pytest.approx(50.0)
    assert candidate["visible_qty_pre_cancel"] == pytest.approx(60.0)
    assert candidate["candidate_visible_qty_pre"] == pytest.approx(50.0)
    assert candidate["attributed_cancel_visible_qty"] == pytest.approx(50.0)
    assert row["matched_deceptive_cancel_visible_qty_window"] == pytest.approx(50.0)
    assert row["matched_deceptive_cancel_fraction_window"] == pytest.approx(1.0)
    assert row["weighted_net_withdrawal_qty_window"] == pytest.approx(50.0 * math.exp(-0.5 / 10.0))


def test_fragmented_passive_fills_become_one_cluster_with_raw_members_and_single_cancellation():
    fill_quantities = [9_000, 9_000, 1_947, 857, 4_196]
    fill_leaves = [16_000, 7_000, 5_053, 4_196, 0]
    offsets = [0, 25, 50, 75, 100]
    rows = [
        raw_event(1, 1, "B0", 1, 100.0, 300_000, 300_000, "C2", bookout="2024-01-02 09:29:59.997"),
        raw_event(2, 1, "BD", 1, 99.9, 219_770, 219_770, "C1", bookout="2024-01-02 09:29:59.998"),
        raw_event(3, 1, "FILL", 2, 100.2, 25_000, 25_000, "C1", bookout="2024-01-02 09:29:59.999"),
    ]
    for index, (qty, leaves, offset) in enumerate(zip(fill_quantities, fill_leaves, offsets), start=4):
        rows.append(
            raw_event(
                index * 3,
                3,
                "FILL",
                2,
                100.2,
                leaves,
                leaves,
                "C1",
                last_shares=qty,
                bookout=f"2024-01-02 09:30:00.{offset:03d}",
            )
        )
        if index < 8:
            rows.extend(
                [
                    raw_event(index * 3 + 1, 3, f"AG{index}", 1, 100.2, 0, 0, "C9", aggressive="Y", bookout=f"2024-01-02 09:30:00.{offset + 5:03d}"),
                    raw_event(index * 3 + 2, 1, f"NEW{index}", 1, 99.8, 10, 10, "C9", bookout=f"2024-01-02 09:30:00.{offset + 10:03d}"),
                ]
            )
    rows.append(raw_event(30, 4, "BD", 1, 99.9, 0, 0, "C1", bookout="2024-01-02 09:30:00.200"))

    result = compute_exploratory_metrics(
        pl.DataFrame(rows),
        top_n=2,
        tick_size=0.1,
        empirical_kernel_weights=EMPIRICAL_KERNEL_WEIGHTS,
        window_seconds=1.0,
        execution_cluster_max_gap_ms=100,
    )

    assert result.execution_metrics.height == 1
    cluster = result.execution_metrics.to_dicts()[0]
    assert cluster["child_fill_count"] == 5
    assert cluster["fill_qty"] == pytest.approx(25_000)
    assert cluster["cluster_start_ts"] == datetime(2024, 1, 2, 9, 30, 0)
    assert cluster["cluster_end_ts"] == datetime(2024, 1, 2, 9, 30, 0, 100_000)
    assert cluster["matched_deceptive_cancel_count_window"] == 1
    assert cluster["withdrawal_to_fill_ratio"] == pytest.approx(8.7908)
    assert result.execution_cluster_members.height == 5
    assert result.execution_cluster_members["child_fill_qty"].sum() == pytest.approx(25_000)
    assert result.execution_cluster_members["execution_cluster_id"].n_unique() == 1
    assert result.execution_cancel_candidates.filter(pl.col("assigned_flag")).height == 1


def test_assign_cancellations_uses_latest_cluster_end_then_first_sort_tiebreak():
    candidate_links = pl.DataFrame(
        {
            "partition_id": ["P", "P", "P"],
            "cancel_sort_index": [30, 30, 30],
            "candidate_order_id": ["BD", "BD", "BD"],
            "execution_cluster_id": ["EC000000010-000000012", "EC000000020-000000022", "EC000000021-000000022"],
            "cluster_end_ts": [
                datetime(2024, 1, 2, 9, 30, 1),
                datetime(2024, 1, 2, 9, 30, 2),
                datetime(2024, 1, 2, 9, 30, 2),
            ],
            "cluster_first_sort_index": [10, 20, 21],
        }
    )

    assigned = assign_cancellations_to_clusters(candidate_links)

    assert assigned.height == 3
    winner = assigned.filter(pl.col("assigned_flag")).to_dicts()
    assert [row["execution_cluster_id"] for row in winner] == ["EC000000020-000000022"]
    assert set(assigned["assignment_rule"].to_list()) == {"latest_prior_cluster_end"}
    assert set(assigned["competing_cluster_count"].to_list()) == {3}


def test_assign_cancellations_selects_one_winning_execution_anchor_branch():
    candidate_links = pl.DataFrame(
        {
            "partition_id": ["P", "P"],
            "actor_key": ["client_original:C1", "client_original:C1"],
            "execution_anchor_mode": ["passive", "aggressive"],
            "cancel_sort_index": [30, 30],
            "candidate_order_id": ["BD", "BD"],
            "execution_cluster_id": ["PASSIVE", "AGGRESSIVE"],
            "cluster_end_ts": [
                datetime(2024, 1, 2, 9, 30, 1),
                datetime(2024, 1, 2, 9, 30, 2),
            ],
            "cluster_first_sort_index": [10, 20],
        }
    )

    assigned = assign_cancellations_to_clusters(candidate_links)

    assert assigned.get_column("assigned_flag").to_list() == [False, True]
    assert assigned.get_column("competing_cluster_count").to_list() == [2, 2]


def test_assign_cancellations_competes_by_physical_cancel_across_actor_namespaces():
    candidate_links = pl.DataFrame(
        {
            "partition_id": ["P", "P"],
            "actor_key": ["client_original:F1", "firm:F1"],
            "execution_anchor_mode": ["passive", "passive"],
            "cancel_sort_index": [30, 30],
            "candidate_order_id": ["BD", "BD"],
            "execution_cluster_id": ["CLIENT", "FIRM"],
            "cluster_end_ts": [
                datetime(2024, 1, 2, 9, 30, 1),
                datetime(2024, 1, 2, 9, 30, 2),
            ],
            "cluster_first_sort_index": [10, 20],
        }
    )

    assigned = assign_cancellations_to_clusters(candidate_links)

    assert assigned.get_column("assigned_flag").to_list() == [False, True]
    assert assigned.get_column("competing_cluster_count").to_list() == [2, 2]



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
        empirical_kernel_weights=EMPIRICAL_KERNEL_WEIGHTS,
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
        df,
        top_n=3,
        tick_size=0.1,
        empirical_kernel_weights=EMPIRICAL_KERNEL_WEIGHTS,
        window_seconds=1.0
    )

    row = result.execution_metrics.to_dicts()[0]
    assert row["candidate_deceptive_order_ids_pre"] == "BD"
    assert row["direct_opposite_cancel_order_ids_window"] == "B_LATE"
    assert row["has_direct_opposite_cancel_window"] is True
    assert row["has_matched_deceptive_cancel_window"] is False
    assert row["matched_deceptive_cancel_count_window"] == 0
    assert row["matched_deceptive_cancel_visible_qty_window"] == pytest.approx(0.0)
    assert row["matched_deceptive_cancel_fraction_window"] == pytest.approx(0.0)