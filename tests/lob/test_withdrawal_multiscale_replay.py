"""Bounded replay opt-ins; synthetic raw rows only, no multiscale scoring."""
from datetime import datetime, timezone
from typing import Any

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from spoofing_detection.lob.spoofing_metrics import (
    _candidate_deceptive_order_rows, _stream_metric_inputs, compute_exploratory_metrics,
)
from tests.lob.test_spoofing_metrics import order
from tests.lob.test_withdrawal_risk_v2 import event

WEIGHTS = {side: {rank: 1.0 for rank in range(1, 11)} for side in ("bid", "ask")}
BASE_NS = 1_704_187_800_000_000_000
DEPTH = "opposite_market_visible_qty_pre_topN"
LIFECYCLE = "cancel_order_first_seen_sort_index"


def raw_fixture(*, age_ns=60_000_000_000, two_fills=False, modify=False):
    rows = [
        event(1, 1, "OWN", 2, 101, 60, 60, ts=BASE_NS),
        event(2, 1, "OTHER", 2, 101, 40, 40, "C2", ts=BASE_NS),
        event(3, 1, "DEEP", 2, 102, 200, 200, "C2", ts=BASE_NS),
        event(4, 1, "BID", 1, 100, 10, 10, ts=BASE_NS),
    ]
    if modify:
        rows.append(event(5, 2, "OWN", 2, 101, 60, 60, ts=BASE_NS + age_ns - 1))
    seq = len(rows) + 1
    fill = event(seq, 3, "BID", 1, 100, 5 if two_fills else 0,
                 5 if two_fills else 0, ts=BASE_NS + age_ns)
    fill["LASTSHARES"] = 5 if two_fills else 10
    rows.append(fill)
    if two_fills:
        rows.append(event(seq + 1, 4, "DEEP", 2, 102, 0, 0, "C2", ts=BASE_NS + age_ns + 1_000_000))
        fill = event(seq + 2, 3, "BID", 1, 100, 0, 0, ts=BASE_NS + age_ns + 2_000_000)
        fill["LASTSHARES"] = 5
        rows.append(fill)
    rows.append(event(len(rows) + 1, 4, "OWN", 2, 101, 0, 0, ts=BASE_NS + age_ns + 1_200_000_000))
    # The metrics display clock parses ISO text; native clock parsing retains ns.
    for row in rows:
        for field in ("SEQUENCETIME", "BOOKIN", "BOOKOUTTIME", "TRADETIME"):
            if row[field] is not None:
                seconds, nanos = divmod(row[field], 1_000_000_000)
                row[field] = datetime.fromtimestamp(seconds, timezone.utc).strftime("%Y-%m-%d %H:%M:%S") + f".{nanos:09d}"
    return pl.DataFrame(rows)


def stream(raw=None, **kwargs):
    return _stream_metric_inputs(
        raw if raw is not None else raw_fixture(), top_n=10, tick_size=1,
        empirical_kernel_weights=WEIGHTS, use_v3_clocks=kwargs.pop("use_v3_clocks", True), **kwargs,
    )


def test_unbounded_age_keeps_members_execution_quantity_and_native_origin():
    bounded = stream(max_deceptive_order_age_seconds=30)
    unbounded = stream(max_deceptive_order_age_seconds=None)
    assert bounded[2].is_empty()
    assert unbounded[2]["deceptive_order_id"].to_list() == ["OWN"]
    assert unbounded[2].item(0, "deceptive_order_first_seen_replay_ns") == BASE_NS
    assert unbounded[2].item(0, "deceptive_order_age_seconds_pre") == 60
    stable = ["execution_cluster_id", "execution_quantity", "ORDERID", "actor_key", "execution_anchor_mode"]
    assert_frame_equal(bounded[1].select(stable), unbounded[1].select(stable))
    assert unbounded[1].item(0, "execution_quantity") == 10
    assert_frame_equal(bounded[5], unbounded[5])


def test_baseline_opt_in_includes_other_participants_and_non_candidate_prices():
    default = stream()
    disabled = stream(retain_withdrawal_baseline=False)
    enabled = stream(retain_withdrawal_baseline=True)
    for before, after in zip(default, disabled):
        assert_frame_equal(before, after)
    assert DEPTH not in default[1].columns
    assert LIFECYCLE not in default[3].columns
    assert enabled[1].item(0, DEPTH) == 300
    assert enabled[3].filter(pl.col("ORDERID") == "OWN").item(0, LIFECYCLE) == 1
    assert_frame_equal(enabled[1].drop(DEPTH), default[1])
    assert_frame_equal(enabled[3].drop(LIFECYCLE), default[3])


def test_cluster_baseline_is_first_fill_snapshot_not_last_fill():
    out = stream(raw_fixture(two_fills=True), retain_withdrawal_baseline=True)
    assert out[1].height == 1
    assert out[1].item(0, "child_fill_count") == 2
    assert out[1].item(0, "execution_quantity") == 10
    assert out[1].item(0, DEPTH) == 300  # last fill sees only 100
    assert out[5].height == 2


@pytest.mark.parametrize("age_ns,expected", [(30_000_000_000, 1), (30_000_000_001, 0)])
@pytest.mark.parametrize("modify", [False, True])
def test_native_age_cutoff_is_inclusive_ns_and_modify_keeps_origin(age_ns, expected, modify):
    out = stream(raw_fixture(age_ns=age_ns, modify=modify), max_deceptive_order_age_seconds=30)
    assert out[2].height == expected
    if expected:
        assert out[2].item(0, "deceptive_order_first_seen_sort_index") == 1
        assert out[2].item(0, "deceptive_order_first_seen_replay_ns") == BASE_NS


def test_legacy_default_age_600_is_unchanged():
    raw = raw_fixture(age_ns=601_000_000_000)
    default = stream(raw, use_v3_clocks=False)
    explicit = stream(raw, use_v3_clocks=False, max_deceptive_order_age_seconds=600)
    for before, after in zip(default, explicit):
        assert_frame_equal(before, after)
    assert default[2].is_empty()
    assert stream(raw, use_v3_clocks=False, max_deceptive_order_age_seconds=None)[2].height == 1


@pytest.mark.parametrize("retain", [False, True])
def test_empty_replay_opt_in_schema_is_typed_without_changing_defaults(retain):
    # Valid raw schema, no rows; exercises replay's actual empty-input path.
    out = stream(raw_fixture().clear(), retain_withdrawal_baseline=retain)
    assert out[1].is_empty() and out[3].is_empty()
    if retain:
        assert out[1].schema[DEPTH] == pl.Float64
        assert out[3].schema[LIFECYCLE] == pl.Int64
    else:
        assert DEPTH not in out[1].columns and LIFECYCLE not in out[3].columns
        for before, after in zip(stream(raw_fixture().clear()), out):
            assert_frame_equal(before, after)


@pytest.mark.parametrize("limit", [True, False, -1, float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("entry", ["candidate", "stream", "exploratory"])
def test_invalid_age_limit_rejected_even_without_rows(limit, entry):
    with pytest.raises(ValueError, match="max_deceptive_order_age_seconds"):
        if entry == "candidate":
            _candidate_deceptive_order_rows({}, {}, top_n=10, tick_size=1,
                order_first_seen_ts={}, max_deceptive_order_age_seconds=limit,
                empirical_kernel_weights=WEIGHTS)
        elif entry == "stream":
            stream(raw_fixture().clear(), max_deceptive_order_age_seconds=limit)
        else:
            compute_exploratory_metrics(raw_fixture().clear(), top_n=10, tick_size=1,
                window_seconds=2, max_deceptive_order_age_seconds=limit,
                empirical_kernel_weights=WEIGHTS)


@pytest.mark.parametrize("origin,execution_clock", [(None, BASE_NS), (BASE_NS, None), (BASE_NS + 1, BASE_NS)])
def test_unbounded_native_age_still_requires_nonnegative_native_clock(origin, execution_clock):
    execution = {"deceptive_side": "ask", "actor_key": "client_original:C1",
                 "event_ts": datetime(2024, 1, 2, 9, 30), "replay_time_ns": execution_clock}
    assert _candidate_deceptive_order_rows(execution, {"OWN": order("OWN", "ask", 101, 60, "C1")},
        top_n=10, tick_size=1, order_first_seen_ts={"OWN": datetime(2024, 1, 2, 9, 29)},
        order_first_seen_ns={"OWN": origin}, require_v3_clocks=True,
        max_deceptive_order_age_seconds=None, empirical_kernel_weights=WEIGHTS) == []


def test_exploratory_api_accepts_unbounded_age_without_baseline_schema_opt_in():
    out = compute_exploratory_metrics(raw_fixture(), top_n=10, tick_size=1,
        window_seconds=2, max_deceptive_order_age_seconds=None,
        empirical_kernel_weights=WEIGHTS, use_v3_clocks=True)
    assert out.candidate_deceptive_orders["deceptive_order_id"].to_list() == ["OWN"]
    assert DEPTH not in out.execution_metrics.columns
    assert LIFECYCLE not in out.direct_cancellations.columns


def test_ask_execution_baseline_uses_opposite_bid_depth():
    raw = raw_fixture().with_columns(
        (3 - pl.col("ORDERSIDE (*)")).alias("ORDERSIDE (*)"),
        (201 - pl.col("ORDERPX")).alias("ORDERPX"),
    )
    out = stream(raw, retain_withdrawal_baseline=True, max_deceptive_order_age_seconds=None)
    assert out[1].item(0, "execution_side") == "ask"
    assert out[1].item(0, "deceptive_side") == "bid"
    assert out[1].item(0, DEPTH) == 300
    assert out[2]["deceptive_order_id"].to_list() == ["OWN"]


def test_unbounded_age_retains_identity_side_visibility_and_top10_filters():
    execution = stream()[1].row(0, named=True)
    active = {f"LEVEL{rank}": order(f"LEVEL{rank}", "ask", 100 + rank, 1, "C2")
              for rank in range(1, 12)}
    active.update({
        "OWN": order("OWN", "ask", 101, 60, "C1"),
        "WRONG_ACTOR": order("WRONG_ACTOR", "ask", 101, 1, "C2"),
        "WRONG_SIDE": order("WRONG_SIDE", "bid", 100, 1, "C1"),
        "INVISIBLE": order("INVISIBLE", "ask", 101, 0, "C1"),
        "OUTSIDE": order("OUTSIDE", "ask", 111, 1, "C1"),
    })
    candidates = _candidate_deceptive_order_rows(execution, active, top_n=10, tick_size=1,
        order_first_seen_ts={}, order_first_seen_ns={key: BASE_NS for key in active},
        require_v3_clocks=True, max_deceptive_order_age_seconds=None,
        empirical_kernel_weights=WEIGHTS)
    assert [row["deceptive_order_id"] for row in candidates] == ["OWN"]


def test_zero_limit_accepts_exact_zero_age():
    out = stream(raw_fixture(age_ns=0), max_deceptive_order_age_seconds=0)
    assert out[2].item(0, "deceptive_order_age_seconds_pre") == 0


def test_baseline_remains_available_without_any_age_eligible_candidate():
    out = stream(max_deceptive_order_age_seconds=30, retain_withdrawal_baseline=True)
    assert out[2].is_empty()
    assert out[1].item(0, DEPTH) == 300


@pytest.mark.parametrize("native", [False, True])
def test_stream_consumes_immutable_order_snapshots_without_mutable_copies(monkeypatch, native):
    from spoofing_detection.lob.replay_observation import ReplayOrder
    original = ReplayOrder.to_active_order
    calls = []
    def counted(order):
        calls.append(order.order_id)
        return original(order)
    monkeypatch.setattr(ReplayOrder, "to_active_order", counted)
    observations = []
    out = stream(raw_fixture(two_fills=True, modify=True), use_v3_clocks=native,
                 max_deceptive_order_age_seconds=None, retain_withdrawal_baseline=True,
                 replay_observer=observations.append)
    assert out[1].height == 1 and out[2].height == 1 and out[3].height == 2
    assert out[1].item(0, "execution_quantity") == 10
    assert out[1].item(0, DEPTH) == 300
    assert out[2].item(0, "deceptive_order_first_seen_sort_index") == 1
    assert observations[0].post_active_orders["OWN"].displayed_qty == 60
    assert observations[-1].post_active_orders.get("OWN") is None
    assert calls == []


def test_empty_actor_selection_does_not_scan_book_but_still_validates(monkeypatch):
    import spoofing_detection.lob.spoofing_metrics as metrics
    original = metrics._market_levels
    scans = []
    def counted(*args, **kwargs):
        scans.append(kwargs["side"])
        return original(*args, **kwargs)
    monkeypatch.setattr(metrics, "_market_levels", counted)
    arguments: dict[str, Any] = dict(top_n=10, tick_size=1, partition_id="p", sort_index=1,
                     event_ts=None, actor_keys=set(),
                     include_zero_actor_keys={"client_original:C1"},
                     empirical_kernel_weights=WEIGHTS)
    assert metrics.compute_actor_top_n_exposures({"OWN": order("OWN", "ask", 101, 60, "C1")},
                                               **arguments) == []
    assert scans == []
    with pytest.raises(ValueError, match="top_n"):
        metrics.compute_actor_top_n_exposures({}, **{**arguments, "top_n": 0})
    with pytest.raises(ValueError, match="tick_size"):
        metrics.compute_actor_top_n_exposures({}, **{**arguments, "tick_size": 0})
    with pytest.raises(ValueError):
        metrics.compute_actor_top_n_exposures({}, **{**arguments, "empirical_kernel_weights": None})
