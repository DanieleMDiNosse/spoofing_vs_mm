from __future__ import annotations

from datetime import datetime
from types import MappingProxyType

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from spoofing_detection.lob.config import LOBConfig
from spoofing_detection.lob.panel import ReplayHooks, replay_events
from spoofing_detection.lob.replay_observation import (
    EVENT_TIMESTAMP_PRECEDENCE,
    choose_event_timestamp,
    replay_lob,
)
from spoofing_detection.lob.spoofing_metrics import (
    choose_event_timestamp as metric_choose_event_timestamp,
    compute_exploratory_metrics,
)


EMPIRICAL_KERNEL_WEIGHTS = {
    "bid": {rank: 1.0 for rank in range(1, 3)},
    "ask": {rank: 1.0 for rank in range(1, 3)},
}


def event(
    seq: int,
    event_type: int,
    order_id: str,
    side: int | None,
    price: float | None,
    leaves: float,
    displayed: float,
    *,
    day: str = "2024-01-02",
    symbol: int = 123,
    client: str | None = "OTHER",
    firm: str = "OTHER_FIRM",
    order_type: int = 2,
    passive: str = "N",
    aggressive: str = "N",
    last_shares: float | None = None,
    last_px: float | None = None,
) -> dict[str, object]:
    timestamp = "2024-01-02 09:30:00"
    return {
        "TRADEDATE": day,
        "MIC": "XMIL",
        "MARKETCODE": "MTA",
        "SYMBOLINDEX": symbol,
        "EMM (*)": 1,
        "SEQUENCETIME": timestamp,
        "BOOKIN": timestamp,
        "BOOKOUTTIME": timestamp,
        "TRADETIME": timestamp if event_type == 3 else None,
        "HDR_APPLKEYSEQUENCENUMBER": seq,
        "HDR_HWMSEQUENCENUMBER": seq,
        "HDR_OFFSETID": seq,
        "ROW_NUMBER": seq,
        "EVENTID": f"E{seq}-{symbol}",
        "ORDEREVENTTYPE (*)": event_type,
        "ORDERID": order_id,
        "ORDERPRIORITY": str(seq),
        "ORDERSIDE (*)": side,
        "ORDERPX": price,
        "ORDERQTY": leaves,
        "DISPLAYEDQTY": displayed,
        "LEAVESQTY": leaves,
        "LASTSHARES": last_shares,
        "LASTTRADEDPX": last_px,
        "ORDERTYPE (*)": order_type,
        "TIMEINFORCE (*)": 0,
        "PASSIVEORDER": passive,
        "AGGRESSIVEORDER": aggressive,
        "FIRMID": firm,
        "NMSC_ORIGINALCLIENTIDSHORTCODE": client,
        "ORDER_TRADINGCAPACITY (*)": 3 if client is not None else 1,
    }


def realistic_events() -> pl.DataFrame:
    return pl.DataFrame(
        [
            event(1, 1, "B0", 1, 100.0, 100, 100),
            event(2, 1, "A0", 2, 100.2, 100, 100),
            # This actor has a top-of-book posture but never executes.
            event(3, 1, "POSTURE", 1, 99.9, 50, 50, client="NO_FILL", firm="POSTURE_FIRM"),
            # Passive execution is observed against its active order.
            event(4, 1, "PASSIVE", 2, 100.1, 5, 5, client="PASSIVE_ACTOR", firm="PASSIVE_FIRM"),
            event(5, 3, "PASSIVE", 2, 100.1, 0, 0, client="PASSIVE_ACTOR", firm="PASSIVE_FIRM", passive="Y", last_shares=5, last_px=100.1),
            # An inactive stop-limit order must not enter visible state.
            event(6, 1, "STOP", 2, 99.0, 10, 10, client="STOP_ACTOR", firm="STOP_FIRM", order_type=4),
            # An aggressive residual remains pending while A0 is present.
            event(7, 3, "RESIDUAL", 1, 100.2, 4, 4, client="AGGRESSIVE_ACTOR", firm="AGGRESSIVE_FIRM", aggressive="Y", last_shares=6, last_px=100.2),
            # Removing A0 lets the pending residual rest after canonical flush ordering.
            event(8, 3, "A0", 2, 100.2, 0, 0, passive="Y", last_shares=100, last_px=100.2),
            # A different instrument is a distinct partition, even at the same timestamp.
            event(9, 1, "P2B", 1, 200.0, 7, 7, symbol=456, client="P2_CLIENT", firm="P2_FIRM"),
        ]
    )


def _order_ids(snapshot) -> set[str]:
    return set(snapshot)


def test_replay_observes_immutable_pre_and_post_states_at_canonical_boundaries():
    observations = []

    replay_lob(realistic_events(), observer=observations.append)

    assert [observation.sort_index for observation in observations] == list(range(1, 10))
    assert observations[4].event["ORDERID"] == "PASSIVE"
    assert "PASSIVE" in _order_ids(observations[4].pre_active_orders)
    assert "PASSIVE" not in _order_ids(observations[4].post_active_orders)

    stop = observations[5]
    assert stop.event["ORDERID"] == "STOP"
    assert "STOP" not in _order_ids(stop.post_active_orders)

    pending_residual = observations[6]
    assert "RESIDUAL" not in _order_ids(pending_residual.post_active_orders)
    released_residual = observations[7]
    assert "RESIDUAL" in _order_ids(released_residual.post_active_orders)

    partition_change = observations[8]
    assert partition_change.partition_id.endswith("|456|1")
    assert _order_ids(partition_change.pre_active_orders) == set()
    p2_order = partition_change.post_active_orders["P2B"]
    assert p2_order.client_original_id == "P2_CLIENT"
    assert p2_order.firm_id == "P2_FIRM"

    assert isinstance(partition_change.event, MappingProxyType)
    with pytest.raises(TypeError):
        partition_change.pre_active_orders["bad"] = p2_order
    with pytest.raises(Exception):
        p2_order.displayed_qty = 999


def test_observation_includes_eligible_posture_actor_without_execution():
    observations = []

    replay_lob(realistic_events(), observer=observations.append)

    posture_observation = observations[2]
    posture = posture_observation.post_active_orders["POSTURE"]
    assert posture.client_original_id == "NO_FILL"
    assert posture.firm_id == "POSTURE_FIRM"
    assert all(observation.event["ORDERID"] != "NO_FILL" for observation in observations)

    result = compute_exploratory_metrics(
        realistic_events(),
        top_n=2,
        tick_size=0.1,
        window_seconds=1.0,
        empirical_kernel_weights=EMPIRICAL_KERNEL_WEIGHTS,
    )
    assert "client_original:NO_FILL" in set(result.state_time_series["actor_key"].to_list())
    assert "client_original:NO_FILL" not in set(result.execution_metrics["actor_key"].to_list())


def test_metrics_are_identical_with_a_non_mutating_replay_observer():
    kwargs = {
        "top_n": 2,
        "tick_size": 0.1,
        "window_seconds": 1.0,
        "empirical_kernel_weights": EMPIRICAL_KERNEL_WEIGHTS,
    }
    baseline = compute_exploratory_metrics(realistic_events(), **kwargs)
    observed = []
    replayed = compute_exploratory_metrics(
        realistic_events(), replay_observer=observed.append, **kwargs
    )

    assert len(observed) == 9
    for artifact in (
        "state_time_series",
        "execution_metrics",
        "candidate_deceptive_orders",
        "direct_cancellations",
        "rejected_executions",
        "execution_cluster_members",
        "execution_cancel_candidates",
        "spoofing_compatible_events",
    ):
        assert_frame_equal(getattr(replayed, artifact), getattr(baseline, artifact))

    assert replayed.episode_result is not None
    assert baseline.episode_result is not None
    for artifact in (
        "episodes",
        "members",
        "withdrawals",
        "anchor_summary",
        "actor_day_summary",
    ):
        assert_frame_equal(
            getattr(replayed.episode_result, artifact),
            getattr(baseline.episode_result, artifact),
        )


def test_observer_cannot_mutate_replay_boundaries_or_detector_outputs():
    mutation_errors = []
    kwargs = {
        "top_n": 2,
        "tick_size": 0.1,
        "window_seconds": 1.0,
        "empirical_kernel_weights": EMPIRICAL_KERNEL_WEIGHTS,
    }

    def attempted_mutation(observation):
        try:
            observation.event["ORDERID"] = "CHANGED"
        except TypeError as exc:
            mutation_errors.append(exc)
        try:
            observation.post_active_orders["CHANGED"] = next(
                iter(observation.post_active_orders.values())
            )
        except (StopIteration, TypeError) as exc:
            mutation_errors.append(exc)

    baseline = compute_exploratory_metrics(realistic_events(), **kwargs)
    result = compute_exploratory_metrics(
        realistic_events(), replay_observer=attempted_mutation, **kwargs
    )

    assert mutation_errors
    assert_frame_equal(result.execution_metrics, baseline.execution_metrics)


def test_central_replay_driver_sorts_before_max_rows_and_orders_hooks():
    callbacks = []

    def record(phase):
        def callback(event, active_orders, partition_id, *extra):
            callbacks.append((phase, event["ORDERID"], tuple(sorted(active_orders))))

        return callback

    canonical_events = replay_events(
        pl.DataFrame(
            [
                event(3, 1, "THIRD", 1, 99.8, 1, 1),
                event(1, 1, "FIRST", 1, 100.0, 1, 1),
                event(2, 1, "SECOND", 1, 99.9, 1, 1),
            ]
        ),
        config=LOBConfig(snapshot_mode="none"),
        max_rows=2,
        hooks=ReplayHooks(
            on_pre_event=record("pre"),
            on_post_event=record("post"),
            on_partition_end=record("partition_end"),
        ),
    )

    assert [event["ORDERID"] for event in canonical_events] == ["FIRST", "SECOND"]
    assert callbacks == [
        ("pre", "FIRST", ()),
        ("post", "FIRST", ("FIRST",)),
        ("pre", "SECOND", ("FIRST",)),
        ("post", "SECOND", ("FIRST", "SECOND")),
        ("partition_end", "SECOND", ("FIRST", "SECOND")),
    ]


def test_replay_callback_exceptions_propagate_without_later_callbacks():
    callbacks = []

    def failing_observer(observation):
        callbacks.append(observation.event["ORDERID"])
        raise RuntimeError("observer failure")

    with pytest.raises(RuntimeError, match="observer failure"):
        replay_lob(
            pl.DataFrame([event(2, 1, "SECOND", 1, 99.9, 1, 1), event(1, 1, "FIRST", 1, 100.0, 1, 1)]),
            observer=failing_observer,
        )

    assert callbacks == ["FIRST"]


def test_observation_timestamp_uses_shared_trade_precedence_and_utc_normalization():
    raw_event = event(1, 3, "FILL", 1, 100.0, 0, 0, passive="Y", last_shares=1, last_px=100.0)
    raw_event.update(
        {
            "TRADETIME": "2024-01-02T09:30:01+02:00",
            "BOOKOUTTIME": "2024-01-02T10:30:02+02:00",
            "BOOKIN": "2024-01-02T10:30:03+02:00",
            "SEQUENCETIME": "2024-01-02T10:30:04+02:00",
        }
    )
    observations = []

    replay_lob(pl.DataFrame([raw_event]), observer=observations.append)

    expected = datetime(2024, 1, 2, 7, 30, 1)
    assert choose_event_timestamp is metric_choose_event_timestamp
    assert choose_event_timestamp(raw_event) == expected
    assert observations[0].event_ts == expected


def test_choose_event_timestamp_uses_exported_event_timestamp_precedence():
    event_with_all_clock_values = {
        key: f"2024-01-02 09:30:0{index}"
        for index, key in enumerate(EVENT_TIMESTAMP_PRECEDENCE, start=1)
    }

    assert EVENT_TIMESTAMP_PRECEDENCE == (
        "TRADETIME", "BOOKOUTTIME", "BOOKIN", "SEQUENCETIME"
    )
    assert choose_event_timestamp(event_with_all_clock_values) == datetime(2024, 1, 2, 9, 30, 1)


def test_canonical_replay_can_skip_retaining_normalized_events_for_streaming_consumers():
    observed = []

    returned = replay_events(
        realistic_events(),
        hooks=ReplayHooks(on_pre_event=lambda event, *_: observed.append(event["sort_index"])),
        retain_events=False,
    )

    assert returned == ()
    assert observed == list(range(1, 10))
