from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl
import pytest

from spoofing_detection.lob.motivating_cancellation_rate import (
    aggregate_actor_cancellation_rates,
    match_post_execution_opposite_side_cancellations,
    reconstruct_direct_cancellations,
)


def _execution(
    cluster_id: str,
    *,
    actor_key: str = "client_original:A",
    actor_id: str = "A",
    identity_level: str = "client_original",
    fallback: bool = False,
    partition: str = "P",
    end: datetime,
    last_sort_index: int,
    side: str = "bid",
    anchor: str = "passive",
) -> dict[str, object]:
    return {
        "execution_cluster_id": cluster_id,
        "partition_id": partition,
        "actor_key": actor_key,
        "actor_id": actor_id,
        "identity_level": identity_level,
        "identity_source": (
            "NMSC_ORIGINALCLIENTIDSHORTCODE" if identity_level == "client_original" else "FIRMID"
        ),
        "identity_fallback_flag": fallback,
        "execution_anchor_mode": anchor,
        "execution_side": side,
        "deceptive_side": "ask" if side == "bid" else "bid",
        "cluster_end_ts": end,
        "cluster_last_sort_index": last_sort_index,
    }


def _cancel(
    *,
    actor_key: str = "client_original:A",
    partition: str = "P",
    side: str = "ask",
    event_ts: datetime,
    sort_index: int,
    order_id: str,
) -> dict[str, object]:
    return {
        "partition_id": partition,
        "sort_index": sort_index,
        "event_ts": event_ts,
        "actor_key": actor_key,
        "side": side,
        "ORDERID": order_id,
    }


def test_broad_match_requires_joint_actor_side_partition_time_and_source_order():
    end = datetime(2024, 1, 2, 9, 30)
    executions = pl.DataFrame(
        [
            _execution("C1", end=end, last_sort_index=10),
            _execution(
                "C2",
                end=end,
                last_sort_index=20,
                side="ask",
                anchor="aggressive",
            ),
            _execution(
                "C3",
                actor_key="firm:F2",
                actor_id="F2",
                identity_level="firm",
                fallback=True,
                end=end,
                last_sort_index=30,
            ),
        ]
    )
    cancellations = pl.DataFrame(
        [
            _cancel(event_ts=end + timedelta(seconds=1), sort_index=11, order_id="A1"),
            _cancel(event_ts=end + timedelta(seconds=1.5), sort_index=12, order_id="A2"),
            _cancel(event_ts=end + timedelta(seconds=2), sort_index=13, order_id="A3"),
            _cancel(event_ts=end, sort_index=14, order_id="SAME-TIME"),
            _cancel(event_ts=end + timedelta(seconds=2.001), sort_index=15, order_id="TOO-LATE"),
            _cancel(event_ts=end + timedelta(seconds=1), sort_index=9, order_id="EARLIER-SOURCE"),
            _cancel(event_ts=end + timedelta(seconds=1), sort_index=16, order_id="WRONG-SIDE", side="bid"),
            _cancel(event_ts=end + timedelta(seconds=1), sort_index=17, order_id="WRONG-ACTOR", actor_key="firm:F2"),
            _cancel(event_ts=end + timedelta(seconds=1), sort_index=18, order_id="WRONG-PARTITION", partition="Q"),
            _cancel(event_ts=end + timedelta(seconds=1), sort_index=21, order_id="B1", side="bid"),
        ]
    )

    clusters, links = match_post_execution_opposite_side_cancellations(
        executions,
        cancellations,
        window_seconds=2.0,
    )

    c1 = clusters.filter(pl.col("execution_cluster_id") == "C1").row(0, named=True)
    c2 = clusters.filter(pl.col("execution_cluster_id") == "C2").row(0, named=True)
    c3 = clusters.filter(pl.col("execution_cluster_id") == "C3").row(0, named=True)
    assert c1["qualifying_cancellation_count"] == 3
    assert c1["has_opposite_side_cancellation_within_2s"] is True
    assert c1["first_qualifying_cancel_order_id"] == "A1"
    assert c1["first_qualifying_cancel_delay_seconds"] == pytest.approx(1.0)
    assert c2["qualifying_cancellation_count"] == 1
    assert c2["has_opposite_side_cancellation_within_2s"] is True
    assert c3["qualifying_cancellation_count"] == 0
    assert c3["has_opposite_side_cancellation_within_2s"] is False
    assert set(links.get_column("cancel_order_id")) == {"A1", "A2", "A3", "B1"}
    assert links.get_column("cancel_delay_seconds").min() > 0
    assert links.get_column("cancel_delay_seconds").max() <= 2.0
    assert links.select((pl.col("cancel_sort_index") > pl.col("cluster_last_sort_index")).all()).item()


def test_actor_rate_counts_each_cluster_once_and_retains_both_execution_anchors():
    end = datetime(2024, 1, 2, 9, 30)
    executions = pl.DataFrame(
        [
            _execution("C1", end=end, last_sort_index=10),
            _execution("C2", end=end, last_sort_index=20, anchor="aggressive"),
            _execution(
                "C3",
                actor_key="firm:F2",
                actor_id="F2",
                identity_level="firm",
                fallback=True,
                end=end,
                last_sort_index=30,
            ),
        ]
    )
    cancellations = pl.DataFrame(
        [
            _cancel(event_ts=end + timedelta(seconds=1), sort_index=31, order_id="ONE"),
            _cancel(event_ts=end + timedelta(seconds=1.5), sort_index=32, order_id="TWO"),
        ]
    )
    clusters, links = match_post_execution_opposite_side_cancellations(executions, cancellations)

    rates = aggregate_actor_cancellation_rates(clusters, instrument="Sample")

    actor = rates.filter(pl.col("actor_key") == "client_original:A").row(0, named=True)
    firm = rates.filter(pl.col("actor_key") == "firm:F2").row(0, named=True)
    assert links.height == 4  # Each broad cancellation can follow both nearby clusters.
    assert actor["execution_cluster_count"] == 2
    assert actor["qualifying_execution_cluster_count"] == 2
    assert actor["cancellation_rate_2s"] == pytest.approx(1.0)
    assert actor["passive_execution_cluster_count"] == 1
    assert actor["aggressive_execution_cluster_count"] == 1
    assert actor["identity_scope"] == "client-level"
    assert firm["execution_cluster_count"] == 1
    assert firm["qualifying_execution_cluster_count"] == 0
    assert firm["identity_scope"] == "firm-fallback"
    assert rates.get_column("rank_within_instrument").to_list() == [1, 2]


def test_broad_statistic_rejects_execution_anchors_outside_passive_and_aggressive():
    end = datetime(2024, 1, 2, 9, 30)
    executions = pl.DataFrame(
        [_execution("C1", end=end, last_sort_index=10, anchor="unknown")]
    )
    cancellations = pl.DataFrame(
        [_cancel(event_ts=end + timedelta(seconds=1), sort_index=11, order_id="A1")]
    )

    with pytest.raises(ValueError, match="unsupported execution_anchor_mode"):
        match_post_execution_opposite_side_cancellations(executions, cancellations)


def _raw_event(
    seq: int,
    event_type: int,
    *,
    order_id: str,
    side: int,
    client: object,
    firm: str,
) -> dict[str, object]:
    timestamp = f"2024-01-02 09:30:{seq:02d}"
    return {
        "TRADEDATE": "2024-01-02",
        "MIC": "XMIL",
        "MARKETCODE": "MTA",
        "SYMBOLINDEX": 123,
        "EMM (*)": 1,
        "SEQUENCETIME": timestamp,
        "BOOKIN": timestamp,
        "BOOKOUTTIME": timestamp,
        "HDR_APPLKEYSEQUENCENUMBER": seq,
        "HDR_HWMSEQUENCENUMBER": seq,
        "HDR_OFFSETID": seq,
        "ROW_NUMBER": seq,
        "ORDEREVENTTYPE (*)": event_type,
        "ORDERID": order_id,
        "ORDERPRIORITY": str(seq),
        "ORDERSIDE (*)": side,
        "ORDERPX": 100.0,
        "ORDERQTY": 50.0,
        "DISPLAYEDQTY": 50.0,
        "LEAVESQTY": 50.0,
        "ORDERTYPE (*)": 2,
        "TIMEINFORCE (*)": 0,
        "FIRMID": firm,
        "NMSC_ORIGINALCLIENTIDSHORTCODE": client,
    }


def test_direct_cancellation_replay_uses_active_order_and_canonical_firm_fallback():
    raw = pl.DataFrame(
        [
            _raw_event(1, 1, order_id="O1", side=2, client="0.00", firm="F1"),
            _raw_event(2, 4, order_id="O1", side=2, client="0.00", firm="F1"),
        ]
    )

    cancellations = reconstruct_direct_cancellations(raw)

    assert cancellations.height == 1
    row = cancellations.row(0, named=True)
    assert row["actor_key"] == "firm:F1"
    assert row["identity_level"] == "firm"
    assert row["identity_fallback_flag"] is True
    assert row["side"] == "ask"
    assert row["visible_qty_pre_cancel"] == pytest.approx(50.0)
