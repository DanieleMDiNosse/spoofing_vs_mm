from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from spoofing_detection.lob.execution_clusters import cluster_passive_execution_fills


BASE_TS = datetime(2024, 1, 2, 9, 30, 0)


def event(
    sort_index: int,
    *,
    order_id: str = "O1",
    client_id: str = "C1",
    side: str = "ask",
    price: float = 100.0,
    qty: float = 10.0,
    offset_ms: int = 0,
    partition_id: str = "P1",
    leaves_qty: float = 10.0,
    event_class: str = "fill",
    passive: bool = True,
) -> dict[str, object]:
    return {
        "sort_index": sort_index,
        "partition_id": partition_id,
        "event_class": event_class,
        "is_passive_fill": passive,
        "ORDERID": order_id,
        "client_original_id": client_id,
        "side_label": side,
        "event_price": price,
        "fill_qty": qty,
        "event_ts": BASE_TS + timedelta(milliseconds=offset_ms),
        "LEAVESQTY": leaves_qty,
        "EVENTID": f"E{sort_index}",
        "EXECUTIONID": f"X{sort_index}",
        "TRADEUNIQUEIDENTIFIER": f"T{sort_index}",
        "metric_row": {"pre_state_marker": f"pre-{sort_index}"},
    }


def test_clusters_fragmented_passive_fills_and_preserves_raw_members():
    clusters, members = cluster_passive_execution_fills(
        [
            event(1, qty=9_000, offset_ms=0, leaves_qty=16_000),
            event(2, event_class="fill", passive=False, offset_ms=10),
            event(3, event_class="new_order", order_id="AGGRESSOR", passive=False, offset_ms=15),
            event(4, qty=9_000, offset_ms=25, leaves_qty=7_000),
            event(5, qty=1_947, offset_ms=50, leaves_qty=5_053),
            event(6, qty=857, offset_ms=75, leaves_qty=4_196),
            event(7, qty=4_196, offset_ms=100, leaves_qty=0),
        ],
        max_gap_ms=100,
    )

    assert len(clusters) == 1
    cluster = clusters[0]
    assert cluster["execution_cluster_id"] == "EC000000001-000000007"
    assert cluster["cluster_first_sort_index"] == 1
    assert cluster["cluster_last_sort_index"] == 7
    assert cluster["cluster_start_ts"] == BASE_TS
    assert cluster["cluster_end_ts"] == BASE_TS + timedelta(milliseconds=100)
    assert cluster["child_fill_count"] == 5
    assert cluster["fill_qty"] == pytest.approx(25_000)
    assert cluster["event_price"] == pytest.approx(100.0)
    assert cluster["pre_state_marker"] == "pre-1"
    assert [member["child_sort_index"] for member in members] == [1, 4, 5, 6, 7]
    assert [member["child_fill_qty"] for member in members] == [9_000, 9_000, 1_947, 857, 4_196]
    assert [member["child_order_id"] for member in members] == ["O1"] * 5
    assert {member["execution_cluster_id"] for member in members} == {cluster["execution_cluster_id"]}


def test_gap_boundary_is_inclusive_and_gap_over_threshold_splits():
    clusters, _ = cluster_passive_execution_fills(
        [
            event(10, offset_ms=0, leaves_qty=20),
            event(11, offset_ms=100, leaves_qty=10),
            event(12, offset_ms=201, leaves_qty=0),
        ],
        max_gap_ms=100,
    )

    assert [(row["cluster_first_sort_index"], row["cluster_last_sort_index"]) for row in clusters] == [(10, 11), (12, 12)]


def test_gap_uses_utc_instants_for_timezone_aware_timestamps():
    first = event(13, leaves_qty=10)
    second = event(14, leaves_qty=0)
    first["event_ts"] = datetime(2024, 1, 2, 10, 0, tzinfo=timezone(timedelta(hours=2)))
    second["event_ts"] = datetime(
        2024,
        1,
        2,
        11,
        0,
        0,
        50_000,
        tzinfo=timezone(timedelta(hours=3)),
    )

    clusters, _ = cluster_passive_execution_fills([first, second], max_gap_ms=100)

    assert len(clusters) == 1
    assert clusters[0]["child_fill_count"] == 2


@pytest.mark.parametrize("lifecycle_class", ["modify_order", "cancel"])
def test_incompatible_lifecycle_event_on_same_order_closes_cluster(lifecycle_class):
    clusters, _ = cluster_passive_execution_fills(
        [
            event(20, offset_ms=0, leaves_qty=20),
            event(21, event_class=lifecycle_class, passive=False, offset_ms=5),
            event(22, offset_ms=10, leaves_qty=0),
        ],
        max_gap_ms=100,
    )

    assert [(row["cluster_first_sort_index"], row["cluster_last_sort_index"]) for row in clusters] == [(20, 20), (22, 22)]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("client_id", "C2"),
        ("side", "bid"),
        ("price", 101.0),
        ("partition_id", "P2"),
    ],
)
def test_identity_changes_never_merge(field, value):
    second_kwargs = {field: value}
    clusters, _ = cluster_passive_execution_fills(
        [event(30, offset_ms=0, leaves_qty=10), event(31, offset_ms=1, leaves_qty=0, **second_kwargs)],
        max_gap_ms=100,
    )

    assert len(clusters) == 2


def test_invalid_fill_data_is_excluded_and_terminal_fill_closes_cluster():
    invalid = event(40, qty=0, leaves_qty=5)
    invalid["event_price"] = None
    clusters, members = cluster_passive_execution_fills(
        [invalid, event(41, qty=3, offset_ms=1, leaves_qty=0), event(42, qty=2, offset_ms=2, leaves_qty=0)],
        max_gap_ms=100,
    )

    assert [row["execution_cluster_id"] for row in clusters] == ["EC000000041-000000041", "EC000000042-000000042"]
    assert [member["child_sort_index"] for member in members] == [41, 42]
