from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from spoofing_detection.lob.execution_clusters import (
    classify_execution_anchor,
    cluster_execution_fills,
    cluster_passive_execution_fills,
)


BASE_TS = datetime(2024, 1, 2, 9, 30, 0)


def event(
    sort_index: int,
    *,
    order_id: str = "O1",
    client_id: str | None = "C1",
    side: str = "ask",
    price: float = 100.0,
    qty: float = 10.0,
    offset_ms: int = 0,
    partition_id: str = "P1",
    leaves_qty: float = 10.0,
    event_class: str = "fill",
    passive: bool = True,
    passive_order: str | None = None,
    aggressive_order: str | None = None,
    firm_id: str | None = None,
    execution_sweep_id: str | None = None,
    execution_price_source: str = "active_order_price",
) -> dict[str, object]:
    row: dict[str, object] = {
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
        "execution_price_source": execution_price_source,
    }
    if passive_order is not None:
        row["PASSIVEORDER"] = passive_order
    if aggressive_order is not None:
        row["AGGRESSIVEORDER"] = aggressive_order
        if aggressive_order.upper() == "Y":
            row["LASTTRADEDPX"] = price
            row["LASTSHARES"] = qty
    if firm_id is not None:
        row["firm_id"] = firm_id
    if execution_sweep_id is not None:
        row["execution_sweep_id"] = execution_sweep_id
    return row


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


@pytest.mark.parametrize(
    ("passive_order", "aggressive_order", "expected"),
    [
        ("Y", "N", "passive"),
        (None, "Y", "aggressive"),
        ("Y", "Y", None),
        (None, None, None),
    ],
)
def test_classify_execution_anchor_uses_exclusive_role_flags(passive_order, aggressive_order, expected):
    assert classify_execution_anchor(
        {"PASSIVEORDER": passive_order, "AGGRESSIVEORDER": aggressive_order}
    ) == expected


def test_generic_clusterer_clusters_firm_fallback_without_mislabelling_it_as_client():
    clusters, members = cluster_execution_fills(
        [
            event(50, client_id=None, firm_id="F1", offset_ms=0, leaves_qty=5, passive_order="Y"),
            event(51, client_id=None, firm_id="F1", offset_ms=5, leaves_qty=0, passive_order="Y"),
        ],
        max_gap_ms=100,
    )

    assert len(clusters) == 1
    assert clusters[0]["actor_key"] == "firm:F1"
    assert clusters[0]["actor_id"] == "F1"
    assert clusters[0]["identity_level"] == "firm"
    assert clusters[0]["identity_source"] == "FIRMID"
    assert clusters[0]["identity_fallback_flag"] is True
    assert "client_id" not in clusters[0]
    assert members[0]["actor_key"] == "firm:F1"


def test_legacy_passive_wrapper_preserves_client_id_alias():
    clusters, _ = cluster_passive_execution_fills([event(55, leaves_qty=0)])

    assert clusters[0]["client_id"] == "C1"


def test_generic_clusterer_never_pools_client_and_firm_with_same_raw_identifier():
    clusters, _ = cluster_execution_fills(
        [
            event(60, client_id="same", firm_id="OTHER", offset_ms=0, leaves_qty=5, passive_order="Y"),
            event(61, client_id=None, firm_id="same", offset_ms=1, leaves_qty=0, passive_order="Y"),
        ],
        max_gap_ms=100,
    )

    assert {row["actor_key"] for row in clusters} == {"client_original:same", "firm:same"}
    assert len(clusters) == 2


def test_generic_clusterer_never_pools_passive_and_aggressive_fills():
    clusters, _ = cluster_execution_fills(
        [
            event(70, offset_ms=0, leaves_qty=5, passive_order="Y"),
            event(
                71,
                offset_ms=1,
                leaves_qty=0,
                passive_order="N",
                aggressive_order="Y",
                execution_price_source="LASTTRADEDPX",
            ),
        ],
        max_gap_ms=100,
    )

    assert {(row["execution_anchor_mode"], row["child_fill_count"]) for row in clusters} == {
        ("passive", 1),
        ("aggressive", 1),
    }


def test_generic_clusterer_aggregates_contiguous_aggressive_sweep_at_vwap():
    clusters, members = cluster_execution_fills(
        [
            event(
                80,
                order_id="A1",
                price=100.0,
                qty=2,
                offset_ms=0,
                leaves_qty=5,
                passive_order="N",
                aggressive_order="Y",
                execution_sweep_id="SWEEP-1",
                execution_price_source="LASTTRADEDPX",
            ),
            event(
                81,
                order_id="A1",
                price=101.0,
                qty=3,
                offset_ms=25,
                leaves_qty=0,
                passive_order="N",
                aggressive_order="Y",
                execution_sweep_id="SWEEP-1",
                execution_price_source="LASTTRADEDPX",
            ),
        ],
        max_gap_ms=100,
    )

    assert len(clusters) == 1
    cluster = clusters[0]
    assert cluster["execution_cluster_id"].startswith("EC-A-")
    assert cluster["execution_anchor_mode"] == "aggressive"
    assert cluster["execution_price_source"] == "LASTTRADEDPX"
    assert cluster["child_fill_count"] == 2
    assert cluster["fill_qty"] == pytest.approx(5.0)
    assert cluster["event_price"] == pytest.approx(100.6)
    assert cluster["execution_quantity"] == pytest.approx(5.0)
    assert cluster["execution_vwap"] == pytest.approx(100.6)
    assert cluster["execution_price_level_count"] == 2
    assert cluster["execution_price_min"] == pytest.approx(100.0)
    assert cluster["execution_price_max"] == pytest.approx(101.0)
    assert cluster["passive_execution_diagnostics_applicable"] is False
    assert cluster["aggressive_execution_diagnostics_applicable"] is True
    assert cluster["passive_execution_quantity"] is None
    assert cluster["passive_execution_vwap"] is None
    assert cluster["passive_child_fill_count"] is None
    assert cluster["aggressive_execution_quantity"] == pytest.approx(5.0)
    assert cluster["aggressive_execution_vwap"] == pytest.approx(100.6)
    assert cluster["aggressive_child_fill_count"] == 2
    assert cluster["aggressive_execution_price_level_count"] == 2
    assert cluster["aggressive_execution_price_min"] == pytest.approx(100.0)
    assert cluster["aggressive_execution_price_max"] == pytest.approx(101.0)
    assert cluster["aggressive_execution_sweep_id"] == "SWEEP-1"
    assert [member["child_fill_price"] for member in members] == [100.0, 101.0]


def test_aggressive_cluster_uses_last_traded_price_not_event_price_alias():
    fill = event(
        82,
        price=10.0,
        leaves_qty=0,
        passive_order="N",
        aggressive_order="Y",
        execution_price_source="LASTTRADEDPX",
    )
    fill["LASTTRADEDPX"] = 101.0

    clusters, members = cluster_execution_fills([fill], max_gap_ms=100)

    assert clusters[0]["event_price"] == pytest.approx(101.0)
    assert members[0]["child_fill_price"] == pytest.approx(101.0)
    assert clusters[0]["execution_price_source"] == "LASTTRADEDPX"


def test_aggressive_cluster_requires_last_shares_instead_of_generic_fill_qty():
    fill = event(
        821,
        qty=7.0,
        passive_order="N",
        aggressive_order="Y",
        execution_price_source="LASTTRADEDPX",
    )
    fill.pop("LASTSHARES")

    clusters, members = cluster_execution_fills([fill], max_gap_ms=100)

    assert clusters == []
    assert members == []


def test_aggressive_same_sweep_different_orders_do_not_merge():
    first = event(
        83,
        order_id="A1",
        price=100.0,
        leaves_qty=5,
        passive_order="N",
        aggressive_order="Y",
        execution_sweep_id="SWEEP-1",
    )
    second = event(
        84,
        order_id="A2",
        price=101.0,
        offset_ms=1,
        leaves_qty=0,
        passive_order="N",
        aggressive_order="Y",
        execution_sweep_id="SWEEP-1",
    )
    first["LASTTRADEDPX"] = 100.0
    second["LASTTRADEDPX"] = 101.0

    clusters, _ = cluster_execution_fills([first, second], max_gap_ms=100)

    assert [cluster["event_order_id"] for cluster in clusters] == ["A1", "A2"]
    assert [cluster["child_fill_count"] for cluster in clusters] == [1, 1]


def test_generic_clusterer_is_deterministic_under_input_permutation():
    fills = [
        event(90, offset_ms=0, leaves_qty=5, passive_order="Y"),
        event(91, offset_ms=10, leaves_qty=0, passive_order="Y"),
        event(92, order_id="A2", offset_ms=20, leaves_qty=0, passive_order="N", aggressive_order="Y"),
    ]

    ordered = cluster_execution_fills(fills, max_gap_ms=100)
    permuted = cluster_execution_fills([fills[2], fills[0], fills[1]], max_gap_ms=100)

    assert permuted == ordered


def test_legacy_adapter_accepts_unflagged_fill_rows():
    fill = event(93, leaves_qty=0)
    fill.pop("is_passive_fill")

    clusters, _ = cluster_passive_execution_fills([fill], max_gap_ms=100)

    assert len(clusters) == 1


def test_interleaved_passive_orders_keep_independent_pending_clusters():
    clusters, _ = cluster_passive_execution_fills(
        [
            event(94, order_id="A", offset_ms=0, leaves_qty=5),
            event(95, order_id="B", offset_ms=1, leaves_qty=0),
            event(96, order_id="A", offset_ms=2, leaves_qty=0),
        ],
        max_gap_ms=100,
    )

    by_order = {cluster["event_order_id"]: cluster for cluster in clusters}
    assert by_order["A"]["child_fill_count"] == 2
    assert by_order["B"]["child_fill_count"] == 1


def test_sort_index_is_primary_causal_order_when_timestamps_regress():
    first = event(97, offset_ms=10, leaves_qty=5)
    second = event(98, offset_ms=0, leaves_qty=0)

    clusters, members = cluster_passive_execution_fills([second, first], max_gap_ms=100)

    assert [cluster["cluster_first_sort_index"] for cluster in clusters] == [97, 98]
    assert [member["child_sort_index"] for member in members] == [97, 98]


def test_equal_primary_keys_have_permutation_independent_tiebreak():
    left = event(99, leaves_qty=5)
    right = event(99, leaves_qty=0)
    left["metric_row"] = {"marker": "left"}
    right["metric_row"] = {"marker": "right"}

    forward = cluster_passive_execution_fills([left, right], max_gap_ms=100)
    reverse = cluster_passive_execution_fills([right, left], max_gap_ms=100)

    assert forward == reverse


def test_generic_clusterer_omits_missing_or_ambiguous_actor_and_anchor_rows():
    clusters, members = cluster_execution_fills(
        [
            event(100, client_id=None, firm_id=None, leaves_qty=0, passive_order="Y"),
            event(101, leaves_qty=0, passive_order="Y", aggressive_order="Y"),
        ],
        max_gap_ms=100,
    )

    assert clusters == []
    assert members == []


@pytest.mark.parametrize("max_gap_ms", [0, -1])
def test_generic_clusterer_rejects_non_positive_max_gap(max_gap_ms):
    with pytest.raises(ValueError, match="positive"):
        cluster_execution_fills([], max_gap_ms=max_gap_ms)


@pytest.mark.parametrize("allowed_anchor_modes", [(), ("passive", "unknown"), ("passive", "passive")])
def test_generic_clusterer_rejects_invalid_allowed_anchor_modes(allowed_anchor_modes):
    with pytest.raises(ValueError):
        cluster_execution_fills([], max_gap_ms=100, allowed_anchor_modes=allowed_anchor_modes)
