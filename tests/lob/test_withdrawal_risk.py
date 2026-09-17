from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
import os
import subprocess
import sys

import polars as pl
import pytest

from spoofing_detection.lob.replay_observation import replay_lob
import spoofing_detection.lob.withdrawal_risk as withdrawal_risk_module
from spoofing_detection.lob.withdrawal_risk import (
    build_withdrawal_risk,
    canonical_execution_cluster_rows,
    compute_withdrawal_risk,
    intersect_observed_execution_exposure,
)


def event(seq, event_type, order_id, side, price, leaves, displayed, client="C1", *, ts=None, firm="F1", day="2024-01-02", symbol=123):
    timestamp = ts or f"{day} 09:30:{seq:02d}"
    return {
        "TRADEDATE": day, "MIC": "XMIL", "MARKETCODE": "MTA", "SYMBOLINDEX": symbol, "EMM (*)": 1,
        "SEQUENCETIME": timestamp, "BOOKIN": timestamp, "BOOKOUTTIME": timestamp,
        "TRADETIME": timestamp if event_type == 3 else None,
        "HDR_APPLKEYSEQUENCENUMBER": seq, "HDR_HWMSEQUENCENUMBER": seq, "HDR_OFFSETID": seq,
        "ROW_NUMBER": seq, "EVENTID": f"E{seq}", "ORDEREVENTTYPE (*)": event_type,
        "ORDERID": order_id, "ORDERPRIORITY": str(seq), "ORDERSIDE (*)": side,
        "ORDERPX": price, "ORDERQTY": leaves, "DISPLAYEDQTY": displayed, "LEAVESQTY": leaves,
        "LASTSHARES": None, "LASTTRADEDPX": None, "ORDERTYPE (*)": 2, "TIMEINFORCE (*)": 0,
        "PASSIVEORDER": "Y" if event_type == 3 else None, "AGGRESSIVEORDER": "N",
        "FIRMID": firm, "NMSC_ORIGINALCLIENTIDSHORTCODE": client,
        "ORDER_TRADINGCAPACITY (*)": 3 if client is not None else 1,
    }


def result(rows, **kwargs):
    observations = []
    replay_lob(pl.DataFrame(rows), observer=observations.append)
    return build_withdrawal_risk(observations, top_n=kwargs.pop("top_n", 1), max_order_age_seconds=kwargs.pop("max_age", 60), **kwargs)


@pytest.mark.parametrize("first_bad_clock", [5, None])
def test_clock_quarantine_does_not_resume_below_previous_high_watermark(first_bad_clock):
    observations = []
    replay_lob(pl.DataFrame([
        event(seq, 1 if seq == 1 else 2, "B", 1, 100.0, 10, 10)
        for seq in range(1, 8)
    ]), observer=observations.append)
    origin = datetime(2024, 1, 2, 9, 30)
    clocks = [0, 10, first_bad_clock, 6, 7, 11, 12]
    observations = [replace(obs, event_ts=None if clock is None else origin + timedelta(seconds=clock))
                    for obs, clock in zip(observations, clocks, strict=True)]
    out = build_withdrawal_risk(observations, top_n=1, max_order_age_seconds=60)
    spans = out.intervals.sort("start_ts")
    assert spans.get_column("duration_seconds").sum() == pytest.approx(11.0)
    rows = spans.to_dicts()
    assert all(a["end_ts"] <= b["start_ts"] for a, b in zip(rows, rows[1:]))
    assert out.diagnostics.height == 3


def test_top_n_entry_exit_builds_one_spell_and_disjoint_constant_membership_intervals():
    out = result([
        event(1, 1, "B0", 1, 100.0, 10, 10, "OTHER", ts="2024-01-02 09:30:00"),
        event(2, 1, "B1", 1, 100.1, 5, 5, "C1", ts="2024-01-02 09:30:01"),
        event(3, 2, "B0", 1, 100.2, 10, 10, "OTHER", ts="2024-01-02 09:30:03"),
        event(4, 2, "B0", 1, 100.0, 10, 10, "OTHER", ts="2024-01-02 09:30:05"),
        event(5, 4, "B1", 1, 100.1, 0, 0, "C1", ts="2024-01-02 09:30:07"),
    ])

    spells = out.spells.filter(pl.col("actor_key") == "client_original:C1")
    intervals = out.intervals.filter(pl.col("actor_key") == "client_original:C1")
    assert spells.height == 2
    spell = spells.row(0, named=True)
    assert spell["actor_key"] == "client_original:C1"
    assert spell["side"] == "bid"
    assert spells.get_column("duration_seconds").sum() == pytest.approx(4.0)
    assert intervals.get_column("duration_seconds").to_list() == [2.0, 2.0]
    assert intervals.get_column("member_count").to_list() == [1, 1]
    assert intervals.select(pl.col("duration_seconds").sum()).item() == pytest.approx(spells.get_column("duration_seconds").sum())
    assert out.transitions.filter((pl.col("actor_key") == "client_original:C1") & (pl.col("transition_reason") == "loss_of_eligibility")).height == 1


def test_age_expiry_splits_between_messages_and_is_not_a_withdrawal():
    out = result([
        event(1, 1, "B", 1, 100.0, 10, 10, ts="2024-01-02 09:30:00"),
        event(2, 1, "A", 2, 100.2, 10, 10, "OTHER", ts="2024-01-02 09:30:10"),
    ], max_age=5)

    assert out.intervals.get_column("end_ts").to_list() == [datetime(2024, 1, 2, 9, 30, 5)]
    assert out.spells.item(0, "duration_seconds") == pytest.approx(5.0)
    assert out.withdrawal_events.is_empty()
    assert out.transitions.item(0, "transition_reason") == "age_expiry"


def test_intervals_report_staggered_member_lifecycle_ages_at_each_interval_start():
    out = result([
        event(1, 1, "B1", 1, 100.0, 10, 10, "C1", ts="2024-01-02 09:30:00"),
        event(2, 1, "B2", 1, 99.0, 10, 10, "C1", ts="2024-01-02 09:30:02"),
        event(3, 1, "A", 2, 100.2, 10, 10, "OTHER", ts="2024-01-02 09:30:05"),
        event(4, 4, "B1", 1, 100.0, 0, 0, "C1", ts="2024-01-02 09:30:06"),
    ], top_n=2)

    two_member_intervals = out.intervals.filter(
        (pl.col("actor_key") == "client_original:C1") & (pl.col("member_count") == 2)
    )
    assert two_member_intervals.get_column("start_ts").to_list() == [
        datetime(2024, 1, 2, 9, 30, 2), datetime(2024, 1, 2, 9, 30, 5),
    ]
    assert two_member_intervals.get_column("mean_member_age_seconds").to_list() == [1.0, 4.0]
    assert two_member_intervals.get_column("oldest_member_age_seconds").to_list() == [2.0, 5.0]


def test_future_events_do_not_change_completed_interval_lifecycle_age_covariates():
    prefix = [
        event(1, 1, "B", 1, 100.0, 10, 10, "C1", ts="2024-01-02 09:30:00"),
        event(2, 1, "A", 2, 100.2, 10, 10, "OTHER", ts="2024-01-02 09:30:02"),
        event(4, 4, "B", 1, 100.0, 0, 0, "C1", ts="2024-01-02 09:30:04"),
    ]
    baseline = result(prefix)
    changed_future = result(prefix + [
        event(5, 1, "LATER", 1, 100.0, 10, 10, "C1", ts="2024-01-02 09:30:05"),
        event(6, 4, "LATER", 1, 100.0, 0, 0, "C1", ts="2024-01-02 09:30:06"),
    ])
    fixed_start = datetime(2024, 1, 2, 9, 30, 2)
    fields = ["mean_member_age_seconds", "oldest_member_age_seconds"]

    expected = baseline.intervals.filter(
        (pl.col("actor_key") == "client_original:C1") & (pl.col("start_ts") == fixed_start)
    ).select(fields)
    actual = changed_future.intervals.filter(
        (pl.col("actor_key") == "client_original:C1") & (pl.col("start_ts") == fixed_start)
    ).select(fields)
    assert expected.to_dicts() == actual.to_dicts() == [{
        "mean_member_age_seconds": 2.0,
        "oldest_member_age_seconds": 2.0,
    }]


def test_intervals_carry_the_latest_real_boundary_sort_index_after_equal_clock_events():
    out = result([
        event(1, 1, "B", 1, 100.0, 10, 10, ts="2024-01-02 09:30:00"),
        event(2, 1, "A", 2, 100.2, 10, 10, "OTHER", ts="2024-01-02 09:30:00"),
        event(3, 2, "B", 1, 100.0, 6, 6, ts="2024-01-02 09:30:01"),
        event(4, 4, "B", 1, 100.0, 0, 0, ts="2024-01-02 09:30:02"),
    ])

    intervals = out.intervals.filter(pl.col("actor_key") == "client_original:C1")
    assert intervals.get_column("start_ts").to_list() == [
        datetime(2024, 1, 2, 9, 30), datetime(2024, 1, 2, 9, 30, 1),
    ]
    assert intervals.get_column("start_sort_index").to_list() == [2, 3]


def test_interval_after_synthetic_age_expiry_has_no_observed_boundary_sort_index():
    out = result([
        event(1, 1, "B", 1, 100.1, 10, 10, ts="2024-01-02 09:30:00"),
        event(2, 1, "A", 1, 100.0, 10, 10, ts="2024-01-02 09:30:02"),
        event(3, 1, "X", 2, 100.2, 10, 10, "OTHER", ts="2024-01-02 09:30:06"),
    ], top_n=2, max_age=5)

    intervals = out.intervals.filter(pl.col("actor_key") == "client_original:C1")
    assert intervals.get_column("start_ts").to_list() == [
        datetime(2024, 1, 2, 9, 30),
        datetime(2024, 1, 2, 9, 30, 2),
        datetime(2024, 1, 2, 9, 30, 5),
    ]
    assert intervals.get_column("start_sort_index").to_list() == [1, 2, None]


def test_order_is_expired_at_feed_event_exactly_on_its_age_boundary():
    out = result([
        event(1, 1, "B", 1, 100.0, 10, 10, ts="2024-01-02 09:30:00"),
        event(2, 1, "A", 2, 100.2, 10, 10, "OTHER", ts="2024-01-02 09:30:05"),
    ], max_age=5, coverage_end=datetime(2024, 1, 2, 9, 30, 10))

    order_spell = out.spells.filter(pl.col("actor_key") == "client_original:C1")
    order_intervals = out.intervals.filter(pl.col("actor_key") == "client_original:C1")
    assert order_spell.item(0, "duration_seconds") == pytest.approx(5.0)
    assert order_spell.item(0, "end_reason") == "age_expiry"
    assert order_intervals.get_column("duration_seconds").to_list() == [5.0]


def test_zero_max_age_never_accrues_positive_risk():
    out = result([
        event(1, 1, "B", 1, 100.0, 10, 10, ts="2024-01-02 09:30:00"),
    ], max_age=0, coverage_end=datetime(2024, 1, 2, 9, 30, 1))

    assert out.spells.is_empty()
    assert out.intervals.is_empty()
    assert out.membership.is_empty()


def test_final_coverage_processes_age_expiry_before_closing_the_spell():
    out = result([
        event(1, 1, "B", 1, 100.0, 10, 10, ts="2024-01-02 09:30:00"),
    ], max_age=5, coverage_end=datetime(2024, 1, 2, 9, 30, 10))

    assert out.spells.item(0, "duration_seconds") == pytest.approx(5.0)
    assert out.spells.item(0, "end_ts") == datetime(2024, 1, 2, 9, 30, 5)
    assert out.spells.item(0, "end_reason") == "age_expiry"


def test_partial_modify_keeps_spell_alive_and_splits_exposure_without_withdrawal():
    out = result([
        event(1, 1, "B", 1, 100.0, 10, 10, ts="2024-01-02 09:30:00"),
        event(2, 2, "B", 1, 100.0, 6, 6, ts="2024-01-02 09:30:02"),
        event(3, 4, "B", 1, 100.0, 0, 0, ts="2024-01-02 09:30:04"),
    ])

    assert out.spells.height == 1
    assert out.intervals.get_column("eligible_visible_qty").to_list() == [10.0, 6.0]
    assert out.withdrawal_events.height == 1
    assert out.withdrawal_events.item(0, "visible_qty_removed") == pytest.approx(6.0)
    assert set(out.transitions.get_column("transition_reason")) == {"modify", "cancellation"}


def test_fill_modify_and_unknown_loss_are_competing_non_cancellation_transitions():
    filled = result([
        event(1, 1, "B", 1, 100.0, 10, 10, ts="2024-01-02 09:30:00"),
        event(2, 3, "B", 1, 100.0, 0, 0, ts="2024-01-02 09:30:02"),
    ])
    assert filled.withdrawal_events.is_empty()
    assert filled.transitions.item(0, "transition_reason") == "fill"


def test_orderid_reuse_creates_new_physical_lifecycle_and_cancel_is_counted_once():
    out = result([
        event(1, 1, "B", 1, 100.0, 10, 10, ts="2024-01-02 09:30:00"),
        event(2, 4, "B", 1, 100.0, 0, 0, ts="2024-01-02 09:30:02"),
        event(3, 1, "B", 1, 100.0, 7, 7, ts="2024-01-02 09:30:03"),
        event(4, 4, "B", 1, 100.0, 0, 0, ts="2024-01-02 09:30:05"),
    ])

    assert out.membership.get_column("first_seen_sort_index").to_list() == [1, 3]
    assert out.withdrawal_events.get_column("physical_order_key").n_unique() == 2
    assert out.withdrawal_events.get_column("visible_qty_removed").to_list() == [10.0, 7.0]


def test_partition_boundary_ends_risk_and_client_and_firm_namespaces_remain_distinct():
    out = result([
        event(1, 1, "B", 1, 100.0, 10, 10, "F2", firm="F1", ts="2024-01-02 09:30:00"),
        event(2, 1, "A", 2, 100.2, 10, 10, None, firm="F2", ts="2024-01-02 09:30:02"),
        event(3, 1, "X", 1, 200.0, 10, 10, "C2", symbol=456, ts="2024-01-02 09:30:04"),
    ])

    assert set(out.spells.get_column("actor_key")) == {"client_original:F2", "firm:F2"}
    assert set(out.spells.get_column("end_reason")) == {"partition_boundary"}


def test_missing_clock_on_new_partition_closes_prior_state_without_carrying_it_forward():
    observations = []
    replay_lob(pl.DataFrame([
        event(1, 1, "B", 1, 100.0, 10, 10, ts="2024-01-02 09:30:00"),
        event(2, 1, "A", 2, 100.2, 10, 10, "OTHER", ts="2024-01-02 09:30:01"),
        event(3, 1, "X", 1, 200.0, 10, 10, "C2", symbol=456, ts="2024-01-02 09:30:02"),
        event(4, 1, "Y", 2, 200.2, 10, 10, "OTHER", symbol=456, ts="2024-01-02 09:30:03"),
    ]), observer=observations.append)
    observations[2] = replace(observations[2], event_ts=None)

    out = build_withdrawal_risk(observations, top_n=1, max_order_age_seconds=60)

    old_partition = observations[0].partition_id
    old_spell = out.spells.filter(pl.col("partition_id") == old_partition)
    assert old_spell.item(0, "duration_seconds") == pytest.approx(1.0)
    assert old_spell.item(0, "end_reason") == "partition_boundary"
    assert out.intervals.filter(pl.col("partition_id") == old_partition).get_column("end_ts").max() == datetime(2024, 1, 2, 9, 30, 1)
    assert out.diagnostics.filter(pl.col("reason") == "missing_event_timestamp").height == 1


def test_scalar_coverage_end_does_not_extend_intermediate_partition_past_next_partition():
    out = result([
        event(1, 1, "B", 1, 100.0, 10, 10, ts="2024-01-02 09:30:00"),
        event(2, 1, "X", 1, 200.0, 10, 10, "C2", symbol=456, ts="2024-01-02 09:30:05"),
    ], max_age=60, coverage_end=datetime(2024, 1, 2, 9, 30, 10))

    old_partition = out.spells.item(0, "partition_id")
    old_spell = out.spells.filter(pl.col("partition_id") == old_partition)
    assert old_spell.item(0, "duration_seconds") == pytest.approx(5.0)
    assert old_spell.item(0, "end_ts") == datetime(2024, 1, 2, 9, 30, 5)
    assert old_spell.item(0, "end_reason") == "partition_boundary"


def test_mapping_coverage_end_after_next_partition_is_capped_at_partition_boundary():
    observations = []
    replay_lob(pl.DataFrame([
        event(1, 1, "B", 1, 100.0, 10, 10, ts="2024-01-02 09:30:00"),
        event(2, 1, "X", 1, 200.0, 10, 10, "C2", symbol=456, ts="2024-01-02 09:30:05"),
    ]), observer=observations.append)
    out = build_withdrawal_risk(observations, top_n=1, max_order_age_seconds=60, coverage_end={
        str(observations[0].partition_id): datetime(2024, 1, 2, 9, 30, 10),
        str(observations[1].partition_id): datetime(2024, 1, 2, 9, 30, 10),
    })

    old_spell = out.spells.filter(pl.col("partition_id") == observations[0].partition_id).row(0, named=True)
    assert old_spell["duration_seconds"] == pytest.approx(5.0)
    assert old_spell["end_ts"] == datetime(2024, 1, 2, 9, 30, 5)


def test_zero_denominator_is_null_and_clock_regression_is_quarantined_without_sorting():
    observations = []
    replay_lob(pl.DataFrame([
        event(1, 1, "B", 1, 100.0, 10, 10, ts="2024-01-02 09:30:02"),
        event(2, 4, "B", 1, 100.0, 0, 0, ts="2024-01-02 09:30:03"),
    ]), observer=observations.append)
    observations[1] = replace(observations[1], event_ts=datetime(2024, 1, 2, 9, 30, 1))
    out = build_withdrawal_risk(observations, top_n=1, max_order_age_seconds=60)

    assert out.spells.is_empty()
    assert out.diagnostics.item(0, "reason") == "clock_regression"
    assert out.actor_summary.is_empty()


@pytest.mark.parametrize(
    ("ambiguous_ts", "diagnostic_reason"),
    [
        (None, "missing_event_timestamp"),
        (datetime(2024, 1, 2, 9, 30, 0), "clock_regression"),
    ],
)
def test_same_partition_ambiguous_clock_censors_risk_before_resuming_left_truncated_coverage(
    ambiguous_ts, diagnostic_reason,
):
    observations = []
    replay_lob(pl.DataFrame([
        event(1, 1, "B", 1, 100.0, 10, 10, "C1", ts="2024-01-02 09:30:00"),
        event(2, 1, "A", 2, 100.2, 10, 10, "OTHER", ts="2024-01-02 09:30:01"),
        event(3, 2, "B", 1, 100.0, 6, 6, "C1", ts="2024-01-02 09:30:02"),
        event(4, 1, "X", 2, 100.3, 10, 10, "OTHER", ts="2024-01-02 09:30:04"),
    ]), observer=observations.append)
    observations[2] = replace(observations[2], event_ts=ambiguous_ts)

    out = build_withdrawal_risk(
        observations,
        top_n=1,
        max_order_age_seconds=60,
        coverage_end=datetime(2024, 1, 2, 9, 30, 5),
    )

    c1_spells = out.spells.filter(pl.col("actor_key") == "client_original:C1")
    c1_intervals = out.intervals.filter(pl.col("actor_key") == "client_original:C1")
    c1_membership = out.membership.filter(pl.col("actor_key") == "client_original:C1")
    assert c1_spells.get_column("duration_seconds").to_list() == [1.0, 1.0]
    assert c1_spells.get_column("end_reason").to_list() == ["clock_ambiguous", "coverage_end"]
    assert c1_intervals.get_column("duration_seconds").to_list() == [1.0, 1.0]
    assert c1_intervals.get_column("eligible_visible_qty").to_list() == [10.0, 6.0]
    assert c1_membership.get_column("eligible_start_ts").to_list() == [
        datetime(2024, 1, 2, 9, 30, 0), datetime(2024, 1, 2, 9, 30, 4),
    ]
    assert c1_membership.get_column("duration_seconds").to_list() == [1.0, 1.0]
    assert c1_membership.get_column("end_reason").to_list() == ["clock_ambiguous", "coverage_end"]
    assert out.withdrawal_events.is_empty()
    assert out.diagnostics.filter(pl.col("reason") == diagnostic_reason).height == 1


def test_retained_physical_order_splits_membership_when_canonical_actor_identity_changes():
    out = result([
        event(1, 1, "B", 1, 100.0, 10, 10, "C1", ts="2024-01-02 09:30:00"),
        event(2, 1, "A", 2, 100.2, 10, 10, "OTHER", ts="2024-01-02 09:30:01"),
        event(3, 2, "B", 1, 100.0, 10, 10, "C2", ts="2024-01-02 09:30:02"),
        event(4, 4, "B", 1, 100.0, 0, 0, "C2", ts="2024-01-02 09:30:04"),
    ])

    membership = out.membership.filter(pl.col("order_id") == "B")
    assert membership.get_column("actor_key").to_list() == ["client_original:C1", "client_original:C2"]
    assert membership.get_column("eligible_start_ts").to_list() == [
        datetime(2024, 1, 2, 9, 30, 0), datetime(2024, 1, 2, 9, 30, 2),
    ]
    assert membership.get_column("duration_seconds").to_list() == [2.0, 2.0]
    assert membership.get_column("end_reason").to_list() == ["identity_change", "cancellation"]
    c1_spells = out.spells.filter(pl.col("actor_key") == "client_original:C1")
    c2_spells = out.spells.filter(pl.col("actor_key") == "client_original:C2")
    assert c1_spells.get_column("duration_seconds").to_list() == [2.0]
    assert c1_spells.item(0, "end_reason") == "identity_change"
    assert c2_spells.get_column("duration_seconds").to_list() == [2.0]
    assert c2_spells.item(0, "end_reason") == "cancellation"
    assert out.transitions.filter(pl.col("transition_reason") == "identity_change").height == 1


def test_empty_outputs_have_explicit_schemas_and_positive_durations_only():
    out = build_withdrawal_risk([], top_n=1, max_order_age_seconds=1)
    assert out.spells.schema["risk_spell_id"] == pl.String
    assert out.intervals.schema["duration_seconds"] == pl.Float64
    assert out.intervals.schema["mean_member_age_seconds"] == pl.Float64
    assert out.intervals.schema["oldest_member_age_seconds"] == pl.Float64
    assert out.membership.schema["physical_order_key"] == pl.String
    assert out.withdrawal_events.schema["visible_qty_removed"] == pl.Float64
    assert out.transitions.schema["transition_reason"] == pl.String


@pytest.mark.parametrize("field, value", [
    ("mean_member_age_seconds", -1.0),
    ("mean_member_age_seconds", float("nan")),
    ("oldest_member_age_seconds", -1.0),
    ("oldest_member_age_seconds", float("inf")),
])
def test_risk_interval_invariants_reject_invalid_lifecycle_age_covariates(field, value):
    out = result([
        event(1, 1, "B", 1, 100.0, 10, 10, ts="2024-01-02 09:30:00"),
    ], coverage_end=datetime(2024, 1, 2, 9, 30, 1))
    intervals = out.intervals.with_columns(pl.lit(value).alias(field))

    with pytest.raises(AssertionError, match="lifecycle-age covariates must be finite and non-negative"):
        withdrawal_risk_module._assert_invariants(out.spells, intervals, out.withdrawal_events)


@pytest.mark.parametrize("top_n", [True, False, 0, -1, 1.0, "1"])
def test_top_n_requires_an_exact_positive_integer(top_n):
    with pytest.raises(ValueError, match="top_n must be a positive integer"):
        build_withdrawal_risk([], top_n=top_n, max_order_age_seconds=1)


@pytest.mark.parametrize("max_age", [True, False, -1, float("nan"), float("inf"), float("-inf"), "1"])
def test_max_order_age_requires_a_finite_nonnegative_real(max_age):
    with pytest.raises(ValueError, match="max_order_age_seconds must be a finite non-negative real"):
        build_withdrawal_risk([], top_n=1, max_order_age_seconds=max_age)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("price", float("nan")),
        ("price", float("inf")),
        ("leaves_qty", float("nan")),
        ("leaves_qty", float("inf")),
        ("displayed_qty", -1.0),
        ("displayed_qty", float("inf")),
    ],
)
def test_invalid_book_numeric_values_are_quarantined_from_risk_outputs(field, value):
    observations = []
    replay_lob(pl.DataFrame([
        event(1, 1, "B", 1, 100.0, 10, 10, ts="2024-01-02 09:30:00"),
    ]), observer=observations.append)
    invalid_order = replace(observations[0].post_active_orders["B"], **{field: value})
    observations[0] = replace(observations[0], post_active_orders={"B": invalid_order})

    out = build_withdrawal_risk(
        observations, top_n=1, max_order_age_seconds=60,
        coverage_end=datetime(2024, 1, 2, 9, 30, 1),
    )

    assert out.spells.is_empty()
    assert out.intervals.is_empty()
    assert out.membership.is_empty()
    assert out.withdrawal_events.is_empty()
    diagnostic = out.diagnostics.row(0, named=True)
    assert diagnostic["reason"] == "invalid_order_numeric"
    assert field in diagnostic["detail"]


def test_build_withdrawal_risk_consumes_a_one_shot_observation_iterable():
    observations = []
    replay_lob(pl.DataFrame([
        event(1, 1, "B", 1, 100.0, 10, 10, ts="2024-01-02 09:30:00"),
    ]), observer=observations.append)

    class OneShot:
        def __init__(self, values):
            self.values = values
            self.iterations = 0

        def __iter__(self):
            self.iterations += 1
            if self.iterations > 1:
                raise AssertionError("observations were iterated more than once")
            yield from self.values

    source = OneShot(observations)
    out = build_withdrawal_risk(source, top_n=1, max_order_age_seconds=60)

    assert source.iterations == 1
    assert out.spells.is_empty()


def test_compute_withdrawal_risk_passes_an_incremental_accumulator_to_replay(monkeypatch):
    observed = []

    def fake_replay(raw_events, *, observer, **kwargs):
        observed.append(observer)
        assert getattr(observer, "__name__", None) == "observe"
        return ()

    monkeypatch.setattr("spoofing_detection.lob.withdrawal_risk.replay_lob", fake_replay)

    out = compute_withdrawal_risk(pl.DataFrame(), top_n=1, max_order_age_seconds=60)

    assert len(observed) == 1
    assert out.spells.is_empty()


def test_raw_replay_output_is_stable_across_python_hash_seeds():
    script = '''
import json
import polars as pl
from spoofing_detection.lob.withdrawal_risk import compute_withdrawal_risk

def event(seq, order_id, side, client):
    ts = f"2024-01-02 09:30:0{seq}"
    return {
        "TRADEDATE": "2024-01-02", "MIC": "XMIL", "MARKETCODE": "MTA", "SYMBOLINDEX": 123, "EMM (*)": 1,
        "SEQUENCETIME": ts, "BOOKIN": ts, "BOOKOUTTIME": ts, "TRADETIME": None,
        "HDR_APPLKEYSEQUENCENUMBER": seq, "HDR_HWMSEQUENCENUMBER": seq, "HDR_OFFSETID": seq,
        "ROW_NUMBER": seq, "EVENTID": f"E{seq}", "ORDEREVENTTYPE (*)": 1, "ORDERID": order_id,
        "ORDERPRIORITY": str(seq), "ORDERSIDE (*)": side, "ORDERPX": 100.0 + seq,
        "ORDERQTY": 10, "DISPLAYEDQTY": 10, "LEAVESQTY": 10, "LASTSHARES": None,
        "LASTTRADEDPX": None, "ORDERTYPE (*)": 2, "TIMEINFORCE (*)": 0,
        "PASSIVEORDER": None, "AGGRESSIVEORDER": "N", "FIRMID": "F1",
        "NMSC_ORIGINALCLIENTIDSHORTCODE": client, "ORDER_TRADINGCAPACITY (*)": 3,
    }

out = compute_withdrawal_risk(pl.DataFrame([
    event(1, "B", 1, "C1"), event(2, "A", 2, "C2"), event(3, "X", 1, "C3"),
]), top_n=1, max_order_age_seconds=60)
print(json.dumps({name: getattr(out, name).to_dicts() for name in ("spells", "intervals", "membership", "withdrawal_events", "transitions", "diagnostics", "actor_summary")}, default=str, sort_keys=True))
'''
    outputs = []
    for seed in ("1", "2"):
        env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": f"{os.getcwd()}/src"}
        completed = subprocess.run(
            [sys.executable, "-c", script], cwd=os.getcwd(), env=env,
            check=True, capture_output=True, text=True,
        )
        outputs.append(json.loads(completed.stdout))

    assert outputs[0] == outputs[1]


def execution_cluster(
    cluster_id, *, actor_key="client_original:C1", level="client_original", side="bid",
    mode="passive", qty=3.0, end_ts=datetime(2024, 1, 2, 9, 30, 1), last_sort_index=1,
    sweep_id=None, partition_id="P",
):
    actor_id = actor_key.partition(":")[2]
    return {
        "execution_cluster_id": cluster_id,
        "partition_id": partition_id,
        "actor_key": actor_key,
        "actor_id": actor_id,
        "identity_level": level,
        "identity_source": "NMSC_ORIGINALCLIENTIDSHORTCODE" if level == "client_original" else "FIRMID",
        "identity_fallback_flag": level == "firm",
        "execution_anchor_mode": mode,
        "execution_quantity": qty,
        "event_side": side,
        "cluster_end_ts": end_ts,
        "cluster_last_sort_index": last_sort_index,
        "execution_sweep_id": sweep_id,
    }


def test_observed_execution_exposure_same_timestamp_outcome_requires_later_canonical_sort_order():
    risk = result([
        event(1, 1, "B", 1, 100.0, 10, 10, "C1", ts="2024-01-02 09:30:00"),
        event(2, 4, "B", 1, 100.0, 0, 0, "C1", ts="2024-01-02 09:30:02"),
    ])
    # The preceding same-clock cancel is not subsequent to the anchor.
    partition_id = risk.intervals.item(0, "partition_id")
    before = execution_cluster("before", partition_id=partition_id, end_ts=datetime(2024, 1, 2, 9, 30, 2), last_sort_index=3)
    later = execution_cluster("later", partition_id=partition_id, end_ts=datetime(2024, 1, 2, 9, 30, 2), last_sort_index=1)

    out = intersect_observed_execution_exposure(risk, [before, later], withdrawal_window_seconds=2)

    # The fixed windows have no positive overlap with this risk interval, but
    # the later canonical cluster is active at the cancellation point.  The
    # outcome therefore belongs to the own-only stratum exactly once.
    assert out.exposure_intervals.is_empty()
    assert out.exposure_contrasts.item(0, "contrast_label") == "own_only"
    assert out.exposure_contrasts.item(0, "withdrawal_event_id") == "withdrawal-1"
    assert out.exposure_contrasts.item(0, "execution_anchor_mode") == "passive"
    assert out.exposure_contrasts.item(0, "execution_quantity") == pytest.approx(3.0)
    assert out.exposure_strata.item(0, "withdrawal_count") == 1


def test_observed_execution_exposure_mask_preserves_mixed_roles_and_dedupes_canonical_sweep_rows():
    risk = result([
        event(1, 1, "B", 1, 100.0, 10, 10, "C1", ts="2024-01-02 09:30:00"),
        event(5, 4, "B", 1, 100.0, 0, 0, "C1", ts="2024-01-02 09:30:05"),
    ])
    partition_id = risk.intervals.item(0, "partition_id")
    own_passive = execution_cluster("p", partition_id=partition_id, end_ts=datetime(2024, 1, 2, 9, 30, 1), sweep_id="S")
    duplicate_sweep = execution_cluster("p-duplicate", partition_id=partition_id, end_ts=datetime(2024, 1, 2, 9, 30, 1), sweep_id="S")
    other_aggressive = execution_cluster(
        "a", partition_id=partition_id, actor_key="client_original:C2", mode="aggressive", end_ts=datetime(2024, 1, 2, 9, 30, 3), last_sort_index=1,
    )

    out = intersect_observed_execution_exposure(
        risk, [own_passive, duplicate_sweep, other_aggressive], withdrawal_window_seconds=3,
    )

    rows = out.exposure_intervals.to_dicts()
    assert [row["duration_seconds"] for row in rows] == [2.0, 1.0, 1.0]
    assert rows[0]["execution_cluster_ids"] == "p"
    assert rows[0]["own_passive"] is True
    assert rows[0]["other_aggressive"] is False
    assert rows[1]["own_passive"] is True
    assert rows[1]["other_aggressive"] is True
    assert rows[1]["contrast_label"] == "mixed"
    assert rows[1]["execution_quantity"] == pytest.approx(6.0)
    assert rows[2]["contrast_label"] == "other_only"
    contrasts = out.exposure_contrasts.to_dicts()
    assert [(row["start_ts"], row["end_ts"], row["contrast_label"]) for row in contrasts] == [
        (datetime(2024, 1, 2, 9, 30, 0), datetime(2024, 1, 2, 9, 30, 1), "no_qualifying_fill"),
        (datetime(2024, 1, 2, 9, 30, 1), datetime(2024, 1, 2, 9, 30, 3), "own_only"),
        (datetime(2024, 1, 2, 9, 30, 3), datetime(2024, 1, 2, 9, 30, 4), "mixed"),
        (datetime(2024, 1, 2, 9, 30, 4), datetime(2024, 1, 2, 9, 30, 5), "other_only"),
    ]
    assert sum(row["duration_seconds"] for row in contrasts) == pytest.approx(5.0)


def test_observed_execution_exposure_treats_client_firm_comparison_as_ambiguous_not_other_and_keeps_no_fill_branch_null():
    risk = result([
        event(1, 1, "B", 1, 100.0, 10, 10, "C1", firm="F1", ts="2024-01-02 09:30:00"),
        event(4, 4, "B", 1, 100.0, 0, 0, "C1", firm="F1", ts="2024-01-02 09:30:04"),
    ])
    ambiguous = execution_cluster(
        "firm", partition_id=risk.intervals.item(0, "partition_id"), actor_key="firm:F1", level="firm", end_ts=datetime(2024, 1, 2, 9, 30, 1), last_sort_index=1,
    )
    out = intersect_observed_execution_exposure(risk, [ambiguous], withdrawal_window_seconds=2)

    assert out.exposure_intervals.is_empty()
    no_fill = out.exposure_contrasts.filter(pl.col("contrast_label") == "no_qualifying_fill").row(0, named=True)
    assert no_fill["execution_anchor_mode"] is None
    assert no_fill["execution_quantity"] is None
    assert no_fill["identity_ambiguous_cluster_count"] == 0


def test_fixed_horizon_exposure_is_independent_of_future_withdrawal_at_fixed_risk_coverage():
    covered = result([
        event(1, 1, "B", 1, 100.0, 10, 10, "C1", ts="2024-01-02 09:30:00"),
        event(2, 1, "A", 2, 100.2, 10, 10, "OTHER", ts="2024-01-02 09:30:01"),
    ], coverage_end=datetime(2024, 1, 2, 9, 30, 10))
    cancellation = result([
        event(1, 1, "B", 1, 100.0, 10, 10, "C1", ts="2024-01-02 09:30:00"),
        event(2, 1, "A", 2, 100.2, 10, 10, "OTHER", ts="2024-01-02 09:30:01"),
        event(8, 4, "B", 1, 100.0, 0, 0, "C1", ts="2024-01-02 09:30:08"),
    ])
    cluster = execution_cluster(
        "p", partition_id=covered.intervals.item(0, "partition_id"),
        end_ts=datetime(2024, 1, 2, 9, 30, 2), last_sort_index=2,
    )
    early = replace(
        covered,
        withdrawal_events=cancellation.withdrawal_events.with_columns(
            pl.lit(datetime(2024, 1, 2, 9, 30, 6)).alias("event_ts"),
        ),
    )
    late = replace(covered, withdrawal_events=cancellation.withdrawal_events)

    first = intersect_observed_execution_exposure(early, [cluster], withdrawal_window_seconds=4)
    second = intersect_observed_execution_exposure(late, [cluster], withdrawal_window_seconds=4)

    fields = ["start_ts", "end_ts", "duration_seconds", "exposure_mask"]
    assert first.exposure_intervals.select(fields).to_dicts() == second.exposure_intervals.select(fields).to_dicts() == [{
        "start_ts": datetime(2024, 1, 2, 9, 30, 2),
        "end_ts": datetime(2024, 1, 2, 9, 30, 6),
        "duration_seconds": 4.0,
        "exposure_mask": "own_passive",
    }]


def test_contrasts_partition_each_risk_interval_and_assign_late_withdrawal_to_no_fill_once():
    risk = result([
        event(1, 1, "B", 1, 100.0, 10, 10, "C1", ts="2024-01-02 09:30:00"),
        event(10, 4, "B", 1, 100.0, 0, 0, "C1", ts="2024-01-02 09:30:10"),
    ])
    cluster = execution_cluster(
        "p", partition_id=risk.intervals.item(0, "partition_id"),
        end_ts=datetime(2024, 1, 2, 9, 30, 2), last_sort_index=2,
    )

    out = intersect_observed_execution_exposure(risk, [cluster], withdrawal_window_seconds=3)

    contrasts = out.exposure_contrasts.to_dicts()
    assert [(row["start_ts"], row["end_ts"], row["contrast_label"]) for row in contrasts] == [
        (datetime(2024, 1, 2, 9, 30, 0), datetime(2024, 1, 2, 9, 30, 2), "no_qualifying_fill"),
        (datetime(2024, 1, 2, 9, 30, 2), datetime(2024, 1, 2, 9, 30, 5), "own_only"),
        (datetime(2024, 1, 2, 9, 30, 5), datetime(2024, 1, 2, 9, 30, 10), "no_qualifying_fill"),
    ]
    assert sum(row["duration_seconds"] for row in contrasts) == pytest.approx(risk.intervals.item(0, "duration_seconds"))
    assert [row["withdrawal_event_id"] for row in contrasts if row["withdrawal_event_id"]] == ["withdrawal-1"]
    assert out.exposure_strata.get_column("withdrawal_count").sum() == 1


def test_cancellation_without_cluster_is_a_no_fill_outcome_once():
    risk = result([
        event(1, 1, "B", 1, 100.0, 10, 10, "C1", ts="2024-01-02 09:30:00"),
        event(4, 4, "B", 1, 100.0, 0, 0, "C1", ts="2024-01-02 09:30:04"),
    ])

    out = intersect_observed_execution_exposure(risk, [], withdrawal_window_seconds=2)

    row = out.exposure_contrasts.row(0, named=True)
    assert row["contrast_label"] == "no_qualifying_fill"
    assert row["withdrawal_event_id"] == "withdrawal-1"
    assert row["execution_anchor_mode"] is None
    assert row["execution_quantity"] is None
    assert out.exposure_strata.item(0, "withdrawal_count") == 1


def test_ambiguous_windows_create_segment_local_boundaries_without_own_or_other_flags():
    risk = result([
        event(1, 1, "B", 1, 100.0, 10, 10, "C1", firm="F1", ts="2024-01-02 09:30:00"),
        event(8, 4, "B", 1, 100.0, 0, 0, "C1", firm="F1", ts="2024-01-02 09:30:08"),
    ])
    partition_id = risk.intervals.item(0, "partition_id")
    ambiguous = execution_cluster(
        "firm", partition_id=partition_id, actor_key="firm:F1", level="firm",
        end_ts=datetime(2024, 1, 2, 9, 30, 2), last_sort_index=2,
    )
    own = execution_cluster(
        "own", partition_id=partition_id, end_ts=datetime(2024, 1, 2, 9, 30, 4), last_sort_index=4,
    )

    out = intersect_observed_execution_exposure(risk, [ambiguous, own], withdrawal_window_seconds=2)

    rows = out.exposure_contrasts.to_dicts()
    assert [(row["duration_seconds"], row["contrast_label"], row["identity_ambiguous_cluster_count"]) for row in rows] == [
        (2.0, "no_qualifying_fill", 0),
        (2.0, "no_qualifying_fill", 1),
        (2.0, "own_only", 0),
        (2.0, "no_qualifying_fill", 0),
    ]
    assert out.exposure_intervals.get_column("exposure_mask").to_list() == ["own_passive"]


@pytest.mark.parametrize("window", [True, False, 0, -1, float("nan"), float("inf"), "2"])
def test_execution_exposure_rejects_non_positive_or_non_finite_fixed_horizon(window):
    risk = result([
        event(1, 1, "B", 1, 100.0, 10, 10, ts="2024-01-02 09:30:00"),
    ], coverage_end=datetime(2024, 1, 2, 9, 30, 1))

    with pytest.raises(ValueError, match="withdrawal_window_seconds must be a finite positive real"):
        intersect_observed_execution_exposure(risk, [], withdrawal_window_seconds=window)


def test_builder_and_compute_reject_execution_clusters_without_an_explicit_fixed_horizon():
    observations = []
    replay_lob(pl.DataFrame([
        event(1, 1, "B", 1, 100.0, 10, 10, ts="2024-01-02 09:30:00"),
    ]), observer=observations.append)

    with pytest.raises(ValueError, match="withdrawal_window_seconds is required when execution_clusters are supplied"):
        build_withdrawal_risk(observations, top_n=1, max_order_age_seconds=60, execution_clusters=[])
    with pytest.raises(ValueError, match="withdrawal_window_seconds is required when execution_clusters are supplied"):
        compute_withdrawal_risk(pl.DataFrame(), top_n=1, max_order_age_seconds=60, execution_clusters=[])


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("execution_cluster_id", "", "execution_cluster_id"),
        ("partition_id", "", "partition_id"),
        ("actor_key", "", "actor_key"),
        ("actor_id", "", "actor_id"),
        ("identity_level", "unknown", "identity_level"),
        ("execution_anchor_mode", "market", "execution_anchor_mode"),
        ("event_side", "buy", "event_side"),
        ("cluster_end_ts", "2024-01-02T09:30:01", "cluster_end_ts"),
        ("cluster_last_sort_index", True, "cluster_last_sort_index"),
        ("cluster_last_sort_index", 0, "cluster_last_sort_index"),
        ("execution_quantity", -1.0, "execution_quantity"),
        ("execution_quantity", float("nan"), "execution_quantity"),
    ],
)
def test_execution_exposure_rejects_malformed_cluster_rows_loudly(field, value, message):
    cluster = execution_cluster("EC1")
    cluster[field] = value

    with pytest.raises(ValueError, match=message):
        intersect_observed_execution_exposure(
            build_withdrawal_risk([], top_n=1, max_order_age_seconds=60),
            [cluster],
            withdrawal_window_seconds=1,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("actor_id", "NOT_C1"),
        ("actor_key", "firm:C1"),
        ("identity_source", "FIRMID"),
        ("identity_fallback_flag", True),
    ],
)
def test_execution_exposure_rejects_incoherent_canonical_identity(field, value):
    cluster = execution_cluster("EC1")
    cluster[field] = value

    with pytest.raises(ValueError, match="canonical identity is inconsistent"):
        intersect_observed_execution_exposure(
            build_withdrawal_risk([], top_n=1, max_order_age_seconds=60),
            [cluster],
            withdrawal_window_seconds=1,
        )


def test_execution_exposure_rejects_non_mapping_cluster_rows_loudly():
    with pytest.raises(ValueError, match="row must be a mapping"):
        intersect_observed_execution_exposure(
            build_withdrawal_risk([], top_n=1, max_order_age_seconds=60),
            ["not-a-cluster"],
            withdrawal_window_seconds=1,
        )


def test_execution_exposure_rejects_conflicting_duplicate_cluster_id():
    first = execution_cluster("EC1")
    conflicting = {**first, "execution_quantity": 4.0}

    with pytest.raises(ValueError, match="conflicting duplicate execution_cluster_id.*EC1"):
        intersect_observed_execution_exposure(
            build_withdrawal_risk([], top_n=1, max_order_age_seconds=60),
            [first, conflicting],
            withdrawal_window_seconds=1,
        )


def test_execution_exposure_dedupes_exact_duplicate_cluster_rows():
    risk = result([
        event(1, 1, "B", 1, 100.0, 10, 10, "C1", ts="2024-01-02 09:30:00"),
        event(3, 4, "B", 1, 100.0, 0, 0, "C1", ts="2024-01-02 09:30:03"),
    ])
    cluster = execution_cluster(
        "EC1", partition_id=risk.intervals.item(0, "partition_id"),
        end_ts=datetime(2024, 1, 2, 9, 30, 1),
    )

    out = intersect_observed_execution_exposure(risk, [cluster, dict(cluster)], withdrawal_window_seconds=1)

    assert out.exposure_intervals.item(0, "execution_cluster_ids") == "EC1"
    assert out.exposure_intervals.item(0, "execution_quantity") == pytest.approx(3.0)


def test_execution_exposure_does_not_scan_windows_from_unrelated_strata(monkeypatch):
    risk = result([
        event(1, 1, "B", 1, 100.0, 10, 10, "C1", ts="2024-01-02 09:30:00"),
        event(3, 4, "B", 1, 100.0, 0, 0, "C1", ts="2024-01-02 09:30:03"),
    ])
    partition_id = risk.intervals.item(0, "partition_id")
    relevant = execution_cluster("relevant", partition_id=partition_id)
    unrelated = [
        execution_cluster(f"other-{index}", partition_id=f"unrelated-{index}")
        for index in range(20)
    ]
    observed_ids = []
    original = withdrawal_risk_module._window_overlaps_interval

    def record_candidate(window, interval):
        observed_ids.append(window["cluster"]["execution_cluster_id"])
        return original(window, interval)

    monkeypatch.setattr(withdrawal_risk_module, "_window_overlaps_interval", record_candidate)

    intersect_observed_execution_exposure(risk, [relevant, *unrelated], withdrawal_window_seconds=1)

    assert observed_ids == ["relevant"]


def test_public_canonical_execution_cluster_rows_normalizes_aware_anchor_without_mutating_input():
    source = execution_cluster(
        "aware",
        end_ts=datetime(2024, 1, 2, 11, 30, 1, tzinfo=timezone(timedelta(hours=2))),
    )

    normalized = canonical_execution_cluster_rows([source])

    assert normalized[0]["cluster_end_ts"] == datetime(2024, 1, 2, 9, 30, 1)
    assert source["cluster_end_ts"].tzinfo == timezone(timedelta(hours=2))
