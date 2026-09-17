from __future__ import annotations

from datetime import date, datetime, timedelta

import polars as pl
import pytest

from spoofing_detection.lob import empirical_controls_v2 as controls_v2
from spoofing_detection.lob.empirical_controls_v2 import analyze_controls_v2
from spoofing_detection.lob.withdrawal_risk_v2 import compute_compact_risk

T0 = datetime(2024, 1, 2, 9, 0)
ACTOR = {"actor_key": "client_original:A", "actor_id": "A", "identity_level": "client_original", "identity_source": "NMSC_ORIGINALCLIENTIDSHORTCODE", "identity_fallback_flag": False}
OTHER = {"actor_key": "client_original:B", "actor_id": "B", "identity_level": "client_original", "identity_source": "NMSC_ORIGINALCLIENTIDSHORTCODE", "identity_fallback_flag": False}


def interval(start, end, *, actor=ACTOR, side="bid", epoch="e1"):
    return {"instrument": "XYZ", "partition_id": "P", "event_date": date(2024, 1, 2), **actor, "side": side, "coverage_epoch_id": epoch, "start_ts": T0 + timedelta(seconds=start), "end_ts": T0 + timedelta(seconds=end), "member_count": 2, "eligible_visible_qty": 10.0, "mean_member_age_seconds": 4.0, "oldest_member_age_seconds": 7.0}


def withdrawal(name, second, sort, *, actor=ACTOR, side="bid", epoch="e1"):
    return {"instrument": "XYZ", "partition_id": "P", "event_date": date(2024, 1, 2), **actor, "side": side, "coverage_epoch_id": epoch, "withdrawal_event_id": name, "sort_index": sort, "event_ts": T0 + timedelta(seconds=second), "physical_order_key": name, "visible_qty_removed": 1.0}


def cluster(name, second, sort, mode, *, actor=ACTOR, side="ask"):
    return {"instrument": "XYZ", "partition_id": "P", "event_date": date(2024, 1, 2), **actor, "event_side": side, "execution_anchor_mode": mode, "cluster_end_ts": T0 + timedelta(seconds=second), "cluster_last_sort_index": sort, "execution_cluster_id": name, "execution_quantity": 1.0}


def coverage(start=0, end=180, epoch="e1"):
    return {"instrument": "XYZ", "partition_id": "P", "event_date": date(2024, 1, 2), "coverage_epoch_id": epoch, "start_ts": T0 + timedelta(seconds=start), "end_ts": T0 + timedelta(seconds=end), "end_reason": "fixture"}


INTERVAL_COLUMNS = tuple(interval(0, 1))
WITHDRAWAL_COLUMNS = tuple(withdrawal("schema", 0, 0))
CLUSTER_COLUMNS = tuple(cluster("schema", 0, 0, "passive"))
COVERAGE_COLUMNS = tuple(coverage())
MARKET_COLUMNS = ("instrument", "partition_id", "event_date", "coverage_epoch_id", "start_ts", "end_ts", "spread", "depth", "prior_event_count_60s")


def frame(rows, columns):
    return pl.DataFrame(rows) if rows else pl.DataFrame(schema={column: pl.String for column in columns})


def run(intervals, withdrawals=(), clusters=(), coverage_rows=None, market_intervals=None):
    return analyze_controls_v2(
        frame(intervals, INTERVAL_COLUMNS),
        frame(withdrawals, WITHDRAWAL_COLUMNS),
        frame(clusters, CLUSTER_COLUMNS),
        frame(coverage_rows or [coverage()], COVERAGE_COLUMNS),
        market_intervals=None if market_intervals is None else frame(market_intervals, MARKET_COLUMNS),
    )


def states(result, kind="observed_full", offset=0):
    return result.state_statistics.filter((pl.col("analysis_kind") == kind) & (pl.col("offset_seconds") == offset)).sort("state")


def observer_event(seq, kind, order_id, side, price, leaves, displayed, client="OTHER", *, ts):
    return {
        "TRADEDATE": "2024-01-02", "MIC": "XMIL", "MARKETCODE": "MTA", "SYMBOLINDEX": 1, "EMM (*)": 1,
        "SEQUENCETIME": ts, "BOOKIN": ts, "BOOKOUTTIME": ts, "TRADETIME": ts if kind == 3 else None,
        "HDR_APPLKEYSEQUENCENUMBER": seq, "HDR_HWMSEQUENCENUMBER": seq, "HDR_OFFSETID": seq,
        "ROW_NUMBER": seq, "EVENTID": f"E{seq}", "ORDEREVENTTYPE (*)": kind, "ORDERID": order_id,
        "ORDERPRIORITY": str(seq), "ORDERSIDE (*)": side, "ORDERPX": price, "ORDERQTY": leaves,
        "DISPLAYEDQTY": displayed, "LEAVESQTY": leaves, "LASTSHARES": None, "LASTTRADEDPX": None,
        "ORDERTYPE (*)": 2, "TIMEINFORCE (*)": 0, "PASSIVEORDER": "N", "AGGRESSIVEORDER": "N",
        "FIRMID": "F1", "NMSC_ORIGINALCLIENTIDSHORTCODE": client, "ORDER_TRADINGCAPACITY (*)": 3,
    }


def test_empty_typed_inputs_return_all_schemas():
    schemas = [
        {x: pl.String for x in ("instrument", "partition_id", "event_date", *ACTOR, "side", "coverage_epoch_id", "start_ts", "end_ts", "member_count", "eligible_visible_qty", "mean_member_age_seconds", "oldest_member_age_seconds")},
        {x: pl.String for x in ("instrument", "partition_id", "event_date", *ACTOR, "side", "coverage_epoch_id", "withdrawal_event_id", "sort_index", "event_ts", "physical_order_key", "visible_qty_removed")},
        {x: pl.String for x in ("instrument", "partition_id", "event_date", *ACTOR, "event_side", "execution_anchor_mode", "cluster_end_ts", "cluster_last_sort_index", "execution_cluster_id", "execution_quantity")},
        {x: pl.String for x in ("instrument", "partition_id", "event_date", "coverage_epoch_id", "start_ts", "end_ts", "end_reason")},
    ]
    out = analyze_controls_v2(*(pl.DataFrame(schema=schema) for schema in schemas))
    for frame in (out.state_statistics, out.contrast_statistics, out.summary, out.coverage, out.balance, out.shift_support):
        assert frame.height == 0 and frame.schema


def test_overlap_ties_endpoints_and_naive_reference_accounting():
    out = run([interval(0, 10)], [withdrawal("tie", 2, 10), withdrawal("post", 2, 11), withdrawal("mixed", 3, 20), withdrawal("upper", 4, 30)], [cluster("p", 2, 10, "passive"), cluster("a", 3, 15, "aggressive")])
    got = {r["state"]: (r["time_at_risk_seconds"], r["withdrawal_count"], r["opening_equality_count"]) for r in states(out).to_dicts()}
    # J=[2,10), time windows p=[2,4), a=[3,5), points include their upper endpoint.
    assert got == {"no_own_observed": (5.0, 1, 1), "own_passive_only": (1.0, 1, 0), "own_aggressive_only": (1.0, 0, 0), "own_mixed": (1.0, 2, 0)}


def test_gaps_bands_and_actor_groups_are_independent():
    out = run([interval(0, 40), interval(60, 70), interval(0, 10, actor=OTHER)], [withdrawal("gap", 50, 1), withdrawal("valid", 62, 2), withdrawal("other", 3, 2, actor=OTHER)], [], [coverage(0, 40), coverage(60, 70)])
    assert states(out).filter(pl.col("state") == "no_own_observed").get_column("withdrawal_count").sum() == 2
    own = states(out).filter(pl.col("actor_key") == ACTOR["actor_key"])
    assert own.get_column("time_at_risk_seconds").sum() == pytest.approx(46.0)
    assert out.coverage.filter(pl.col("actor_key") == ACTOR["actor_key"]).item(0, "observed_domain_seconds") == pytest.approx(46.0)


def test_zero_time_point_is_support_anomaly_and_contrast_weights_are_common():
    anomalous = run([interval(2, 2)], [withdrawal("z", 2, 2)], [cluster("p", 2, 1, "passive")], [coverage(0, 10)])
    row = states(anomalous).filter(pl.col("state") == "own_passive_only").row(0, named=True)
    assert row["withdrawal_intensity"] is None and row["support_anomaly"] is True
    out = run([interval(0, 12)], [withdrawal("p", 3, 2), withdrawal("a", 7, 3), withdrawal("n", 10, 4)], [cluster("p", 2, 1, "passive"), cluster("a", 6, 1, "aggressive")])
    assert out.contrast_statistics.filter(pl.col("analysis_kind") == "observed_full").get_column("weight_seconds").to_list() == [2.0, 2.0]


def test_authoritative_withdrawal_at_exact_risk_endpoint_is_not_dropped():
    out = run([interval(2, 4)], [withdrawal("expiry", 4, 2)], [cluster("p", 2, 1, "passive")], [coverage(0, 10)])
    row = states(out).filter(pl.col("state") == "own_passive_only").row(0, named=True)
    assert (row["time_at_risk_seconds"], row["withdrawal_count"]) == (2.0, 1)


def test_observer_point_only_same_timestamp_cancel_reaches_accounting_without_time():
    risk = compute_compact_risk(pl.DataFrame([
        observer_event(1, 1, "A", 2, 101.0, 10, 10, ts="2024-01-02 09:00:00"),
        observer_event(2, 1, "B", 1, 100.0, 10, 10, "C1", ts="2024-01-02 09:00:03"),
        observer_event(3, 4, "B", 1, 100.0, 0, 0, "C1", ts="2024-01-02 09:00:03"),
        observer_event(4, 1, "L", 1, 99.0, 10, 10, ts="2024-01-02 09:00:06"),
    ]), instrument="XYZ", top_n=1)

    assert risk.intervals.filter(pl.col("actor_key") == "client_original:C1").is_empty()
    assert risk.withdrawal_events.filter(pl.col("actor_key") == "client_original:C1").height == 1

    accounted = analyze_controls_v2(
        risk.intervals, risk.withdrawal_events, frame([], CLUSTER_COLUMNS), risk.coverage_epochs,
    )
    row = states(accounted).filter(
        (pl.col("actor_key") == "client_original:C1") & (pl.col("state") == "no_own_observed")
    ).row(0, named=True)
    assert (row["time_at_risk_seconds"], row["withdrawal_count"]) == (0.0, 1)
    assert row["withdrawal_intensity"] is None and row["support_anomaly"] is True


def test_withdrawal_epoch_mismatch_cannot_cross_a_coverage_gap():
    out = run(
        [interval(2, 4, epoch="e1")], [withdrawal("wrong-epoch", 3, 1, epoch="e2")],
        coverage_rows=[coverage(0, 5, "e1"), coverage(10, 15, "e2")],
    )
    assert states(out).get_column("withdrawal_count").sum() == 0


def test_zero_duration_coverage_is_valid_but_has_no_observed_point_domain():
    out = run([], [withdrawal("zero-coverage", 0, 1)], coverage_rows=[coverage(0, 0)])
    assert out.state_statistics.is_empty()


def test_adding_an_ordinary_withdrawal_point_does_not_change_risk_time_or_weight():
    baseline = run([interval(0, 12)], clusters=[cluster("p", 2, 1, "passive")])
    with_point = run(
        [interval(0, 12)], [withdrawal("ordinary", 3, 2)], [cluster("p", 2, 1, "passive")],
    )
    state_times = lambda out: {(row["analysis_kind"], row["offset_seconds"], row["state"]): row["time_at_risk_seconds"] for row in out.state_statistics.to_dicts()}
    contrast_weights = lambda out: {(row["analysis_kind"], row["offset_seconds"], row["comparison_mode"]): row["weight_seconds"] for row in out.contrast_statistics.to_dicts()}
    assert state_times(with_point) == state_times(baseline)
    assert contrast_weights(with_point) == contrast_weights(baseline)


def test_shift_summary_reuses_zero_weights_when_risk_overlap_changes():
    out = run(
        [interval(0, 93), interval(94, 300), interval(0, 300, actor=OTHER)],
        [withdrawal('a', 92.5, 20), withdrawal('b', 123, 30, actor=OTHER)],
        [cluster('pa', 122, 1, 'passive'), cluster('pb', 122, 1, 'passive', actor=OTHER)],
        [coverage(0, 300)],
    )
    support = out.shift_support.filter(pl.col('comparison_mode') == 'passive')
    assert support['supported_all_offsets'].all()
    zero_weights = dict(zip(support['actor_key'], support['zero_weight_seconds']))
    summary = out.summary.filter((pl.col('analysis_kind') == 'timing_shift_common_support') & (pl.col('comparison_mode') == 'passive'))
    for row in summary.iter_rows(named=True):
        contrasts = out.contrast_statistics.filter(
            (pl.col('analysis_kind') == 'timing_shift_common_support') &
            (pl.col('comparison_mode') == 'passive') & (pl.col('offset_seconds') == row['offset_seconds'])
        ).to_dicts()
        expected_weight = sum(zero_weights.values())
        assert row['weight_seconds'] == pytest.approx(expected_weight)
        expected = sum(zero_weights[c['actor_key']] * c['intensity_difference'] for c in contrasts) / expected_weight
        assert row['weighted_intensity_difference'] == pytest.approx(expected)


def test_shift_schedule_is_deterministic_and_requires_all_offsets():
    out = run([interval(0, 300)], [withdrawal("opening", 122, 1)], [cluster("p", 122, 1, "passive")], [coverage(0, 300)])
    shifts = out.state_statistics.filter(pl.col("analysis_kind") == "timing_shift_common_support")
    assert set(shifts.get_column("offset_seconds")) == {-60, -30, 0, 30, 60}
    assert out.shift_support.filter(pl.col("comparison_mode") == "passive").item(0, "supported_all_offsets") is True
    zero = shifts.filter((pl.col("offset_seconds") == 0) & (pl.col("state") == "no_own_observed"))
    assert zero.item(0, "opening_equality_count") == 1 and zero.item(0, "withdrawal_count") == 1
    assert out.state_statistics.to_dicts() == run([interval(0, 300)], [withdrawal("opening", 122, 1)], [cluster("p", 122, 1, "passive")], [coverage(0, 300)]).state_statistics.to_dicts()


def test_market_balance_is_time_weighted_and_explicitly_missing_when_absent():
    market = [
        {"instrument": "XYZ", "partition_id": "P", "event_date": date(2024, 1, 2), "coverage_epoch_id": "e1", "start_ts": T0 + timedelta(seconds=2), "end_ts": T0 + timedelta(seconds=3), "spread": 2.0, "depth": 10.0, "prior_event_count_60s": 4.0},
        {"instrument": "XYZ", "partition_id": "P", "event_date": date(2024, 1, 2), "coverage_epoch_id": "e1", "start_ts": T0 + timedelta(seconds=3), "end_ts": T0 + timedelta(seconds=4), "spread": 4.0, "depth": 30.0, "prior_event_count_60s": 8.0},
    ]
    present = run([interval(0, 8)], clusters=[cluster("p", 2, 1, "passive")], market_intervals=market)
    absent = run([interval(0, 8)], clusters=[cluster("p", 2, 1, "passive")])
    exposed = present.balance.filter((pl.col("analysis_kind") == "observed_full") & (pl.col("state") == "own_passive_only")).row(0, named=True)
    missing = absent.balance.filter((pl.col("analysis_kind") == "observed_full") & (pl.col("state") == "own_passive_only")).row(0, named=True)
    assert (exposed["spread_time_seconds"], exposed["mean_spread"], exposed["mean_depth"], exposed["mean_prior_event_count_60s"]) == pytest.approx((2.0, 3.0, 20.0, 6.0))
    assert missing["spread_time_seconds"] == 0.0 and missing["mean_spread"] is None
    assert missing["market_covariate_support"] == "market_intervals_absent"


def test_market_intervals_reject_overlap_within_an_epoch():
    overlapping = [
        {"instrument": "XYZ", "partition_id": "P", "event_date": date(2024, 1, 2), "coverage_epoch_id": "e1", "start_ts": T0 + timedelta(seconds=2), "end_ts": T0 + timedelta(seconds=4), "spread": 2.0, "depth": 10.0, "prior_event_count_60s": 4.0},
        {"instrument": "XYZ", "partition_id": "P", "event_date": date(2024, 1, 2), "coverage_epoch_id": "e1", "start_ts": T0 + timedelta(seconds=3), "end_ts": T0 + timedelta(seconds=5), "spread": 4.0, "depth": 30.0, "prior_event_count_60s": 8.0},
    ]
    with pytest.raises(ValueError, match="market_intervals overlap"):
        run([interval(0, 8)], market_intervals=overlapping)


class CountingRows:
    """Sequence probe that records concrete rows examined by a range scan."""

    def __init__(self, rows):
        self._rows = rows
        self.examined = []

    def __getitem__(self, index):
        if isinstance(index, slice):
            raise AssertionError("the index must not materialize a prefix slice")
        self.examined.append(index)
        return self._rows[index]


def test_schedule_index_examines_only_time_local_rows():
    rows = [
        (T0 + timedelta(seconds=second), T0 + timedelta(seconds=second + 2), second)
        for second in range(10_000)
    ]
    examined = CountingRows(rows)
    windows = {
        "passive": ([row[0] for row in rows], examined),
        "aggressive": ([], CountingRows([])),
    }

    passive, aggressive, equality = controls_v2._mode_at_time(
        windows, T0 + timedelta(seconds=9_000, milliseconds=500), observed=False, sort_index=None,
    )

    assert (passive, aggressive, equality) == (True, False, False)
    assert examined.examined == [8_999, 9_000]


def test_market_index_examines_only_overlapping_rows_without_prefix_slice():
    rows = [
        {
            "start_ts": T0 + timedelta(seconds=second),
            "end_ts": T0 + timedelta(seconds=second + 1),
            "spread": float(second),
            "depth": 10.0,
            "prior_event_count_60s": 1.0,
        }
        for second in range(10_000)
    ]
    examined = CountingRows(rows)

    values = list(controls_v2._market_values(
        ([row["start_ts"] for row in rows], examined),
        T0 + timedelta(seconds=9_000, milliseconds=500),
        T0 + timedelta(seconds=9_001, milliseconds=500),
    ))

    assert [(seconds, covariates["spread"]) for seconds, covariates in values] == [(0.5, 9000.0), (0.5, 9001.0)]
    assert examined.examined == [9_000, 9_001]


def test_balance_integrates_ages_and_tracks_finite_market_support_separately():
    market = [
        {"instrument": "XYZ", "partition_id": "P", "event_date": date(2024, 1, 2), "coverage_epoch_id": "e1", "start_ts": T0 + timedelta(seconds=2), "end_ts": T0 + timedelta(seconds=3), "spread": None, "depth": 10.0, "prior_event_count_60s": 4.0},
        {"instrument": "XYZ", "partition_id": "P", "event_date": date(2024, 1, 2), "coverage_epoch_id": "e1", "start_ts": T0 + timedelta(seconds=3), "end_ts": T0 + timedelta(seconds=4), "spread": 4.0, "depth": float("nan"), "prior_event_count_60s": 8.0},
    ]
    out = run([interval(0, 8)], clusters=[cluster("p", 2, 1, "passive")], market_intervals=market)
    row = out.balance.filter((pl.col("analysis_kind") == "observed_full") & (pl.col("state") == "own_passive_only")).row(0, named=True)
    # Risk-interval ages are measured at opening, so their mean over [2, 4) advances by elapsed time.
    assert (row["analytical_mean_member_age_seconds"], row["analytical_oldest_member_age_seconds"]) == pytest.approx((7.0, 10.0))
    assert (row["spread_time_seconds"], row["spread_missing_time_seconds"], row["mean_spread"]) == pytest.approx((1.0, 1.0, 4.0))
    assert (row["depth_time_seconds"], row["depth_missing_time_seconds"], row["mean_depth"]) == pytest.approx((1.0, 1.0, 10.0))
    assert (row["prior_event_count_60s_time_seconds"], row["prior_event_count_60s_missing_time_seconds"], row["mean_prior_event_count_60s"]) == pytest.approx((2.0, 0.0, 6.0))


def test_overlapping_risk_or_coverage_intervals_are_rejected_before_accounting():
    with pytest.raises(ValueError, match="risk intervals overlap"):
        run([interval(0, 8), interval(4, 10)], coverage_rows=[coverage(0, 10)])
    with pytest.raises(ValueError, match="coverage rows overlap"):
        run([interval(0, 10)], coverage_rows=[coverage(0, 8), coverage(4, 10)])
    with pytest.raises(ValueError, match="coverage rows overlap"):
        run([interval(0, 8)], coverage_rows=[coverage(0, 8, "e1"), coverage(4, 10, "e2")])
