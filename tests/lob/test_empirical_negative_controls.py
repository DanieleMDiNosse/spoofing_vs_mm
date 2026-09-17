from __future__ import annotations

from collections.abc import Iterable
from datetime import date, datetime, timedelta, timezone

import polars as pl
import pytest

import spoofing_detection.lob.negative_controls as negative_controls_module
from spoofing_detection.lob.negative_controls import _block_id, rescore_empirical_placebos
from spoofing_detection.lob.withdrawal_risk import (
    ACTOR_SUMMARY_SCHEMA,
    DIAGNOSTIC_SCHEMA,
    EXPOSURE_CONTRAST_SCHEMA,
    EXPOSURE_INTERVAL_SCHEMA,
    EXPOSURE_STRATUM_SCHEMA,
    INTERVAL_SCHEMA,
    MEMBERSHIP_SCHEMA,
    SPELL_SCHEMA,
    TRANSITION_SCHEMA,
    WITHDRAWAL_SCHEMA,
    WithdrawalRiskResult,
)


T0 = datetime(2024, 1, 2, 9, 30)
ACTOR = {
    "actor_key": "client_original:C1",
    "actor_id": "C1",
    "identity_level": "client_original",
    "identity_source": "NMSC_ORIGINALCLIENTIDSHORTCODE",
    "identity_fallback_flag": False,
}


def _empty(schema):
    return pl.DataFrame(schema=schema)


def _risk(
    *,
    withdrawal_at: int | None = 3,
    boundaries: Iterable[int] = range(0, 10),
    spell_by_second: dict[int, str] | None = None,
    start_sort_index_by_second: dict[int, int | None] | None = None,
    panel_end_second: int = 10,
) -> WithdrawalRiskResult:
    intervals = []
    for index, second in enumerate(boundaries, start=1):
        if second + 1 > panel_end_second:
            continue
        intervals.append({
            "risk_interval_id": f"risk-{index}",
            "risk_spell_id": (spell_by_second or {}).get(second, "spell-1"),
            "partition_id": "P",
            "event_date": date(2024, 1, 2),
            **ACTOR,
            "side": "bid",
            "start_ts": T0 + timedelta(seconds=second),
            "start_sort_index": (start_sort_index_by_second or {}).get(second, second + 1),
            "end_ts": T0 + timedelta(seconds=second + 1),
            "duration_seconds": 1.0,
            "member_count": 1,
            "eligible_visible_qty": 10.0,
            "exposure_quantity": None,
            "membership_signature": "B:10",
        })
    withdrawals = []
    if withdrawal_at is not None:
        withdrawals.append({
            "withdrawal_event_id": "withdrawal-1",
            "physical_order_key": "P|2024-01-02|B|1",
            "partition_id": "P",
            "event_date": date(2024, 1, 2),
            "sort_index": 99,
            "event_ts": T0 + timedelta(seconds=withdrawal_at),
            **ACTOR,
            "side": "bid",
            "order_id": "B",
            "first_seen_sort_index": 1,
            "visible_qty_removed": 10.0,
            "transition_reason": "cancellation",
        })
    return WithdrawalRiskResult(
        _empty(SPELL_SCHEMA),
        pl.DataFrame(intervals, schema=INTERVAL_SCHEMA),
        _empty(MEMBERSHIP_SCHEMA),
        pl.DataFrame(withdrawals, schema=WITHDRAWAL_SCHEMA),
        _empty(TRANSITION_SCHEMA),
        _empty(DIAGNOSTIC_SCHEMA),
        _empty(ACTOR_SUMMARY_SCHEMA),
        _empty(EXPOSURE_INTERVAL_SCHEMA),
        _empty(EXPOSURE_CONTRAST_SCHEMA),
        _empty(EXPOSURE_STRATUM_SCHEMA),
    )


def _cluster(cluster_id: str, second: int, *, band: str = "open") -> dict:
    return {
        "execution_cluster_id": cluster_id,
        "partition_id": "P",
        **ACTOR,
        "execution_anchor_mode": "passive",
        "execution_quantity": 3.0,
        "event_side": "bid",
        "cluster_end_ts": T0 + timedelta(seconds=second),
        "cluster_last_sort_index": second + 1,
        "execution_sweep_id": f"sweep-{cluster_id}",
        "activity_band": band,
    }


def _run(*, seed=7, clusters=None, risk=None, draws=8, band_windows=None):
    return rescore_empirical_placebos(
        risk or _risk(),
        clusters if clusters is not None else [_cluster("a", 2), _cluster("b", 4)],
        withdrawal_window_seconds=1,
        draw_count=draws,
        seed=seed,
        activity_band_windows=band_windows or {"open": (T0, T0 + timedelta(seconds=10))},
    )


def test_placebos_require_declared_activity_band_boundaries():
    with pytest.raises(ValueError, match="activity_band_windows must define every cluster activity_band"):
        rescore_empirical_placebos(
            _risk(), [_cluster("a", 2)], withdrawal_window_seconds=1, draw_count=1, seed=1,
        )


def test_placebo_seed_is_reproducible_sensitive_and_preserves_ordered_schedule_gaps():
    first = _run(seed=7)
    second = _run(seed=7)
    changed = _run(seed=8)

    assert first.placebo_schedules.to_dicts() == second.placebo_schedules.to_dicts()
    assert first.placebo_statistics.to_dicts() == second.placebo_statistics.to_dicts()
    assert first.placebo_schedules.to_dicts() != changed.placebo_schedules.to_dicts()
    accepted = first.placebo_schedules.filter(pl.col("accepted"))
    for _, draw in accepted.group_by("draw_id", maintain_order=True):
        anchors = draw.sort("source_anchor_order").get_column("pseudo_anchor_ts").to_list()
        assert anchors == sorted(anchors)
        assert anchors[1] - anchors[0] == timedelta(seconds=2)


def test_placebo_block_substreams_are_invariant_to_an_unrelated_earlier_block():
    target = [_cluster("target-a", 2), _cluster("target-b", 4)]
    common = {
        "withdrawal_window_seconds": 1,
        "draw_count": 12,
        "seed": 7,
        "activity_band_windows": {
            "aaa": (T0, T0 + timedelta(seconds=10)),
            "open": (T0, T0 + timedelta(seconds=10)),
        },
    }
    without_unrelated = rescore_empirical_placebos(_risk(), target, **common)
    with_unrelated = rescore_empirical_placebos(
        _risk(), [*target, _cluster("unrelated", 1, band="aaa")], **common,
    )

    def target_schedule(result):
        return result.placebo_schedules.filter(
            pl.col("activity_band") == "open"
        ).sort(["draw_id", "source_anchor_order"])

    def target_support(result):
        return result.placebo_support.filter(
            pl.col("activity_band") == "open"
        ).sort("draw_id")

    assert target_schedule(without_unrelated).to_dicts() == target_schedule(with_unrelated).to_dicts()
    assert target_support(without_unrelated).to_dicts() == target_support(with_unrelated).to_dicts()


def test_placebos_rebuild_exposure_and_rescore_aligned_outcome_instead_of_copying_observed_score():
    out = _run(seed=7, draws=12)

    observed = out.placebo_statistics.filter(
        (pl.col("draw_id") == 0)
        & (pl.col("row_kind") == "state")
        & (pl.col("contrast_label") == "own_only")
        & (pl.col("actor_key") == ACTOR["actor_key"])
    ).row(0, named=True)
    placebo = out.placebo_statistics.filter(
        (pl.col("draw_id") > 0)
        & (pl.col("row_kind") == "state")
        & (pl.col("contrast_label") == "own_only")
        & (pl.col("actor_key") == ACTOR["actor_key"])
    )

    assert observed["withdrawal_count"] == 1
    assert observed["time_at_risk_seconds"] == pytest.approx(2.0)
    assert (placebo.get_column("withdrawal_count") != observed["withdrawal_count"]).any() or (
        placebo.get_column("time_at_risk_seconds") != observed["time_at_risk_seconds"]
    ).any()
    assert out.observed_risk.exposure_contrasts.height > 0


def test_unaligned_nonzero_outcome_outside_restricted_band_has_no_schedule_separation():
    # The real withdrawal is in the same risk panel but outside every observed
    # and possible placebo horizon in the prespecified activity band.
    out = rescore_empirical_placebos(
        _risk(withdrawal_at=9),
        [_cluster("a", 1), _cluster("b", 3)],
        withdrawal_window_seconds=1,
        draw_count=6,
        seed=7,
        activity_band_windows={"open": (T0, T0 + timedelta(seconds=6))},
    )
    normalized_statistics = out.placebo_statistics.drop("draw_id").unique().to_dicts()
    own = out.placebo_statistics.filter(
        (pl.col("row_kind") == "state") & (pl.col("contrast_label") == "own_only")
    )

    assert len(normalized_statistics) == out.placebo_statistics.height // 7
    assert own.get_column("withdrawal_count").sum() == 0


def test_no_wrapping_or_boundary_crossing_is_rejected_with_support_audit_reason():
    out = _run(
        risk=_risk(boundaries=range(0, 3)),
        clusters=[_cluster("a", 0), _cluster("b", 4)],
        draws=2,
    )

    assert out.placebo_schedules.filter(~pl.col("accepted")).height == 4
    assert set(out.placebo_schedules.get_column("rejection_reason")) == {"no_supported_schedule_start"}
    assert out.placebo_support.get_column("accepted").to_list() == [False, False]


def test_translated_multi_anchor_schedule_rejects_an_observed_at_risk_gap():
    out = _run(
        risk=_risk(withdrawal_at=None, boundaries=(0, 4)),
        clusters=[_cluster("a", 0), _cluster("b", 4)],
        draws=2,
    )

    assert out.placebo_support.get_column("accepted").to_list() == [False, False]
    assert set(out.placebo_support.get_column("rejection_reason")) == {"at_risk_gap_or_spell_closure"}
    assert set(out.placebo_schedules.get_column("rejection_reason")) == {"at_risk_gap_or_spell_closure"}


def test_translated_multi_anchor_schedule_rejects_a_risk_spell_closure():
    out = _run(
        risk=_risk(
            withdrawal_at=None,
            boundaries=(0, 1),
            spell_by_second={0: "spell-1", 1: "spell-2"},
        ),
        clusters=[_cluster("a", 0), _cluster("b", 1)],
        draws=1,
    )

    assert out.placebo_support.item(0, "accepted") is False
    assert out.placebo_support.item(0, "rejection_reason") == "at_risk_gap_or_spell_closure"


def test_source_schedule_rejects_reversed_input_before_canonical_sorting():
    with pytest.raises(ValueError, match="nonordered_schedule"):
        _run(clusters=[_cluster("b", 4), _cluster("a", 2)], draws=1)


def test_source_schedule_rejects_timestamp_sort_index_inconsistency_before_draws():
    first = _cluster("a", 2)
    second = _cluster("b", 4)
    second["cluster_last_sort_index"] = first["cluster_last_sort_index"] - 1

    with pytest.raises(ValueError, match="nonordered_schedule"):
        _run(clusters=[first, second], draws=1)


def test_exact_duplicate_cluster_rows_do_not_create_a_source_order_failure():
    first = _cluster("a", 2)
    out = _run(clusters=[first, dict(first), _cluster("b", 4)], draws=1)

    assert out.placebo_support.item(0, "accepted") is True
    assert out.placebo_schedules.filter(pl.col("accepted")).height == 2


def test_source_order_validation_allows_interleaved_blocks():
    alpha_first = _cluster("alpha-a", 0, band="alpha")
    beta_first = _cluster("beta-a", 0, band="beta")
    alpha_second = _cluster("alpha-b", 2, band="alpha")
    beta_second = _cluster("beta-b", 2, band="beta")
    out = rescore_empirical_placebos(
        _risk(),
        [alpha_first, beta_first, alpha_second, beta_second],
        withdrawal_window_seconds=1,
        draw_count=1,
        seed=7,
        activity_band_windows={
            "alpha": (T0, T0 + timedelta(seconds=10)),
            "beta": (T0, T0 + timedelta(seconds=10)),
        },
    )

    assert out.placebo_support.get_column("accepted").to_list() == [True, True]


def test_schedule_support_can_end_at_its_last_anchor_before_the_outcome_horizon():
    out = rescore_empirical_placebos(
        _risk(withdrawal_at=None, boundaries=(0, 1)),
        [_cluster("a", 0), _cluster("b", 1)],
        withdrawal_window_seconds=5,
        draw_count=1,
        seed=7,
        activity_band_windows={"open": (T0, T0 + timedelta(seconds=10))},
    )

    assert out.placebo_support.item(0, "accepted") is True


def test_placebo_selection_ignores_future_panel_activity_beyond_the_latest_anchor():
    clusters = [_cluster("a", 2), _cluster("b", 4)]
    common = {
        "withdrawal_window_seconds": 1,
        "draw_count": 5,
        "seed": 7,
        "activity_band_windows": {"open": (T0, T0 + timedelta(seconds=10))},
    }
    before_future_activity = rescore_empirical_placebos(
        _risk(withdrawal_at=None), clusters, **common,
    )
    after_future_activity = rescore_empirical_placebos(
        _risk(
            withdrawal_at=12,
            boundaries=range(0, 12),
            panel_end_second=12,
        ),
        clusters,
        **common,
    )

    assert before_future_activity.placebo_schedules.to_dicts() == after_future_activity.placebo_schedules.to_dicts()
    assert before_future_activity.placebo_support.to_dicts() == after_future_activity.placebo_support.to_dicts()
    assert before_future_activity.placebo_statistics.to_dicts() != after_future_activity.placebo_statistics.to_dicts()


def test_activity_band_boundary_crossing_is_rejected_before_outcome_rescoring():
    with pytest.raises(ValueError, match="activity_band_windows must contain positive datetime intervals"):
        rescore_empirical_placebos(
            _risk(), [_cluster("a", 2)], withdrawal_window_seconds=1, draw_count=1, seed=1,
            activity_band_windows={"open": (T0, T0)},
        )
    out = rescore_empirical_placebos(
        _risk(), [_cluster("a", 2), _cluster("b", 4)], withdrawal_window_seconds=1,
        draw_count=1, seed=1, activity_band_windows={"open": (T0 + timedelta(seconds=2), T0 + timedelta(seconds=4))},
    )
    assert out.placebo_support.item(0, "rejection_reason") == "activity_band_boundary_crossing"


def test_placebo_same_clock_outcome_uses_mapped_risk_boundary_not_source_anchor_order():
    risk = _risk(
        withdrawal_at=1,
        boundaries=(0, 1),
        panel_end_second=2,
        start_sort_index_by_second={0: 10, 1: 50},
    )
    first_source = _cluster("source", 2)
    first_source["cluster_last_sort_index"] = 1
    second_source = dict(first_source)
    second_source["cluster_last_sort_index"] = 100

    band_windows = {"open": (T0 + timedelta(seconds=1), T0 + timedelta(seconds=2))}
    first = _run(risk=risk, clusters=[first_source], draws=1, band_windows=band_windows)
    second = _run(risk=risk, clusters=[second_source], draws=1, band_windows=band_windows)

    first_schedule = first.placebo_schedules.row(0, named=True)
    second_schedule = second.placebo_schedules.row(0, named=True)
    assert first_schedule["pseudo_anchor_ts"] == second_schedule["pseudo_anchor_ts"] == T0 + timedelta(seconds=1)
    assert first_schedule["source_anchor_sort_index"] == 1
    assert second_schedule["source_anchor_sort_index"] == 100
    assert first_schedule["pseudo_anchor_sort_index"] == second_schedule["pseudo_anchor_sort_index"] == 50
    own_first = first.placebo_statistics.filter(
        (pl.col("draw_id") == 1)
        & (pl.col("row_kind") == "state")
        & (pl.col("contrast_label") == "own_only")
    ).item(0, "withdrawal_count")
    own_second = second.placebo_statistics.filter(
        (pl.col("draw_id") == 1)
        & (pl.col("row_kind") == "state")
        & (pl.col("contrast_label") == "own_only")
    ).item(0, "withdrawal_count")
    assert own_first == own_second == 1


def test_placebo_excludes_synthetic_boundary_without_observed_ordering_support():
    out = _run(
        risk=_risk(
            withdrawal_at=None,
            boundaries=(1,),
            panel_end_second=2,
            start_sort_index_by_second={1: None},
        ),
        clusters=[_cluster("source", 2)],
        draws=1,
    )

    assert out.placebo_support.item(0, "accepted") is False
    assert out.placebo_support.item(0, "rejection_reason") == "no_observed_boundary_ordering_support"
    schedule = out.placebo_schedules.row(0, named=True)
    assert schedule["pseudo_anchor_ts"] is None
    assert schedule["pseudo_anchor_sort_index"] is None
    assert schedule["rejection_reason"] == "no_observed_boundary_ordering_support"


def test_zero_state_denominators_are_null_and_invalid_draw_inputs_are_rejected():
    out = _run(clusters=[], draws=1)
    own = out.placebo_statistics.filter(
        (pl.col("row_kind") == "state") & (pl.col("contrast_label") == "own_only")
    )
    assert own.get_column("time_at_risk_seconds").to_list() == [0.0, 0.0]
    assert own.get_column("withdrawal_intensity").to_list() == [None, None]

    for value in (True, False, 0, -1, 1.5, float("nan"), "2"):
        with pytest.raises(ValueError, match="draw_count must be a positive integer"):
            _run(draws=value)


def test_placebo_seed_accepts_only_unsigned_64_bit_integers():
    for seed in (0, 2**64 - 1):
        assert _run(seed=seed, draws=1).placebo_schedules.height == 2

    for seed in (True, False, -1, 2**64, 2**100, 1.5, "7"):
        with pytest.raises(ValueError, match=r"seed must be an integer from 0 through 2\*\*64-1"):
            _run(seed=seed, draws=1)


def test_placebo_block_id_is_injective_when_components_contain_delimiters():
    # These deliberately malformed *keys* collide under delimiter joining. The
    # identifier helper must still be safe for any future widening of key fields.
    first = ("P", date(2024, 1, 2), "actor|identity", "level", "bid", "open")
    second = ("P", date(2024, 1, 2), "actor", "identity|level", "bid", "open")

    assert "|".join((first[0], first[1].isoformat(), *first[2:])) == "|".join(
        (second[0], second[1].isoformat(), *second[2:])
    )
    assert _block_id(first) != _block_id(second)
    assert _block_id(first).startswith("placebo-block-")


def test_exact_duplicate_risk_support_boundaries_are_deduplicated():
    risk = _risk(withdrawal_at=None, boundaries=(0, 1))
    duplicate = risk.intervals.row(0, named=True)
    risk = WithdrawalRiskResult(
        risk.spells,
        pl.DataFrame([*risk.intervals.to_dicts(), duplicate], schema=INTERVAL_SCHEMA),
        risk.membership,
        risk.withdrawal_events,
        risk.transitions,
        risk.diagnostics,
        risk.actor_summary,
        risk.exposure_intervals,
        risk.exposure_contrasts,
        risk.exposure_strata,
    )

    out = _run(risk=risk, clusters=[_cluster("a", 0)], draws=1)

    assert out.placebo_support.item(0, "support_boundary_count") == 2
    assert out.placebo_schedules.item(0, "accepted") is True


@pytest.mark.parametrize(
    "changes",
    [
        {"start_sort_index": 99},
        {"end_ts": T0 + timedelta(seconds=2)},
        {"risk_spell_id": "different-spell"},
    ],
)
def test_conflicting_same_start_risk_support_boundaries_fail_before_observed_rescore(monkeypatch, changes):
    risk = _risk(withdrawal_at=None, boundaries=(0, 1))
    conflicting = {**risk.intervals.row(0, named=True), "risk_interval_id": "conflict", **changes}
    risk = WithdrawalRiskResult(
        risk.spells,
        pl.DataFrame([*risk.intervals.to_dicts(), conflicting], schema=INTERVAL_SCHEMA),
        risk.membership,
        risk.withdrawal_events,
        risk.transitions,
        risk.diagnostics,
        risk.actor_summary,
        risk.exposure_intervals,
        risk.exposure_contrasts,
        risk.exposure_strata,
    )

    def observed_rescore_must_not_run(*args, **kwargs):
        raise AssertionError("observed rescore ran before support-boundary validation")

    monkeypatch.setattr(
        "spoofing_detection.lob.negative_controls.intersect_observed_execution_exposure",
        observed_rescore_must_not_run,
    )
    with pytest.raises(ValueError, match="conflicting risk support boundaries share start_ts"):
        _run(risk=risk, clusters=[_cluster("a", 0)], draws=1)


def _aware_risk() -> WithdrawalRiskResult:
    risk = _risk()
    intervals = []
    for row in risk.intervals.to_dicts():
        intervals.append({
            **row,
            "start_ts": row["start_ts"].replace(tzinfo=timezone.utc),
            "end_ts": row["end_ts"].replace(tzinfo=timezone.utc),
        })
    withdrawals = [
        {**row, "event_ts": row["event_ts"].replace(tzinfo=timezone.utc)}
        for row in risk.withdrawal_events.to_dicts()
    ]
    return WithdrawalRiskResult(
        risk.spells,
        pl.DataFrame(intervals),
        risk.membership,
        pl.DataFrame(withdrawals),
        risk.transitions,
        risk.diagnostics,
        risk.actor_summary,
        risk.exposure_intervals,
        risk.exposure_contrasts,
        risk.exposure_strata,
    )


def test_placebo_rescore_normalizes_all_aware_clocks_to_naive_utc_without_date_drift():
    utc = timezone.utc
    plus_two = timezone(timedelta(hours=2))
    first = _cluster("a", 2)
    second = _cluster("b", 4)
    first["cluster_end_ts"] = (T0 + timedelta(seconds=2)).replace(tzinfo=utc).astimezone(plus_two)
    second["cluster_end_ts"] = (T0 + timedelta(seconds=4)).replace(tzinfo=utc)
    out = _run(
        risk=_aware_risk(),
        clusters=[first, second],
        draws=1,
        band_windows={
            "open": (
                T0.replace(tzinfo=utc).astimezone(plus_two),
                (T0 + timedelta(seconds=10)).replace(tzinfo=utc),
            )
        },
    )

    schedule = out.placebo_schedules.sort("source_anchor_order")
    assert schedule.get_column("source_anchor_ts").to_list() == [T0 + timedelta(seconds=2), T0 + timedelta(seconds=4)]
    assert schedule.get_column("event_date").to_list() == [date(2024, 1, 2), date(2024, 1, 2)]


def test_placebo_rescore_rejects_mixed_naive_and_aware_clocks_before_observed_rescore(monkeypatch):
    def observed_rescore_must_not_run(*args, **kwargs):
        raise AssertionError("observed rescore ran despite mixed clock awareness")

    monkeypatch.setattr(
        "spoofing_detection.lob.negative_controls.intersect_observed_execution_exposure",
        observed_rescore_must_not_run,
    )
    with pytest.raises(ValueError, match="consistently all naive or all timezone-aware"):
        _run(risk=_aware_risk(), clusters=[_cluster("a", 2)], draws=1)


def test_placebo_rescore_accepts_all_naive_clocks():
    out = _run(draws=1)

    assert out.placebo_schedules.get_column("source_anchor_ts").dtype == pl.Datetime("us")


def test_placebo_support_interval_index_is_built_once_for_many_unrelated_strata(monkeypatch):
    risk = _risk(withdrawal_at=None)
    unrelated = [
        {
            **row,
            "partition_id": f"unrelated-{index}",
            "risk_interval_id": f"unrelated-{index}-{row['risk_interval_id']}",
            "risk_spell_id": f"unrelated-{index}-{row['risk_spell_id']}",
        }
        for index in range(50)
        for row in risk.intervals.to_dicts()
    ]
    risk = WithdrawalRiskResult(
        risk.spells,
        pl.DataFrame([*risk.intervals.to_dicts(), *unrelated], schema=INTERVAL_SCHEMA),
        risk.membership,
        risk.withdrawal_events,
        risk.transitions,
        risk.diagnostics,
        risk.actor_summary,
        risk.exposure_intervals,
        risk.exposure_contrasts,
        risk.exposure_strata,
    )
    calls = 0
    original = negative_controls_module._risk_support_interval_index

    def indexed_once(result):
        nonlocal calls
        calls += 1
        return original(result)

    monkeypatch.setattr(negative_controls_module, "_risk_support_interval_index", indexed_once)
    out = _run(risk=risk, clusters=[_cluster("a", 2), _cluster("b", 4)], draws=2)

    assert calls == 1
    assert out.placebo_support.height == 2
