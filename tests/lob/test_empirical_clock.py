from __future__ import annotations

from datetime import datetime

import pytest

from spoofing_detection.lob.empirical_clock import (
    CLOCK_POLICY_ID,
    EmpiricalClockDecision,
    resolve_empirical_clocks,
)


def event(kind: int, *, sequence, bookin=None, bookout=None, trade=None):
    return {
        "ORDEREVENTTYPE (*)": kind,
        "SEQUENCETIME": sequence,
        "BOOKIN": bookin,
        "BOOKOUTTIME": bookout,
        "TRADETIME": trade,
    }


def test_policy_separates_replay_fill_and_cancellation_removal_clocks_without_mutation():
    passive = event(
        3,
        sequence="2024-01-02 09:00:00.123456789",
        bookin="2024-01-02 08:59:00.000000001",
        bookout="2024-01-02 09:00:00.123457000",
        trade="2024-01-02 09:00:00.123000000",
    )
    original = dict(passive)

    decision = resolve_empirical_clocks(passive)

    assert decision == EmpiricalClockDecision(
        policy_id=CLOCK_POLICY_ID,
        replay_time_ns=1_704_186_000_123_456_789,
        fill_anchor_ns=1_704_186_000_123_000_000,
        cancellation_removal_ns=None,
        incompatibilities=(),
    )
    assert passive == original


def test_aggressive_fill_uses_tradetime_not_bookin_or_bookout():
    decision = resolve_empirical_clocks(
        event(
            3,
            sequence="2024-01-02 09:00:00.200000000",
            bookin="2024-01-02 09:00:00.100000000",
            bookout="2024-01-02 09:00:00.300000000",
            trade="2024-01-02 09:00:00.100000000",
        )
    )

    assert decision.replay_time_ns == 1_704_186_000_200_000_000
    assert decision.fill_anchor_ns == 1_704_186_000_100_000_000


def test_cancel_uses_sequencetime_without_bookin_requirement():
    with_bookin = resolve_empirical_clocks(
        event(
            4,
            sequence="2024-01-02 09:00:01.000000001",
            bookin="2024-01-02 09:00:01.000000000",
            bookout="2024-01-02 09:00:01.000000900",
        )
    )
    without_bookin = resolve_empirical_clocks(
        event(
            4,
            sequence="2024-01-02 09:00:01.000000002",
            bookin=None,
            bookout="2024-01-02 09:00:01.000000900",
        )
    )

    assert with_bookin.cancellation_removal_ns == with_bookin.replay_time_ns
    assert with_bookin.incompatibilities == ()
    assert without_bookin.cancellation_removal_ns == without_bookin.replay_time_ns
    assert without_bookin.incompatibilities == ()


def test_normalized_event_class_preserves_semantic_clock_rules():
    normalized_cancel = {
        "event_class": "cancel",
        "SEQUENCETIME": "2024-01-02 09:00:01",
        "BOOKIN": None,
        "BOOKOUTTIME": "2024-01-02 09:00:02",
        "TRADETIME": None,
    }

    decision = resolve_empirical_clocks(normalized_cancel)

    assert decision.cancellation_removal_ns == decision.replay_time_ns
    assert decision.incompatibilities == ()


def test_missing_semantic_clocks_are_explicit_without_fallback():
    fill = resolve_empirical_clocks(
        event(3, sequence="2024-01-02 09:00:00", trade=None, bookout="2024-01-02 09:00:01")
    )
    cancel = resolve_empirical_clocks(
        event(4, sequence=None, bookin="2024-01-02 09:00:00", bookout="2024-01-02 09:00:01")
    )
    missing_replay = resolve_empirical_clocks(
        event(1, sequence=None, bookin="2024-01-02 09:00:00", bookout="2024-01-02 09:00:01")
    )

    assert fill.fill_anchor_ns is None
    assert fill.incompatibilities == ("fill_missing_tradetime",)
    assert cancel.cancellation_removal_ns is None
    assert cancel.incompatibilities == ("missing_sequencetime",)
    assert missing_replay.replay_time_ns is None
    assert missing_replay.incompatibilities == ("missing_sequencetime",)


def test_fill_after_sequence_is_diagnosed_not_clamped():
    decision = resolve_empirical_clocks(
        event(
            3,
            sequence="2024-01-02 09:00:00.000000001",
            trade="2024-01-02 09:00:00.000000002",
        )
    )

    assert decision.fill_anchor_ns > decision.replay_time_ns
    assert decision.incompatibilities == ("fill_tradetime_after_sequencetime",)


def test_native_datetime_precision_is_normalized_and_invalid_values_are_missing():
    decision = resolve_empirical_clocks(
        event(
            1,
            sequence=datetime(2024, 1, 2, 9, 0, 0, 123456),
            bookin="not-a-time",
        )
    )

    assert decision.replay_time_ns == 1_704_186_000_123_456_000
    assert decision.fill_anchor_ns is None
    assert decision.cancellation_removal_ns is None


def test_native_nanoseconds_are_utc_instants_not_ambiguous_rome_local_times():
    utc = datetime(2024, 10, 27, 2, 30)
    epoch = datetime(1970, 1, 1)
    native_ns = int((utc - epoch).total_seconds() * 1_000_000_000)

    decision = resolve_empirical_clocks(event(1, sequence=native_ns))

    assert decision.replay_time_ns == native_ns
    assert decision.incompatibilities == ()


@pytest.mark.parametrize(
    "sequence",
    [
        "2024-01-02T09:00:00+02:00",
        "2024-01-02 09:00:00.1234567899",
    ],
)
def test_policy_rejects_aware_ambiguous_nonexistent_and_super_nanosecond_values(sequence):
    decision = resolve_empirical_clocks(event(1, sequence=sequence))

    assert decision.replay_time_ns is None
    assert decision.incompatibilities == ("invalid_sequencetime",)


@pytest.mark.parametrize("value", ["", "NaN", None])
def test_missing_sequence_markers_are_explicit(value):
    decision = resolve_empirical_clocks(event(1, sequence=value))
    assert decision.replay_time_ns is None
    assert decision.incompatibilities == ("missing_sequencetime",)
