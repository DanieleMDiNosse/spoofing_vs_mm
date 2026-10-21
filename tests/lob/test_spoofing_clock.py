from __future__ import annotations


def test_fill_uses_sequence_for_replay_and_trade_for_economic_anchor():
    from spoofing_detection.lob.spoofing_clock import (
        SPOOFING_DETECTION_CLOCK_POLICY_ID,
        resolve_spoofing_clocks,
    )

    decision = resolve_spoofing_clocks(
        {
            "ORDEREVENTTYPE (*)": 3,
            "SEQUENCETIME": 1_704_186_000_579_000_456,
            "TRADETIME": 1_704_186_000_579_000_123,
        }
    )

    assert decision.policy_id == SPOOFING_DETECTION_CLOCK_POLICY_ID
    assert decision.replay_time_ns == 1_704_186_000_579_000_456
    assert decision.fill_anchor_ns == 1_704_186_000_579_000_123
    assert decision.cancellation_removal_ns is None


def test_cancel_uses_sequence_and_never_falls_back_to_book_fields():
    from spoofing_detection.lob.spoofing_clock import resolve_spoofing_clocks

    decision = resolve_spoofing_clocks(
        {
            "ORDEREVENTTYPE (*)": 4,
            "SEQUENCETIME": 1_704_186_001_000_000_007,
            "BOOKIN": 1_704_186_000_000_000_000,
            "BOOKOUTTIME": 1_704_186_002_000_000_000,
        }
    )

    assert decision.replay_time_ns == 1_704_186_001_000_000_007
    assert decision.cancellation_removal_ns == 1_704_186_001_000_000_007


def test_fill_without_tradetime_is_incompatible_not_fallback():
    from spoofing_detection.lob.spoofing_clock import resolve_spoofing_clocks

    decision = resolve_spoofing_clocks(
        {"ORDEREVENTTYPE (*)": 3, "SEQUENCETIME": 1_704_186_000_000_000_000}
    )

    assert decision.fill_anchor_ns is None
    assert "fill_missing_tradetime" in decision.incompatibilities


def test_native_nanoseconds_are_preserved_exactly():
    from spoofing_detection.lob.spoofing_clock import resolve_spoofing_clocks

    decision = resolve_spoofing_clocks(
        {
            "ORDEREVENTTYPE (*)": 3,
            "SEQUENCETIME": 1_704_186_000_579_000_456,
            "TRADETIME": 1_704_186_000_579_000_123,
        }
    )

    assert decision.fill_anchor_ns is not None
    assert decision.fill_anchor_ns % 1_000 == 123


def test_panel_preserved_clock_columns_take_precedence_over_truncated_display_values():
    from spoofing_detection.lob.spoofing_clock import resolve_spoofing_clocks

    decision = resolve_spoofing_clocks(
        {
            "ORDEREVENTTYPE (*)": 3,
            "SEQUENCETIME": 1_700_000_000_123_456_000,
            "TRADETIME": 1_700_000_000_123_456_000,
            "__source_SEQUENCETIME_ns": 1_700_000_000_123_456_789,
            "__source_TRADETIME_ns": 1_700_000_000_123_456_123,
        }
    )

    assert decision.replay_time_ns == 1_700_000_000_123_456_789
    assert decision.fill_anchor_ns == 1_700_000_000_123_456_123