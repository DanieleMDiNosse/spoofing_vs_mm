"""Explicit clocks for the v3 spoofing-detection contract.

Canonical message ordering is owned by the replay driver.  This module only
resolves temporal meanings; no field can substitute for another.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .empirical_clock import _timestamp_ns
from .enums import normalize_enum_code

SPOOFING_DETECTION_CLOCK_POLICY_ID = (
    "spoofing_detection_sequence_replay_trade_fill_sequence_cancel_v3_utc_provisional"
)
SPOOFING_DETECTION_CLOCK_CONTRACT = {
    "replay_clock": "SEQUENCETIME",
    "duration_clock": "SEQUENCETIME",
    "cancellation_removal_clock": "SEQUENCETIME",
    "fill_anchor_clock": "TRADETIME",
    "timestamp_representation": "UTC epoch nanoseconds",
    "semantic_fallback": "forbidden",
    "interval_convention": "left_closed_right_open",
}


@dataclass(frozen=True)
class SpoofingClockDecision:
    """Independent temporal values for one normalized source event."""

    policy_id: str
    replay_time_ns: int | None
    fill_anchor_ns: int | None
    cancellation_removal_ns: int | None
    incompatibilities: tuple[str, ...]


def resolve_spoofing_clocks(event: Mapping[str, Any]) -> SpoofingClockDecision:
    """Resolve detector clocks at native precision without semantic fallback."""

    incompatibilities: list[str] = []

    def parse(field: str, invalid_reason: str) -> int | None:
        try:
            native_value = event.get(f"__source_{field}_ns")
            if native_value is None:
                # Compatibility with external callers that predate panel's internal name.
                native_value = event.get(f"source_{field.lower()}_ns")
            if native_value is not None:
                return _timestamp_ns(native_value)
            return _timestamp_ns(event.get(field))
        except ValueError:
            incompatibilities.append(invalid_reason)
            return None

    sequence_ns = parse("SEQUENCETIME", "invalid_sequencetime")
    if sequence_ns is None and "invalid_sequencetime" not in incompatibilities:
        incompatibilities.append("missing_sequencetime")

    event_code = normalize_enum_code(event.get("ORDEREVENTTYPE (*)"))
    event_class = event.get("event_class")
    fill_anchor_ns: int | None = None
    cancellation_removal_ns: int | None = None
    if event_code == 3 or event_class == "fill":
        fill_anchor_ns = parse("TRADETIME", "fill_invalid_tradetime")
        if fill_anchor_ns is None and "fill_invalid_tradetime" not in incompatibilities:
            incompatibilities.append("fill_missing_tradetime")
    elif event_code == 4 or event_class == "cancel":
        cancellation_removal_ns = sequence_ns

    return SpoofingClockDecision(
        policy_id=SPOOFING_DETECTION_CLOCK_POLICY_ID,
        replay_time_ns=sequence_ns,
        fill_anchor_ns=fill_anchor_ns,
        cancellation_removal_ns=cancellation_removal_ns,
        incompatibilities=tuple(incompatibilities),
    )