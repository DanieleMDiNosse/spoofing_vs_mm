"""Explicit clocks for the provisional empirical-controls policy.

Canonical message ordering remains the responsibility of the replay driver. This
module only assigns independent semantic clocks and never substitutes one field
for another.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
import re
from typing import Any, Mapping

from .enums import normalize_enum_code

CLOCK_POLICY_ID = "empirical_clock_sequence_replay_trade_fill_sequence_cancel_v3_utc_provisional"
_FRACTION_RE = re.compile(r"\.(\d+)(?:Z|[+-]\d\d:\d\d)?$")


@dataclass(frozen=True)
class EmpiricalClockDecision:
    policy_id: str
    replay_time_ns: int | None
    fill_anchor_ns: int | None
    cancellation_removal_ns: int | None
    incompatibilities: tuple[str, ...]


def _timestamp_ns(value: Any) -> int | None:
    """Parse one UTC source-clock representation to integer epoch nanoseconds."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            raise ValueError("timezone-aware timestamp is incompatible with UTC source-clock policy")
        parsed = value
        nanosecond_remainder = 0
    else:
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none", "null"}:
            return None
        if text.endswith("Z") or re.search(r"[+-]\d\d:\d\d$", text):
            raise ValueError("timezone-aware timestamp is incompatible with UTC source-clock policy")
        fraction_match = _FRACTION_RE.search(text)
        fraction = fraction_match.group(1) if fraction_match else ""
        if len(fraction) > 9:
            raise ValueError("timestamp precision finer than nanoseconds")
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            raise ValueError("malformed timestamp") from None

        # datetime retains six digits; carry only source digits seven through nine.
        nanosecond_remainder = int((fraction[6:9] + "000")[:3]) if len(fraction) > 6 else 0
    epoch = datetime(1970, 1, 1)
    delta = parsed - epoch
    microseconds = ((delta.days * 86_400 + delta.seconds) * 1_000_000) + delta.microseconds
    return microseconds * 1_000 + nanosecond_remainder


def resolve_empirical_clocks(event: Mapping[str, Any]) -> EmpiricalClockDecision:
    """Resolve replay, fill-anchor and cancellation-removal clocks without fallback."""
    event_code = normalize_enum_code(event.get("ORDEREVENTTYPE (*)"))
    event_class = event.get("event_class")
    incompatibilities: list[str] = []

    def parse(field: str, invalid_reason: str) -> int | None:
        try:
            return _timestamp_ns(event.get(field))
        except ValueError:
            incompatibilities.append(invalid_reason)
            return None

    sequence_ns = parse("SEQUENCETIME", "invalid_sequencetime")
    trade_ns = parse("TRADETIME", "fill_invalid_tradetime")

    if sequence_ns is None and "invalid_sequencetime" not in incompatibilities:
        incompatibilities.append("missing_sequencetime")

    fill_anchor_ns: int | None = None
    cancellation_removal_ns: int | None = None
    if event_code == 3 or event_class == "fill":
        fill_anchor_ns = trade_ns
        if trade_ns is None and "fill_invalid_tradetime" not in incompatibilities:
            incompatibilities.append("fill_missing_tradetime")
        elif trade_ns is not None and sequence_ns is not None and trade_ns > sequence_ns:
            incompatibilities.append("fill_tradetime_after_sequencetime")
    elif event_code == 4 or event_class == "cancel":
        cancellation_removal_ns = sequence_ns

    return EmpiricalClockDecision(
        policy_id=CLOCK_POLICY_ID,
        replay_time_ns=sequence_ns,
        fill_anchor_ns=fill_anchor_ns,
        cancellation_removal_ns=cancellation_removal_ns,
        incompatibilities=tuple(incompatibilities),
    )
