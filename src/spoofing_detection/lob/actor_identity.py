"""Canonical, namespaced actor identity resolution for LOB records."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

from .models import ActiveOrder


@dataclass(frozen=True)
class ActorIdentity:
    actor_key: str
    actor_id: str
    identity_level: str
    identity_source: str
    identity_fallback_flag: bool


def normalize_identity_value(value: Any) -> str | None:
    """Return a stable identity string, or ``None`` for missing values."""
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip()
        return None if normalized.lower() in {"", "null", "none", "nan"} else normalized
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if value.is_integer():
            return str(int(value))
    return str(value)


def normalize_client_original_identity_value(value: Any) -> str | None:
    """Normalize the client shortcode, treating the feed's zero sentinel as missing."""
    normalized = normalize_identity_value(value)
    return None if is_zero_client_original_identity_sentinel(normalized) else normalized


def is_zero_client_original_identity_sentinel(value: Any) -> bool:
    """Return whether a client identity is any finite numeric representation of zero."""
    normalized = normalize_identity_value(value)
    if normalized is None:
        return False
    try:
        numeric = Decimal(normalized)
    except (InvalidOperation, ValueError):
        return False
    return numeric.is_finite() and numeric == 0


def resolve_actor_identity(*, client_original_id: Any, firm_id: Any) -> ActorIdentity | None:
    """Prefer original-client identity, falling back only to firm identity."""
    client = normalize_client_original_identity_value(client_original_id)
    if client is not None:
        return ActorIdentity(
            actor_key=f"client_original:{client}",
            actor_id=client,
            identity_level="client_original",
            identity_source="NMSC_ORIGINALCLIENTIDSHORTCODE",
            identity_fallback_flag=False,
        )

    firm = normalize_identity_value(firm_id)
    if firm is not None:
        return ActorIdentity(
            actor_key=f"firm:{firm}",
            actor_id=firm,
            identity_level="firm",
            identity_source="FIRMID",
            identity_fallback_flag=True,
        )
    return None


def actor_identity_from_event(event: Mapping[str, Any]) -> ActorIdentity | None:
    """Resolve identity from a normalized project event mapping."""
    return resolve_actor_identity(
        client_original_id=event.get("client_original_id"),
        firm_id=event.get("firm_id"),
    )


def actor_identity_from_order(order: ActiveOrder) -> ActorIdentity | None:
    """Resolve identity from an active resting order."""
    return resolve_actor_identity(
        client_original_id=order.client_original_id,
        firm_id=order.firm_id,
    )


def same_actor(left: ActorIdentity | None, right: ActorIdentity | None) -> bool:
    """Return whether two resolved identities have the same namespace-aware key."""
    return (
        isinstance(left, ActorIdentity)
        and isinstance(right, ActorIdentity)
        and left.actor_key == right.actor_key
    )
