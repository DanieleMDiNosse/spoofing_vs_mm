"""Deterministic, actor-aware clustering of passive and aggressive executions.

The generic API operates on normalized fill mappings and keeps both aggregate
cluster rows and one provenance row per child fill.  The passive-only function
is retained as a legacy adapter for consumers that have not yet migrated to
role flags and actor-aware cluster identifiers.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .actor_identity import ActorIdentity, actor_identity_from_event


LIFECYCLE_BREAK_CLASSES = {
    "cancel",
    "modify_order",
    "new_order",
    "session_reload",
    "iceberg_refill",
    "move_dark_to_cob",
}
ANCHOR_MODES = frozenset({"passive", "aggressive"})


@dataclass
class PendingExecutionCluster:
    partition_id: str | None
    actor: ActorIdentity
    execution_anchor_mode: str
    side: str
    order_id: str
    sweep_id: str | None
    first_sort_index: int
    last_sort_index: int
    start_ts: datetime
    end_ts: datetime
    first_event_id: Any = None
    last_event_id: Any = None
    weighted_price_notional: float = 0.0
    fill_qty: float = 0.0
    child_fill_count: int = 0
    first_metric_row: dict[str, Any] = field(default_factory=dict)
    first_fill: dict[str, Any] = field(default_factory=dict)
    price_sources: set[str] = field(default_factory=set)
    members: list[dict[str, Any]] = field(default_factory=list)
    legacy_cluster_id: bool = False


@dataclass(frozen=True)
class _FillDetails:
    partition_id: str | None
    actor: ActorIdentity
    anchor: str
    side: str
    order_id: str
    sweep_id: str | None
    price: float
    price_source: str
    qty: float
    timestamp: datetime
    sort_index: int


def _as_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            value = value.astimezone(timezone.utc)
        return value.replace(tzinfo=None)
    return None


def _as_sort_index(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _finite_positive(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) and numeric > 0 else None


def _normalized_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text and text.lower() not in {"null", "none", "nan"} else None


def classify_execution_anchor(fill: Mapping[str, Any]) -> str | None:
    """Classify exclusive execution role flags as ``passive`` or ``aggressive``.

    Both flags set, neither flag set, and non-``Y`` values are deliberately not
    assigned to either analytical population.
    """
    passive = str(fill.get("PASSIVEORDER") or "").upper() == "Y"
    aggressive = str(fill.get("AGGRESSIVEORDER") or "").upper() == "Y"
    if passive == aggressive:
        return None
    return "passive" if passive else "aggressive"


def _actor_from_fill(fill: Mapping[str, Any]) -> ActorIdentity | None:
    """Resolve and validate canonical actor fields, including pre-resolved rows."""
    resolved = actor_identity_from_event(fill)
    supplied_key = _normalized_text(fill.get("actor_key"))
    if supplied_key is None:
        return resolved

    prefix, separator, actor_id = supplied_key.partition(":")
    if separator != ":" or prefix not in {"client_original", "firm"} or not actor_id:
        return None
    expected = ActorIdentity(
        actor_key=supplied_key,
        actor_id=actor_id,
        identity_level=prefix,
        identity_source=("NMSC_ORIGINALCLIENTIDSHORTCODE" if prefix == "client_original" else "FIRMID"),
        identity_fallback_flag=prefix == "firm",
    )
    if resolved is not None and resolved != expected:
        return None
    for field_name, expected_value in (
        ("actor_id", expected.actor_id),
        ("identity_level", expected.identity_level),
        ("identity_source", expected.identity_source),
        ("identity_fallback_flag", expected.identity_fallback_flag),
    ):
        supplied_value = fill.get(field_name)
        if supplied_value is not None and supplied_value != expected_value:
            return None
    return expected


def _fill_price(fill: Mapping[str, Any], anchor: str) -> tuple[float, str] | None:
    candidates: tuple[tuple[str, Any | None], ...]
    if anchor == "aggressive":
        candidates = (("LASTTRADEDPX", fill.get("LASTTRADEDPX")),)
        default_source = "LASTTRADEDPX"
    else:
        candidates = (("event_price", fill.get("event_price")), ("ORDERPX", fill.get("ORDERPX")))
        default_source = "active_order_price"
    for field_name, value in candidates:
        price = _finite_positive(value)
        if price is not None:
            return price, str(fill.get("execution_price_source") or default_source if field_name == "event_price" else field_name)
    return None


def _fill_details(fill: Mapping[str, Any], *, allowed_anchor_modes: frozenset[str]) -> _FillDetails | None:
    if fill.get("event_class") != "fill":
        return None
    anchor = classify_execution_anchor(fill)
    if anchor not in allowed_anchor_modes:
        return None
    actor = _actor_from_fill(fill)
    side = fill.get("side_label")
    order_id = _normalized_text(fill.get("ORDERID"))
    qty = _finite_positive(fill.get("LASTSHARES") if anchor == "aggressive" else fill.get("fill_qty"))
    timestamp = _as_datetime(fill.get("event_ts"))
    sort_index = _as_sort_index(fill.get("sort_index"))
    price_and_source = _fill_price(fill, anchor)
    if actor is None or side not in {"bid", "ask"} or order_id is None or qty is None or timestamp is None or sort_index is None or price_and_source is None:
        return None
    price, price_source = price_and_source
    return _FillDetails(
        partition_id=fill.get("partition_id"),
        actor=actor,
        anchor=anchor,
        side=str(side),
        order_id=order_id,
        sweep_id=_normalized_text(fill.get("execution_sweep_id")),
        price=price,
        price_source=price_source,
        qty=qty,
        timestamp=timestamp,
        sort_index=sort_index,
    )


def _base_group(details: _FillDetails) -> tuple[str | None, str, str, str, str]:
    return details.partition_id, details.actor.actor_key, details.order_id, details.anchor, details.side


def _same_actor(cluster: PendingExecutionCluster, actor: ActorIdentity) -> bool:
    return cluster.actor == actor


def _compatible_sweep(cluster: PendingExecutionCluster, details: _FillDetails) -> bool:
    """Prevent unrelated orders from joining a same-actor/side temporal run."""
    if cluster.order_id != details.order_id:
        return False
    if cluster.sweep_id is not None or details.sweep_id is not None:
        compatible = cluster.sweep_id is not None and cluster.sweep_id == details.sweep_id
    else:
        compatible = True
    # Passive resting executions retain their historical price-sensitive rule.
    return compatible and (cluster.execution_anchor_mode != "passive" or cluster.members[0]["_cluster_price"] == details.price)


def can_extend_execution_cluster(
    cluster: PendingExecutionCluster,
    fill: Mapping[str, Any],
    *,
    max_gap_ms: int,
    intervening_lifecycle_break: bool = False,
) -> bool:
    """Whether a valid child fill can join an existing cluster."""
    if max_gap_ms < 0 or intervening_lifecycle_break:
        return False
    details = _fill_details(fill, allowed_anchor_modes=frozenset({cluster.execution_anchor_mode}))
    if details is None or _base_group(details) != (
        cluster.partition_id,
        cluster.actor.actor_key,
        cluster.order_id,
        cluster.execution_anchor_mode,
        cluster.side,
    ):
        return False
    if not _same_actor(cluster, details.actor) or not _compatible_sweep(cluster, details):
        return False
    if details.timestamp.date() != cluster.start_ts.date():
        return False
    gap_ms = (details.timestamp - cluster.end_ts).total_seconds() * 1_000.0
    return 0 <= gap_ms <= max_gap_ms


def _cluster_id(first_sort_index: int, last_sort_index: int, *, anchor: str, legacy: bool) -> str:
    if legacy:
        return f"EC{first_sort_index:09d}-{last_sort_index:09d}"
    return f"EC-{'P' if anchor == 'passive' else 'A'}-{first_sort_index:09d}-{last_sort_index:09d}"


def _start_cluster(fill: Mapping[str, Any], details: _FillDetails, *, legacy_cluster_id: bool) -> PendingExecutionCluster:
    cluster = PendingExecutionCluster(
        partition_id=details.partition_id,
        actor=details.actor,
        execution_anchor_mode=details.anchor,
        side=details.side,
        order_id=details.order_id,
        sweep_id=details.sweep_id,
        first_sort_index=details.sort_index,
        last_sort_index=details.sort_index,
        start_ts=details.timestamp,
        end_ts=details.timestamp,
        first_event_id=fill.get("EVENTID"),
        last_event_id=fill.get("EVENTID"),
        weighted_price_notional=details.qty * details.price,
        fill_qty=details.qty,
        child_fill_count=1,
        first_metric_row=dict(fill.get("metric_row") or {}),
        first_fill=dict(fill),
        price_sources={details.price_source},
        legacy_cluster_id=legacy_cluster_id,
    )
    cluster.members.append({**dict(fill), "_cluster_price": details.price, "_cluster_qty": details.qty, "_cluster_ts": details.timestamp, "_cluster_price_source": details.price_source})
    return cluster


def _append_child(cluster: PendingExecutionCluster, fill: Mapping[str, Any], details: _FillDetails) -> None:
    cluster.last_sort_index = details.sort_index
    cluster.end_ts = details.timestamp
    cluster.last_event_id = fill.get("EVENTID")
    cluster.weighted_price_notional += details.qty * details.price
    cluster.fill_qty += details.qty
    cluster.child_fill_count += 1
    cluster.price_sources.add(details.price_source)
    cluster.members.append({**dict(fill), "_cluster_price": details.price, "_cluster_qty": details.qty, "_cluster_ts": details.timestamp, "_cluster_price_source": details.price_source})


def _member_row(cluster_id: str, cluster: PendingExecutionCluster, fill: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "execution_cluster_id": cluster_id,
        "partition_id": cluster.partition_id,
        "actor_key": cluster.actor.actor_key,
        "actor_id": cluster.actor.actor_id,
        "identity_level": cluster.actor.identity_level,
        "identity_source": cluster.actor.identity_source,
        "identity_fallback_flag": cluster.actor.identity_fallback_flag,
        "execution_anchor_mode": cluster.execution_anchor_mode,
        "child_order_id": str(fill.get("ORDERID")) if fill.get("ORDERID") is not None else None,
        "child_event_id": fill.get("EVENTID"),
        "child_sort_index": int(fill["sort_index"]),
        "child_execution_id": fill.get("EXECUTIONID"),
        "child_trade_uid": fill.get("TRADEUNIQUEIDENTIFIER"),
        "child_fill_qty": fill["_cluster_qty"],
        "child_fill_price": fill["_cluster_price"],
        "child_event_ts": fill["_cluster_ts"],
        "child_execution_anchor_mode": cluster.execution_anchor_mode,
        "child_execution_price_source": fill["_cluster_price_source"],
        "child_execution_sweep_id": fill.get("execution_sweep_id"),
        "child_event_client_original_id": fill.get("client_original_id"),
        "child_event_firm_id": fill.get("firm_id"),
    }


def finalize_execution_cluster(cluster: PendingExecutionCluster, *, gap_ms: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Materialize one aggregate row and its raw child-fill provenance rows."""
    cluster_id = _cluster_id(
        cluster.first_sort_index,
        cluster.last_sort_index,
        anchor=cluster.execution_anchor_mode,
        legacy=cluster.legacy_cluster_id,
    )
    first = cluster.first_fill
    vwap = cluster.weighted_price_notional / cluster.fill_qty
    price_sources = sorted(cluster.price_sources)
    execution_prices = [float(member["_cluster_price"]) for member in cluster.members]
    price_min = min(execution_prices)
    price_max = max(execution_prices)
    price_level_count = len(set(execution_prices))
    passive = cluster.execution_anchor_mode == "passive"
    aggressive = cluster.execution_anchor_mode == "aggressive"
    smallness = {}
    for scope in ("market", "actor"):
        denominator = _finite_positive(cluster.first_metric_row.get(f"same_level_{scope}_visible_qty_pre"))
        # Fixed first-fill queue benchmark; replenishment can make this exceed 1.
        smallness[f"smallness_fraction_{scope}_level"] = (
            cluster.fill_qty / denominator if passive and denominator is not None else None
        )
    cluster_row = {
        **cluster.first_metric_row,
        **smallness,
        "smallness_definition": "cluster_quantity_over_first_fill_pre_queue_v1",
        "execution_cluster_id": cluster_id,
        "partition_id": cluster.partition_id,
        "actor_key": cluster.actor.actor_key,
        "actor_id": cluster.actor.actor_id,
        "identity_level": cluster.actor.identity_level,
        "identity_source": cluster.actor.identity_source,
        "identity_fallback_flag": cluster.actor.identity_fallback_flag,
        "execution_anchor_mode": cluster.execution_anchor_mode,
        "execution_price_source": price_sources[0] if len(price_sources) == 1 else "|".join(price_sources),
        "execution_sweep_id": cluster.sweep_id,
        "event_client_original_id": first.get("client_original_id"),
        "event_firm_id": first.get("firm_id"),
        "cluster_first_event_id": cluster.first_event_id,
        "cluster_last_event_id": cluster.last_event_id,
        "cluster_first_sort_index": cluster.first_sort_index,
        "cluster_last_sort_index": cluster.last_sort_index,
        "cluster_start_ts": cluster.start_ts,
        "cluster_end_ts": cluster.end_ts,
        "child_fill_count": cluster.child_fill_count,
        "event_order_id": cluster.order_id,
        "event_side": cluster.side,
        "event_price": vwap,
        "fill_qty": cluster.fill_qty,
        # Canonical execution fields shared by both anchor modes.  ``fill_qty``
        # and ``event_price`` remain as compatibility aliases.
        "execution_quantity": cluster.fill_qty,
        "execution_vwap": vwap,
        "execution_price_level_count": price_level_count,
        "execution_price_min": price_min,
        "execution_price_max": price_max,
        # Branch-specific diagnostics are intentionally null outside their
        # applicable anchor; non-applicability is not encoded as zero.
        "passive_execution_diagnostics_applicable": passive,
        "aggressive_execution_diagnostics_applicable": aggressive,
        "passive_execution_quantity": cluster.fill_qty if passive else None,
        "passive_execution_vwap": vwap if passive else None,
        "passive_child_fill_count": cluster.child_fill_count if passive else None,
        "passive_same_level_market_visible_qty_pre": (
            cluster.first_metric_row.get("same_level_market_visible_qty_pre") if passive else None
        ),
        "passive_same_level_actor_visible_qty_pre": (
            cluster.first_metric_row.get("same_level_actor_visible_qty_pre") if passive else None
        ),
        "passive_smallness_fraction_market_level": (
            smallness["smallness_fraction_market_level"]
        ),
        "passive_smallness_fraction_actor_level": (
            smallness["smallness_fraction_actor_level"]
        ),
        "aggressive_execution_quantity": cluster.fill_qty if aggressive else None,
        "aggressive_execution_vwap": vwap if aggressive else None,
        "aggressive_child_fill_count": cluster.child_fill_count if aggressive else None,
        "aggressive_execution_price_level_count": price_level_count if aggressive else None,
        "aggressive_execution_price_min": price_min if aggressive else None,
        "aggressive_execution_price_max": price_max if aggressive else None,
        "aggressive_execution_sweep_id": cluster.sweep_id if aggressive else None,
        "execution_cluster_gap_ms": gap_ms,
        "execution_cluster_quality_flags": "",
        # Stable event aliases consumed by downstream metric assembly.
        "sort_index": cluster.first_sort_index,
        "event_ts": cluster.start_ts,
        "ORDERID": cluster.order_id,
        "execution_side": cluster.side,
    }
    return cluster_row, [_member_row(cluster_id, cluster, member) for member in cluster.members]


def _is_terminal_fill(fill: Mapping[str, Any]) -> bool:
    leaves = fill.get("LEAVESQTY")
    try:
        return leaves is not None and float(leaves) <= 0
    except (TypeError, ValueError):
        return False


def _lifecycle_break_key(event: Mapping[str, Any]) -> tuple[str | None, str] | None:
    if event.get("event_class") not in LIFECYCLE_BREAK_CLASSES:
        return None
    order_id = _normalized_text(event.get("ORDERID"))
    return (event.get("partition_id"), order_id) if order_id is not None else None


def _event_sort_key(event: Mapping[str, Any]) -> tuple[int | float, datetime, str, str, str, str]:
    timestamp = _as_datetime(event.get("event_ts")) or datetime.max
    sort_index = _as_sort_index(event.get("sort_index"))
    return (
        sort_index if sort_index is not None else math.inf,
        timestamp,
        str(event.get("partition_id") or ""),
        str(event.get("ORDERID") or ""),
        str(event.get("EVENTID") or ""),
        repr(sorted((str(key), repr(value)) for key, value in event.items())),
    )


def _validate_max_gap_ms(max_gap_ms: int, *, allow_zero: bool = False) -> int:
    try:
        numeric = int(max_gap_ms)
    except (TypeError, ValueError) as exc:
        raise ValueError("max_gap_ms must be positive") from exc
    if numeric < 0 or (numeric == 0 and not allow_zero):
        raise ValueError("max_gap_ms must be positive")
    return numeric


def _validate_allowed_anchor_modes(allowed_anchor_modes: Iterable[str]) -> frozenset[str]:
    try:
        modes = tuple(allowed_anchor_modes)
    except TypeError as exc:
        raise ValueError("allowed_anchor_modes must be a non-empty iterable of known modes") from exc
    if not modes or len(set(modes)) != len(modes) or not set(modes).issubset(ANCHOR_MODES):
        raise ValueError("allowed_anchor_modes must be a non-empty unique subset of passive/aggressive")
    return frozenset(modes)


def cluster_execution_fills(
    fills: Iterable[Mapping[str, Any]],
    *,
    max_gap_ms: int,
    allowed_anchor_modes: Iterable[str] = ("passive", "aggressive"),
    _legacy_cluster_ids: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Cluster valid fills by actor, role branch, side, time, and sweep identity.

    Invalid/missing actor identities and ambiguous/missing role flags are omitted
    deterministically because this API returns only materialized clusters.
    """
    gap_ms = _validate_max_gap_ms(max_gap_ms, allow_zero=_legacy_cluster_ids)
    allowed = _validate_allowed_anchor_modes(allowed_anchor_modes)
    pending: dict[tuple[str | None, str, str, str, str], PendingExecutionCluster] = {}
    clusters: list[dict[str, Any]] = []
    members: list[dict[str, Any]] = []

    def finalize_key(key: tuple[str | None, str, str, str, str]) -> None:
        cluster = pending.pop(key, None)
        if cluster is not None:
            row, member_rows = finalize_execution_cluster(cluster, gap_ms=gap_ms)
            clusters.append(row)
            members.extend(member_rows)

    for event in sorted((dict(fill) for fill in fills), key=_event_sort_key):
        lifecycle_key = _lifecycle_break_key(event)
        if lifecycle_key is not None:
            for key, cluster in list(pending.items()):
                if (cluster.partition_id, cluster.order_id) == lifecycle_key:
                    finalize_key(key)
            continue
        details = _fill_details(event, allowed_anchor_modes=allowed)
        if details is None:
            continue
        key = _base_group(details)
        existing = pending.get(key)
        if existing is not None and not can_extend_execution_cluster(existing, event, max_gap_ms=gap_ms):
            finalize_key(key)
            existing = None
        if existing is None:
            pending[key] = _start_cluster(event, details, legacy_cluster_id=_legacy_cluster_ids)
        else:
            _append_child(existing, event, details)
        if _is_terminal_fill(event):
            finalize_key(key)

    for key in sorted(pending, key=lambda item: tuple("" if value is None else str(value) for value in item)):
        finalize_key(key)
    clusters.sort(key=lambda row: (int(row["cluster_first_sort_index"]), row["cluster_start_ts"], row["execution_cluster_id"]))
    members.sort(key=lambda row: (int(row["child_sort_index"]), row["child_event_ts"], str(row["child_event_id"] or "")))
    return clusters, members


def cluster_passive_execution_fills(
    events: Iterable[Mapping[str, Any]], *, max_gap_ms: int = 100
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Legacy adapter that selects passive fills and preserves legacy IDs.

    Older callers may only provide ``is_passive_fill`` instead of the exclusive
    role flags.  The adapter translates that historical marker before delegating
    all clustering behavior to :func:`cluster_execution_fills`.
    """
    if max_gap_ms < 0:
        raise ValueError("max_gap_ms must be non-negative")
    normalized: list[dict[str, Any]] = []
    for event in events:
        row = dict(event)
        if (
            "PASSIVEORDER" not in row
            and "AGGRESSIVEORDER" not in row
            and row.get("event_class") == "fill"
            and ("is_passive_fill" not in row or bool(row.get("is_passive_fill")))
        ):
            row["PASSIVEORDER"] = "Y"
            row.setdefault("AGGRESSIVEORDER", "N")
        normalized.append(row)
    clusters, members = cluster_execution_fills(
        normalized,
        max_gap_ms=max_gap_ms,
        allowed_anchor_modes=("passive",),
        _legacy_cluster_ids=True,
    )
    for cluster in clusters:
        cluster["client_id"] = cluster["actor_id"] if cluster["identity_level"] == "client_original" else None
    return clusters, members
