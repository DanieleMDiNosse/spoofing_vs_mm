#!/usr/bin/env python3
"""Generate a restricted CONSOB-facing register of strict detector events."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from spoofing_detection.lob.actor_identity import (  # noqa: E402
    is_zero_client_original_identity_sentinel,
)


INSTRUMENT_ORDER = ("FERRARI", "NEXI", "RISANAMENTO")
MAR_ARTICLE_15_URL = (
    "https://eur-lex.europa.eu/legal-content/IT/TXT/?uri=CELEX:32014R0596"
)
STRICT_GATE_FIELDS = (
    "episode_has_matched_withdrawal",
    "episode_gate_joint_smallness",
    "episode_price_path_observed",
    "spoofing_compatible_episode",
    "episode_strict_detection",
)
REQUIRED_EVENT_FIELDS = (
    "episode_id",
    "episode_start_ts",
    "episode_end_ts",
    "actor_key",
    "identity_level",
    "identity_fallback_flag",
    "execution_cluster_id",
    "episode_representative_cluster_id",
    "episode_cluster_count",
    "episode_mixed_anchor",
    "execution_anchor_mode",
    "execution_side",
    "deceptive_side",
    "episode_total_execution_quantity",
    "episode_execution_vwap",
    "episode_unique_withdrawal_count",
    "episode_withdrawal_min_delay_seconds",
    "episode_withdrawal_max_delay_seconds",
    "episode_withdrawn_quantity",
    "episode_withdrawal_to_execution_ratio",
    "episode_favorable_mid_move",
    "episode_post_cancel_mid_reversion",
    "candidate_deceptive_order_count_pre",
    "candidate_deceptive_visible_qty_pre",
    "candidate_deceptive_min_age_seconds_pre",
    "candidate_deceptive_qty_weighted_depth_distance_ticks_pre",
    "MSCI_resting_profile",
    "withdrawal_profile_scale_event",
    *STRICT_GATE_FIELDS,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _episode_withdrawal_delay_summary(
    withdrawals: pl.DataFrame,
    execution_metrics: pl.DataFrame,
) -> pl.DataFrame:
    cluster_times = execution_metrics.select(
        "execution_cluster_id", "cluster_end_ts"
    ).unique()
    if cluster_times["execution_cluster_id"].n_unique() != cluster_times.height:
        raise ValueError("execution metrics contain conflicting cluster end times")
    delays = withdrawals.join(
        cluster_times,
        on="execution_cluster_id",
        how="left",
        validate="m:1",
    ).with_columns(
        (
            (pl.col("cancel_event_ts") - pl.col("cluster_end_ts")).dt.total_microseconds()
            / 1_000_000.0
        ).alias("cancel_delay_seconds")
    )
    if delays["cluster_end_ts"].null_count() or delays["cancel_delay_seconds"].null_count():
        raise ValueError("episode withdrawal lacks its execution-cluster end time")
    if delays.filter(pl.col("cancel_delay_seconds") < 0).height:
        raise ValueError("episode withdrawal precedes its execution cluster")
    return delays.group_by("episode_id").agg(
        pl.col("cancel_delay_seconds").min().alias(
            "episode_withdrawal_min_delay_seconds"
        ),
        pl.col("cancel_delay_seconds").max().alias(
            "episode_withdrawal_max_delay_seconds"
        ),
    )


def _as_bool(value: object, *, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    raise ValueError(f"{field} must be boolean, got {value!r}")


def _as_float(value: Any, *, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric, got {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite, got {value!r}")
    return result


def _as_int(value: object, *, field: str) -> int:
    number = _as_float(value, field=field)
    rounded = round(number)
    if not math.isclose(number, rounded, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"{field} must be integral, got {value!r}")
    return int(rounded)


def _parse_timestamp(value: object, *, field: str) -> datetime:
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty ISO timestamp")
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"invalid {field}: {value!r}") from exc


_ITALIAN_MONTHS = (
    "gennaio",
    "febbraio",
    "marzo",
    "aprile",
    "maggio",
    "giugno",
    "luglio",
    "agosto",
    "settembre",
    "ottobre",
    "novembre",
    "dicembre",
)
_REPORT_DATE_PATTERN = re.compile(
    rf"(?P<day>[1-9]|[12][0-9]|3[01]) "
    rf"(?P<month>{'|'.join(_ITALIAN_MONTHS)}) (?P<year>[0-9]{{4}})"
)


def _validate_report_date(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("report_date must use the Italian form 'D mese YYYY'")
    match = _REPORT_DATE_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError("report_date must use the Italian form 'D mese YYYY'")
    try:
        datetime(
            int(match.group("year")),
            _ITALIAN_MONTHS.index(match.group("month")) + 1,
            int(match.group("day")),
        )
    except ValueError as exc:
        raise ValueError(f"report_date is not a valid calendar date: {value!r}") from exc
    return value


def _validate_strict_events(
    events_by_instrument: dict[str, list[dict[str, Any]]],
    *,
    withdrawal_window_seconds: float,
) -> None:
    if not math.isfinite(withdrawal_window_seconds) or withdrawal_window_seconds <= 0:
        raise ValueError("withdrawal window must be positive and finite")
    seen_keys: set[tuple[str, str]] = set()
    for instrument, events in events_by_instrument.items():
        for event in events:
            missing = [field for field in REQUIRED_EVENT_FIELDS if field not in event]
            if missing:
                raise ValueError(
                    f"{instrument} strict-event row is missing fields: {', '.join(missing)}"
                )
            if not all(_as_bool(event[field], field=field) for field in STRICT_GATE_FIELDS):
                raise ValueError("strict-event register contains a failed gate")
            episode_id = str(event["episode_id"])
            if re.fullmatch(r"EP-[0-9a-f]{24}", episode_id) is None:
                raise ValueError("strict-event row has noncanonical episode_id")
            execution_quantity = _as_float(
                event["episode_total_execution_quantity"],
                field="episode_total_execution_quantity",
            )
            execution_vwap = _as_float(
                event["episode_execution_vwap"], field="episode_execution_vwap"
            )
            withdrawn_quantity = _as_float(
                event["episode_withdrawn_quantity"],
                field="episode_withdrawn_quantity",
            )
            favorable_move = _as_float(
                event["episode_favorable_mid_move"],
                field="episode_favorable_mid_move",
            )
            reversion = _as_float(
                event["episode_post_cancel_mid_reversion"],
                field="episode_post_cancel_mid_reversion",
            )
            if execution_quantity <= 0 or execution_vwap <= 0:
                raise ValueError("strict-event execution quantity and VWAP must be positive")
            if not withdrawn_quantity > execution_quantity:
                raise ValueError("strict-event row violates withdrawn quantity > execution quantity")
            if not favorable_move > 0.0:
                raise ValueError("strict episode has non-positive favorable post-publication move")
            if not reversion > 0.0:
                raise ValueError("strict-event row has non-positive cancel-anchored reversion")
            episode_start = _parse_timestamp(
                event["episode_start_ts"], field="episode_start_ts"
            )
            episode_end = _parse_timestamp(event["episode_end_ts"], field="episode_end_ts")
            if episode_end < episode_start:
                raise ValueError("strict-event row has an episode end before its start")
            _as_bool(event["episode_mixed_anchor"], field="episode_mixed_anchor")
            anchor = str(event["execution_anchor_mode"])
            if anchor not in {"passive", "aggressive"}:
                raise ValueError(f"unsupported representative anchor: {anchor!r}")
            execution_side = str(event["execution_side"]).lower()
            deceptive_side = str(event["deceptive_side"]).lower()
            expected_deceptive_side = {"bid": "ask", "ask": "bid"}.get(execution_side)
            if expected_deceptive_side is None or deceptive_side != expected_deceptive_side:
                raise ValueError("strict-event deceptive side must be opposite to execution side")
            identity_level = str(event["identity_level"])
            if identity_level not in {"client_original", "firm"}:
                raise ValueError(f"unsupported identity level: {identity_level!r}")
            identity_fallback = _as_bool(
                event["identity_fallback_flag"], field="identity_fallback_flag"
            )
            if identity_fallback != (identity_level == "firm"):
                raise ValueError("strict-event identity fallback flag is inconsistent")
            if not str(event["actor_key"]).startswith(f"{identity_level}:"):
                raise ValueError("strict-event actor key and identity level are inconsistent")
            original_identifier = str(event["actor_key"]).split(":", 1)[1]
            if re.fullmatch(r"[A-Za-z0-9_.-]+", original_identifier) is None:
                raise ValueError("strict-event original actor identifier is not safe to render")
            for count_field in (
                "episode_cluster_count",
                "episode_unique_withdrawal_count",
            ):
                if _as_int(event[count_field], field=count_field) <= 0:
                    raise ValueError(f"{count_field} must be a positive integer")
            min_delay = _as_float(
                event["episode_withdrawal_min_delay_seconds"],
                field="episode_withdrawal_min_delay_seconds",
            )
            max_delay = _as_float(
                event["episode_withdrawal_max_delay_seconds"],
                field="episode_withdrawal_max_delay_seconds",
            )
            if min_delay < 0 or max_delay < min_delay:
                raise ValueError("strict-event cancellation delays must satisfy 0 <= min <= max")
            if max_delay > withdrawal_window_seconds + 1e-9:
                raise ValueError("strict-event cancellation delay exceeds the withdrawal window")
            reported_ratio = _as_float(
                event["episode_withdrawal_to_execution_ratio"],
                field="episode_withdrawal_to_execution_ratio",
            )
            if not math.isclose(
                reported_ratio,
                withdrawn_quantity / execution_quantity,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError("strict episode withdrawal ratio is inconsistent")
            if str(event["episode_representative_cluster_id"]) != str(
                event["execution_cluster_id"]
            ):
                raise ValueError("strict episode row is not its representative cluster")
            key = (instrument, episode_id)
            if key in seen_keys:
                raise ValueError(f"duplicate canonical strict episode: {key!r}")
            seen_keys.add(key)


def _same_number(left: object, right: object) -> bool:
    return math.isclose(
        _as_float(left, field="comparison value"),
        _as_float(right, field="comparison value"),
        rel_tol=0.0,
        abs_tol=1e-9,
    )


def _episode_anchor_mode(event: dict[str, Any]) -> str:
    if _as_bool(event["episode_mixed_anchor"], field="episode_mixed_anchor"):
        return "mixed"
    return str(event["execution_anchor_mode"])


def _event_matches_audit(event: dict[str, Any], audit_event: dict[str, Any]) -> bool:
    return (
        _parse_timestamp(event["episode_start_ts"], field="episode_start_ts")
        == _parse_timestamp(audit_event["cluster_start"], field="cluster_start")
        and _parse_timestamp(event["episode_end_ts"], field="episode_end_ts")
        == _parse_timestamp(audit_event["cluster_end"], field="cluster_end")
        and _episode_anchor_mode(event) == str(audit_event["execution_anchor_mode"])
        and str(event["execution_side"]) == str(audit_event["execution_side"])
        and _same_number(
            event["episode_total_execution_quantity"],
            audit_event["execution_quantity"],
        )
        and _same_number(event["episode_execution_vwap"], audit_event["execution_vwap"])
    )


def _validated_period_identifier(
    value: object, *, instrument: str, union_period: bool
) -> str:
    identifier = str(value)
    kind = "union" if union_period else "source"
    separator = "-union-" if union_period else "-"
    pattern = rf"{re.escape(instrument)}{separator}[0-9]{{3,}}"
    if re.fullmatch(pattern, identifier) is None:
        raise ValueError(
            f"invalid {kind} period identifier for {instrument}: {identifier!r}"
        )
    return identifier


def _strict_audit_events(audit: dict[str, Any]) -> list[dict[str, Any]]:
    datasets = audit.get("datasets")
    if not isinstance(datasets, dict):
        raise ValueError("audit datasets mapping is missing")
    strict: dict[str, dict[str, Any]] = {}
    for instrument, dataset in datasets.items():
        if not isinstance(dataset, dict):
            raise ValueError(f"audit dataset {instrument!r} must be an object")
        source_rows = dataset.get("source_period_results", [])
        union_rows = dataset.get("union_timed_results", [])
        if not isinstance(source_rows, list) or not isinstance(union_rows, list):
            raise ValueError(f"audit dataset {instrument!r} has invalid period collections")
        for source_row in source_rows:
            detector_events = source_row.get("detector_events") or []
            for audit_event in detector_events:
                if not _as_bool(
                    audit_event.get("strict_detection"), field="strict_detection"
                ):
                    continue
                event_alias = str(audit_event.get("event_alias", ""))
                if not event_alias.startswith("detector_event_"):
                    raise ValueError("strict audit event is missing its report-local alias")
                source_actor_alias = str(source_row.get("actor_alias", ""))
                if not source_actor_alias:
                    raise ValueError("strict audit event is missing its source-subject alias")
                identity_granularity_aligned = _as_bool(
                    source_row.get("identity_granularity_aligned"),
                    field="identity_granularity_aligned",
                )
                entry = strict.setdefault(
                    event_alias,
                    {
                        "instrument": str(instrument),
                        "event_alias": event_alias,
                        "actor_alias": source_actor_alias,
                        "identity_granularity_aligned": identity_granularity_aligned,
                        "details": audit_event,
                        "source_period_ids": [],
                    },
                )
                if (
                    entry["instrument"] != str(instrument)
                    or entry["actor_alias"] != source_actor_alias
                    or entry["identity_granularity_aligned"]
                    != identity_granularity_aligned
                    or entry["details"] != audit_event
                ):
                    raise ValueError(f"inconsistent duplicate audit event {event_alias}")
                period_id = _validated_period_identifier(
                    source_row.get("period_id", ""),
                    instrument=str(instrument),
                    union_period=False,
                )
                if period_id not in entry["source_period_ids"]:
                    entry["source_period_ids"].append(period_id)
        for entry in [value for value in strict.values() if value["instrument"] == instrument]:
            start = _parse_timestamp(entry["details"]["cluster_start"], field="cluster_start")
            end = _parse_timestamp(entry["details"]["cluster_end"], field="cluster_end")
            candidates = []
            for union_row in union_rows:
                if union_row.get("actor_alias") != entry["actor_alias"]:
                    continue
                union_start = _parse_timestamp(union_row.get("start"), field="union start")
                union_end = _parse_timestamp(union_row.get("end"), field="union end")
                if start <= union_end and end >= union_start:
                    candidates.append(
                        _validated_period_identifier(
                            union_row.get("union_period_id", ""),
                            instrument=str(instrument),
                            union_period=True,
                        )
                    )
            if len(candidates) != 1 or not candidates[0]:
                raise ValueError(
                    f"strict audit event {entry['event_alias']} must map to exactly one union interval"
                )
            entry["union_period_id"] = candidates[0]
    return sorted(strict.values(), key=lambda row: (row["instrument"], row["event_alias"]))


def _resolve_consob_tags(
    events_by_instrument: dict[str, list[dict[str, Any]]],
    audit: dict[str, Any],
) -> dict[tuple[str, int], dict[str, Any]]:
    tags: dict[tuple[str, int], dict[str, Any]] = {}
    for audit_event in _strict_audit_events(audit):
        instrument = audit_event["instrument"]
        matches = [
            index
            for index, event in enumerate(events_by_instrument.get(instrument, []))
            if _event_matches_audit(event, audit_event["details"])
        ]
        if len(matches) != 1:
            raise ValueError(
                f"audit event {audit_event['event_alias']} must match exactly one canonical strict event; "
                f"found {len(matches)}"
            )
        key = (instrument, matches[0])
        if key in tags:
            raise ValueError("multiple audit aliases resolve to one canonical strict event")
        tags[key] = audit_event
    return tags


def _instrument_sort_key(instrument: str) -> tuple[int, str]:
    try:
        return (INSTRUMENT_ORDER.index(instrument), instrument)
    except ValueError:
        return (len(INSTRUMENT_ORDER), instrument)


def _event_sort_key(event: dict[str, Any]) -> tuple[datetime, datetime, str, str, str]:
    return (
        _parse_timestamp(event["episode_start_ts"], field="episode_start_ts"),
        _parse_timestamp(event["episode_end_ts"], field="episode_end_ts"),
        _episode_anchor_mode(event),
        str(event["episode_id"]),
        str(event["actor_key"]),
    )


def _it_number(value: float, decimals: int) -> str:
    rendered = f"{value:,.{decimals}f}"
    return rendered.replace(",", "\x00").replace(".", ",").replace("\x00", ".")


def _it_compact_number(value: float, *, min_decimals: int, max_decimals: int = 6) -> str:
    if min_decimals > max_decimals:
        raise ValueError("min_decimals cannot exceed max_decimals")
    rendered = _it_number(value, max_decimals)
    if "," not in rendered:
        return rendered
    integer, fraction = rendered.rsplit(",", 1)
    fraction = fraction.rstrip("0")
    if len(fraction) < min_decimals:
        fraction += "0" * (min_decimals - len(fraction))
    if not fraction:
        return integer
    return f"{integer},{fraction}"


def _seconds_label(value: float) -> str:
    unit = "secondo" if math.isclose(value, 1.0, rel_tol=0.0, abs_tol=1e-9) else "secondi"
    return f"{_it_compact_number(value, min_decimals=0)} {unit}"


def _fmt_quantity(value: object) -> str:
    number = _as_float(value, field="quantity")
    decimals = 0 if math.isclose(number, round(number), abs_tol=1e-9) else 2
    return _it_number(number, decimals)


def _price_decimals(tick_size: float) -> int:
    if tick_size <= 0.0 or not math.isfinite(tick_size):
        raise ValueError(f"invalid tick size: {tick_size!r}")
    return max(2, min(6, int(math.ceil(-math.log10(tick_size)))))


def _fmt_timestamp(value: object) -> str:
    timestamp = _parse_timestamp(value, field="event timestamp")
    return timestamp.strftime("%d/%m/%Y %H:%M:%S.%f")[:-3]


def _count_label(value: object, *, field: str, singular: str, plural: str) -> str:
    count = _as_int(value, field=field)
    return f"{count} {singular if count == 1 else plural}"


def _median(values: Iterable[float]) -> float:
    materialized = list(values)
    if not materialized:
        raise ValueError("cannot compute a median of an empty collection")
    return float(statistics.median(materialized))


def _is_zero_client_actor_key(value: object) -> bool:
    namespace, separator, actor_id = str(value).partition(":")
    return (
        bool(separator)
        and namespace == "client_original"
        and is_zero_client_original_identity_sentinel(actor_id)
    )


def _identity_label(event: dict[str, Any]) -> str:
    original_identifier = str(event["actor_key"]).split(":", 1)[1]
    if _is_zero_client_actor_key(event["actor_key"]):
        return (
            "Identificativo non attribuibile: `0`<br>"
            "il valore tecnico non identifica un cliente o un intermediario"
        )
    identity_level = str(event["identity_level"])
    if identity_level == "firm":
        return (
            f"Intermediario: `{original_identifier}`<br>"
            "Codice cliente non disponibile per l'evento"
        )
    if identity_level == "client_original":
        return f"Cliente: `{original_identifier}`"
    raise ValueError(f"unsupported identity level: {identity_level!r}")


def _book_side(value: object) -> str:
    side = str(value).lower()
    if side not in {"bid", "ask"}:
        raise ValueError(f"unsupported book side: {value!r}")
    return side


def _transaction_label(value: object) -> str:
    return {"bid": "acquisto", "ask": "vendita"}[_book_side(value)]


def _orders_label(value: object) -> str:
    return {"bid": "ordini di acquisto", "ask": "ordini di vendita"}[
        _book_side(value)
    ]


def _best_quote_label(value: object) -> str:
    return {
        "bid": "migliore proposta di acquisto",
        "ask": "migliore proposta di vendita",
    }[_book_side(value)]


def _audit_count(audit: dict[str, Any], field: str) -> int:
    overall = audit.get("overall")
    if not isinstance(overall, dict) or field not in overall:
        raise ValueError(f"audit overall count is missing: {field}")
    value = overall[field]
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"audit overall count {field} must be a nonnegative integer")
    return value


def _dataset_counts(dataset: dict[str, Any]) -> dict[str, int]:
    source_rows = dataset.get("source_period_results", [])
    union_rows = dataset.get("union_timed_results", [])
    date_only = sum(row.get("scope") == "date" for row in source_rows)
    return {
        "source": int(dataset.get("source_external_periods", len(source_rows))),
        "date_only": int(dataset.get("source_date_only_periods", date_only)),
        "union": int(dataset.get("union_timed_periods", len(union_rows))),
        "execution": int(
            dataset.get(
                "union_periods_with_recovered_execution",
                sum((row.get("recovered_execution_clusters") or 0) > 0 for row in union_rows),
            )
        ),
        "withdrawal": int(
            dataset.get(
                "union_periods_with_matched_withdrawal",
                sum((row.get("clusters_with_matched_withdrawal") or 0) > 0 for row in union_rows),
            )
        ),
        "strict_scope": int(
            dataset.get(
                "union_periods_with_strict_subject_scope_detection",
                sum((row.get("episodes_with_strict_detection") or 0) > 0 for row in union_rows),
            )
        ),
        "aligned": int(
            dataset.get(
                "identity_aligned_union_timed_periods",
                sum(bool(row.get("identity_granularity_aligned")) for row in union_rows),
            )
        ),
        "strict_exact": int(
            dataset.get(
                "identity_aligned_union_periods_with_strict_detection",
                sum(
                    bool(row.get("identity_granularity_aligned"))
                    and (row.get("episodes_with_strict_detection") or 0) > 0
                    for row in union_rows
                ),
            )
        ),
    }


def _date_only_alert_notes(datasets: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    for instrument in sorted(datasets, key=_instrument_sort_key):
        dataset = datasets[instrument]
        if not isinstance(dataset, dict):
            raise ValueError(f"audit dataset {instrument!r} must be an object")
        source_rows = dataset.get("source_period_results", [])
        if not isinstance(source_rows, list):
            raise ValueError(f"audit dataset {instrument!r} has invalid source periods")
        for row in source_rows:
            if row.get("scope") != "date":
                continue
            event_date = _parse_timestamp(
                row.get("start"), field=f"{instrument} date-only alert start"
            )
            counts = {
                "execution": _as_int(
                    row.get("date_level_candidate_posture_episodes"),
                    field="date_level_candidate_posture_episodes",
                ),
                "withdrawal": _as_int(
                    row.get("date_level_episodes_with_matched_withdrawal"),
                    field="date_level_episodes_with_matched_withdrawal",
                ),
                "strict": _as_int(
                    row.get("date_level_episodes_with_strict_detection"),
                    field="date_level_episodes_with_strict_detection",
                ),
            }
            if any(value < 0 for value in counts.values()):
                raise ValueError("date-only alert counts must be nonnegative")
            execution_text = (
                "non si osservano episodi con una configurazione candidata"
                if counts["execution"] == 0
                else (
                    "si osserva un episodio con una configurazione candidata"
                    if counts["execution"] == 1
                    else f"si osservano {counts['execution']} episodi con una configurazione candidata"
                )
            )
            withdrawal_text = (
                "Nessuno presenta cancellazioni successive che il sistema associa all'esecuzione."
                if counts["withdrawal"] == 0
                else (
                    "Uno presenta cancellazioni successive che il sistema associa all'esecuzione."
                    if counts["withdrawal"] == 1
                    else f"{counts['withdrawal']} presentano cancellazioni successive che il sistema associa alle esecuzioni."
                )
            )
            strict_text = (
                "Nessuno soddisfa tutti i criteri di selezione."
                if counts["strict"] == 0
                else (
                    "Uno soddisfa tutti i criteri di selezione."
                    if counts["strict"] == 1
                    else f"{counts['strict']} soddisfano tutti i criteri di selezione."
                )
            )
            notes.append(
                f"La segnalazione {instrument} del {event_date:%d/%m/%Y} riporta soltanto la data. "
                f"Per quella giornata, nei dati interni riferiti al soggetto segnalato {execution_text}. "
                f"{withdrawal_text} {strict_text} Poiché la segnalazione non specifica un orario, "
                "non viene associato alcun riscontro a un evento specifico."
            )
    return notes


_EVENT_ROW_ID_PATTERN = re.compile(r"^\| ([A-Z][A-Z0-9_-]*-E[0-9]{3}) \|")


def _unescaped_pipe_count(value: str) -> int:
    count = 0
    preceding_backslashes = 0
    for character in value:
        if character == "\\":
            preceding_backslashes += 1
            continue
        if character == "|" and preceding_backslashes % 2 == 0:
            count += 1
        preceding_backslashes = 0
    return count


def _validate_rendered_event_rows(
    rendered_rows: list[str], expected_event_ids: list[str]
) -> None:
    rendered_ids = []
    for row in rendered_rows:
        match = _EVENT_ROW_ID_PATTERN.match(row)
        if match is None:
            raise ValueError("rendered event row does not begin with a valid event-row ID")
        rendered_ids.append(match.group(1))
    if len(rendered_ids) != len(set(rendered_ids)) or rendered_ids != expected_event_ids:
        raise ValueError(
            "rendered event-row IDs or order do not exactly match the canonical register"
        )
    if any(_unescaped_pipe_count(row) != 8 for row in rendered_rows):
        raise ValueError("rendered event row does not contain exactly seven Markdown cells")


def _safe_report_check(
    report: str,
    events_by_instrument: dict[str, list[dict[str, Any]]],
    audit: dict[str, Any],
) -> None:
    forbidden_markers = ("/home/", "--input", "execution_cluster_id", "actor_key")
    for marker in forbidden_markers:
        if marker in report:
            raise ValueError(f"distributable report contains forbidden marker {marker!r}")
    for events in events_by_instrument.values():
        for event in events:
            actor_key = str(event["actor_key"])
            if actor_key in report:
                raise ValueError("distributable report contains a raw canonical actor key")
    for dataset in audit.get("datasets", {}).values():
        for source_row in dataset.get("source_period_results", []):
            actor_alias = source_row.get("actor_alias")
            if isinstance(actor_alias, str) and actor_alias and actor_alias in report:
                raise ValueError("distributable report contains an upstream actor alias")


def _reject_legacy_cluster_strict_audit(value: object) -> None:
    if isinstance(value, dict):
        if any(
            key in {"clusters_with_strict_detection", "date_level_clusters_with_strict_detection"}
            for key in value
        ):
            raise ValueError(
                "external audit predates candidate-posture episode semantics"
            )
        for child in value.values():
            _reject_legacy_cluster_strict_audit(child)
    elif isinstance(value, list):
        for child in value:
            _reject_legacy_cluster_strict_audit(child)


def build_report(
    *,
    events_by_instrument: dict[str, list[dict[str, Any]]],
    audit: dict[str, Any],
    run_info: dict[str, Any],
    report_date: str,
) -> str:
    """Render the complete restricted strict-event register."""
    report_date = _validate_report_date(report_date)
    _reject_legacy_cluster_strict_audit(audit)
    total_events = sum(len(events) for events in events_by_instrument.values())
    if total_events == 0:
        raise ValueError("the strict-event register is empty")
    if run_info.get("validation_status") != "passed":
        raise ValueError("canonical run validation status is not passed")
    instruments_info = run_info.get("instruments")
    if not isinstance(instruments_info, dict):
        raise ValueError("run_info instruments mapping is missing")
    parameters = run_info.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError("run_info parameters mapping is missing")
    withdrawal_window_seconds = _as_float(
        parameters.get("withdrawal_window_seconds"),
        field="withdrawal_window_seconds",
    )
    reversion_horizon_seconds = _as_float(
        parameters.get("reversion_horizon_seconds"),
        field="reversion_horizon_seconds",
    )
    if not math.isfinite(reversion_horizon_seconds) or reversion_horizon_seconds <= 0:
        raise ValueError("reversion horizon must be positive and finite")
    _validate_strict_events(
        events_by_instrument,
        withdrawal_window_seconds=withdrawal_window_seconds,
    )
    tags = _resolve_consob_tags(events_by_instrument, audit)
    subject_tag_count = sum(
        tag["identity_granularity_aligned"]
        and events_by_instrument[instrument][original_index]["identity_level"]
        == "client_original"
        for (instrument, original_index), tag in tags.items()
    )
    intermediary_tag_count = len(tags) - subject_tag_count
    sentinel_total = sum(
        _is_zero_client_actor_key(event["actor_key"])
        for events in events_by_instrument.values()
        for event in events
    )
    client_total = sum(
        str(event["identity_level"]) == "client_original"
        and not _is_zero_client_actor_key(event["actor_key"])
        for events in events_by_instrument.values()
        for event in events
    )
    intermediary_total = sum(
        str(event["identity_level"]) == "firm"
        for events in events_by_instrument.values()
        for event in events
    )
    if client_total + intermediary_total + sentinel_total != total_events:
        raise ValueError("identity categories do not form a partition of the event register")

    sorted_events: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    event_ids: dict[tuple[str, int], str] = {}
    for instrument in sorted(events_by_instrument, key=_instrument_sort_key):
        indexed = list(enumerate(events_by_instrument[instrument]))
        indexed.sort(key=lambda item: _event_sort_key(item[1]))
        sorted_events[instrument] = indexed
        for rank, (original_index, _) in enumerate(indexed, start=1):
            event_ids[(instrument, original_index)] = f"{instrument}-E{rank:03d}"

    ranked_events = [
        (instrument, original_index, event)
        for instrument, indexed in sorted_events.items()
        for original_index, event in indexed
    ]
    ranked_events.sort(
        key=lambda item: (
            -_as_float(
                item[2]["withdrawal_profile_scale_event"],
                field="withdrawal_profile_scale_event",
            ),
            _event_sort_key(item[2]),
            _instrument_sort_key(item[0]),
        )
    )

    lines = [
        "# Relazione sugli eventi compatibili con un possibile schema di spoofing",
        "",
        f"**Data della relazione:** {report_date}",
        "",
        "**Destinatario:** Commissione Nazionale per le Società e la Borsa (CONSOB)",
        "",
        "**Classificazione:** Riservato — contiene identificativi originali di clienti o intermediari; distribuzione limitata alla finalità di vigilanza.",
        "",
        "## 1. Oggetto e perimetro",
        "",
        (
            f"Il registro raccoglie **{total_events}** sequenze che soddisfano tutti i criteri di selezione "
            "descritti di seguito. Ogni sequenza è un episodio costruito attorno a una stessa configurazione candidata "
            "e può comprendere più cluster di esecuzioni; non coincide quindi con un singolo messaggio di mercato. "
            "La selezione serve a indirizzare la revisione umana e, da sola, non dimostra né un intento "
            "manipolativo né una violazione."
        ),
        "",
        "### Condizioni di ingresso nel registro",
        "",
        (
            "Per entrare in questo registro una sequenza deve presentare congiuntamente cinque elementi osservabili:"
        ),
        "",
        (
            "1. **Ordini di segno opposto:** prima dell'esecuzione sono visibili ordini di segno opposto "
            "all'operazione eseguita, raggruppati in base all'identificativo disponibile nei dati."
        ),
        (
            "2. **Ritiro successivo:** almeno una parte di tali ordini viene cancellata entro "
            f"{_seconds_label(withdrawal_window_seconds)} dal cluster di esecuzioni al quale la cancellazione è attribuita."
        ),
        (
            "3. **Dimensione del ritiro:** la quantità complessivamente cancellata supera quella eseguita."
        ),
        (
            "4. **Movimento precedente all'esecuzione:** tra la prima osservazione degli ordini e l'esecuzione, "
            "il midprice scende prima di un acquisto oppure sale prima di una vendita."
        ),
        (
            "5. **Movimento successivo alle cancellazioni:** dopo le cancellazioni, il midprice mostra "
            "in media un movimento inverso rispetto a quello precedente all'esecuzione."
        ),
        "",
        (
            "Le cinque condizioni sono cumulative e volutamente restrittive come filtro operativo: servono a "
            "isolare sequenze complete e facilmente verificabili, non a definire tutti i casi potenzialmente "
            "rilevanti. In particolare, non costituiscono requisiti giuridici necessari né una definizione "
            "esaustiva della manipolazione di mercato. Il regolamento europeo sugli abusi di mercato dispone che "
            "«Non è consentito effettuare manipolazioni di mercato o tentare di effettuare manipolazioni di "
            "mercato».[1] Di conseguenza, l'assenza di un'esecuzione, di un movimento favorevole del prezzo o di "
            "una successiva inversione non esclude di per sé che una condotta tentata possa meritare "
            "esame. Il mancato superamento di una delle cinque condizioni significa soltanto che la sequenza resta "
            "fuori da questo registro ristretto."
        ),
        "",
        (
            "Il richiamo al tentativo non implica che l'analisi possa provare l'intenzione. Il mero "
            "intento, non accompagnato da condotte osservabili, non è rilevabile dai dati e non viene inferito dalla "
            "procedura automatica; l'eventuale valutazione dell'intenzionalità richiede elementi ulteriori e una "
            "revisione del contesto complessivo."
        ),
        "",
        (
            "Il midprice è la media tra la migliore proposta di acquisto e la migliore proposta di vendita."
            + (
                f" In {sentinel_total} eventi l'unico valore identificativo disponibile è il codice tecnico `0`. "
                "Questi casi restano nel registro, ma il valore non identifica un cliente o un intermediario."
                if sentinel_total > 0
                else ""
            )
        ),
        "",
        (
            "Per consentire la riconciliazione con i dati operativi, il registro riporta l'identificativo "
            "originale usato per attribuire ciascun evento: il codice cliente, quando disponibile, oppure il "
            "codice dell'intermediario. Restano esclusi i codici degli ordini e dei gruppi di esecuzioni. I "
            "codici di evento sono riferimenti creati per questa relazione. Poiché contiene identificativi "
            "originali, il documento deve essere trattato e distribuito come materiale riservato."
        ),
        "",
        "### Legenda dei riscontri CONSOB",
        "",
        (
            "Nel seguito, una sovrapposizione temporale indica soltanto che gli intervalli si sovrappongono "
            "confrontando gli orari così come sono registrati nei due archivi. Non stabilisce una coincidenza "
            "temporale assoluta: il fuso orario e la sincronizzazione degli orologi devono ancora essere verificati."
        ),
        "",
        (
            "- `CONSOB_MATCH_SOGGETTO`: la sequenza ricade in un intervallo CONSOB e i dati consentono di collegare "
            "in modo univoco il cliente nei due archivi, mantenendo lo stesso livello di dettaglio identificativo. "
            "Il tag non prova, da solo, identità economica, intenzione o illecito."
        ),
        (
            "- `CONSOB_MATCH_INTERMEDIARIO`: la sequenza ricade nell'intervallo CONSOB, ma il collegamento è "
            "disponibile soltanto per l'intermediario oppure non è riconducibile in modo univoco a un cliente. "
            "Il riscontro non identifica quale cliente o operatore economico sottostante abbia generato l'attività."
        ),
        (
            "- `NUOVO_EVENTO`: secondo gli orari registrati, la sequenza non ricade in alcun intervallo CONSOB per "
            "il quale siano disponibili data e ora. Il tag indica quindi un evento nuovo rispetto agli intervalli temporali forniti, "
            "ma non esclude un'eventuale relazione sostanziale."
        ),
        "",
        "## 2. Quadro sintetico degli eventi",
        "",
        "| Titolo | Periodo osservato | Episodi | Solo passivi | Solo aggressivi | Misti | Attribuzione tramite codice cliente | Attribuzione tramite codice intermediario | Identificativo tecnico 0 | Riscontri CONSOB |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for instrument in sorted(sorted_events, key=_instrument_sort_key):
        events = [event for _, event in sorted_events[instrument]]
        dates = sorted(
            {
                _parse_timestamp(event["episode_start_ts"], field="episode_start_ts").date()
                for event in events
            }
        )
        period = (
            dates[0].strftime("%d/%m/%Y")
            if len(dates) == 1
            else f"{dates[0].strftime('%d/%m/%Y')}–{dates[-1].strftime('%d/%m/%Y')} ({len(dates)} sedute con eventi)"
        )
        anchor_counts = Counter(_episode_anchor_mode(event) for event in events)
        identity_counts = Counter(str(event["identity_level"]) for event in events)
        sentinel_count = sum(_is_zero_client_actor_key(event["actor_key"]) for event in events)
        client_attribution_count = identity_counts["client_original"] - sentinel_count
        tag_count = sum(key[0] == instrument for key in tags)
        lines.append(
            "| "
            + " | ".join(
                [
                    instrument,
                    period,
                    str(len(events)),
                    str(anchor_counts["passive"]),
                    str(anchor_counts["aggressive"]),
                    str(anchor_counts["mixed"]),
                    str(client_attribution_count),
                    str(identity_counts["firm"]),
                    str(sentinel_count),
                    str(tag_count),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "### Valori mediani degli indicatori",
            "",
            "La tabella riporta, per ciascun titolo, le mediane dei singoli indicatori. I valori di una stessa riga non descrivono necessariamente un unico evento osservato. I movimenti di prezzo sono espressi in tick, ossia in multipli del passo minimo di quotazione.",
            "",
            "| Titolo | Quantità eseguita nell'episodio | Quantità visibile nel cluster rappresentante | Rapporto cancellato/eseguito | Ritardo massimo delle cancellazioni | Variazione dalla pubblicazione | Inversione dopo le cancellazioni | WMSCI del cluster rappresentante | MSCI del cluster rappresentante |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for instrument in sorted(sorted_events, key=_instrument_sort_key):
        events = [event for _, event in sorted_events[instrument]]
        info = instruments_info.get(instrument)
        if not isinstance(info, dict):
            raise ValueError(f"run_info is missing instrument {instrument}")
        tick_size = _as_float(info.get("tick_size"), field=f"{instrument} tick_size")
        values = [
            instrument,
            _fmt_quantity(
                _median(
                    _as_float(
                        e["episode_total_execution_quantity"],
                        field="episode_total_execution_quantity",
                    )
                    for e in events
                )
            ),
            _fmt_quantity(
                _median(
                    _as_float(
                        e["candidate_deceptive_visible_qty_pre"],
                        field="candidate_deceptive_visible_qty_pre",
                    )
                    for e in events
                )
            ),
            _it_number(
                _median(
                    _as_float(
                        e["episode_withdrawal_to_execution_ratio"],
                        field="episode_withdrawal_to_execution_ratio",
                    )
                    for e in events
                ),
                2,
            ),
            _it_number(
                1000.0
                * _median(
                    _as_float(
                        e["episode_withdrawal_max_delay_seconds"],
                        field="episode_withdrawal_max_delay_seconds",
                    )
                    for e in events
                ),
                1,
            )
            + " ms",
            _it_number(
                _median(
                    _as_float(
                        e["episode_favorable_mid_move"],
                        field="episode_favorable_mid_move",
                    )
                    / tick_size
                    for e in events
                ),
                2,
            )
            + " tick",
            _it_number(
                _median(
                    _as_float(
                        e["episode_post_cancel_mid_reversion"],
                        field="episode_post_cancel_mid_reversion",
                    )
                    / tick_size
                    for e in events
                ),
                2,
            )
            + " tick",
            _it_number(
                _median(
                    _as_float(
                        e["withdrawal_profile_scale_event"],
                        field="withdrawal_profile_scale_event",
                    )
                    for e in events
                ),
                3,
            ),
            _it_number(
                _median(
                    _as_float(e["MSCI_resting_profile"], field="MSCI_resting_profile")
                    for e in events
                ),
                3,
            ),
        ]
        lines.append("| " + " | ".join(values) + " |")

    source = _audit_count(audit, "source_external_periods")
    timed = _audit_count(audit, "source_timed_periods")
    date_only = _audit_count(audit, "source_date_only_periods")
    union = _audit_count(audit, "union_timed_periods")
    recovered = _audit_count(audit, "union_periods_with_recovered_execution")
    withdrawals = _audit_count(audit, "union_periods_with_matched_withdrawal")
    strict_scope = _audit_count(
        audit, "union_periods_with_strict_subject_scope_detection"
    )
    aligned = _audit_count(audit, "identity_aligned_union_timed_periods")
    strict_exact = _audit_count(
        audit, "identity_aligned_union_periods_with_strict_detection"
    )
    unaligned = _audit_count(audit, "identity_unaligned_union_timed_periods")
    strict_unaligned = _audit_count(
        audit, "identity_unaligned_union_periods_with_strict_subject_scope_detection"
    )
    if aligned + unaligned != union:
        raise ValueError("aligned and unaligned identity counts do not sum to timed intervals")
    if strict_exact + strict_unaligned != strict_scope:
        raise ValueError("strict identity counts do not sum to strict subject-scope intervals")
    if strict_exact > aligned or strict_unaligned > unaligned:
        raise ValueError("strict identity counts exceed their corresponding interval populations")

    def counted_intervals(count: int) -> str:
        return "1 intervallo" if count == 1 else f"{count} intervalli"

    def percentage(count: int, denominator: int) -> str:
        if denominator <= 0:
            raise ValueError("percentage denominator must be positive")
        return _it_number(100.0 * count / denominator, 1) + "%"

    strict_identity_summary = (
        f"{strict_exact}/{strict_scope} con identificazione cliente allineata; "
        f"{strict_unaligned}/{strict_scope} solo a livello più ampio"
        if strict_scope > 0
        else "Nessuna sequenza completa negli intervalli intraday"
    )
    strict_identity_implication = (
        f"I {strict_unaligned} riscontri a livello di intermediario richiedono l'attribuzione del cliente "
        "sottostante prima di qualsiasi conclusione soggettiva."
        if strict_unaligned > 0
        else "Non vi sono riscontri completi a cui attribuire un soggetto."
    )

    if aligned == 0:
        aligned_strict_text = "non vi sono intervalli con identificazione allineata"
    elif strict_exact == 0:
        aligned_strict_text = (
            f"nessuno dei {aligned} contiene una sequenza che soddisfa tutti i criteri"
            if aligned > 1
            else "quell'intervallo non contiene una sequenza che soddisfa tutti i criteri"
        )
    else:
        aligned_strict_text = (
            f"{strict_exact} contengono una sequenza che soddisfa tutti i criteri"
        )
    if strict_unaligned == 0:
        unaligned_strict_text = (
            "Negli intervalli rimanenti non compaiono riscontri completi a un livello identificativo più ampio."
        )
    elif strict_unaligned == 1:
        unaligned_strict_text = (
            f"Il riscontro completo appartiene invece agli altri {unaligned} intervalli, nei quali "
            "il confronto è possibile soltanto a un livello più ampio, generalmente quello "
            "dell'intermediario."
        )
    else:
        unaligned_strict_text = (
            f"I {strict_unaligned} riscontri completi appartengono invece agli altri {unaligned} intervalli, nei quali "
            "il confronto è possibile soltanto a un livello più ampio, generalmente quello "
            "dell'intermediario."
        )
    lines.extend(
        [
            "",
            "### Lettura operativa per la prima istruttoria",
            "",
            "La tabella seguente separa ciò che il registro documenta da ciò che richiede ancora una verifica istruttoria. I valori non costituiscono una graduatoria di responsabilità né un giudizio sulla natura manipolativa delle condotte.",
            "",
            "| Elemento | Esito osservato | Implicazione per la lettura |",
            "|---|---|---|",
            f"| Registro ristretto | {total_events} sequenze complete | Ogni riga soddisfa i cinque criteri tecnici cumulativi; il registro individua piste di revisione, non violazioni. |",
            f"| Qualità dell'attribuzione | {client_total}/{total_events} ({percentage(client_total, total_events)}) con codice cliente; {intermediary_total}/{total_events} ({percentage(intermediary_total, total_events)}) solo tramite intermediario; {sentinel_total}/{total_events} ({percentage(sentinel_total, total_events)}) non attribuibili | L'identità economica del soggetto non è disponibile per la maggior parte delle sequenze. |",
            f"| Confronto intraday CONSOB | {strict_scope}/{union} ({percentage(strict_scope, union)}) intervalli con una sequenza completa | Il confronto individua sovrapposizioni comportamentali nel perimetro disponibile; non misura precisione o falsi positivi. |",
            f"| Identità nei riscontri completi | {strict_identity_summary} | {strict_identity_implication} |",
            "| Temporalità | Fuso orario, sincronizzazione, precisione e latenze non verificati | Ogni sovrapposizione è una corrispondenza a orologio registrato, da confermare prima di trattarla come coincidenza temporale assoluta. |",
            "",
            "**Passi istruttori suggeriti.** Per i riscontri a livello di intermediario, verificare prima la base temporale dei due archivi e poi acquisire i dati che consentano di attribuire l'attività al cliente sottostante. Per le sequenze senza riscontro CONSOB, usare il registro come coda di revisione e valutare il contesto operativo, inclusi eventuali comportamenti ripetuti, attività di market making o gestione dell'inventario ed elementi non contenuti nel libro degli ordini.",
            "",
            "## 3. Confronto con le segnalazioni CONSOB",
            "",
            (
                f"La fonte CONSOB comprende {source} segnalazioni. Di queste, {timed} indicano un intervallo con "
                f"data e ora e possono entrare nel confronto intraday; {date_only} indica soltanto la data e "
                "viene quindi esaminata separatamente. Quando più segnalazioni dello stesso soggetto e dello stesso "
                f"titolo si sovrappongono o si toccano, vengono riunite. Le {timed} segnalazioni con orario formano "
                f"così {union} intervalli intraday distinti, che costituiscono il denominatore del confronto seguente."
            ),
            "",
            "### Cosa coincide, e a quale livello",
            "",
            (
                f"- **Presenza di attività:** in {counted_intervals(recovered)} su {union} i dati interni contengono "
                "almeno un gruppo di esecuzioni attribuibile al soggetto segnalato, al livello identificativo "
                "disponibile. Questo conferma che l'attività è osservabile nell'intervallo, ma non che la sequenza "
                "soddisfi i criteri di selezione."
            ),
            (
                f"- **Sequenza comportamentale:** in {counted_intervals(withdrawals)} su {union} alle esecuzioni "
                "seguono cancellazioni che il sistema riesce ad associare. "
                f"In {counted_intervals(strict_scope)} su {union} compare almeno una sequenza che soddisfa tutti i "
                "criteri temporali, quantitativi e di andamento del prezzo. Questi sono riscontri comportamentali "
                "all'interno delle finestre CONSOB, non accertamenti di spoofing."
            ),
            (
                f"- **Identificazione:** in {counted_intervals(aligned)} su {union} il soggetto CONSOB corrisponde a "
                "un solo identificativo interno allo stesso livello di dettaglio; "
                f"{aligned_strict_text}. {unaligned_strict_text} In tali casi non è possibile stabilire quale "
                "cliente sottostante abbia generato l'attività."
            ),
            "",
            (
                f"Il risultato finale è quindi: **{strict_unaligned} sovrapposizioni temporali e comportamentali "
                f"a livello di intermediario, ma {strict_exact} corrispondenze complete con identificazione allineata**. "
                "Questi due numeri rispondono a domande diverse e non devono essere sommati o interpretati come una "
                "misura validata della capacità di individuare i singoli clienti."
            ),
            "",
            (
                "Due intervalli riferiti allo stesso soggetto e titolo vengono uniti quando si sovrappongono o si "
                "toccano, estremi compresi. Il registro riporta un riscontro soltanto quando la sequenza interna è "
                "collegata senza ambiguità al cluster rappresentante tramite l'inizio e la fine del cluster, il tipo di "
                "esecuzione, la direzione dell'ordine (acquisto o vendita), la quantità e il prezzo. In caso di "
                "ambiguità, il riscontro non è riportato nel registro. Questa riconciliazione collega il cluster "
                "rappresentante dell'episodio ai due artefatti interni usati per il confronto; non è una conferma "
                "indipendente dell'evento segnalato da CONSOB. "
                "I codici degli intervalli sono riferimenti della relazione e non identificano persone."
            ),
            "",
            (
                "Né la fonte CONSOB né i dati di mercato specificano il fuso orario. I riscontri si basano quindi "
                "sugli orari così come sono registrati nei due archivi. Prima di interpretarli come corrispondenze "
                "temporali occorre verificare fuso orario, sincronizzazione degli orologi, precisione dei timestamp "
                "ed eventuali latenze."
            ),
            "",
            "| Titolo | Segnalazioni CONSOB | Con sola data | Intervalli intraday distinti | Con esecuzione | Con cancellazioni associate | Con tutti i criteri nel perimetro del soggetto | Tutti i criteri tra quelli con identificazione allineata |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    datasets = audit.get("datasets", {})
    for instrument in sorted(datasets, key=_instrument_sort_key):
        counts = _dataset_counts(datasets[instrument])
        if counts["union"] == 0:
            comparison_values = ["—", "—", "—", "—", "—"]
        else:
            comparison_values = [
                str(counts["union"]),
                f"{counts['execution']}/{counts['union']}",
                f"{counts['withdrawal']}/{counts['union']}",
                f"{counts['strict_scope']}/{counts['union']}",
                (
                    f"{counts['strict_exact']}/{counts['aligned']}"
                    if counts["aligned"] > 0
                    else "—"
                ),
            ]
        lines.append(
            "| "
            + " | ".join(
                [
                    instrument,
                    str(counts["source"]),
                    str(counts["date_only"]),
                    *comparison_values,
                ]
            )
            + " |"
        )

    lines.extend(["", "### Riscontri riportati nel registro", ""])
    if tags:
        lines.append(
            f"Nel registro compaiono {len(tags)} riscontri: {subject_tag_count} "
            f"`CONSOB_MATCH_SOGGETTO` e {intermediary_tag_count} `CONSOB_MATCH_INTERMEDIARIO`. "
            "Per ciascun riscontro, la relativa riga riporta anche l'intervallo CONSOB."
        )
    else:
        lines.append("Nessun evento del registro coincide con una finestra CONSOB dotata di data e ora.")

    intermediary_dossiers = [
        (instrument, original_index, event, tags[(instrument, original_index)])
        for instrument, original_index, event in ranked_events
        if (instrument, original_index) in tags
        and not tags[(instrument, original_index)]["identity_granularity_aligned"]
    ]
    if intermediary_dossiers:
        lines.extend(
            [
                "",
                "### Schede dei riscontri a livello di intermediario",
                "",
                "Le schede seguenti sono una coda di primo esame. La priorità indicata riguarda la verificabilità della sequenza e non esprime una valutazione di responsabilità.",
                "",
                "| Evento | Intervallo CONSOB e sequenza interna | Attribuzione disponibile | Verifiche istruttorie prioritarie |",
                "|---|---|---|---|",
            ]
        )
        for instrument, original_index, event, tag in intermediary_dossiers:
            source_periods = ", ".join(tag["source_period_ids"])
            event_id = event_ids[(instrument, original_index)]
            interval_and_sequence = (
                f"Intervallo: {tag['union_period_id']}<br>Fonte: {source_periods}<br>"
                f"Episodio interno: {_fmt_timestamp(event['episode_start_ts'])}; "
                f"{_transaction_label(_book_side(event['execution_side'])).lower()} di "
                f"{_fmt_quantity(event['episode_total_execution_quantity'])} unità."
            )
            lines.append(
                "| "
                + " | ".join(
                    cell.replace("|", "\\|")
                    for cell in [
                        f"Riferimento: `{event_id}`",
                        interval_and_sequence,
                        _identity_label(event),
                        "Confermare base temporale e sincronizzazione; acquisire l'attribuzione al cliente sottostante; esaminare il contesto operativo e l'eventuale ripetizione nella seduta.",
                    ]
                )
                + " |"
            )

    lines.append("")
    lines.extend(_date_only_alert_notes(datasets))
    lines.extend(
        [
            "",
            "## 4. Registro completo degli eventi",
            "",
            (
                "Gli eventi sono ordinati per WMSCI, dal valore più alto al più basso. A parità di WMSCI prevale "
                "l'ordine temporale; gli eventuali ulteriori pareggi sono risolti con regole fisse, in modo da "
                "ottenere sempre lo stesso ordinamento. "
                "Il codice dell'evento è assegnato in ordine cronologico all'interno di ciascun titolo e resta "
                "invariato quando cambia l'ordinamento della tabella."
            ),
            "",
            "### Come leggere WMSCI e MSCI",
            "",
            (
                "**WMSCI — dimensione e rapidità del ritiro collegato all'evento.** In questa relazione WMSCI è "
                "la metrica principale. Si calcola come:"
            ),
            "",
            (
                "**WMSCI = log(1 + quantità visibile / quantità eseguita) × log(1 + quantità ritirata "
                "ponderata / quantità eseguita) × quota ritirata**"
            ),
            "",
            (
                "Nella formula, `log` indica il logaritmo naturale; la **quantità visibile** è il totale degli ordini "
                "di segno opposto presenti prima dell'esecuzione; la **quantità ritirata ponderata** somma soltanto "
                "le cancellazioni attribuite all'evento. Nel calcolo, le cancellazioni ricevono un peso "
                "progressivamente minore all'aumentare del ritardo, secondo un decadimento esponenziale con scala "
                "di 10 secondi. La **quota ritirata** è la parte della quantità inizialmente visibile che viene "
                "cancellata. I due logaritmi attenuano l'effetto dei rapporti estremamente grandi, mentre il prodotto "
                "resta elevato soltanto se profilo visibile, ritiro ponderato e quota ritirata sono tutti rilevanti "
                "rispetto all'esecuzione."
            ),
            "",
            (
                "La WMSCI è sempre non negativa, non ha un massimo prefissato e non è una probabilità né una "
                "percentuale. Un valore doppio non significa quindi un rischio doppio. Valori più alti indicano "
                "soltanto un ritiro attribuito più ampio, più rapido e più esteso rispetto alla quantità eseguita."
            ),
            "",
            (
                "**MSCI — cambiamento relativo del profilo visibile.** La MSCI confronta la struttura degli ordini "
                "visibili prima dell'esecuzione con quella osservata alla fine della finestra successiva. La formula "
                "è `MSCI = SCI / 2 + C_opposta - C_stessa`: **SCI** misura quanto cambia lo sbilanciamento complessivo "
                "tra ordini di acquisto e di vendita, tenendo conto della loro profondità nei primi dieci livelli; "
                "**C_opposta** è la riduzione relativa della liquidità visibile di segno opposto all'operazione; "
                "**C_stessa** è la riduzione relativa dalla parte dell'operazione eseguita."
            ),
            "",
            (
                "Il valore della MSCI è compreso tra -1 e 2. Un valore positivo e alto descrive una contrazione "
                "relativamente maggiore degli ordini di segno opposto, accompagnata da un cambiamento della "
                "composizione complessiva. Un valore negativo indica che prevale la contrazione dalla parte "
                "dell'operazione eseguita. Un valore vicino a zero non implica necessariamente assenza di "
                "cambiamenti: può anche derivare da variazioni simili nelle due direzioni o da componenti che si "
                "compensano. La MSCI descrive una variazione relativa del profilo e non la dimensione assoluta o la "
                "rapidità delle cancellazioni attribuite."
            ),
            "",
            (
                "I due indicatori possono quindi divergere e non devono essere sostituiti l'uno con l'altro: la "
                "WMSCI può essere alta quando il ritiro attribuito è grande rispetto a una piccola esecuzione, anche "
                "se la struttura residua cambia poco; la MSCI può essere alta quando il profilo visibile cambia in "
                "modo asimmetrico, anche senza un ritiro attribuito altrettanto grande. Nessuno dei due comprende il "
                "movimento del prezzo, dimostra un effetto causale o misura l'intenzione. La WMSCI determina soltanto "
                "l'ordine di presentazione del registro: non è prevista una soglia minima di WMSCI o MSCI per "
                "includere una sequenza e nessuno dei due indicatori, da solo, determina un riscontro CONSOB."
            ),
            "",
            "### Come leggere la variazione del midprice",
            "",
            (
                "Per ogni cancellazione si confronta il midprice nell'ultimo stato precedente con quello "
                "dell'ultimo stato disponibile nell'intervallo di "
                f"{_seconds_label(reversion_horizon_seconds)} successivo a ciascuna cancellazione. "
                "Si considera soltanto il movimento che annulla quello osservato prima dell'esecuzione e soltanto "
                "quando l'intero intervallo è osservabile. Il valore dell'evento è la media delle misure disponibili, "
                "ponderata per la quantità cancellata e per la rapidità della cancellazione. "
                "Gli orari sono riportati esattamente come compaiono nei dati sorgente."
            ),
            "",
            "| Evento | Data, soggetto e riscontro CONSOB | Operazione eseguita | Ordini visibili prima | Cancellazioni successive | Variazione del midprice | Indicatori |",
            "|---|---|---|---|---|---|---|",
        ]
    )

    rendered_event_rows: list[str] = []
    for instrument, original_index, event in ranked_events:
        info = instruments_info[instrument]
        tick_size = _as_float(info.get("tick_size"), field=f"{instrument} tick_size")
        price_decimals = _price_decimals(tick_size)
        key = (instrument, original_index)
        tag = tags.get(key)
        if tag is None:
            tag_text = "NUOVO_EVENTO"
        else:
            label = (
                "CONSOB_MATCH_SOGGETTO"
                if tag["identity_granularity_aligned"]
                and event["identity_level"] == "client_original"
                else "CONSOB_MATCH_INTERMEDIARIO"
            )
            source_periods = ", ".join(tag["source_period_ids"])
            tag_text = (
                f"{label}<br>Intervallo: {tag['union_period_id']}<br>"
                f"Fonte: {source_periods}"
            )
        context_cell = (
            f"{tag_text}<br>{_fmt_timestamp(event['episode_start_ts'])}<br>"
            f"{_identity_label(event)}"
        )
        execution_side = _book_side(event["execution_side"])
        deceptive_side = _book_side(event["deceptive_side"])
        transaction = _transaction_label(execution_side)
        anchor_mode = _episode_anchor_mode(event)
        if anchor_mode == "passive":
            executed = "eseguito" if execution_side == "bid" else "eseguita"
            execution_intro = (
                f"{transaction.capitalize()} {executed} mediante un ordine di {transaction} "
                "già presente nel libro degli ordini"
            )
        elif anchor_mode == "aggressive":
            execution_intro = (
                f"{transaction.capitalize()} contro {_orders_label(deceptive_side)} già presenti "
                "nel libro degli ordini"
            )
        elif anchor_mode == "mixed":
            execution_intro = (
                f"Episodio di {transaction.lower()} composto da cluster passivi e aggressivi"
            )
        else:
            raise ValueError(f"unsupported execution anchor mode: {anchor_mode!r}")
        episode_duration_ms = 1000.0 * (
            _parse_timestamp(event["episode_end_ts"], field="episode_end_ts")
            - _parse_timestamp(event["episode_start_ts"], field="episode_start_ts")
        ).total_seconds()
        cluster_count = _as_int(event["episode_cluster_count"], field="episode_cluster_count")
        cluster_noun = "cluster di esecuzione" if cluster_count == 1 else "cluster di esecuzioni"
        fill_description = f"L'episodio comprende {cluster_count} {cluster_noun}"
        execution_cell = (
            f"{execution_intro}: "
            f"{_fmt_quantity(event['episode_total_execution_quantity'])} unità al prezzo medio ponderato di "
            f"{_it_number(_as_float(event['episode_execution_vwap'], field='episode_execution_vwap'), price_decimals)}. "
            f"{fill_description} e dura {_it_number(episode_duration_ms, 1)} ms."
        )
        profile_cell = (
            "Nel cluster rappresentante, prima dell'esecuzione erano visibili "
            f"{_count_label(event['candidate_deceptive_order_count_pre'], field='candidate_deceptive_order_count_pre', singular='ordine', plural='ordini')} "
            f"di {'acquisto' if deceptive_side == 'bid' else 'vendita'}, "
            f"per un totale di {_fmt_quantity(event['candidate_deceptive_visible_qty_pre'])} unità. "
            "Al momento dell'esecuzione, il più recente era nel libro degli ordini da "
            f"{_it_number(_as_float(event['candidate_deceptive_min_age_seconds_pre'], field='candidate_deceptive_min_age_seconds_pre'), 3)} s; "
            f"la distanza media, ponderata per quantità, dalla {_best_quote_label(deceptive_side)} era di "
            f"{_it_number(_as_float(event['candidate_deceptive_qty_weighted_depth_distance_ticks_pre'], field='candidate_deceptive_qty_weighted_depth_distance_ticks_pre'), 2)} tick."
        )
        cancel_count = _as_int(
            event["episode_unique_withdrawal_count"],
            field="episode_unique_withdrawal_count",
        )
        min_delay_text = _it_number(
            1000.0
            * _as_float(
                event["episode_withdrawal_min_delay_seconds"],
                field="episode_withdrawal_min_delay_seconds",
            ),
            1,
        )
        max_delay_text = _it_number(
            1000.0
            * _as_float(
                event["episode_withdrawal_max_delay_seconds"],
                field="episode_withdrawal_max_delay_seconds",
            ),
            1,
        )
        if min_delay_text == max_delay_text:
            delay_text = (
                "La cancellazione avviene"
                if cancel_count == 1
                else "Le cancellazioni avvengono"
            ) + f" {min_delay_text} ms dopo l'esecuzione."
        else:
            delay_text = (
                f"Le cancellazioni avvengono tra {min_delay_text} e {max_delay_text} ms "
                "dopo l'esecuzione."
            )
        if cancel_count == 1:
            cancel_observation = (
                "Dopo l'esecuzione è stata rilevata, tra gli ordini considerati, una cancellazione"
            )
        else:
            cancel_observation = (
                "Dopo l'esecuzione sono state rilevate, tra gli ordini considerati, "
                f"{cancel_count} cancellazioni"
            )
        withdrawal_cell = (
            f"{cancel_observation}, per un totale di "
            f"{_fmt_quantity(event['episode_withdrawn_quantity'])} unità. "
            f"{delay_text} "
            "Il rapporto tra quantità cancellata ed eseguita è "
            f"{_it_number(_as_float(event['episode_withdrawal_to_execution_ratio'], field='episode_withdrawal_to_execution_ratio'), 2)}."
        )
        favorable = _as_float(
            event["episode_favorable_mid_move"], field="episode_favorable_mid_move"
        )
        reversion = _as_float(
            event["episode_post_cancel_mid_reversion"],
            field="episode_post_cancel_mid_reversion",
        )
        if execution_side == "bid":
            transaction_context = "dell'acquisto"
            favorable_verb = "è sceso"
            reverse_verb = "è risalito"
        else:
            transaction_context = "della vendita"
            favorable_verb = "è salito"
            reverse_verb = "è sceso"
        price_cell = (
            f"Dalla pubblicazione osservata della configurazione candidata alle relative esecuzioni, "
            f"il midprice {favorable_verb} di "
            f"{_it_compact_number(favorable, min_decimals=price_decimals)} "
            f"({_it_number(favorable / tick_size, 2)} tick); dopo le cancellazioni "
            f"{reverse_verb} in media di "
            f"{_it_compact_number(reversion, min_decimals=price_decimals)} "
            f"({_it_number(reversion / tick_size, 2)} tick)."
        )
        diagnostic_cell = (
            f"**WMSCI del cluster rappresentante:** {_it_number(_as_float(event['withdrawal_profile_scale_event'], field='withdrawal_profile_scale_event'), 3)}<br>"
            f"**MSCI del cluster rappresentante:** {_it_number(_as_float(event['MSCI_resting_profile'], field='MSCI_resting_profile'), 3)}"
        )
        cells = [
            event_ids[key],
            context_cell,
            execution_cell,
            profile_cell,
            withdrawal_cell,
            price_cell,
            diagnostic_cell,
        ]
        rendered_row = (
            "| " + " | ".join(cell.replace("|", "\\|") for cell in cells) + " |"
        )
        lines.append(rendered_row)
        rendered_event_rows.append(rendered_row)

    source_info = audit.get("external_source", {})
    lines.extend(
        [
            "",
            "## 5. Parametri, provenienza e limiti",
            "",
            "### Parametri operativi",
            "",
            f"- Il libro degli ordini è ricostruito sui primi {parameters.get('top_n', 10)} livelli, con pesi calibrati separatamente per ciascun titolo.",
            f"- Le esecuzioni dello stesso ordine sono raggruppate se distano al massimo {_it_compact_number(_as_float(parameters.get('execution_cluster_max_gap_ms', 100), field='execution_cluster_max_gap_ms'), min_decimals=0)} ms. Le esecuzioni di ordini già presenti nel libro degli ordini e quelle contro ordini già presenti sono trattate separatamente.",
            f"- Gli ordini di segno opposto all'operazione eseguita possono essere presenti nel libro degli ordini da non più di {_it_compact_number(_as_float(parameters.get('max_deceptive_order_age_seconds', 90), field='max_deceptive_order_age_seconds'), min_decimals=0)} s. Non è imposto un tempo minimo di permanenza; devono tuttavia precedere l'esecuzione nell'ordine dei dati sorgente.",
            f"- Le cancellazioni associate sono ricercate nei {_it_compact_number(_as_float(parameters.get('withdrawal_window_seconds', 2), field='withdrawal_window_seconds'), min_decimals=0)} s successivi.",
            f"- Il movimento inverso del midprice è osservato nei {_it_compact_number(_as_float(parameters.get('reversion_horizon_seconds', 2), field='reversion_horizon_seconds'), min_decimals=0)} s successivi a ciascuna cancellazione.",
            "- Ogni riga riporta il codice cliente originale quando disponibile; in sua assenza riporta il codice originale dell'intermediario.",
            "",
            "### Provenienza verificabile",
            "",
            f"- Elaborazione canonica: `{run_info.get('run_id')}`.",
            f"- Esito della validazione: `{run_info.get('validation_status')}` ({run_info.get('validated_at_utc')}).",
            f"- Audit CONSOB SHA-256: `{run_info.get('audit_sha256')}`.",
            f"- Fonte primaria CONSOB: `{source_info.get('source_kind')}`, SHA-256 `{source_info.get('sha256')}`.",
        ]
    )
    for instrument in sorted(events_by_instrument, key=_instrument_sort_key):
        info = instruments_info[instrument]
        lines.append(
            f"- {instrument}, `spoofing_compatible_events.parquet`: SHA-256 `{info.get('artifact_sha256')}`."
        )
    identity_limit_lines: list[str] = []
    if intermediary_tag_count == 1:
        identity_limit_lines.append(
            "- Il riscontro riportato nel registro è classificato a livello di intermediario: coincidono "
            "l'intervallo temporale e l'intermediario registrato, ma il codice cliente non è disponibile. "
            "La sequenza interna è stata riconciliata senza ambiguità tra gli artefatti del confronto, ma il "
            "riscontro non dimostra che essa riguardi lo stesso cliente o la stessa condotta sottostante alla "
            "segnalazione CONSOB."
        )
    elif intermediary_tag_count > 1:
        identity_limit_lines.append(
            f"- I {intermediary_tag_count} riscontri riportati nel registro sono classificati a livello di "
            "intermediario: coincidono l'intervallo temporale e l'intermediario registrato, ma il codice cliente "
            "non è disponibile. Non dimostrano quindi l'identità del cliente o dell'operatore economico "
            "sottostante. Le sequenze interne sono state riconciliate senza ambiguità tra gli artefatti del "
            "confronto, ma i riscontri non dimostrano che riguardino gli stessi clienti o le stesse condotte "
            "sottostanti alle segnalazioni CONSOB."
        )
    if subject_tag_count == 1:
        identity_limit_lines.append(
            "- Il riscontro classificato a livello del soggetto indica che il medesimo codice cliente è stato "
            "associato senza ambiguità nei due archivi; non dimostra da solo l'identità economica, l'intenzione, "
            "una strategia o un illecito."
        )
    elif subject_tag_count > 1:
        identity_limit_lines.append(
            f"- I {subject_tag_count} riscontri classificati a livello del soggetto indicano che il medesimo "
            "codice cliente è stato associato senza ambiguità nei due archivi; non dimostrano da soli l'identità "
            "economica, l'intenzione, una strategia o un illecito."
        )
    sentinel_limit_lines = (
        [
            "- Gli eventi per i quali l'unico valore identificativo disponibile è il codice tecnico `0` "
            "restano separati; il valore non identifica un cliente o un intermediario."
        ]
        if sentinel_total > 0
        else []
    )
    lines.extend(
        [
            "",
            "### Limiti di interpretazione",
            "",
            "- Le segnalazioni CONSOB costituiscono un termine di confronto esterno; non esprimono un giudizio indipendente e definitivo sulla natura manipolativa degli eventi.",
            "- Poiché il confronto non include intervalli temporali privi di segnalazioni, non è possibile stimare né la precisione né il tasso di falsi positivi.",
        ]
        + identity_limit_lines
        + [
            "- La classificazione delle esecuzioni contro ordini già presenti dipende dalla ricostruzione tecnica del ruolo degli ordini e resta sperimentale finché tale ricostruzione non viene verificata separatamente.",
            "- Le differenze tra titoli hanno valore descrittivo e non sono direttamente comparabili, perché periodo osservato, composizione dei partecipanti e liquidità non sono omogenei.",
        ]
        + sentinel_limit_lines
        + [
            "- Gli identificativi originali di clienti e intermediari sono riportati esclusivamente per la riconciliazione autorizzata; il documento non è destinato alla diffusione pubblica.",
            "",
            "### Fonti normative",
            "",
            f"[1] {MAR_ARTICLE_15_URL} — Regolamento (UE) n. 596/2014, articolo 15.",
            "",
        ]
    )
    expected_event_ids = [
        event_ids[(instrument, original_index)]
        for instrument, original_index, _event in ranked_events
    ]
    _validate_rendered_event_rows(rendered_event_rows, expected_event_ids)
    report = "\n".join(lines)
    _safe_report_check(report, events_by_instrument, audit)
    return report


def load_canonical_inputs(
    run_root: Path, audit_path: Path
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any], dict[str, Any]]:
    validation_path = run_root / "validation_report.json"
    if not validation_path.is_file():
        raise FileNotFoundError(validation_path)
    validation = _read_json(validation_path)
    if validation.get("status") != "passed":
        raise ValueError("canonical run validation_report.json is not passed")
    audit = _read_json(audit_path)
    events_by_instrument: dict[str, list[dict[str, Any]]] = {}
    instruments: dict[str, dict[str, Any]] = {}
    common_parameters: dict[str, Any] | None = None
    for instrument in INSTRUMENT_ORDER:
        run_dirs = sorted(path for path in run_root.glob(f"{instrument}_*") if path.is_dir())
        if len(run_dirs) != 1:
            raise ValueError(f"expected exactly one {instrument} run directory, found {len(run_dirs)}")
        run_dir = run_dirs[0]
        metadata = _read_json(run_dir / "metadata.json")
        episode_semantics = metadata.get("episode_semantics", {})
        if episode_semantics.get("primary_unit") != "candidate_posture_episode":
            raise ValueError(f"{instrument} metadata do not declare episode semantics")
        parquet_path = run_dir / "spoofing_compatible_events.parquet"
        expected_hash = metadata.get("artifact_hashes", {}).get("spoofing_compatible_events")
        observed_hash = _sha256(parquet_path)
        if expected_hash != observed_hash:
            raise ValueError(f"{instrument} compatible-event artifact hash mismatch")
        frame = pl.read_parquet(parquet_path)
        expected_rows = metadata.get("row_counts", {}).get("spoofing_compatible_events")
        if frame.height != expected_rows:
            raise ValueError(f"{instrument} compatible-event row-count mismatch")
        episode_path = run_dir / "candidate_episodes.parquet"
        withdrawal_path = run_dir / "episode_withdrawals.parquet"
        execution_metrics_path = run_dir / "execution_metrics.parquet"
        for artifact_name, artifact_path in (
            ("candidate_episodes", episode_path),
            ("episode_withdrawals", withdrawal_path),
            ("execution_metrics", execution_metrics_path),
        ):
            expected_artifact_hash = metadata.get("artifact_hashes", {}).get(artifact_name)
            if expected_artifact_hash != _sha256(artifact_path):
                raise ValueError(f"{instrument} {artifact_name} artifact hash mismatch")
        episodes = pl.read_parquet(episode_path).filter("spoofing_compatible_episode")
        withdrawals = pl.read_parquet(withdrawal_path)
        execution_metrics = pl.read_parquet(execution_metrics_path)
        expected_episode_rows = metadata.get("row_counts", {}).get("candidate_episodes")
        expected_withdrawal_rows = metadata.get("row_counts", {}).get("episode_withdrawals")
        if pl.read_parquet(episode_path).height != expected_episode_rows:
            raise ValueError(f"{instrument} candidate-episode row-count mismatch")
        if withdrawals.height != expected_withdrawal_rows:
            raise ValueError(f"{instrument} episode-withdrawal row-count mismatch")
        if frame.height != episodes.height:
            raise ValueError(f"{instrument} strict-event and strict-episode counts disagree")
        episode_fields = episodes.select(
            "episode_id",
            pl.col("episode_start_ts").alias("episode_start_ts"),
            pl.col("episode_end_ts").alias("episode_end_ts"),
            "actor_key",
            pl.col("cluster_count").alias("episode_cluster_count"),
            pl.col("unique_withdrawal_count").alias("episode_unique_withdrawal_count"),
            pl.col("total_execution_quantity").alias("episode_total_execution_quantity"),
            pl.col("execution_vwap").alias("episode_execution_vwap"),
            pl.col("withdrawn_quantity").alias("episode_withdrawn_quantity"),
            pl.col("withdrawal_to_execution_ratio").alias(
                "episode_withdrawal_to_execution_ratio"
            ),
            pl.col("favorable_mid_move").alias("episode_favorable_mid_move"),
            pl.col("post_cancel_mid_reversion").alias(
                "episode_post_cancel_mid_reversion"
            ),
            pl.col("price_path_observed").alias("episode_price_path_observed"),
            (pl.col("unique_withdrawal_count") > 0).alias(
                "episode_has_matched_withdrawal"
            ),
            pl.col("gate_joint_smallness").alias("episode_gate_joint_smallness"),
            "spoofing_compatible_episode",
            pl.col("mixed_anchor").alias("episode_mixed_anchor"),
        )
        delay_summary = _episode_withdrawal_delay_summary(
            withdrawals, execution_metrics
        )
        diagnostic_fields = frame.select(
            "episode_id",
            "identity_level",
            "identity_fallback_flag",
            "execution_side",
            "deceptive_side",
            "execution_cluster_id",
            "episode_representative_cluster_id",
            "execution_anchor_mode",
            "candidate_deceptive_order_count_pre",
            "candidate_deceptive_visible_qty_pre",
            "candidate_deceptive_min_age_seconds_pre",
            "candidate_deceptive_qty_weighted_depth_distance_ticks_pre",
            "MSCI_resting_profile",
            "withdrawal_profile_scale_event",
            pl.col("episode_strict_detection"),
        )
        authoritative = episode_fields.join(
            diagnostic_fields, on="episode_id", how="inner", validate="1:1"
        ).join(delay_summary, on="episode_id", how="inner", validate="1:1")
        if authoritative.height != episodes.height:
            raise ValueError(f"{instrument} strict episodes lack authoritative detail")
        events_by_instrument[instrument] = authoritative.to_dicts()
        instruments[instrument] = {
            "artifact_sha256": observed_hash,
            "tick_size": metadata.get("tick_size"),
        }
        parameters = {
            "top_n": metadata.get("top_n"),
            "execution_cluster_max_gap_ms": metadata.get("execution_cluster_max_gap_ms"),
            "max_deceptive_order_age_seconds": metadata.get(
                "max_deceptive_order_age_seconds"
            ),
            "withdrawal_window_seconds": metadata.get("withdrawal_window_seconds"),
            "reversion_horizon_seconds": metadata.get("reversion_horizon_seconds"),
            "actor_identity_mode": metadata.get("actor_identity_mode"),
            "execution_anchor_modes": metadata.get("execution_anchor_modes"),
            "kernel_mode": metadata.get("kernel_mode"),
        }
        if common_parameters is None:
            common_parameters = parameters
        elif parameters != common_parameters:
            raise ValueError("instrument metadata do not share one common parameter contract")
    run_info = {
        "run_id": run_root.name,
        "validation_status": validation.get("status"),
        "validated_at_utc": validation.get("validated_at_utc"),
        "audit_sha256": _sha256(audit_path),
        "instruments": instruments,
        "parameters": common_parameters or {},
    }
    return events_by_instrument, audit, run_info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the complete CONSOB-facing strict-event Markdown report."
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-date", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    events, audit, run_info = load_canonical_inputs(args.run_root, args.audit)
    report = build_report(
        events_by_instrument=events,
        audit=audit,
        run_info=run_info,
        report_date=args.report_date,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8", newline="\n")
    tag_count = len(_resolve_consob_tags(events, audit))
    print(
        json.dumps(
            {
                "output": str(args.output),
                "event_rows": sum(len(rows) for rows in events.values()),
                "consob_tagged_events": tag_count,
                "sha256": _sha256(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
