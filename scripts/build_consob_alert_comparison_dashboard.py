#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any, Mapping, Sequence


DASHBOARD_SCHEMA_VERSION = "consob_alert_comparison_dashboard_v2"
_ALLOWED_SOURCE_KINDS = frozenset({"primary_pdf"})
_ALLOWED_ACTOR_IDENTITY_MODES = frozenset({"client_then_firm"})
_ALLOWED_EXECUTION_ANCHOR_MODES = frozenset({"passive", "aggressive"})
_ALLOWED_OUTPUT_SCHEMA_VERSIONS = frozenset({"actor_execution_anchor_v2"})
_ALLOWED_EXECUTION_SIDES = frozenset({"ask", "bid"})
_ALLOWED_IDENTITY_NAMESPACES = frozenset({"client_original", "firm"})
_DETECTOR_EVENT_FIELDS = frozenset(
    {
        "event_alias",
        "cluster_start",
        "cluster_end",
        "execution_anchor_mode",
        "execution_side",
        "execution_quantity",
        "execution_vwap",
        "has_matched_withdrawal",
        "gate_rapid_matched_withdrawal",
        "gate_small_fill_relative_to_withdrawal",
        "gate_favorable_pre_fill_move",
        "gate_cancel_anchored_reversion",
        "strict_detection",
    }
)
_COUNT_KEYS = (
    "source_external_periods",
    "source_timed_periods",
    "source_date_only_periods",
    "union_timed_periods",
    "union_periods_with_raw_subject_event",
    "union_periods_with_recovered_execution",
    "union_periods_with_matched_withdrawal",
    "union_periods_with_strict_subject_scope_detection",
    "identity_aligned_union_timed_periods",
    "identity_aligned_union_periods_with_strict_detection",
    "identity_unaligned_union_timed_periods",
    "identity_unaligned_union_periods_with_strict_subject_scope_detection",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a self-contained dashboard comparing CONSOB alerts with detector outcomes.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--audit-json",
        type=Path,
        required=True,
        help="JSON produced by audit_external_alert_detector_overlap.py",
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        required=True,
        help="Destination for the self-contained HTML dashboard",
    )
    parser.add_argument(
        "--generated-at-utc",
        default=None,
        help="Optional ISO-8601 build timestamp, mainly for deterministic regeneration",
    )
    return parser.parse_args(argv)


def _mapping(value: Any, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    return value


def _rows(value: Any, *, context: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(row, Mapping) for row in value):
        raise ValueError(f"{context} must be a list of objects")
    return value


def _count(mapping: Mapping[str, Any], key: str, *, context: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{context}.{key} must be a non-negative integer")
    return value


def _numeric_count(mapping: Mapping[str, Any], key: str, *, context: str) -> int:
    value = mapping.get(key, 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{context}.{key} must be a non-negative integer")
    return value


def _normalize_utc_timestamp(value: str | None) -> str:
    if value is None:
        return datetime.now(timezone.utc).isoformat()
    if not isinstance(value, str):
        raise ValueError("generated_at_utc must be an ISO-8601 timestamp with timezone")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(
            "generated_at_utc must be an ISO-8601 timestamp with timezone"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("generated_at_utc must be an ISO-8601 timestamp with timezone")
    return parsed.astimezone(timezone.utc).isoformat()


def _parsed_timestamp(value: Any, *, context: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{context} must be an ISO-8601 timestamp")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{context} must be an ISO-8601 timestamp") from exc


def _finite_number(value: Any, *, context: str, positive: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        qualifier = "positive finite" if positive else "finite"
        raise ValueError(f"{context} must be a {qualifier} number")
    return result


def _validate_detector_events(row: Mapping[str, Any], *, context: str) -> None:
    events = _rows(row.get("detector_events"), context=f"{context}.detector_events")
    recovered = _numeric_count(row, "recovered_execution_clusters", context=context)
    if len(events) != recovered:
        raise ValueError(
            f"{context}.detector_events must match recovered_execution_clusters"
        )

    alert_start = _parsed_timestamp(row.get("start"), context=f"{context}.start")
    alert_end = _parsed_timestamp(row.get("end"), context=f"{context}.end")
    aliases: set[str] = set()
    matched_count = 0
    strict_count = 0
    anchor_counts = {mode: 0 for mode in _ALLOWED_EXECUTION_ANCHOR_MODES}
    for event_index, event in enumerate(events):
        event_context = f"{context}.detector_events[{event_index}]"
        unexpected = sorted(set(event) - _DETECTOR_EVENT_FIELDS)
        missing = sorted(_DETECTOR_EVENT_FIELDS - set(event))
        if unexpected or missing:
            raise ValueError(
                f"{event_context} has unexpected fields {unexpected} or missing fields {missing}"
            )

        alias = event.get("event_alias")
        if not isinstance(alias, str) or re.fullmatch(
            r"detector_event_[0-9a-f]{12}", alias
        ) is None:
            raise ValueError(f"{event_context}.event_alias is not a report-local pseudonym")
        if alias in aliases:
            raise ValueError(f"{context}.detector_events contains duplicate event aliases")
        aliases.add(alias)

        cluster_start = _parsed_timestamp(
            event.get("cluster_start"), context=f"{event_context}.cluster_start"
        )
        cluster_end = _parsed_timestamp(
            event.get("cluster_end"), context=f"{event_context}.cluster_end"
        )
        if cluster_start > cluster_end:
            raise ValueError(f"{event_context} has cluster_start after cluster_end")
        if cluster_end < alert_start or cluster_start > alert_end:
            raise ValueError(f"{event_context} does not overlap its source alert")

        anchor = event.get("execution_anchor_mode")
        if anchor not in _ALLOWED_EXECUTION_ANCHOR_MODES:
            raise ValueError(f"{event_context}.execution_anchor_mode is invalid")
        anchor_counts[str(anchor)] += 1
        if event.get("execution_side") not in _ALLOWED_EXECUTION_SIDES:
            raise ValueError(f"{event_context}.execution_side is invalid")
        _finite_number(
            event.get("execution_quantity"),
            context=f"{event_context}.execution_quantity",
            positive=True,
        )
        _finite_number(
            event.get("execution_vwap"),
            context=f"{event_context}.execution_vwap",
            positive=True,
        )

        boolean_fields = (
            "has_matched_withdrawal",
            "gate_rapid_matched_withdrawal",
            "gate_small_fill_relative_to_withdrawal",
            "gate_favorable_pre_fill_move",
            "gate_cancel_anchored_reversion",
            "strict_detection",
        )
        if any(not isinstance(event.get(field), bool) for field in boolean_fields):
            raise ValueError(f"{event_context} detector flags must be boolean")
        matched_count += int(event["has_matched_withdrawal"])
        strict_count += int(event["strict_detection"])
        if event["strict_detection"] and not all(
            event[field] for field in boolean_fields[:-1]
        ):
            raise ValueError(f"{event_context} strict detection has unsatisfied gates")

    if matched_count != _numeric_count(
        row, "clusters_with_matched_withdrawal", context=context
    ):
        raise ValueError(f"{context}.detector_events withdrawal count does not reconcile")
    if strict_count != _numeric_count(
        row, "clusters_with_strict_detection", context=context
    ):
        raise ValueError(f"{context}.detector_events strict count does not reconcile")
    expected_anchor_counts = row.get("recovered_execution_clusters_by_anchor")
    observed_anchor_counts = {key: value for key, value in anchor_counts.items() if value}
    if expected_anchor_counts != observed_anchor_counts:
        raise ValueError(f"{context}.detector_events anchor counts do not reconcile")


def _observed_union_counts(
    rows: Sequence[Mapping[str, Any]], *, context_prefix: str
) -> dict[str, int]:
    observed = {
        "union_periods_with_raw_subject_event": 0,
        "union_periods_with_recovered_execution": 0,
        "union_periods_with_matched_withdrawal": 0,
        "union_periods_with_strict_subject_scope_detection": 0,
        "identity_aligned_union_timed_periods": 0,
        "identity_aligned_union_periods_with_strict_detection": 0,
        "identity_unaligned_union_timed_periods": 0,
        "identity_unaligned_union_periods_with_strict_subject_scope_detection": 0,
    }
    for row_index, row in enumerate(rows):
        context = f"{context_prefix}[{row_index}]"
        if not isinstance(row.get("start"), str) or not isinstance(row.get("end"), str):
            raise ValueError(f"{context} requires string start and end timestamps")
        raw_events = _numeric_count(row, "raw_actor_rows_in_period", context=context)
        executions = _numeric_count(row, "recovered_execution_clusters", context=context)
        withdrawals = _numeric_count(row, "clusters_with_matched_withdrawal", context=context)
        strict = _numeric_count(row, "clusters_with_strict_detection", context=context)
        if strict > withdrawals or withdrawals > executions:
            raise ValueError(f"{context} has inconsistent detector-stage counts")
        if not isinstance(row.get("identity_granularity_aligned"), bool):
            raise ValueError(f"{context}.identity_granularity_aligned must be boolean")
        aligned = row["identity_granularity_aligned"]
        observed["union_periods_with_raw_subject_event"] += raw_events > 0
        observed["union_periods_with_recovered_execution"] += executions > 0
        observed["union_periods_with_matched_withdrawal"] += withdrawals > 0
        observed["union_periods_with_strict_subject_scope_detection"] += strict > 0
        observed["identity_aligned_union_timed_periods"] += aligned
        observed["identity_aligned_union_periods_with_strict_detection"] += (
            aligned and strict > 0
        )
        observed["identity_unaligned_union_timed_periods"] += not aligned
        observed[
            "identity_unaligned_union_periods_with_strict_subject_scope_detection"
        ] += not aligned and strict > 0
    return observed


def _validate_report(report: Mapping[str, Any]) -> None:
    overall = _mapping(report.get("overall"), context="overall")
    datasets = _mapping(report.get("datasets"), context="datasets")
    if not datasets:
        raise ValueError("datasets must not be empty")

    for key in _COUNT_KEYS:
        _count(overall, key, context="overall")

    source = _mapping(report.get("external_source"), context="external_source")
    source_hash = source.get("sha256")
    if (
        not isinstance(source_hash, str)
        or len(source_hash) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in source_hash)
    ):
        raise ValueError("external_source.sha256 must be a 64-character hexadecimal hash")
    if source.get("local_path_disclosed") is not False:
        raise ValueError("external_source.local_path_disclosed must be false")
    if source.get("source_kind") not in _ALLOWED_SOURCE_KINDS:
        raise ValueError("external_source.source_kind is not an allowed value")

    sums = {key: 0 for key in _COUNT_KEYS}
    all_union_rows: list[Mapping[str, Any]] = []
    date_rows = 0
    provenance_signature: tuple[str, tuple[str, ...], str] | None = None
    for instrument, raw_dataset in datasets.items():
        dataset = _mapping(raw_dataset, context=f"datasets.{instrument}")
        for key in _COUNT_KEYS:
            value = _count(dataset, key, context=f"datasets.{instrument}")
            sums[key] += value

        union_rows = _rows(
            dataset.get("union_timed_results"),
            context=f"datasets.{instrument}.union_timed_results",
        )
        for row_index, row in enumerate(union_rows):
            if row.get("identity_namespace") not in _ALLOWED_IDENTITY_NAMESPACES:
                raise ValueError(
                    f"datasets.{instrument}.union_timed_results[{row_index}]."
                    "identity_namespace is invalid"
                )
        if len(union_rows) != dataset["union_timed_periods"]:
            raise ValueError(
                f"datasets.{instrument}.union_timed_periods does not match union_timed_results"
            )
        all_union_rows.extend(union_rows)

        source_rows = _rows(
            dataset.get("source_period_results"),
            context=f"datasets.{instrument}.source_period_results",
        )
        dataset_date_rows = sum(row.get("scope") == "date" for row in source_rows)
        dataset_timed_rows = len(source_rows) - dataset_date_rows
        if len(source_rows) != dataset["source_external_periods"]:
            raise ValueError(
                f"datasets.{instrument}.source_external_periods does not match source rows"
            )
        if dataset_timed_rows != dataset["source_timed_periods"]:
            raise ValueError(
                f"datasets.{instrument}.source_timed_periods does not match intraday rows"
            )
        if dataset_date_rows != dataset["source_date_only_periods"]:
            raise ValueError(
                f"datasets.{instrument}.source_date_only_periods does not match date rows"
            )
        for row_index, row in enumerate(source_rows):
            if row.get("identity_namespace") not in _ALLOWED_IDENTITY_NAMESPACES:
                raise ValueError(
                    f"datasets.{instrument}.source_period_results[{row_index}]."
                    "identity_namespace is invalid"
                )
            if row.get("scope") == "date":
                if row.get("detector_events") is not None:
                    raise ValueError(
                        f"datasets.{instrument}.source_period_results[{row_index}].detector_events "
                        "must be null for date-only alerts"
                    )
                continue
            _validate_detector_events(
                row,
                context=f"datasets.{instrument}.source_period_results[{row_index}]",
            )
        date_rows += dataset_date_rows
        if dataset["source_external_periods"] != (
            dataset["source_timed_periods"] + dataset["source_date_only_periods"]
        ):
            raise ValueError(
                f"datasets.{instrument}.source_external_periods must equal timed plus date-only periods"
            )

        timed = dataset["union_timed_periods"]
        raw = dataset["union_periods_with_raw_subject_event"]
        execution = dataset["union_periods_with_recovered_execution"]
        withdrawal = dataset["union_periods_with_matched_withdrawal"]
        strict = dataset["union_periods_with_strict_subject_scope_detection"]
        if not 0 <= strict <= withdrawal <= execution <= raw <= timed:
            raise ValueError(
                f"datasets.{instrument} has inconsistent raw-subject/detector-stage counts"
            )
        aligned = dataset["identity_aligned_union_timed_periods"]
        unaligned = dataset["identity_unaligned_union_timed_periods"]
        if aligned + unaligned != timed:
            raise ValueError(
                f"datasets.{instrument} identity aligned/unaligned windows do not sum to total"
            )
        aligned_strict = dataset[
            "identity_aligned_union_periods_with_strict_detection"
        ]
        unaligned_strict = dataset[
            "identity_unaligned_union_periods_with_strict_subject_scope_detection"
        ]
        if aligned_strict + unaligned_strict != strict:
            raise ValueError(
                f"datasets.{instrument} strict identity aligned/unaligned counts do not sum to subject-scope total"
            )

        provenance = _mapping(
            dataset.get("metrics_provenance"),
            context=f"datasets.{instrument}.metrics_provenance",
        )
        actor_mode = provenance.get("actor_identity_mode")
        if actor_mode not in _ALLOWED_ACTOR_IDENTITY_MODES:
            raise ValueError(
                f"datasets.{instrument}.metrics_provenance.actor_identity_mode is not allowed"
            )
        anchor_modes = provenance.get("execution_anchor_modes")
        if (
            not isinstance(anchor_modes, list)
            or not anchor_modes
            or any(mode not in _ALLOWED_EXECUTION_ANCHOR_MODES for mode in anchor_modes)
            or len(anchor_modes) != len(set(anchor_modes))
        ):
            raise ValueError(
                f"datasets.{instrument}.metrics_provenance.execution_anchor_modes is invalid"
            )
        output_schema = provenance.get("output_schema_version")
        if output_schema not in _ALLOWED_OUTPUT_SCHEMA_VERSIONS:
            raise ValueError(
                f"datasets.{instrument}.metrics_provenance.output_schema_version is not allowed"
            )
        signature = (actor_mode, tuple(anchor_modes), output_schema)
        if provenance_signature is None:
            provenance_signature = signature
        elif signature != provenance_signature:
            raise ValueError("metrics_provenance must be consistent across datasets")

        dataset_observed = _observed_union_counts(
            union_rows,
            context_prefix=f"datasets.{instrument}.union_timed_results",
        )
        for key, value in dataset_observed.items():
            if dataset[key] != value:
                raise ValueError(f"datasets.{instrument}.{key} does not match union rows")

    for key, expected in sums.items():
        if overall[key] != expected:
            raise ValueError(
                f"overall.{key}={overall[key]} does not equal dataset sum {expected}"
            )

    if len(all_union_rows) != overall["union_timed_periods"]:
        raise ValueError("overall.union_timed_periods does not match union result rows")
    if date_rows != overall["source_date_only_periods"]:
        raise ValueError("overall.source_date_only_periods does not match date-only rows")
    if overall["source_external_periods"] != (
        overall["source_timed_periods"] + overall["source_date_only_periods"]
    ):
        raise ValueError(
            "overall.source_external_periods must equal timed plus date-only periods"
        )
    if not (
        0
        <= overall["union_periods_with_strict_subject_scope_detection"]
        <= overall["union_periods_with_matched_withdrawal"]
        <= overall["union_periods_with_recovered_execution"]
        <= overall["union_periods_with_raw_subject_event"]
        <= overall["union_timed_periods"]
    ):
        raise ValueError("overall has inconsistent raw-subject/detector-stage counts")
    if (
        overall["identity_aligned_union_timed_periods"]
        + overall["identity_unaligned_union_timed_periods"]
        != overall["union_timed_periods"]
    ):
        raise ValueError("overall identity aligned/unaligned windows do not sum to total")
    if (
        overall["identity_aligned_union_periods_with_strict_detection"]
        + overall[
            "identity_unaligned_union_periods_with_strict_subject_scope_detection"
        ]
        != overall["union_periods_with_strict_subject_scope_detection"]
    ):
        raise ValueError(
            "overall strict identity aligned/unaligned counts do not sum to subject-scope total"
        )

    observed = _observed_union_counts(
        all_union_rows, context_prefix="union_timed_results"
    )
    for key, value in observed.items():
        if overall[key] != value:
            raise ValueError(f"overall.{key} does not match union result rows")


def _format_number(value: int | float) -> str:
    if isinstance(value, float) and not value.is_integer():
        return f"{value:,.1f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{int(value):,}".replace(",", ".")


def _format_decimal(value: int | float) -> str:
    rendered = f"{float(value):,.6f}".rstrip("0").rstrip(".")
    return rendered.replace(",", "X").replace(".", ",").replace("X", ".")


def _format_percent(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "n.d."
    value = 100.0 * numerator / denominator
    if value.is_integer():
        return f"{int(value)}%"
    return f"{value:.1f}%".replace(".", ",")


def _format_timestamp(value: Any) -> str:
    if not isinstance(value, str):
        return "Non disponibile"
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return escape(value)
    rendered = parsed.strftime("%d/%m/%Y %H:%M:%S")
    if parsed.microsecond:
        rendered += "." + f"{parsed.microsecond:06d}".rstrip("0")
    return rendered


def _format_time(value: Any) -> str:
    if not isinstance(value, str):
        return "Non disponibile"
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return escape(value)
    rendered = parsed.strftime("%H:%M:%S")
    if parsed.microsecond:
        rendered += "." + f"{parsed.microsecond:06d}".rstrip("0")
    return rendered


def _format_date(value: Any) -> str:
    if not isinstance(value, str):
        return "Data non disponibile"
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return escape(value)
    return parsed.strftime("%d/%m/%Y")


def _anchor_label(by_anchor: Any) -> str:
    if not isinstance(by_anchor, Mapping):
        return "Non disponibile"
    labels = []
    for key, singular, plural in (
        ("passive", "passivo", "passivi"),
        ("aggressive", "aggressivo", "aggressivi"),
    ):
        value = by_anchor.get(key, 0)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
            label = singular if value == 1 else plural
            labels.append(f"{_format_number(value)} {label}")
    return ", ".join(labels) if labels else "Nessun cluster"


def _outcome(row: Mapping[str, Any]) -> tuple[str, str]:
    strict = _numeric_count(row, "clusters_with_strict_detection", context="window")
    withdrawal = _numeric_count(row, "clusters_with_matched_withdrawal", context="window")
    execution = _numeric_count(row, "recovered_execution_clusters", context="window")
    if strict > 0:
        return "strict", "Sequenza stretta"
    if withdrawal > 0:
        return "withdrawal", "Ritiro recuperato; gate stretti non superati"
    if execution > 0:
        return "execution", "Solo esecuzione recuperata"
    return "none", "Nessuna esecuzione recuperata"


def _badge(value: bool, *, yes: str = "Sì", no: str = "No") -> str:
    css = "yes" if value else "no"
    label = yes if value else no
    return f'<span class="badge {css}">{escape(label)}</span>'


def _render_detector_events(events: Sequence[Mapping[str, Any]]) -> str:
    if not events:
        return '<span class="muted">Nessun cluster recuperato</span>'
    cards: list[str] = []
    for event in events:
        anchor = {
            "passive": "ancora passiva",
            "aggressive": "ancora aggressiva",
        }.get(str(event["execution_anchor_mode"]), "ancora non disponibile")
        gates = (
            ("ritiro rapido", event["gate_rapid_matched_withdrawal"]),
            ("fill piccolo/ritiro", event["gate_small_fill_relative_to_withdrawal"]),
            ("movimento pre-fill", event["gate_favorable_pre_fill_move"]),
            ("reversione post-cancel", event["gate_cancel_anchored_reversion"]),
        )
        gate_text = " · ".join(
            f"{escape(label)}: {'sì' if passed else 'no'}" for label, passed in gates
        )
        cards.append(
            '<article class="detector-event">'
            f'<code>{escape(str(event["event_alias"]))}</code>'
            f'<strong>{_format_time(event["cluster_start"])} → '
            f'{_format_time(event["cluster_end"])}</strong>'
            f'<span>{escape(anchor)} · lato {escape(str(event["execution_side"]))} · '
            f'qty {_format_decimal(event["execution_quantity"])} @ '
            f'VWAP {_format_decimal(event["execution_vwap"])}</span>'
            '<div class="event-badges">'
            f'<span>Ritiro attribuito {_badge(event["has_matched_withdrawal"])}</span>'
            f'<span>Sequenza stretta {_badge(event["strict_detection"])}</span>'
            "</div>"
            f'<details><summary>Gate comportamentali</summary><span>{gate_text}</span></details>'
            "</article>"
        )
    return '<div class="detector-events">' + "".join(cards) + "</div>"


def _common_provenance(datasets: Mapping[str, Any]) -> dict[str, Any]:
    provenances = []
    for dataset in datasets.values():
        if isinstance(dataset, Mapping) and isinstance(dataset.get("metrics_provenance"), Mapping):
            provenances.append(dataset["metrics_provenance"])
    if not provenances:
        return {}
    allowlist = ("actor_identity_mode", "execution_anchor_modes", "output_schema_version")
    common: dict[str, Any] = {}
    for key in allowlist:
        values = [provenance.get(key) for provenance in provenances]
        if values and all(value == values[0] for value in values):
            common[key] = values[0]
    return common


def build_dashboard(
    report: Mapping[str, Any], *, generated_at_utc: str | None = None
) -> str:
    _validate_report(report)
    overall = _mapping(report["overall"], context="overall")
    datasets = _mapping(report["datasets"], context="datasets")
    source = _mapping(report["external_source"], context="external_source")
    provenance = _common_provenance(datasets)
    generated_at = _normalize_utc_timestamp(generated_at_utc)

    total_source = _count(overall, "source_external_periods", context="overall")
    timed_source = _count(overall, "source_timed_periods", context="overall")
    date_only = _count(overall, "source_date_only_periods", context="overall")
    union_total = _count(overall, "union_timed_periods", context="overall")
    raw = _count(overall, "union_periods_with_raw_subject_event", context="overall")
    executions = _count(overall, "union_periods_with_recovered_execution", context="overall")
    withdrawals = _count(overall, "union_periods_with_matched_withdrawal", context="overall")
    strict = _count(
        overall, "union_periods_with_strict_subject_scope_detection", context="overall"
    )
    aligned = _count(overall, "identity_aligned_union_timed_periods", context="overall")
    aligned_strict = _count(
        overall, "identity_aligned_union_periods_with_strict_detection", context="overall"
    )

    instrument_options = []
    source_subject_options = []
    coverage_rows = []
    source_alert_rows = []
    window_rows = []
    date_cards = []
    source_alert_index = 0
    window_index = 0
    for instrument in sorted(datasets):
        dataset = _mapping(datasets[instrument], context=f"datasets.{instrument}")
        timed = _count(dataset, "union_timed_periods", context=f"datasets.{instrument}")
        instrument_options.append(
            f'<option value="{escape(instrument)}">{escape(instrument)}</option>'
        )
        if timed:
            coverage_rows.append(
                "<tr>"
                f"<th>{escape(instrument)}</th>"
                f"<td>{_format_number(_count(dataset, 'source_timed_periods', context=instrument))}</td>"
                f"<td>{_format_number(timed)}</td>"
                f"<td>{_format_number(_count(dataset, 'union_periods_with_recovered_execution', context=instrument))}/{_format_number(timed)}</td>"
                f"<td>{_format_number(_count(dataset, 'union_periods_with_matched_withdrawal', context=instrument))}/{_format_number(timed)}</td>"
                f"<td>{_format_number(_count(dataset, 'union_periods_with_strict_subject_scope_detection', context=instrument))}/{_format_number(timed)}</td>"
                "</tr>"
            )

        union_rows = _rows(
            dataset["union_timed_results"],
            context=f"datasets.{instrument}.union_timed_results",
        )
        for row in union_rows:
            window_index += 1
            outcome_key, outcome_label = _outcome(row)
            recovered_clusters = _numeric_count(
                row, "recovered_execution_clusters", context="window"
            )
            matched_clusters = _numeric_count(
                row, "clusters_with_matched_withdrawal", context="window"
            )
            strict_clusters = _numeric_count(
                row, "clusters_with_strict_detection", context="window"
            )
            identity_aligned = row.get("identity_granularity_aligned") is True
            window_rows.append(
                f'<tr data-instrument="{escape(instrument)}" data-outcome="{outcome_key}">'
                f"<td>{window_index}</td>"
                f"<th>{escape(instrument)}</th>"
                f"<td><span class=\"nowrap\">{_format_timestamp(row.get('start'))}</span><br>"
                f"<span class=\"muted nowrap\">→ {_format_timestamp(row.get('end'))}</span></td>"
                f"<td>{_format_number(_numeric_count(row, 'source_windows_merged', context='window'))}</td>"
                f"<td>{escape(str(row.get('identity_namespace', 'Non disponibile')))}<br>"
                f"{_badge(identity_aligned, yes='Allineata', no='Non allineata')}</td>"
                f"<td>{_format_number(recovered_clusters)}<br><span class=\"muted\">{escape(_anchor_label(row.get('recovered_execution_clusters_by_anchor')))}</span></td>"
                f"<td>{_badge(matched_clusters > 0)}<br><span class=\"muted\">{_format_number(matched_clusters)} cluster</span></td>"
                f"<td>{_badge(strict_clusters > 0)}<br><span class=\"muted\">{_format_number(strict_clusters)} cluster</span></td>"
                f'<td><span class="outcome {outcome_key}">{escape(outcome_label)}</span></td>'
                "</tr>"
            )

        source_rows = _rows(
            dataset["source_period_results"],
            context=f"datasets.{instrument}.source_period_results",
        )
        timed_source_rows = [row for row in source_rows if row.get("scope") != "date"]
        actor_aliases = list(
            dict.fromkeys(str(row.get("actor_alias")) for row in timed_source_rows)
        )
        subject_keys = {
            actor_alias: f"{instrument}-{subject_index}"
            for subject_index, actor_alias in enumerate(actor_aliases, start=1)
        }
        subject_labels = {
            actor_alias: f"Soggetto {subject_index}"
            for subject_index, actor_alias in enumerate(actor_aliases, start=1)
        }
        for actor_alias in actor_aliases:
            source_subject_options.append(
                f'<option value="{escape(subject_keys[actor_alias])}">'
                f'{escape(instrument)} · {escape(subject_labels[actor_alias])}</option>'
            )

        for row in timed_source_rows:
            source_alert_index += 1
            actor_alias = str(row.get("actor_alias"))
            subject_key = subject_keys[actor_alias]
            subject_label = subject_labels[actor_alias]
            outcome_key, outcome_label = _outcome(row)
            identity_aligned = row.get("identity_granularity_aligned") is True
            identity_label = (
                "Granularità coincidente"
                if identity_aligned
                else "Granularità non coincidente"
            )
            events = _rows(row["detector_events"], context="source alert detector events")
            source_alert_rows.append(
                f'<tr data-source-instrument="{escape(instrument)}" '
                f'data-source-subject="{escape(subject_key)}" '
                f'data-source-outcome="{outcome_key}">'
                f"<td>{source_alert_index}</td>"
                f"<th>{escape(instrument)}</th>"
                f"<td><strong>{escape(subject_label)}</strong><br>"
                f'<span class="muted">{escape(str(row.get("identity_namespace", "Non disponibile")))}</span></td>'
                f"<td><span class=\"nowrap\">{_format_timestamp(row.get('start'))}</span><br>"
                f"<span class=\"muted nowrap\">→ {_format_timestamp(row.get('end'))}</span></td>"
                f"<td>{_render_detector_events(events)}</td>"
                f"<td>{_badge(identity_aligned, yes=identity_label, no=identity_label)}</td>"
                f'<td><span class="outcome {outcome_key}">{escape(outcome_label)}</span></td>'
                "</tr>"
            )
        for row in source_rows:
            if row.get("scope") != "date":
                continue
            daily_clusters = _numeric_count(
                row, "date_level_recovered_execution_clusters", context="date row"
            )
            daily_withdrawals = _numeric_count(
                row, "date_level_clusters_with_matched_withdrawal", context="date row"
            )
            daily_strict = _numeric_count(
                row, "date_level_clusters_with_strict_detection", context="date row"
            )
            date_cards.append(
                '<article class="date-card">'
                f'<div><span class="eyebrow">{escape(instrument)}</span>'
                f"<h3>{_format_date(row.get('start'))}</h3></div>"
                f'<div class="date-stat"><strong>{_format_number(daily_clusters)}</strong><span>cluster nella giornata</span></div>'
                f'<div class="date-stat"><strong>{escape(_anchor_label(row.get("date_level_recovered_execution_clusters_by_anchor")))}</strong><span>per ancora</span></div>'
                f'<div class="date-stat"><strong>{_format_number(daily_withdrawals)}</strong><span>con ritiro attribuito</span></div>'
                f'<div class="date-stat"><strong>{_format_number(daily_strict)}</strong><span>sequenze strette</span></div>'
                '<p class="date-note">La fonte non fornisce un orario: questi valori descrivono la giornata, ma non misurano il richiamo dell’evento specifico.</p>'
                "</article>"
            )

    stages = (
        ("Attività del soggetto", raw, "stage-raw"),
        ("Cluster di esecuzione", executions, "stage-execution"),
        ("Ritiro attribuito", withdrawals, "stage-withdrawal"),
        ("Sequenza stretta", strict, "stage-strict"),
    )
    stage_rows = []
    for label, value, css in stages:
        width = 100.0 * value / union_total if union_total else 0.0
        stage_rows.append(
            '<div class="stage-row">'
            f'<div class="stage-label"><span>{escape(label)}</span><strong>{_format_number(value)}/{_format_number(union_total)}</strong></div>'
            f'<div class="stage-track"><div class="stage-fill {css}" style="width:{width:.4f}%"></div></div>'
            f'<div class="stage-percent">{_format_percent(value, union_total)}</div>'
            "</div>"
        )

    actor_mode = escape(str(provenance.get("actor_identity_mode", "Non disponibile")))
    schema_version = escape(str(provenance.get("output_schema_version", "Non disponibile")))
    anchor_modes_value = provenance.get("execution_anchor_modes")
    if isinstance(anchor_modes_value, list):
        anchor_modes = ", ".join(escape(str(value)) for value in anchor_modes_value)
    else:
        anchor_modes = "Non disponibile"

    html = f"""<!doctype html>
<html lang="it">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Confronto alert CONSOB e detector</title>
<style>
:root {{
  --ink: #17212b; --muted: #667585; --paper: #f4f7f9; --card: #ffffff;
  --line: #dce4e9; --navy: #12344d; --blue: #1f6f8b; --cyan: #69b3b0;
  --gold: #d6a84b; --orange: #d97b43; --red: #b84a5a; --green: #39856b;
  --shadow: 0 14px 36px rgba(18, 52, 77, .08);
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--paper); color: var(--ink); font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; line-height: 1.45; }}
header {{ color: white; background: linear-gradient(125deg, #0d2b3d 0%, #174e68 60%, #24757c 100%); padding: 44px max(24px, calc((100vw - 1280px) / 2)); }}
header .eyebrow {{ color: #aee1df; }}
h1 {{ margin: 6px 0 10px; font-size: clamp(2rem, 5vw, 3.8rem); line-height: 1.02; letter-spacing: -.04em; max-width: 900px; }}
header p {{ max-width: 860px; margin: 0; color: #d7e8ee; font-size: 1.05rem; }}
main {{ max-width: 1280px; margin: 0 auto; padding: 30px 24px 56px; }}
section {{ margin-bottom: 30px; }}
.eyebrow {{ display: block; text-transform: uppercase; letter-spacing: .11em; font-size: .75rem; font-weight: 800; color: var(--blue); }}
h2 {{ margin: 4px 0 6px; font-size: 1.65rem; letter-spacing: -.02em; }}
h3 {{ margin: 4px 0; }}
.section-intro {{ margin: 0 0 18px; color: var(--muted); max-width: 920px; }}
.kpis {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; margin-top: -58px; position: relative; }}
.kpi, .card, .date-card {{ background: var(--card); border: 1px solid rgba(220, 228, 233, .92); border-radius: 16px; box-shadow: var(--shadow); }}
.kpi {{ padding: 20px; min-height: 142px; }}
.kpi strong {{ display: block; color: var(--navy); font-size: 2.4rem; letter-spacing: -.04em; }}
.kpi span {{ display: block; font-weight: 750; }}
.kpi small {{ display: block; margin-top: 8px; color: var(--muted); }}
.story-grid {{ display: grid; grid-template-columns: minmax(0, 1.45fr) minmax(280px, .55fr); gap: 18px; }}
.card {{ padding: 22px; }}
.stage-row {{ display: grid; grid-template-columns: minmax(170px, .8fr) minmax(220px, 1.8fr) 64px; align-items: center; gap: 12px; margin: 17px 0; }}
.stage-label {{ display: flex; justify-content: space-between; gap: 10px; }}
.stage-label strong {{ color: var(--navy); }}
.stage-track {{ height: 14px; background: #edf2f4; overflow: hidden; border-radius: 99px; }}
.stage-fill {{ height: 100%; border-radius: inherit; }}
.stage-raw {{ background: var(--blue); }} .stage-execution {{ background: var(--cyan); }}
.stage-withdrawal {{ background: var(--gold); }} .stage-strict {{ background: var(--red); }}
.stage-percent {{ font-weight: 800; text-align: right; }}
.callout {{ display: flex; flex-direction: column; justify-content: space-between; background: var(--navy); color: white; }}
.callout .eyebrow {{ color: #aee1df; }}
.callout strong {{ font-size: 3rem; line-height: 1; color: #f3c969; }}
.callout p {{ color: #d7e8ee; }}
.table-card {{ padding: 0; overflow: hidden; }}
.table-scroll {{ overflow-x: auto; }}
table {{ width: 100%; border-collapse: collapse; font-size: .9rem; }}
th, td {{ padding: 13px 14px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }}
thead th {{ background: #eef3f5; color: var(--navy); font-size: .75rem; text-transform: uppercase; letter-spacing: .04em; white-space: nowrap; }}
tbody tr:hover {{ background: #f8fbfc; }}
.coverage-table th:first-child {{ color: var(--navy); }}
.source-alert-table {{ min-width: 1180px; }}
.source-alert-table td:nth-child(5) {{ min-width: 440px; }}
.detector-events {{ display: grid; gap: 9px; }}
.detector-event {{ display: grid; gap: 5px; border: 1px solid var(--line); border-left: 4px solid var(--blue); border-radius: 10px; padding: 10px 11px; background: #fbfdfe; }}
.detector-event code {{ color: var(--navy); font-size: .74rem; overflow-wrap: anywhere; }}
.detector-event strong {{ color: var(--navy); font-size: .88rem; }}
.detector-event > span, .detector-event details {{ color: var(--muted); font-size: .78rem; }}
.event-badges {{ display: flex; flex-wrap: wrap; gap: 7px 14px; font-size: .78rem; }}
.event-badges > span {{ display: inline-flex; align-items: center; gap: 5px; }}
.detector-event summary {{ cursor: pointer; color: var(--blue); font-weight: 750; }}
.detector-event details > span {{ display: block; padding-top: 5px; }}
.filters {{ display: flex; flex-wrap: wrap; align-items: end; gap: 12px; margin: 16px 0; }}
.filter {{ display: grid; gap: 5px; }}
.filter label {{ font-weight: 750; font-size: .82rem; color: var(--navy); }}
select {{ min-width: 210px; border: 1px solid #bdcbd3; border-radius: 9px; padding: 9px 34px 9px 11px; background: white; color: var(--ink); font: inherit; }}
.visible-count {{ margin-left: auto; color: var(--muted); font-size: .9rem; }}
.badge, .outcome {{ display: inline-flex; align-items: center; border-radius: 99px; padding: 3px 8px; font-size: .75rem; font-weight: 800; white-space: nowrap; }}
.badge.yes {{ color: #176348; background: #e2f3ec; }} .badge.no {{ color: #8c3845; background: #f8e5e8; }}
.outcome.strict {{ color: #792d3a; background: #f8dfe3; }}
.outcome.withdrawal {{ color: #785815; background: #f8edcf; }}
.outcome.execution {{ color: #195e72; background: #dff1f5; }}
.outcome.none {{ color: #596573; background: #e9edf0; }}
.muted {{ color: var(--muted); font-size: .8rem; }} .nowrap {{ white-space: nowrap; }}
.date-card {{ display: grid; grid-template-columns: 1.1fr repeat(4, minmax(100px, .7fr)); align-items: center; gap: 18px; padding: 22px; }}
.date-stat strong, .date-stat span {{ display: block; }}
.date-stat strong {{ color: var(--navy); font-size: 1.15rem; }}
.date-stat span {{ color: var(--muted); font-size: .78rem; }}
.date-note {{ grid-column: 1 / -1; background: #fff8e8; border-left: 4px solid var(--gold); padding: 11px 13px; margin: 0; color: #705722; }}
.provenance-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }}
.provenance-list {{ margin: 12px 0 0; padding-left: 20px; }}
.provenance-list li {{ margin: 8px 0; overflow-wrap: anywhere; }}
.hash {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .78rem; word-break: break-all; }}
.footnote {{ margin-top: 18px; color: var(--muted); font-size: .8rem; }}
[hidden] {{ display: none !important; }}
@media (max-width: 900px) {{
  .kpis {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
  .story-grid, .provenance-grid {{ grid-template-columns: 1fr; }}
  .date-card {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
  .date-note {{ grid-column: 1 / -1; }}
}}
@media (max-width: 600px) {{
  header {{ padding-bottom: 74px; }} main {{ padding-left: 14px; padding-right: 14px; }}
  .kpis {{ grid-template-columns: 1fr; }}
  .stage-row {{ grid-template-columns: 1fr 58px; }} .stage-track {{ grid-column: 1; }} .stage-percent {{ grid-column: 2; grid-row: 2; }}
  .date-card {{ grid-template-columns: 1fr; }} .date-note {{ grid-column: 1; }}
  .visible-count {{ width: 100%; margin-left: 0; }}
}}
</style>
</head>
<body>
<header>
  <span class="eyebrow">Audit esterno · sorveglianza di mercato</span>
  <h1>Confronto alert CONSOB e detector</h1>
  <p>Dalla segnalazione esterna alla sequenza comportamentale stretta: copertura temporale, recupero delle esecuzioni, ritiro attribuito e gate finali, senza trasformare gli alert esterni in etichette di intento.</p>
</header>
<main>
  <section class="kpis" aria-label="Numeri principali">
    <article class="kpi"><span>Segnalazioni CONSOB</span><strong>{_format_number(total_source)}</strong><small>{_format_number(timed_source)} con orario, {_format_number(date_only)} solo data</small></article>
    <article class="kpi"><span>Finestre intraday distinte</span><strong>{_format_number(union_total)}</strong><small>Intervalli sovrapposti uniti prima del confronto</small></article>
    <article class="kpi"><span>Ritiro attribuito</span><strong>{_format_number(withdrawals)}</strong><small>{_format_percent(withdrawals, union_total)} delle finestre intraday</small></article>
    <article class="kpi"><span>Sequenze strette</span><strong>{_format_number(strict)}</strong><small>{_format_percent(strict, union_total)} a livello di soggetto</small></article>
  </section>

  <section>
    <span class="eyebrow">Lettura in quattro passaggi</span>
    <h2>Sintesi del confronto</h2>
    <p class="section-intro">Il {_format_percent(executions, union_total)} delle finestre intraday contiene almeno un cluster di esecuzione. La selezione si restringe quando si richiede un ritiro attribuito e, infine, il superamento congiunto di tutti i gate comportamentali.</p>
    <div class="story-grid">
      <article class="card">{''.join(stage_rows)}</article>
      <aside class="card callout">
        <div><span class="eyebrow">Exact actor</span><h3>Identità allineata</h3></div>
        <strong>{_format_number(aligned_strict)}/{_format_number(aligned)}</strong>
        <p>finestre con identità esterna e chiave del detector alla stessa granularità che superano tutti i gate. I match stretti a identità non allineata restano evidenza a livello di soggetto/firm, non recall exact-actor.</p>
      </aside>
    </div>
  </section>

  <section>
    <span class="eyebrow">Confronto aggregato</span>
    <h2>Copertura per titolo</h2>
    <p class="section-intro">Le segnalazioni con orari sovrapposti sono unite prima del calcolo; i denominatori sono quindi finestre distinte, non righe originarie della fonte.</p>
    <div class="card table-card table-scroll">
      <table class="coverage-table">
        <thead><tr><th>Titolo</th><th>Alert con orario</th><th>Finestre distinte</th><th>Esecuzione</th><th>Ritiro</th><th>Stretta</th></tr></thead>
        <tbody>{''.join(coverage_rows)}</tbody>
      </table>
    </div>
  </section>

  <section>
    <span class="eyebrow">Righe originali della fonte</span>
    <h2>Alert originali ed eventi detector</h2>
    <p class="section-intro">Una riga per ogni alert con orario riportato nel PDF. La colonna “Evento detector corrispondente” elenca tutti i cluster del soggetto con almeno un fill nell’intervallo; lo stesso alias evento può quindi ricomparire in alert originali sovrapposti. Questa tabella documenta il mapping riga-per-riga, mentre i tassi scientifici continuano a usare le finestre unite mostrate nella sezione successiva.</p>
    <div class="filters">
      <div class="filter"><label for="source-instrument-filter">Titolo</label><select id="source-instrument-filter"><option value="all">Tutti i titoli</option>{''.join(instrument_options)}</select></div>
      <div class="filter"><label for="source-subject-filter">Soggetto del PDF</label><select id="source-subject-filter"><option value="all">Tutti i soggetti</option>{''.join(source_subject_options)}</select></div>
      <div class="filter"><label for="source-outcome-filter">Esito</label><select id="source-outcome-filter"><option value="all">Tutti gli esiti</option><option value="strict">Sequenza stretta</option><option value="withdrawal">Ritiro senza sequenza stretta</option><option value="execution">Solo esecuzione</option><option value="none">Nessuna esecuzione</option></select></div>
      <div class="visible-count" id="visible-source-count" aria-live="polite">{_format_number(timed_source)} alert visualizzati</div>
    </div>
    <div class="card table-card table-scroll">
      <table id="source-alert-table" class="source-alert-table">
        <thead><tr><th>#</th><th>Titolo</th><th>Soggetto</th><th>Intervallo alert PDF</th><th>Evento detector corrispondente</th><th>Identità</th><th>Esito alert</th></tr></thead>
        <tbody>{''.join(source_alert_rows)}</tbody>
      </table>
    </div>
  </section>

  <section>
    <span class="eyebrow">Dettaglio investigativo</span>
    <h2>Finestre CONSOB</h2>
    <p class="section-intro">Una riga per ogni intervallo intraday unito. I filtri modificano solo la vista: i conteggi scientifici restano quelli dell’intera popolazione.</p>
    <div class="filters">
      <div class="filter"><label for="instrument-filter">Titolo</label><select id="instrument-filter"><option value="all">Tutti i titoli</option>{''.join(instrument_options)}</select></div>
      <div class="filter"><label for="outcome-filter">Esito</label><select id="outcome-filter"><option value="all">Tutti gli esiti</option><option value="strict">Sequenza stretta</option><option value="withdrawal">Ritiro senza sequenza stretta</option><option value="execution">Solo esecuzione</option><option value="none">Nessuna esecuzione</option></select></div>
      <div class="visible-count" id="visible-window-count" aria-live="polite">{_format_number(union_total)} finestre visualizzate</div>
    </div>
    <div class="card table-card table-scroll">
      <table id="window-table">
        <thead><tr><th>#</th><th>Titolo</th><th>Intervallo</th><th>Alert uniti</th><th>Identità</th><th>Cluster esecuzione</th><th>Ritiro</th><th>Stretta</th><th>Esito</th></tr></thead>
        <tbody>{''.join(window_rows)}</tbody>
      </table>
    </div>
  </section>

  <section>
    <span class="eyebrow">Fuori dal denominatore intraday</span>
    <h2>Evidenza giornaliera non localizzabile</h2>
    <p class="section-intro">Questa sezione documenta l’attività nella data segnalata, ma non entra nei numeratori o denominatori del confronto evento-per-evento.</p>
    {''.join(date_cards) if date_cards else '<div class="card"><p>Nessuna segnalazione solo-data.</p></div>'}
  </section>

  <section>
    <span class="eyebrow">Interpretazione e tracciabilità</span>
    <h2>Limiti e provenienza</h2>
    <div class="provenance-grid">
      <article class="card">
        <h3>Come leggere il risultato</h3>
        <ul class="provenance-list">
          <li>Gli alert CONSOB sono periodi segnalati dalla fonte, non etichette di intento manipolativo indipendentemente riadjudicate.</li>
          <li>La sovrapposizione a livello di soggetto può usare provenienza client/firm; il recall exact-actor è identificabile solo con granularità d’identità allineata.</li>
          <li>Una sequenza stretta richiede esecuzione recuperata, ritiro attribuito e superamento di tutti i gate comportamentali configurati.</li>
          <li>Le finestre solo-data sono evidenza di copertura giornaliera, non recall dell’evento specifico.</li>
        </ul>
      </article>
      <article class="card">
        <h3>Provenienza tecnica</h3>
        <ul class="provenance-list">
          <li>Fonte primaria: <strong>{escape(str(source.get('source_kind', 'Non disponibile')))}</strong></li>
          <li>SHA-256 fonte: <span class="hash">{escape(str(source['sha256']))}</span></li>
          <li>Modalità identità: <strong>{actor_mode}</strong></li>
          <li>Ancore configurate: <strong>{anchor_modes}</strong></li>
          <li>Schema output detector: <strong>{schema_version}</strong></li>
          <li>Schema dashboard: <strong>{DASHBOARD_SCHEMA_VERSION}</strong></li>
          <li>Generata il: <strong>{escape(generated_at)}</strong></li>
        </ul>
      </article>
    </div>
    <p class="footnote">La dashboard include solo campi aggregati e finestre temporali necessarie al confronto. Identificativi esterni pseudonimizzati, path locali e comandi di esecuzione non sono incorporati.</p>
  </section>
</main>
<script>
(() => {{
  const sourceInstrument = document.getElementById('source-instrument-filter');
  const sourceSubject = document.getElementById('source-subject-filter');
  const sourceOutcome = document.getElementById('source-outcome-filter');
  const sourceRows = Array.from(document.querySelectorAll('#source-alert-table tbody tr'));
  const sourceCount = document.getElementById('visible-source-count');

  function applySourceFilters() {{
    let visible = 0;
    sourceRows.forEach((row) => {{
      const instrumentMatch = sourceInstrument.value === 'all' || row.dataset.sourceInstrument === sourceInstrument.value;
      const subjectMatch = sourceSubject.value === 'all' || row.dataset.sourceSubject === sourceSubject.value;
      const outcomeMatch = sourceOutcome.value === 'all' || row.dataset.sourceOutcome === sourceOutcome.value;
      const show = instrumentMatch && subjectMatch && outcomeMatch;
      row.hidden = !show;
      if (show) visible += 1;
    }});
    sourceCount.textContent = visible + (visible === 1 ? ' alert visualizzato' : ' alert visualizzati');
  }}

  [sourceInstrument, sourceSubject, sourceOutcome].forEach((control) =>
    control.addEventListener('change', applySourceFilters)
  );
  applySourceFilters();

  const instrumentFilter = document.getElementById('instrument-filter');
  const outcomeFilter = document.getElementById('outcome-filter');
  const rows = Array.from(document.querySelectorAll('#window-table tbody tr'));
  const count = document.getElementById('visible-window-count');

  function applyFilters() {{
    const instrument = instrumentFilter.value;
    const outcome = outcomeFilter.value;
    let visible = 0;
    rows.forEach(function (row) {{
      const showInstrument = instrument === 'all' || row.dataset.instrument === instrument;
      const showOutcome = outcome === 'all' || row.dataset.outcome === outcome;
      row.hidden = !(showInstrument && showOutcome);
      if (!row.hidden) visible += 1;
    }});
    count.textContent = visible + (visible === 1 ? ' finestra visualizzata' : ' finestre visualizzate');
  }}

  instrumentFilter.addEventListener('change', applyFilters);
  outcomeFilter.addEventListener('change', applyFilters);
  applyFilters();
}})();
</script>
</body>
</html>
"""
    return html


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    audit_bytes = args.audit_json.read_bytes()
    report = json.loads(audit_bytes)
    generated_at = _normalize_utc_timestamp(args.generated_at_utc)
    html = build_dashboard(report, generated_at_utc=generated_at)

    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    args.output_html.write_text(html, encoding="utf-8")
    dashboard_bytes = args.output_html.read_bytes()
    metadata_path = args.output_html.with_suffix(".metadata.json")
    overall = _mapping(report["overall"], context="overall")
    source = _mapping(report["external_source"], context="external_source")
    metadata = {
        "schema_version": DASHBOARD_SCHEMA_VERSION,
        "generated_at_utc": generated_at,
        "output_html": args.output_html.name,
        "audit_json": args.audit_json.name,
        "audit_sha256": hashlib.sha256(audit_bytes).hexdigest(),
        "dashboard_sha256": hashlib.sha256(dashboard_bytes).hexdigest(),
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "external_source_sha256": source["sha256"],
        "source_external_periods": overall["source_external_periods"],
        "source_timed_periods": overall["source_timed_periods"],
        "source_date_only_periods": overall["source_date_only_periods"],
        "union_timed_periods": overall["union_timed_periods"],
        "union_periods_with_recovered_execution": overall[
            "union_periods_with_recovered_execution"
        ],
        "union_periods_with_matched_withdrawal": overall[
            "union_periods_with_matched_withdrawal"
        ],
        "union_periods_with_strict_subject_scope_detection": overall[
            "union_periods_with_strict_subject_scope_detection"
        ],
        "identity_aligned_union_timed_periods": overall[
            "identity_aligned_union_timed_periods"
        ],
        "identity_aligned_union_periods_with_strict_detection": overall[
            "identity_aligned_union_periods_with_strict_detection"
        ],
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(f"Dashboard written: {args.output_html}")
    print(f"Metadata written: {metadata_path}")
    print(
        "Timed union windows: "
        f"{overall['union_timed_periods']} | "
        f"execution: {overall['union_periods_with_recovered_execution']} | "
        f"withdrawal: {overall['union_periods_with_matched_withdrawal']} | "
        f"strict: {overall['union_periods_with_strict_subject_scope_detection']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
