#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import re
import secrets
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from spoofing_detection.lob.actor_identity import (  # noqa: E402
    normalize_client_original_identity_value,
    normalize_identity_value,
)

MONTH = {
    "JAN": 1,
    "FEB": 2,
    "MAR": 3,
    "APR": 4,
    "MAY": 5,
    "JUN": 6,
    "JUL": 7,
    "AUG": 8,
    "SEP": 9,
    "OCT": 10,
    "NOV": 11,
    "DEC": 12,
}
TRUTHY = ("Y", "1", "TRUE", "T")
SOURCE_CLIENT_ID = "_source_client_original_id"
SOURCE_FIRM_ID = "_source_firm_id"


@dataclass(frozen=True)
class ExternalAlert:
    dataset: str
    external_actor_id: str
    start: datetime
    end: datetime
    date_level: bool = False


@dataclass(frozen=True)
class DatasetPaths:
    raw: Path
    metrics: Path


@dataclass(frozen=True)
class ExternalIdentityResolution:
    identity_namespace: str | None
    raw_rows: int
    detector_actor_key_count: int
    identity_granularity_aligned: bool


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit external alert coverage and detector overlap without exporting source identifiers."
    )
    parser.add_argument("--external-source", type=Path, required=True, help="Source PDF or extracted text file")
    for dataset in ("nexi", "risanamento", "ferrari"):
        parser.add_argument(f"--{dataset}-raw", type=Path, required=True)
        parser.add_argument(f"--{dataset}-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def _read_source_text(path: Path) -> str:
    if path.suffix.lower() != ".pdf":
        return path.read_text(encoding="utf-8")
    completed = subprocess.run(
        ["pdftotext", "-layout", str(path), "-"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_metrics_provenance(metadata_path: Path) -> dict[str, Any]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    allowed_input_hashes = {
        "config_sha256",
        "empirical_depth_kernel_sha256",
        "quote_panel_sha256",
        "raw_events_sha256",
    }
    allowed_artifacts = {
        "actor_mcps_scores",
        "candidate_deceptive_orders",
        "execution_cancel_candidates",
        "execution_cluster_members",
        "execution_metrics",
        "rejected_executions",
        "spoofing_compatible_events",
        "state_time_series",
    }

    def safe_hashes(value: Any, allowed_keys: set[str]) -> dict[str, str]:
        if not isinstance(value, dict):
            return {}
        return {
            key: digest
            for key, digest in value.items()
            if key in allowed_keys
            and isinstance(digest, str)
            and re.fullmatch(r"[0-9a-fA-F]{64}", digest) is not None
        }

    row_counts = metadata.get("row_counts")
    safe_row_counts = (
        {
            key: int(value)
            for key, value in row_counts.items()
            if key in allowed_artifacts | {"input_rows_for_compute"}
            and isinstance(value, int)
            and not isinstance(value, bool)
            and value >= 0
        }
        if isinstance(row_counts, dict)
        else {}
    )
    provenance: dict[str, Any] = {
        "metadata_sha256": _sha256(metadata_path),
        "input_hashes": safe_hashes(metadata.get("input_hashes"), allowed_input_hashes),
        "artifact_hashes": safe_hashes(metadata.get("artifact_hashes"), allowed_artifacts),
        "row_counts": safe_row_counts,
    }
    if metadata.get("output_schema_version") in {
        "actor_execution_anchor_v1",
        "actor_execution_anchor_v2",
    }:
        provenance["output_schema_version"] = metadata["output_schema_version"]
    if metadata.get("actor_identity_mode") == "client_then_firm":
        provenance["actor_identity_mode"] = "client_then_firm"
    for key in ("execution_anchor_modes", "observed_execution_anchor_modes"):
        value = metadata.get(key)
        if isinstance(value, list) and all(mode in {"passive", "aggressive"} for mode in value):
            provenance[key] = value
    if metadata.get("state_client_mode") in {"all", "execution-actors", "passive-fill-clients"}:
        provenance["state_client_mode"] = metadata["state_client_mode"]
    if isinstance(metadata.get("compact_state"), bool):
        provenance["compact_state"] = metadata["compact_state"]
    return provenance


def _pseudonym(value: str, alias_key: bytes, *, prefix: str) -> str:
    digest = hmac.new(alias_key, value.encode(), hashlib.sha256).hexdigest()
    return prefix + digest[:12]


def _alias(actor_id: str, alias_key: bytes) -> str:
    return _pseudonym(actor_id, alias_key, prefix="actor_")


def _detector_event_alias(actor_key: str, cluster_id: str, alias_key: bytes) -> str:
    return _pseudonym(
        f"{actor_key}\x1f{cluster_id}", alias_key, prefix="detector_event_"
    )


def _parse_interval(day: int, month: int, year: int, value: str) -> datetime:
    if not re.fullmatch(r"\d{9}", value):
        raise ValueError(f"expected HHMMSSmmm time, got {value!r}")
    parsed = datetime(
        year,
        month,
        day,
        int(value[0:2]),
        int(value[2:4]),
        int(value[4:6]),
        int(value[6:9]) * 1_000,
    )
    round_trip = parsed.strftime("%H%M%S") + f"{parsed.microsecond // 1_000:03d}"
    if round_trip != value:
        raise ValueError(f"compact time failed round-trip validation: {value!r}")
    return parsed


def parse_external_alerts(text: str) -> dict[str, list[ExternalAlert]]:
    alerts: dict[str, list[ExternalAlert]] = {dataset: [] for dataset in ("NEXI", "RISANAMENTO", "FERRARI")}

    risa_match = re.search(
        r"RISANAMENTO.*?giornata del (\d{2})/(\d{2})/(\d{4}) relativo al\s+committente\s+(\d+)",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if risa_match is None:
        raise ValueError("RISANAMENTO alert date and actor were not found in the external source")
    day, month, year, actor_id = risa_match.groups()
    start = datetime(int(year), int(month), int(day))
    alerts["RISANAMENTO"].append(
        ExternalAlert(
            dataset="RISANAMENTO",
            external_actor_id=actor_id,
            start=start,
            end=start,
            date_level=True,
        )
    )

    ferrari_match = re.search(
        r"AZIONI\s+FERRARI\s*\.?(.*?)(?=AZIONI\s+NEXI\b)",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if ferrari_match is None:
        raise ValueError("FERRARI alert section was not found in the external source")
    ferrari_section = ferrari_match.group(1)
    current_actor: str | None = None
    for raw_line in ferrari_section.splitlines():
        line = raw_line.strip()
        if re.fullmatch(r"\d+_\d+", line):
            current_actor = line
            continue
        interval = re.search(r"time interval:\s*\[(\d{9})\]-\[(\d{9})\]", line)
        if current_actor is None or interval is None:
            continue
        alerts["FERRARI"].append(
            ExternalAlert(
                dataset="FERRARI",
                external_actor_id=current_actor,
                start=_parse_interval(13, 6, 2024, interval.group(1)),
                end=_parse_interval(13, 6, 2024, interval.group(2)),
            )
        )
    if not alerts["FERRARI"]:
        raise ValueError("no FERRARI alert windows were found in the external source")

    nexi_match = re.search(
        r"AZIONI\s+NEXI\s*:.*?committente\s+(\d+).*?\[(\d{2})/([A-Z]{3})/(\d{4})\].*?"
        r"\[(\d{9})\]-\[(\d{9})\]",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if nexi_match is None:
        raise ValueError("NEXI alert window and actor were not found in the external source")
    actor_id, day, month_text, year, start_value, end_value = nexi_match.groups()
    month = MONTH[month_text.upper()]
    alerts["NEXI"].append(
        ExternalAlert(
            dataset="NEXI",
            external_actor_id=actor_id,
            start=_parse_interval(int(day), month, int(year), start_value),
            end=_parse_interval(int(day), month, int(year), end_value),
        )
    )
    return alerts


def _normalized_text(column: str) -> pl.Expr:
    return pl.col(column).cast(pl.String, strict=False).str.strip_chars()


def _normalized_identity(column: str) -> pl.Expr:
    return pl.col(column).map_elements(normalize_identity_value, return_dtype=pl.String)


def _normalized_client_identity(column: str) -> pl.Expr:
    return pl.col(column).map_elements(
        normalize_client_original_identity_value,
        return_dtype=pl.String,
    )


def _valid_identity(text: pl.Expr) -> pl.Expr:
    return text.is_not_null() & ~text.str.to_lowercase().is_in(("", "nan", "none", "null"))


def _datetime_candidate(column: str) -> pl.Expr:
    text = _normalized_text(column)
    return pl.coalesce(
        pl.col(column).cast(pl.Datetime("us"), strict=False),
        text.str.strptime(
            pl.Datetime("us"),
            format="%Y%m%d %H:%M:%S%.f",
            strict=False,
        ),
        text.str.strptime(
            pl.Datetime("us"),
            format="%Y-%m-%d %H:%M:%S%.f",
            strict=False,
        ),
    )


def _with_canonical_fields(raw: pl.DataFrame) -> pl.DataFrame:
    required = {
        "TRADETIME",
        "BOOKOUTTIME",
        "BOOKIN",
        "SEQUENCETIME",
        "NMSC_ORIGINALCLIENTIDSHORTCODE",
        "FIRMID",
        "ORDEREVENTTYPE (*)",
        "PASSIVEORDER",
        "AGGRESSIVEORDER",
        "LASTSHARES",
        "LASTTRADEDPX",
    }
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"raw input is missing required audit columns: {', '.join(missing)}")

    client = _normalized_client_identity("NMSC_ORIGINALCLIENTIDSHORTCODE")
    firm = _normalized_identity("FIRMID")
    actor_key = (
        pl.when(_valid_identity(client))
        .then(pl.concat_str(pl.lit("client_original:"), client))
        .when(_valid_identity(firm))
        .then(pl.concat_str(pl.lit("firm:"), firm))
        .otherwise(None)
    )
    timestamp_candidates = [
        _datetime_candidate(column)
        for column in ("TRADETIME", "BOOKOUTTIME", "BOOKIN", "SEQUENCETIME")
    ]
    return raw.with_columns(
        actor_key.alias("_actor_key"),
        client.alias(SOURCE_CLIENT_ID),
        firm.alias(SOURCE_FIRM_ID),
        pl.coalesce(timestamp_candidates).alias("_event_ts"),
    )


def _period_filter(column: str, alert: ExternalAlert) -> pl.Expr:
    if alert.date_level:
        return pl.col(column).dt.date() == alert.start.date()
    return pl.col(column).is_between(alert.start, alert.end, closed="both")


def _role_counts(rows: pl.DataFrame) -> dict[str, int]:
    fill = pl.col("ORDEREVENTTYPE (*)") == 3
    passive = _normalized_text("PASSIVEORDER").str.to_uppercase().is_in(TRUTHY)
    aggressive = _normalized_text("AGGRESSIVEORDER").str.to_uppercase().is_in(TRUTHY)
    valid_aggressive_trade = (
        pl.col("LASTSHARES").cast(pl.Float64, strict=False).fill_null(0.0) > 0
    ) & pl.col("LASTTRADEDPX").cast(pl.Float64, strict=False).is_finite()
    return {
        "passive_fill_rows": rows.filter(fill & passive & ~aggressive).height,
        "aggressive_fill_rows": rows.filter(fill & aggressive & ~passive).height,
        "aggressive_fill_rows_with_lastshares_and_lasttradedpx": rows.filter(
            fill & aggressive & ~passive & valid_aggressive_trade
        ).height,
    }


def _resolve_external_identity(
    raw: pl.DataFrame,
    external_actor_id: str,
) -> ExternalIdentityResolution:
    matching_namespaces = [
        (namespace, column)
        for namespace, column in (
            ("client_original", SOURCE_CLIENT_ID),
            ("firm", SOURCE_FIRM_ID),
        )
        if raw.filter(pl.col(column) == external_actor_id).height
    ]
    if not matching_namespaces:
        return ExternalIdentityResolution(None, 0, 0, False)
    if len(matching_namespaces) > 1:
        raise ValueError("external actor identifier is ambiguous across identity namespaces")

    identity_namespace, source_column = matching_namespaces[0]
    subject_rows = raw.filter(pl.col(source_column) == external_actor_id)
    actor_keys = subject_rows.get_column("_actor_key").drop_nulls().unique().to_list()
    expected_single_key = f"{identity_namespace}:{external_actor_id}"
    return ExternalIdentityResolution(
        identity_namespace=identity_namespace,
        raw_rows=subject_rows.height,
        detector_actor_key_count=len(actor_keys),
        identity_granularity_aligned=actor_keys == [expected_single_key],
    )


def _with_source_identities(
    frame: pl.DataFrame,
    *,
    client_column: str,
    firm_column: str,
) -> pl.DataFrame:
    missing = sorted({client_column, firm_column} - set(frame.columns))
    if missing:
        raise ValueError(f"artifact is missing source identity columns: {', '.join(missing)}")
    return frame.with_columns(
        _normalized_client_identity(client_column).alias(SOURCE_CLIENT_ID),
        _normalized_identity(firm_column).alias(SOURCE_FIRM_ID),
    )


def _subject_rows(
    frame: pl.DataFrame,
    resolution: ExternalIdentityResolution,
    external_actor_id: str,
) -> pl.DataFrame:
    source_column = {
        "client_original": SOURCE_CLIENT_ID,
        "firm": SOURCE_FIRM_ID,
    }.get(resolution.identity_namespace)
    if source_column is None:
        return frame.head(0)
    return frame.filter(pl.col(source_column) == external_actor_id)


def _filter_alerts(frame: pl.DataFrame, column: str, alerts: list[ExternalAlert]) -> pl.DataFrame:
    if not alerts:
        return frame.head(0)
    return frame.filter(pl.any_horizontal([_period_filter(column, alert) for alert in alerts]))


def _metrics_for_members(metrics: pl.DataFrame, members: pl.DataFrame) -> pl.DataFrame:
    keys = members.select("actor_key", "execution_cluster_id").unique()
    return metrics.join(keys, on=["actor_key", "execution_cluster_id"], how="semi")


def _counts_by(frame: pl.DataFrame, column: str) -> dict[str, int]:
    if frame.is_empty():
        return {}
    return {
        str(row[column]): int(row["len"])
        for row in frame.group_by(column).len().sort(column).to_dicts()
    }


def _required_timestamp(value: Any, *, field: str) -> str:
    if not isinstance(value, datetime):
        raise ValueError(f"detector event {field} must be a timestamp")
    return value.isoformat()


def _required_nonempty_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"detector event {field} must be a non-empty string")
    return value


def _required_bool(value: Any, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"detector event {field} must be boolean")
    return value


def _required_positive_finite_float(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"detector event {field} must be positive and finite")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"detector event {field} must be positive and finite")
    return result


def _episode_metrics(metrics: pl.DataFrame) -> pl.DataFrame:
    if metrics.filter(
        pl.col("spoofing_compatible_episode").fill_null(False) & pl.col("episode_id").is_null()
    ).height:
        raise ValueError("strict detector episode must have a non-null episode_id")
    return (
        metrics.filter(pl.col("episode_id").is_not_null())
        .with_columns(
            pl.when(pl.col("episode_mixed_anchor").fill_null(False))
            .then(pl.lit("mixed"))
            .otherwise(pl.col("execution_anchor_mode"))
            .alias("episode_anchor_mode")
        )
        .sort(
        ["cluster_start_ts", "cluster_end_ts", "execution_anchor_mode", "execution_cluster_id"]
        )
        .unique(subset="episode_id", keep="first", maintain_order=True)
    )


def _detector_event_details(metrics: pl.DataFrame, alias_key: bytes) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    ordered = _episode_metrics(metrics)
    for metric in ordered.to_dicts():
        actor_key = _required_nonempty_string(metric["actor_key"], field="actor_key")
        episode_id = _required_nonempty_string(
            metric["episode_id"], field="episode_id"
        )
        details.append(
            {
                "event_alias": _detector_event_alias(actor_key, episode_id, alias_key),
                "analytical_unit": "candidate_posture_episode",
                "cluster_start": _required_timestamp(
                    metric["episode_start_ts"], field="episode_start_ts"
                ),
                "cluster_end": _required_timestamp(
                    metric["episode_end_ts"], field="episode_end_ts"
                ),
                "execution_anchor_mode": _required_nonempty_string(
                    metric["episode_anchor_mode"], field="episode_anchor_mode"
                ),
                "execution_side": _required_nonempty_string(
                    metric["execution_side"], field="execution_side"
                ),
                "execution_quantity": _required_positive_finite_float(
                    metric["episode_total_execution_quantity"],
                    field="episode_total_execution_quantity",
                ),
                "execution_vwap": _required_positive_finite_float(
                    metric["episode_execution_vwap"], field="episode_execution_vwap"
                ),
                "has_matched_withdrawal": _required_bool(
                    metric["episode_has_matched_withdrawal"],
                    field="episode_has_matched_withdrawal",
                ),
                "gate_rapid_matched_withdrawal": _required_bool(
                    metric["episode_has_matched_withdrawal"],
                    field="episode_has_matched_withdrawal",
                ),
                "gate_small_fill_relative_to_withdrawal": _required_bool(
                    metric["episode_gate_joint_smallness"],
                    field="episode_gate_joint_smallness",
                ),
                "gate_favorable_pre_fill_move": _required_bool(
                    metric["episode_price_path_observed"]
                    and metric["episode_favorable_mid_move"] > 0,
                    field="episode_favorable_mid_move_positive",
                ),
                "gate_cancel_anchored_reversion": _required_bool(
                    metric["episode_price_path_observed"]
                    and metric["episode_post_cancel_mid_reversion"] > 0,
                    field="episode_post_cancel_mid_reversion_positive",
                ),
                "strict_detection": _required_bool(
                    metric["spoofing_compatible_episode"],
                    field="spoofing_compatible_episode",
                ),
            }
        )
    return details


def _merge_timed_alerts(alerts: list[ExternalAlert]) -> list[ExternalAlert]:
    if any(alert.date_level for alert in alerts):
        raise ValueError("date-level alerts cannot be merged as timed windows")
    ordered = sorted(alerts, key=lambda alert: (alert.start, alert.end))
    merged: list[ExternalAlert] = []
    for alert in ordered:
        if not merged or alert.start > merged[-1].end:
            merged.append(alert)
            continue
        previous = merged[-1]
        merged[-1] = ExternalAlert(
            dataset=previous.dataset,
            external_actor_id=previous.external_actor_id,
            start=previous.start,
            end=max(previous.end, alert.end),
        )
    return merged


def _audit_scope(
    alerts: list[ExternalAlert],
    resolution: ExternalIdentityResolution,
    external_actor_id: str,
    raw: pl.DataFrame,
    members: pl.DataFrame,
    metrics: pl.DataFrame,
    rejected: pl.DataFrame,
    *,
    detector_event_alias_key: bytes | None = None,
) -> dict[str, Any]:
    all_raw = _filter_alerts(raw, "_event_ts", alerts)
    actor_raw = _subject_rows(all_raw, resolution, external_actor_id)
    all_members = _filter_alerts(members, "child_event_ts", alerts)
    actor_members = _subject_rows(all_members, resolution, external_actor_id)
    all_metrics = _metrics_for_members(metrics, all_members)
    actor_metrics = _subject_rows(
        _metrics_for_members(metrics, actor_members),
        resolution,
        external_actor_id,
    )
    all_episodes = _episode_metrics(all_metrics)
    actor_episodes = _episode_metrics(actor_metrics)
    all_matched = all_metrics.filter(pl.col("has_matched_deceptive_cancel_window").fill_null(False))
    matched = actor_metrics.filter(pl.col("has_matched_deceptive_cancel_window").fill_null(False))
    all_matched_episodes = all_episodes.filter(
        pl.col("episode_has_matched_withdrawal").fill_null(False)
    )
    matched_episodes = actor_episodes.filter(
        pl.col("episode_has_matched_withdrawal").fill_null(False)
    )
    all_strict = all_episodes.filter(pl.col("spoofing_compatible_episode").fill_null(False))
    strict = actor_episodes.filter(pl.col("spoofing_compatible_episode").fill_null(False))
    all_rejected = _filter_alerts(rejected, "event_ts", alerts)
    actor_rejected = _subject_rows(all_rejected, resolution, external_actor_id)
    detector_actor_keys = (
        pl.concat(
            [
                actor_members.select("actor_key"),
                actor_rejected.select("actor_key"),
            ],
            how="vertical",
        )
        .get_column("actor_key")
        .drop_nulls()
        .unique()
    )
    actor_roles = _role_counts(actor_raw)
    all_roles = _role_counts(all_raw)
    result = {
        "raw_actor_rows_in_period": actor_raw.height,
        "detector_actor_key_count_in_period": len(detector_actor_keys),
        "identity_granularity_aligned": resolution.identity_granularity_aligned,
        **actor_roles,
        "all_actor_raw_rows_in_period": all_raw.height,
        **{f"all_actor_{key}": value for key, value in all_roles.items()},
        "recovered_child_fill_rows": actor_members.height,
        "recovered_child_fill_rows_by_anchor": _counts_by(actor_members, "execution_anchor_mode"),
        "recovered_execution_clusters": actor_metrics.height,
        "recovered_execution_clusters_by_anchor": _counts_by(
            actor_metrics, "execution_anchor_mode"
        ),
        "candidate_posture_episodes": actor_episodes.height,
        "candidate_posture_episodes_by_anchor": _counts_by(
            actor_episodes, "episode_anchor_mode"
        ),
        "clusters_with_matched_withdrawal": matched.height,
        "episodes_with_matched_withdrawal": matched_episodes.height,
        "episodes_with_strict_detection": strict.height,
        "rejected_execution_rows": actor_rejected.height,
        "rejected_execution_rows_by_anchor": _counts_by(actor_rejected, "execution_anchor_mode"),
        "rejected_execution_rows_by_reason": _counts_by(actor_rejected, "reject_reason"),
        "all_actor_recovered_child_fill_rows": all_members.height,
        "all_actor_recovered_execution_clusters": all_metrics.height,
        "all_actor_candidate_posture_episodes": all_episodes.height,
        "all_actor_clusters_with_matched_withdrawal": all_matched.height,
        "all_actor_episodes_with_matched_withdrawal": all_matched_episodes.height,
        "all_actor_episodes_with_strict_detection": all_strict.height,
        "all_actor_rejected_execution_rows": all_rejected.height,
        "all_actor_rejected_execution_rows_by_reason": _counts_by(
            all_rejected, "reject_reason"
        ),
    }
    if detector_event_alias_key is not None:
        result["detector_events"] = _detector_event_details(
            actor_metrics, detector_event_alias_key
        )
    return result


def _detector_outcome(row: dict[str, Any], *, date_level: bool) -> str:
    if date_level:
        return "date_only_event_recall_not_identifiable"
    if row["raw_actor_rows_in_period"] == 0:
        return "no_raw_subject_event"
    if row["recovered_execution_clusters"] == 0:
        if row["rejected_execution_rows"] > 0:
            return "no_recovered_execution_rejected_rows_present"
        return "no_recovered_execution_cluster"
    if row["clusters_with_matched_withdrawal"] == 0:
        return "no_matched_withdrawal"
    if row["episodes_with_strict_detection"] == 0:
        return "strict_behavioral_gates_not_satisfied"
    return "strict_detection"


def _detector_outcomes(row: dict[str, Any], *, date_level: bool) -> dict[str, str]:
    subject_scope_outcome = _detector_outcome(row, date_level=date_level)
    if date_level or row["identity_granularity_aligned"]:
        exact_actor_outcome = subject_scope_outcome
    else:
        exact_actor_outcome = (
            "identity_granularity_not_aligned_exact_actor_recall_not_identifiable"
        )
    return {
        "subject_scope_detector_outcome": subject_scope_outcome,
        "exact_actor_detector_outcome": exact_actor_outcome,
    }


def _audit_dataset(
    dataset: str,
    alerts: list[ExternalAlert],
    paths: DatasetPaths,
    *,
    alias_key: bytes | None = None,
) -> dict[str, Any]:
    if alias_key is None:
        alias_key = secrets.token_bytes(32)
    raw = _with_canonical_fields(pl.read_parquet(paths.raw))
    members = _with_source_identities(
        pl.read_parquet(paths.metrics / "execution_cluster_members.parquet"),
        client_column="child_event_client_original_id",
        firm_column="child_event_firm_id",
    )
    metrics = _with_source_identities(
        pl.read_parquet(paths.metrics / "execution_metrics.parquet"),
        client_column="event_client_original_id",
        firm_column="event_firm_id",
    )
    rejected = _with_source_identities(
        pl.read_parquet(paths.metrics / "rejected_executions.parquet"),
        client_column="client_original_id",
        firm_column="firm_id",
    )
    required_member = {"actor_key", "execution_cluster_id", "execution_anchor_mode", "child_event_ts"}
    required_metric = {
        "actor_key",
        "cluster_end_ts",
        "cluster_start_ts",
        "execution_cluster_id",
        "execution_anchor_mode",
        "execution_quantity",
        "execution_side",
        "execution_vwap",
        "gate_cancel_anchored_reversion",
        "gate_favorable_pre_fill_move",
        "gate_rapid_matched_withdrawal",
        "gate_small_fill_relative_to_withdrawal",
        "has_matched_deceptive_cancel_window",
        "episode_id",
        "episode_start_ts",
        "episode_end_ts",
        "episode_total_execution_quantity",
        "episode_execution_vwap",
        "episode_mixed_anchor",
        "episode_has_matched_withdrawal",
        "episode_gate_joint_smallness",
        "episode_price_path_observed",
        "episode_favorable_mid_move",
        "episode_post_cancel_mid_reversion",
        "spoofing_compatible_episode",
    }
    required_rejected = {"actor_key", "event_ts", "execution_anchor_mode", "reject_reason"}
    if missing := sorted(required_member - set(members.columns)):
        raise ValueError(f"{dataset} cluster-member artifact is missing: {', '.join(missing)}")
    if missing := sorted(required_metric - set(metrics.columns)):
        raise ValueError(f"{dataset} execution-metric artifact is missing: {', '.join(missing)}")
    if missing := sorted(required_rejected - set(rejected.columns)):
        raise ValueError(f"{dataset} rejected-execution artifact is missing: {', '.join(missing)}")

    actor_lookup = {
        actor_id: _resolve_external_identity(raw, actor_id)
        for actor_id in sorted({alert.external_actor_id for alert in alerts})
    }
    period_rows: list[dict[str, Any]] = []
    actor_totals: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "source_external_periods": 0,
            "source_timed_periods": 0,
            "source_date_only_periods": 0,
            "source_periods_with_raw_subject_event": 0,
            "union_timed_periods": 0,
            "union_periods_with_recovered_execution": 0,
            "union_periods_with_matched_withdrawal": 0,
            "union_periods_with_strict_subject_scope_detection": 0,
            "identity_aligned_union_periods_with_strict_detection": 0,
        }
    )
    for index, alert in enumerate(alerts, start=1):
        alias = _alias(alert.external_actor_id, alias_key)
        resolution = actor_lookup[alert.external_actor_id]
        row = {
            "period_id": f"{dataset}-{index:03d}",
            "actor_alias": alias,
            "identity_namespace": resolution.identity_namespace,
            "scope": "date" if alert.date_level else "intraday_window",
            "event_recall_identifiable": not alert.date_level,
            "start": alert.start.isoformat(),
            "end": None if alert.date_level else alert.end.isoformat(),
            **_audit_scope(
                [alert],
                resolution,
                alert.external_actor_id,
                raw,
                members,
                metrics,
                rejected,
                detector_event_alias_key=None if alert.date_level else alias_key,
            ),
        }
        row.update(_detector_outcomes(row, date_level=alert.date_level))
        if alert.date_level:
            row["detector_events"] = None
            for key in (
                "detector_actor_key_count_in_period",
                "recovered_child_fill_rows",
                "recovered_child_fill_rows_by_anchor",
                "recovered_execution_clusters",
                "recovered_execution_clusters_by_anchor",
                "candidate_posture_episodes",
                "candidate_posture_episodes_by_anchor",
                "clusters_with_matched_withdrawal",
                "episodes_with_matched_withdrawal",
                "episodes_with_strict_detection",
                "rejected_execution_rows",
                "rejected_execution_rows_by_anchor",
                "rejected_execution_rows_by_reason",
                "all_actor_recovered_child_fill_rows",
                "all_actor_recovered_execution_clusters",
                "all_actor_candidate_posture_episodes",
                "all_actor_clusters_with_matched_withdrawal",
                "all_actor_episodes_with_matched_withdrawal",
                "all_actor_episodes_with_strict_detection",
                "all_actor_rejected_execution_rows",
                "all_actor_rejected_execution_rows_by_reason",
            ):
                row[f"date_level_{key}"] = row[key]
                row[key] = None
        period_rows.append(row)
        totals = actor_totals[alias]
        totals["identity_namespace"] = row["identity_namespace"]
        totals["raw_actor_rows_in_dataset"] = resolution.raw_rows
        totals["detector_actor_key_count_in_dataset"] = resolution.detector_actor_key_count
        totals["identity_granularity_aligned"] = resolution.identity_granularity_aligned
        totals["source_external_periods"] += 1
        totals["source_timed_periods"] += int(not alert.date_level)
        totals["source_date_only_periods"] += int(alert.date_level)
        totals["source_periods_with_raw_subject_event"] += int(
            row["raw_actor_rows_in_period"] > 0
        )

    timed_by_actor: dict[str, list[ExternalAlert]] = defaultdict(list)
    for alert in alerts:
        if not alert.date_level:
            timed_by_actor[alert.external_actor_id].append(alert)
    union_rows: list[dict[str, Any]] = []
    for external_actor_id, actor_alerts in sorted(timed_by_actor.items()):
        resolution = actor_lookup[external_actor_id]
        alias = _alias(external_actor_id, alias_key)
        for merged_alert in _merge_timed_alerts(actor_alerts):
            union_row = {
                "union_period_id": f"{dataset}-union-{len(union_rows) + 1:03d}",
                "actor_alias": alias,
                "identity_namespace": resolution.identity_namespace,
                "scope": "unioned_intraday_window",
                "event_recall_identifiable": True,
                "start": merged_alert.start.isoformat(),
                "end": merged_alert.end.isoformat(),
                "duration_seconds": (merged_alert.end - merged_alert.start).total_seconds(),
                "source_windows_merged": sum(
                    alert.start <= merged_alert.end and alert.end >= merged_alert.start
                    for alert in actor_alerts
                ),
                **_audit_scope(
                    [merged_alert],
                    resolution,
                    external_actor_id,
                    raw,
                    members,
                    metrics,
                    rejected,
                ),
            }
            union_row.update(_detector_outcomes(union_row, date_level=False))
            union_rows.append(union_row)
            totals = actor_totals[alias]
            totals["union_timed_periods"] += 1
            totals["union_periods_with_recovered_execution"] += int(
                union_row["recovered_execution_clusters"] > 0
            )
            totals["union_periods_with_matched_withdrawal"] += int(
                union_row["clusters_with_matched_withdrawal"] > 0
            )
            totals["union_periods_with_strict_subject_scope_detection"] += int(
                union_row["episodes_with_strict_detection"] > 0
            )
            totals["identity_aligned_union_periods_with_strict_detection"] += int(
                union_row["identity_granularity_aligned"]
                and union_row["episodes_with_strict_detection"] > 0
            )

    return {
        "source_external_periods": len(period_rows),
        "source_timed_periods": sum(not alert.date_level for alert in alerts),
        "source_date_only_periods": sum(alert.date_level for alert in alerts),
        "union_timed_periods": len(union_rows),
        "distinct_external_actors": len(actor_totals),
        "source_periods_with_raw_subject_event": sum(
            row["raw_actor_rows_in_period"] > 0 for row in period_rows
        ),
        "union_periods_with_raw_subject_event": sum(
            row["raw_actor_rows_in_period"] > 0 for row in union_rows
        ),
        "union_periods_with_recovered_execution": sum(
            row["recovered_execution_clusters"] > 0 for row in union_rows
        ),
        "union_periods_with_matched_withdrawal": sum(
            row["clusters_with_matched_withdrawal"] > 0 for row in union_rows
        ),
        "union_periods_with_strict_subject_scope_detection": sum(
            row["episodes_with_strict_detection"] > 0 for row in union_rows
        ),
        "identity_aligned_union_timed_periods": sum(
            row["identity_granularity_aligned"] for row in union_rows
        ),
        "identity_aligned_union_periods_with_strict_detection": sum(
            row["identity_granularity_aligned"]
            and row["episodes_with_strict_detection"] > 0
            for row in union_rows
        ),
        "identity_unaligned_union_timed_periods": sum(
            not row["identity_granularity_aligned"] for row in union_rows
        ),
        "identity_unaligned_union_periods_with_strict_subject_scope_detection": sum(
            not row["identity_granularity_aligned"]
            and row["episodes_with_strict_detection"] > 0
            for row in union_rows
        ),
        "per_pseudonymized_actor": dict(sorted(actor_totals.items())),
        "source_period_results": period_rows,
        "union_timed_results": union_rows,
        "metrics_provenance": _safe_metrics_provenance(paths.metrics / "metadata.json"),
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    text = _read_source_text(args.external_source)
    alerts = parse_external_alerts(text)
    alias_key = secrets.token_bytes(32)
    dataset_paths = {
        "NEXI": DatasetPaths(args.nexi_raw, args.nexi_metrics),
        "RISANAMENTO": DatasetPaths(args.risanamento_raw, args.risanamento_metrics),
        "FERRARI": DatasetPaths(args.ferrari_raw, args.ferrari_metrics),
    }
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "verification_tier": "external_subject_temporal_coverage_execution_recovery_and_strict_detector_overlap",
        "detector_recall_evaluated": (
            "exact_actor_recall_on_identity_aligned_unioned_timed_windows; "
            "broader_subject_scope_overlap_reported_separately"
        ),
        "recall_unit": (
            "identity-aligned unioned timed external-alert window with at least one strict "
            "detector execution"
        ),
        "timestamp_precedence": ["TRADETIME", "BOOKOUTTIME", "BOOKIN", "SEQUENCETIME"],
        "identifier_policy": (
            "External identifiers use report-local keyed HMAC-SHA-256 pseudonyms; "
            "the key and local source path are not serialized."
        ),
        "external_source": {
            "sha256": _sha256(args.external_source),
            "source_kind": (
                "primary_pdf"
                if args.external_source.suffix.lower() == ".pdf"
                else "cached_extracted_text"
            ),
            "local_path_disclosed": False,
        },
        "datasets": {
            dataset: _audit_dataset(
                dataset,
                alerts[dataset],
                dataset_paths[dataset],
                alias_key=alias_key,
            )
            for dataset in ("NEXI", "RISANAMENTO", "FERRARI")
        },
        "limitations": [
            "External alert labels establish source-provided periods, not manipulative intent independently re-adjudicated here.",
            "RISANAMENTO is coverage-only at date level because the source provides no intraday interval; it is excluded from event-level recall numerators and denominators.",
            "Subject-scope overlap matches raw client/firm provenance. Exact-actor recall is identifiable only when the external identity and detector actor key have aligned granularity.",
            "Strict subject-scope detection requires matched provenance, a recovered execution, matched withdrawal, and all configured behavioral gates.",
            "source_period_results preserves every source window; union_timed_results contains one row per merged interval and is the sole basis of timed overlap and recall totals.",
            "Embedded metrics provenance is restricted to allowlisted hashes, enum-valued modes, and row counts; upstream paths and commands are omitted.",
            *(
                [
                    "The primary PDF was unavailable at audit time; alert definitions were parsed from cached extracted text and are not newly verified against the primary source."
                ]
                if args.external_source.suffix.lower() != ".pdf"
                else []
            ),
        ],
    }
    report["overall"] = {
        key: sum(int(dataset_report[key]) for dataset_report in report["datasets"].values())
        for key in (
            "source_external_periods",
            "source_timed_periods",
            "source_date_only_periods",
            "source_periods_with_raw_subject_event",
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
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps(report["overall"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
