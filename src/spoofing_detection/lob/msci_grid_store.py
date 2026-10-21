"""Immutable local completion store for one MSCI grid partition/configuration."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

_EXECUTION_KEYS = ("instrument", "partition_id", "actor_key", "execution_cluster_id")
TRAJECTORY_SCHEMA: dict[str, pl.DataType] = {
    "instrument": pl.String, "partition_id": pl.String, "actor_key": pl.String,
    "execution_cluster_id": pl.String, "grid_offset_ns": pl.Int64,
    "target_replay_ns": pl.Int64, "selected_sort_index": pl.Int64,
    "selected_replay_ns": pl.Int64, "baseline_sort_index": pl.Int64,
    "DWI": pl.Float64, "L_bid_topN": pl.Float64, "L_ask_topN": pl.Float64,
    "SCI": pl.Float64, "C_opposite": pl.Float64, "C_same": pl.Float64,
    "MSCI": pl.Float64, "valid": pl.Boolean, "reason": pl.String,
}
SUMMARY_SCHEMA: dict[str, pl.DataType] = {
    "instrument": pl.String, "partition_id": pl.String, "actor_key": pl.String,
    "execution_cluster_id": pl.String, "J": pl.Float64,
    "J_observed_partial": pl.Float64, "argmax_end_offset_ns": pl.Int64,
    "argmax_start_offset_ns": pl.Int64, "complete_grid": pl.Boolean,
    "valid_pair_count": pl.Int64,
}


@dataclass(frozen=True)
class PartitionCompletion:
    did_compute: bool
    complete_dir: Path
    trajectory: pl.DataFrame
    summary: pl.DataFrame
    manifest: dict[str, Any]


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_hash(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _validate_hash_tree(value: Any, *, label: str) -> None:
    if _is_hash(value):
        return
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{label} must be a non-empty mapping of SHA-256 hashes")
    for key, child in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{label} hash keys must be non-empty strings")
        _validate_hash_tree(child, label=label)


def _validate_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(identity, Mapping):
        raise ValueError("identity must be a mapping")
    required = {"instrument", "partition_id", "configuration_key", "input_hashes", "config_hashes", "code_hashes"}
    missing, unknown = sorted(required - set(identity)), sorted(set(identity) - required)
    if missing or unknown:
        raise ValueError(f"identity keys mismatch: missing={missing}, unknown={unknown}")
    normalized = dict(identity)
    for key in ("instrument", "partition_id", "configuration_key"):
        value = normalized[key]
        if not isinstance(value, str) or not value:
            raise ValueError(f"identity {key} must be a non-empty string")
        if value in {".", ".."} or "/" in value or "\\" in value:
            raise ValueError(f"identity {key} must be a single path component")
    _validate_hash_tree(normalized["input_hashes"], label="input_hashes")
    _validate_hash_tree(normalized["code_hashes"], label="code_hashes")
    configs = normalized["config_hashes"]
    if not isinstance(configs, Mapping) or not configs:
        raise ValueError("config_hashes must be a non-empty mapping of configuration SHA-256 hashes")
    if normalized["configuration_key"] not in configs:
        raise ValueError("configuration_key must be present in config_hashes")
    for key, config_hash in configs.items():
        if not isinstance(key, str) or not key:
            raise ValueError("config_hashes keys must be non-empty strings")
        if not _is_hash(config_hash):
            raise ValueError(f"config_hashes.{key} must be a SHA-256 hash")
    try:
        json.loads(_canonical_json(normalized))
    except (TypeError, ValueError) as exc:
        raise ValueError("identity must be JSON-serializable") from exc
    return normalized


def _require_schema(frame: Any, schema: Mapping[str, pl.DataType], *, label: str) -> pl.DataFrame:
    if not isinstance(frame, pl.DataFrame):
        raise ValueError(f"{label} must be a polars DataFrame")
    if frame.schema != schema:
        raise ValueError(f"{label} schema mismatch")
    return frame


def _members(frame: pl.DataFrame, *, label: str) -> set[tuple[str, str, str, str]]:
    keys = frame.select(list(_EXECUTION_KEYS))
    _require_schema(keys, {key: pl.String for key in _EXECUTION_KEYS}, label=label)
    if keys.null_count().sum_horizontal().item() != 0:
        raise ValueError(f"{label} execution keys must be non-null")
    if keys.is_duplicated().any():
        raise ValueError(f"{label} has duplicate execution keys")
    return set(tuple(row) for row in keys.iter_rows())


def _validate_requested(identity: Mapping[str, Any], requested: Any) -> set[tuple[str, str, str, str]]:
    requested = _require_schema(requested, {key: pl.String for key in _EXECUTION_KEYS}, label="requested_executions")
    members = _members(requested, label="requested_executions")
    if any(instrument != identity["instrument"] or partition != identity["partition_id"] for instrument, partition, _, _ in members):
        raise ValueError("requested_executions do not match identity")
    return members


def _validate_outputs(identity: Mapping[str, Any], requested: set[tuple[str, str, str, str]], trajectory: Any, summary: Any) -> tuple[pl.DataFrame, pl.DataFrame]:
    trajectory = _require_schema(trajectory, TRAJECTORY_SCHEMA, label="trajectory")
    summary = _require_schema(summary, SUMMARY_SCHEMA, label="summary")
    node_keys = trajectory.select([*_EXECUTION_KEYS, "grid_offset_ns"])
    if node_keys.is_duplicated().any():
        raise ValueError("trajectory has duplicate execution/grid node")
    trajectory_members = set(tuple(row) for row in trajectory.select(list(_EXECUTION_KEYS)).unique().iter_rows())
    if trajectory_members != requested:
        raise ValueError("trajectory membership does not equal requested executions")
    summary_members = _members(summary, label="summary")
    if summary_members != requested:
        raise ValueError("summary membership does not equal requested executions")
    for members, label in ((trajectory_members, "trajectory"), (summary_members, "summary")):
        if any(instrument != identity["instrument"] or partition != identity["partition_id"] for instrument, partition, _, _ in members):
            raise ValueError(f"{label} does not match identity")
    return trajectory, summary


def _schema_manifest(schema: Mapping[str, pl.DataType]) -> dict[str, str]:
    return {name: str(dtype) for name, dtype in schema.items()}


def _manifest(identity: Mapping[str, Any], requested: set[tuple[str, str, str, str]], trajectory_path: Path, summary_path: Path, trajectory: pl.DataFrame, summary: pl.DataFrame) -> dict[str, Any]:
    return {
        "format": "msci_grid_partition_store_v1", "identity": identity,
        "counts": {"requested_executions": len(requested), "trajectory_rows": trajectory.height, "summary_rows": summary.height},
        "outputs": {
            "trajectory": {"file": "trajectory.parquet", "sha256": _sha256(trajectory_path), "schema": _schema_manifest(trajectory.schema)},
            "summary": {"file": "summary.parquet", "sha256": _sha256(summary_path), "schema": _schema_manifest(summary.schema)},
        },
    }


def _load_verified(complete_dir: Path, identity: Mapping[str, Any], requested: set[tuple[str, str, str, str]]) -> PartitionCompletion:
    manifest_path = complete_dir / "manifest.json"
    if not (complete_dir / "COMPLETE").is_file() or not manifest_path.is_file():
        raise ValueError("existing final partition is incomplete or corrupt")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("existing manifest is corrupt") from exc
    if manifest.get("identity") != identity:
        raise ValueError("existing final partition identity mismatch")
    if manifest.get("format") != "msci_grid_partition_store_v1":
        raise ValueError("existing manifest format mismatch")
    try:
        trajectory_path = complete_dir / manifest["outputs"]["trajectory"]["file"]
        summary_path = complete_dir / manifest["outputs"]["summary"]["file"]
        if _sha256(trajectory_path) != manifest["outputs"]["trajectory"]["sha256"]:
            raise ValueError("trajectory output hash mismatch")
        if _sha256(summary_path) != manifest["outputs"]["summary"]["sha256"]:
            raise ValueError("summary output hash mismatch")
        trajectory, summary = pl.read_parquet(trajectory_path), pl.read_parquet(summary_path)
    except (KeyError, OSError, pl.exceptions.PolarsError) as exc:
        raise ValueError("existing final partition outputs are corrupt") from exc
    trajectory, summary = _validate_outputs(identity, requested, trajectory, summary)
    if manifest != _manifest(identity, requested, trajectory_path, summary_path, trajectory, summary):
        raise ValueError("existing final partition manifest validation mismatch")
    return PartitionCompletion(False, complete_dir, trajectory, summary, manifest)


def read_or_compute_partition(root: str | Path, identity: Mapping[str, Any], requested_executions: pl.DataFrame, compute: Callable[[], tuple[pl.DataFrame, pl.DataFrame]]) -> PartitionCompletion:
    """Reuse only a fully verified exact unit; otherwise validate then publish once."""
    identity = _validate_identity(identity)
    requested = _validate_requested(identity, requested_executions)
    root_path = Path(root)
    unit_dir = root_path / identity["instrument"] / identity["partition_id"] / identity["configuration_key"]
    complete_dir = unit_dir / "complete"
    unit_dir.mkdir(parents=True, exist_ok=True)
    if complete_dir.exists():
        if (complete_dir / "COMPLETE").is_file():
            return _load_verified(complete_dir, identity, requested)
        interrupted_dir = unit_dir / f".interrupted-{uuid.uuid4().hex}"
        os.replace(complete_dir, interrupted_dir)
    attempt = unit_dir / f".inprogress-{uuid.uuid4().hex}"
    attempt.mkdir()
    trajectory, summary = _validate_outputs(identity, requested, *compute())
    trajectory_path, summary_path = attempt / "trajectory.parquet", attempt / "summary.parquet"
    trajectory.write_parquet(trajectory_path)
    summary.write_parquet(summary_path)
    manifest = _manifest(identity, requested, trajectory_path, summary_path, trajectory, summary)
    (attempt / "manifest.json").write_text(_canonical_json(manifest) + "\n", encoding="utf-8")
    (attempt / "COMPLETE").write_text("complete\n", encoding="utf-8")
    os.replace(attempt, complete_dir)
    return PartitionCompletion(True, complete_dir, trajectory, summary, manifest)
