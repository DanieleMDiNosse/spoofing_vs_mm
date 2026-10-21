#!/usr/bin/env python
"""Fail-closed validator for persisted v3 spoofing-detection bundles."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping

import polars as pl


V3_OUTPUT_SCHEMA_VERSION = "actor_execution_anchor_v3_clocked"
V3_CLOCK_POLICY_ID = "spoofing_detection_sequence_replay_trade_fill_sequence_cancel_v3_utc_provisional"
V3_CLOCK_CONTRACT = {
    "replay_clock": "SEQUENCETIME",
    "duration_clock": "SEQUENCETIME",
    "cancellation_removal_clock": "SEQUENCETIME",
    "fill_anchor_clock": "TRADETIME",
    "timestamp_representation": "UTC epoch nanoseconds",
    "semantic_fallback": "forbidden",
    "interval_convention": "left_closed_right_open",
}

# These are the persisted rows whose timestamps carry the detector's v3 clock
# meanings. Every row in each table is relevant to the indicated meaning.
REQUIRED_NS_FIELDS: dict[str, tuple[str, ...]] = {
    "execution_metrics": ("replay_time_ns", "fill_anchor_ns"),
    "execution_cluster_members": ("child_replay_time_ns", "child_fill_anchor_ns"),
    "execution_cancel_candidates": (
        "cluster_end_fill_anchor_ns",
        "cancellation_removal_ns",
    ),
}
_UNHASHED_PATH_KEYS = frozenset({"metadata", "summary_report"})
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_metadata(root: Path) -> dict[str, Any]:
    path = root / "metadata.json"
    if not path.is_file():
        raise ValueError("metadata.json is required")
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("metadata.json is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("metadata.json must contain an object")
    return payload


def _require_exact_provenance(metadata: Mapping[str, Any]) -> None:
    if metadata.get("output_schema_version") != V3_OUTPUT_SCHEMA_VERSION:
        raise ValueError(
            "output_schema_version must be " + repr(V3_OUTPUT_SCHEMA_VERSION)
        )
    if metadata.get("clock_policy_id") != V3_CLOCK_POLICY_ID:
        raise ValueError("clock_policy_id does not match the v3 detector policy")
    if metadata.get("clock_contract") != V3_CLOCK_CONTRACT:
        raise ValueError("clock_contract does not exactly match the v3 detector contract")


def _safe_artifact_path(root: Path, value: object, *, artifact: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"artifact path is invalid: {artifact}")
    declared = Path(value)
    # Current producers historically persisted paths relative to the repository
    # working directory (for example ``outputs/.../bundles/NEXI/file.parquet``),
    # whereas the bundle contract also permits paths relative to ``root``.  Accept
    # either representation, but only after canonicalizing it and proving that it
    # remains beneath this bundle root.
    candidates = [declared.resolve()] if declared.is_absolute() else [
        (root / declared).resolve(),
        declared.resolve(),
    ]
    contained: list[Path] = []
    for target in candidates:
        try:
            target.relative_to(root)
        except ValueError:
            continue
        contained.append(target)
    for target in contained:
        if target.is_file():
            return target
    if contained:
        return contained[0]
    raise ValueError(f"artifact path escapes bundle root: {artifact}")


def _validated_artifacts(root: Path, metadata: Mapping[str, Any]) -> dict[str, Path]:
    paths = metadata.get("paths")
    hashes = metadata.get("artifact_hashes")
    if not isinstance(paths, dict):
        raise ValueError("metadata paths mapping is required")
    if not isinstance(hashes, dict):
        raise ValueError("metadata artifact_hashes mapping is required")

    expected_hash_keys = set(paths) - _UNHASHED_PATH_KEYS
    if set(hashes) != expected_hash_keys:
        raise ValueError("artifact_hashes must cover exactly the declared detector artifacts")
    missing_required = set(REQUIRED_NS_FIELDS) - set(hashes)
    if missing_required:
        raise ValueError("required detector artifacts are missing: " + ", ".join(sorted(missing_required)))

    artifacts: dict[str, Path] = {}
    for artifact, expected_hash in hashes.items():
        if not isinstance(artifact, str) or not artifact:
            raise ValueError("artifact_hashes contains an invalid artifact name")
        if not isinstance(expected_hash, str) or _SHA256_RE.fullmatch(expected_hash) is None:
            raise ValueError(f"artifact hash must be a lowercase SHA-256 hex digest: {artifact}")
        target = _safe_artifact_path(root, paths[artifact], artifact=artifact)
        if not target.is_file():
            raise ValueError(f"artifact file missing: {artifact}")
        if _sha256(target) != expected_hash:
            raise ValueError(f"artifact hash mismatch: {artifact}")
        artifacts[artifact] = target
    return artifacts


def _validate_ns_fields(artifacts: Mapping[str, Path]) -> None:
    for artifact, fields in REQUIRED_NS_FIELDS.items():
        path = artifacts[artifact]
        try:
            schema = dict(pl.scan_parquet(path).collect_schema())
        except Exception as exc:
            raise ValueError(f"cannot read parquet artifact: {artifact}") from exc
        for field in fields:
            if schema.get(field) != pl.Int64:
                raise ValueError(f"{artifact}.{field} must have Int64 dtype")
        try:
            null_counts = (
                pl.scan_parquet(path)
                .select([pl.col(field).null_count().alias(field) for field in fields])
                .collect()
                .row(0, named=True)
            )
        except Exception as exc:
            raise ValueError(f"cannot inspect parquet artifact: {artifact}") from exc
        for field, null_count in null_counts.items():
            if null_count:
                raise ValueError(f"{artifact}.{field} must be non-null on relevant rows")


def validate_bundle(bundle: str | Path) -> dict[str, Any]:
    """Validate a v3 detector bundle or raise ``ValueError`` on any mismatch."""
    root = Path(bundle).resolve()
    if not root.is_dir():
        raise ValueError(f"bundle directory does not exist: {root}")
    metadata = _load_metadata(root)
    _require_exact_provenance(metadata)
    artifacts = _validated_artifacts(root, metadata)
    _validate_ns_fields(artifacts)
    return {"artifact_count": len(artifacts), "bundle": str(root)}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fail closed unless a spoofing-detection bundle satisfies the v3 clock contract."
    )
    parser.add_argument("bundle", type=Path, help="Directory containing metadata.json and detector artifacts")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = validate_bundle(args.bundle)
    except ValueError as exc:
        print(f"validation failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
