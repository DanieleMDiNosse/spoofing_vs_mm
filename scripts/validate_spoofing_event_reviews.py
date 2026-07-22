#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import polars as pl


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def validate_review_artifacts(review_dir: Path, execution_metrics_path: Path) -> list[str]:
    errors: list[str] = []
    metrics = pl.read_parquet(execution_metrics_path)
    if "execution_cluster_id" not in metrics.columns:
        return ["execution metrics do not contain execution_cluster_id"]
    if "has_matched_deceptive_cancel_window" in metrics.columns:
        metrics = metrics.filter(pl.col("has_matched_deceptive_cancel_window").fill_null(False))
    expected_ids = {str(value) for value in metrics.get_column("execution_cluster_id").drop_nulls().to_list()}

    candidates_path = review_dir / "execution_cancel_candidates.parquet"
    if not candidates_path.exists():
        errors.append("missing execution_cancel_candidates.parquet")
    else:
        candidates = pl.read_parquet(candidates_path)
        if candidates.is_empty():
            for missing_id in sorted(expected_ids):
                errors.append(
                    f"cluster {missing_id} is marked matched but has no assigned cancellation"
                )
        else:
            required = {
                "partition_id",
                "candidate_order_id",
                "execution_cluster_id",
                "cluster_last_sort_index",
                "cancel_sort_index",
                "assigned_flag",
            }
            missing = sorted(required - set(candidates.columns))
            if missing:
                errors.append(
                    "execution_cancel_candidates.parquet missing columns: "
                    + ", ".join(missing)
                )
            else:
                noncausal = candidates.filter(
                    pl.col("cancel_sort_index") <= pl.col("cluster_last_sort_index")
                )
                if not noncausal.is_empty():
                    errors.append(
                        "noncausal cancellation assignment/candidate rows: "
                        f"{noncausal.height}"
                    )
                assigned = candidates.filter(pl.col("assigned_flag").fill_null(False))
                duplicate_assignments = assigned.group_by(
                    ["partition_id", "cancel_sort_index", "candidate_order_id"]
                ).len().filter(pl.col("len") > 1)
                if not duplicate_assignments.is_empty():
                    errors.append(
                        "duplicate assigned cancellation keys: "
                        f"{duplicate_assignments.height}"
                    )
                assigned_ids = {
                    str(value)
                    for value in assigned.get_column("execution_cluster_id").drop_nulls().to_list()
                }
                for missing_id in sorted(expected_ids - assigned_ids):
                    errors.append(
                        f"cluster {missing_id} is marked matched but has no assigned cancellation"
                    )

    reviews_root = review_dir / "llm_reviews"
    found_ids: set[str] = set()
    if reviews_root.exists():
        for item in sorted(reviews_root.iterdir()):
            if not item.is_dir():
                continue
            review_id = item.name
            if review_id.startswith("S"):
                errors.append(f"stale message-level review directory: {review_id}")
            manifest_path = item / "manifest.json"
            if not manifest_path.exists():
                errors.append(f"missing manifest for {review_id}")
                continue
            try:
                manifest = _read_json(manifest_path)
            except ValueError as exc:
                errors.append(str(exc))
                continue
            status = str(manifest.get("status") or "missing")
            if status != "complete":
                errors.append(f"review {review_id} manifest status is {status}")
                continue
            found_ids.add(review_id)
            for filename in ("dossier.md", "prompt.md", "response.md", "metadata.json"):
                if not (item / filename).exists():
                    errors.append(f"review {review_id} missing {filename}")
            for filename, key in (("dossier.md", "dossier_sha256"), ("prompt.md", "prompt_sha256")):
                path = item / filename
                if path.exists() and manifest.get(key) and sha256_text(path.read_text()) != manifest[key]:
                    errors.append(f"review {review_id} {key} mismatch")
            response_path = item / "response.md"
            if response_path.exists():
                response = response_path.read_text()
                if not response.startswith(f"# Surveillance review for event {review_id}"):
                    errors.append(f"review {review_id} response heading mismatch")
                if "## Intent limitation" not in response:
                    errors.append(f"review {review_id} missing intent limitation")

    for missing_id in sorted(expected_ids - found_ids):
        errors.append(f"missing complete review for cluster {missing_id}")

    metadata_path = review_dir / "metadata.json"
    if not metadata_path.exists():
        errors.append("missing review metadata.json")
    else:
        try:
            metadata = _read_json(metadata_path)
            if not metadata.get("dashboard_refreshed_at_utc"):
                errors.append("dashboard_refreshed_at_utc is missing or null")
        except ValueError as exc:
            errors.append(str(exc))
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate cluster-first spoofing review artifacts.")
    parser.add_argument("--review-dir", type=Path, required=True)
    parser.add_argument("--execution-metrics", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    errors = validate_review_artifacts(args.review_dir, args.execution_metrics)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    print("review artifacts valid")


if __name__ == "__main__":
    main()
