#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import polars as pl

UNKNOWN_CLIENT_IDS = {"", "0", "null", "none", "nan"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_run(run_dir: Path) -> dict[str, object]:
    metadata_path = run_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    paths = {key: Path(value) for key, value in metadata["paths"].items() if key not in {"metadata", "summary_report"}}
    artifacts = {
        "execution_metrics": pl.read_parquet(paths["execution_metrics"]),
        "execution_cluster_members": pl.read_parquet(paths["execution_cluster_members"]),
        "execution_cancel_candidates": pl.read_parquet(paths["execution_cancel_candidates"]),
        "client_mcps_scores": pl.read_parquet(paths["client_mcps_scores"]),
    }
    executions = artifacts["execution_metrics"]
    members = artifacts["execution_cluster_members"]
    candidates = artifacts["execution_cancel_candidates"]
    scores = artifacts["client_mcps_scores"]

    assert not executions.get_column("execution_cluster_id").is_duplicated().any()
    child_key = [column for column in ("partition_id", "child_sort_index") if column in members.columns]
    assert child_key and not members.select(child_key).is_duplicated().any()
    assert executions.select(pl.col("child_fill_count").sum()).item() == members.height
    assert abs(
        float(executions.select(pl.col("fill_qty").sum()).item())
        - float(members.select(pl.col("child_fill_qty").sum()).item())
    ) < 1e-6

    assert candidates.filter(
        pl.col("cancel_sort_index") <= pl.col("cluster_last_sort_index")
    ).is_empty()
    assigned = candidates.filter(pl.col("assigned_flag").fill_null(False))
    cancel_key = ["partition_id", "cancel_sort_index", "candidate_order_id"]
    assert not assigned.select(cancel_key).is_duplicated().any()
    matched = executions.filter(pl.col("has_matched_deceptive_cancel_window").fill_null(False))
    assert set(matched.get_column("execution_cluster_id").to_list()) == set(
        assigned.get_column("execution_cluster_id").to_list()
    )

    normalized_score_ids = (
        scores.get_column("client_id").cast(pl.Utf8).fill_null("").str.to_lowercase()
        if not scores.is_empty()
        else pl.Series("client_id", [], dtype=pl.Utf8)
    )
    unknown_score_ids = sorted(
        {value for value in normalized_score_ids.to_list() if value in UNKNOWN_CLIENT_IDS}
    )
    assert not unknown_score_ids, f"unattributable client buckets in MCPS scores: {unknown_score_ids}"

    for key, expected in metadata["artifact_hashes"].items():
        assert sha256(paths[key]) == expected

    observed_post = int(
        executions.select(pl.col("has_post_window_state").fill_null(False).sum()).item()
    )
    finite_msci = int(executions.select(pl.col("MSCI").is_not_null().sum()).item())
    return {
        "run_dir": str(run_dir),
        "execution_clusters": executions.height,
        "child_fill_messages": members.height,
        "matched_clusters": matched.height,
        "assigned_cancellations": assigned.height,
        "candidate_links": candidates.height,
        "client_score_rows": scores.height,
        "observed_post_window_states": observed_post,
        "finite_MSCI": finite_msci,
        "post_window_coverage": observed_post / executions.height if executions.height else None,
        "finite_MSCI_coverage": finite_msci / executions.height if executions.height else None,
        "population_metadata": {
            key: metadata[key]
            for key in (
                "analytical_event_population",
                "mcps_population",
                "review_event_selection",
                "event_metrics_include_unattributable_rows",
                "client_scores_exclude_unattributable_rows",
            )
        },
        "hashes_verified": len(metadata["artifact_hashes"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", type=Path)
    args = parser.parse_args()
    print(json.dumps([audit_run(path) for path in args.run_dirs], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
