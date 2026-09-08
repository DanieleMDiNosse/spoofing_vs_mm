#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from spoofing_detection.lob.actor_identity import (  # noqa: E402
    is_zero_client_original_identity_sentinel,
)

UNKNOWN_ACTOR_KEYS = {
    "",
    "0",
    "null",
    "none",
    "nan",
    "client_original:0",
    "client_original:0.0",
}
VALID_IDENTITY_LEVELS = {"client_original", "firm"}
VALID_EXECUTION_ANCHORS = {"passive", "aggressive"}
SUPPORTED_SCHEMA_VERSIONS = {
    "actor_execution_anchor_v1",
    "actor_execution_anchor_v2",
    "actor_execution_anchor_metric_separation_v2",
}
CLUSTER_PROVENANCE_COLUMNS = (
    "actor_key",
    "actor_id",
    "identity_level",
    "identity_source",
    "identity_fallback_flag",
    "execution_anchor_mode",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _counts(frame: pl.DataFrame, column: str) -> dict[str, int]:
    if frame.is_empty() or column not in frame.columns:
        return {}
    rows = frame.group_by(column).len(name="count").sort(column).to_dicts()
    return {str(row[column]): int(row["count"]) for row in rows}


def _validate_actor_anchor_frame(frame: pl.DataFrame, *, artifact: str) -> None:
    required = {
        "actor_key",
        "actor_id",
        "identity_level",
        "identity_source",
        "identity_fallback_flag",
        "execution_anchor_mode",
    }
    missing = sorted(required - set(frame.columns))
    assert not missing, f"{artifact} missing actor/anchor columns: {missing}"
    if frame.is_empty():
        return
    assert not frame.select(
        pl.any_horizontal(pl.col(column).is_null() for column in sorted(required)).any()
    ).item(), f"null actor/anchor fields in {artifact}"
    actor_keys = frame.get_column("actor_key").cast(pl.String).fill_null("").str.to_lowercase()
    invalid_actor_keys = sorted(
        {
            value
            for value in actor_keys.to_list()
            if value in UNKNOWN_ACTOR_KEYS
            or not any(value.startswith(f"{prefix}:") for prefix in VALID_IDENTITY_LEVELS)
            or (
                value.startswith("client_original:")
                and is_zero_client_original_identity_sentinel(value.partition(":")[2])
            )
        }
    )
    assert not invalid_actor_keys, f"invalid actor buckets in {artifact}: {invalid_actor_keys}"
    identity_levels = set(frame.get_column("identity_level").drop_nulls().cast(pl.String).to_list())
    assert identity_levels <= VALID_IDENTITY_LEVELS, f"invalid identity levels in {artifact}: {identity_levels}"
    anchors = set(frame.get_column("execution_anchor_mode").drop_nulls().cast(pl.String).to_list())
    assert anchors <= VALID_EXECUTION_ANCHORS, f"invalid execution anchors in {artifact}: {anchors}"
    expected_source = (
        pl.when(pl.col("identity_level") == "firm")
        .then(pl.lit("FIRMID"))
        .otherwise(pl.lit("NMSC_ORIGINALCLIENTIDSHORTCODE"))
    )
    mismatched = frame.filter(
        (
            (pl.col("identity_level") == "firm")
            != pl.col("identity_fallback_flag")
        )
        | (
            pl.col("actor_key").cast(pl.String)
            != pl.col("identity_level") + ":" + pl.col("actor_id").cast(pl.String)
        )
        | (pl.col("identity_source") != expected_source)
    )
    assert mismatched.is_empty(), f"actor identity fields disagree in {artifact}"


def _validate_cluster_provenance(
    executions: pl.DataFrame,
    linked_rows: pl.DataFrame,
    *,
    artifact: str,
) -> None:
    """Require linked child/cancel rows to retain the cluster's actor and anchor."""
    if linked_rows.is_empty():
        return
    reference = executions.select(
        "execution_cluster_id",
        *CLUSTER_PROVENANCE_COLUMNS,
    ).rename(
        {column: f"execution_{column}" for column in CLUSTER_PROVENANCE_COLUMNS}
    )
    joined = linked_rows.join(
        reference,
        on="execution_cluster_id",
        how="left",
        validate="m:1",
    )
    mismatch = pl.col("execution_actor_key").is_null()
    for column in CLUSTER_PROVENANCE_COLUMNS:
        mismatch |= pl.col(column) != pl.col(f"execution_{column}")
    assert joined.filter(mismatch).is_empty(), (
        f"cluster provenance disagrees in {artifact}"
    )


def _validate_score_provenance(executions: pl.DataFrame, scores: pl.DataFrame) -> None:
    """Require every scored actor/anchor stratum to exist in event metrics."""

    keys = ["actor_key", "execution_anchor_mode"]
    execution_population = executions.select(keys).unique()
    score_population = scores.select(keys).unique()
    invalid = score_population.join(execution_population, on=keys, how="anti")
    assert not invalid.height, (
        "score provenance contains actor/anchor strata absent from execution metrics"
    )


def audit_run(run_dir: Path) -> dict[str, object]:
    metadata_path = run_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    paths = {key: Path(value) for key, value in metadata["paths"].items() if key not in {"metadata", "summary_report"}}
    score_path_key = "actor_mcps_scores" if "actor_mcps_scores" in paths else "client_mcps_scores"
    artifacts = {
        "execution_metrics": pl.read_parquet(paths["execution_metrics"]),
        "execution_cluster_members": pl.read_parquet(paths["execution_cluster_members"]),
        "execution_cancel_candidates": pl.read_parquet(paths["execution_cancel_candidates"]),
        "actor_mcps_scores": pl.read_parquet(paths[score_path_key]),
    }
    executions = artifacts["execution_metrics"]
    members = artifacts["execution_cluster_members"]
    candidates = artifacts["execution_cancel_candidates"]
    scores = artifacts["actor_mcps_scores"]
    schema_version = metadata.get("output_schema_version")
    assert schema_version in SUPPORTED_SCHEMA_VERSIONS, (
        f"unsupported output_schema_version: {schema_version!r}"
    )
    if schema_version in {
        "actor_execution_anchor_v2",
        "actor_execution_anchor_metric_separation_v2",
    }:
        execution_quantity_column = "execution_quantity"
        msci_column = "MSCI_resting_profile"
        mcps_column = "MCPS_resting_profile"
    else:
        execution_quantity_column = "fill_qty"
        msci_column = "MSCI"
        mcps_column = "MCPS"
    for frame, column, artifact in (
        (executions, execution_quantity_column, "execution_metrics"),
        (executions, msci_column, "execution_metrics"),
        (scores, mcps_column, "actor_mcps_scores"),
    ):
        assert column in frame.columns, f"{artifact} missing schema field: {column}"

    _validate_actor_anchor_frame(executions, artifact="execution_metrics")
    _validate_actor_anchor_frame(members, artifact="execution_cluster_members")
    _validate_actor_anchor_frame(candidates, artifact="execution_cancel_candidates")
    _validate_actor_anchor_frame(scores, artifact="actor_mcps_scores")
    _validate_score_provenance(executions, scores)
    _validate_cluster_provenance(
        executions,
        members,
        artifact="execution_cluster_members",
    )
    _validate_cluster_provenance(
        executions,
        candidates,
        artifact="execution_cancel_candidates",
    )

    if executions.is_empty():
        assert members.is_empty(), "empty executions must have no child fill members"
        assert candidates.is_empty(), "empty executions must have no cancellation candidates"
        assert scores.is_empty(), "empty executions must have no actor scores"
        assigned = candidates
        matched = executions
    else:
        assert not executions.get_column("execution_cluster_id").is_duplicated().any()
        child_key = [
            column
            for column in ("partition_id", "child_sort_index")
            if column in members.columns
        ]
        assert child_key and not members.select(child_key).is_duplicated().any()
        assert executions.select(pl.col("child_fill_count").sum()).item() == members.height
        assert abs(
            float(executions.select(pl.col(execution_quantity_column).sum()).item())
            - float(members.select(pl.col("child_fill_qty").sum()).item())
        ) < 1e-6

        assert candidates.filter(
            pl.col("cancel_sort_index") <= pl.col("cluster_last_sort_index")
        ).is_empty()
        assigned = candidates.filter(pl.col("assigned_flag").fill_null(False))
        cancel_key = ["partition_id", "cancel_sort_index", "candidate_order_id"]
        assert not assigned.select(cancel_key).is_duplicated().any()
        matched = executions.filter(
            pl.col("has_matched_deceptive_cancel_window").fill_null(False)
        )
        assert set(matched.get_column("execution_cluster_id").to_list()) == set(
            assigned.get_column("execution_cluster_id").to_list()
        )


    for key, expected in metadata["artifact_hashes"].items():
        assert sha256(paths[key]) == expected

    observed_post = (
        int(
            executions.select(
                pl.col("has_post_window_state").fill_null(False).sum()
            ).item()
        )
        if not executions.is_empty()
        else 0
    )
    finite_msci = int(executions.select(pl.col(msci_column).is_not_null().sum()).item())
    configured_anchors = metadata.get("execution_anchor_modes")
    assert isinstance(configured_anchors, list) and configured_anchors, (
        "metadata execution_anchor_modes must be a non-empty list"
    )
    assert len(configured_anchors) == len(set(configured_anchors)), (
        "metadata execution_anchor_modes must not contain duplicates"
    )
    assert set(configured_anchors) <= VALID_EXECUTION_ANCHORS, (
        "metadata contains invalid configured execution anchors"
    )
    configured_anchors = [mode for mode in ("passive", "aggressive") if mode in configured_anchors]
    observed_anchors = sorted(
        set(executions.get_column("execution_anchor_mode").drop_nulls().cast(pl.String).to_list()),
        key=("passive", "aggressive").index,
    )
    declared_observed_anchors = metadata.get("observed_execution_anchor_modes")
    assert isinstance(declared_observed_anchors, list), (
        "metadata observed_execution_anchor_modes must be a list"
    )
    assert declared_observed_anchors == observed_anchors, (
        "metadata observed execution anchors do not match execution_metrics"
    )
    assert set(observed_anchors) <= set(configured_anchors), (
        f"observed execution anchors are not configured: {set(observed_anchors) - set(configured_anchors)}"
    )
    return {
        "run_dir": str(run_dir),
        "execution_clusters": executions.height,
        "child_fill_messages": members.height,
        "matched_clusters": matched.height,
        "assigned_cancellations": assigned.height,
        "candidate_links": candidates.height,
        "actor_score_rows": scores.height,
        "execution_clusters_by_anchor": _counts(executions, "execution_anchor_mode"),
        "execution_clusters_by_identity_level": _counts(executions, "identity_level"),
        "child_fills_by_execution_anchor_mode": _counts(
            members, "execution_anchor_mode"
        ),
        "child_fills_by_identity_level": _counts(members, "identity_level"),
        "candidate_links_by_execution_anchor_mode": _counts(
            candidates, "execution_anchor_mode"
        ),
        "candidate_links_by_identity_level": _counts(candidates, "identity_level"),
        "score_rows_by_anchor": _counts(scores, "execution_anchor_mode"),
        "score_rows_by_identity_level": _counts(scores, "identity_level"),
        "configured_execution_anchor_modes": configured_anchors,
        "observed_execution_anchor_modes": observed_anchors,
        "observed_post_window_states": observed_post,
        "finite_MSCI_resting_profile": finite_msci,
        "post_window_coverage": observed_post / executions.height if executions.height else None,
        "finite_MSCI_resting_profile_coverage": finite_msci / executions.height
        if executions.height
        else None,
        "population_metadata": {
            key: metadata[key]
            for key in (
                "analytical_event_population",
                "mcps_population",
                "review_event_selection",
                "event_metrics_include_unattributable_rows",
                "actor_scores_exclude_unattributable_rows",
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
