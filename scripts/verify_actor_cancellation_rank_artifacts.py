#!/usr/bin/env python
"""Validate the broad actor cancellation-rate artifacts and audit raw source rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from spoofing_detection.lob.config import LOBConfig  # noqa: E402
from spoofing_detection.lob.normalize import normalize_event  # noqa: E402
from spoofing_detection.lob.panel import _partition_id, sort_events  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _read_artifacts(metadata_path: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    metadata = json.loads(metadata_path.read_text())
    generator_path = REPO_ROOT / metadata["generator"]
    _assert(generator_path.exists(), f"missing recorded generator: {generator_path}")
    _assert(
        _sha256(generator_path) == metadata["generator_sha256"],
        "metadata generator hash does not match the current source",
    )
    paths = {}
    for key, value in metadata["paths"].items():
        if key == "metadata":
            continue
        paths[key] = _resolve_repo_path(value)
    for key, path in paths.items():
        _assert(path.exists(), f"missing {key}: {path}")
        expected_hash = metadata["artifact_hashes"][key]
        _assert(_sha256(path) == expected_hash, f"hash mismatch for {key}: {path}")
    return metadata, paths


def _validate_public_figure_metadata(metadata: dict[str, Any], paths: dict[str, Path]) -> None:
    public = json.loads(paths["figure_metadata"].read_text())
    expected_fields = {
        "schema_version",
        "generator",
        "generator_sha256",
        "actor_rate_artifact_sha256",
        "figure_pdf_sha256",
        "figure_png_sha256",
        "window_seconds",
        "instrument_order",
        "instrument_counts",
        "privacy",
        "support_encoding",
        "identity_encoding",
        "plot_rank_rule",
        "x_axis_scale",
    }
    _assert(set(public) == expected_fields, "public figure metadata fields differ from the privacy allowlist")
    _assert(public["schema_version"] == "actor_cancellation_rank_figure_v1", "public schema mismatch")
    _assert(public["generator"] == metadata["generator"], "public generator path mismatch")
    _assert(public["generator_sha256"] == metadata["generator_sha256"], "public generator hash mismatch")
    _assert(
        public["actor_rate_artifact_sha256"] == metadata["artifact_hashes"]["actor_rates"],
        "public actor-rate hash mismatch",
    )
    _assert(public["figure_pdf_sha256"] == metadata["artifact_hashes"]["figure_pdf"], "public PDF hash mismatch")
    _assert(public["figure_png_sha256"] == metadata["artifact_hashes"]["figure_png"], "public PNG hash mismatch")
    _assert(public["window_seconds"] == metadata["window_seconds"], "public window mismatch")
    _assert(public["instrument_order"] == metadata["instrument_order"], "public instrument order mismatch")
    expected_counts = {
        instrument: {
            "actor_count": provenance["actor_count"],
            "execution_cluster_count": provenance["execution_cluster_count"],
            "qualifying_execution_cluster_count": provenance["qualifying_execution_cluster_count"],
        }
        for instrument, provenance in metadata["provenance"].items()
    }
    _assert(public["instrument_counts"] == expected_counts, "public aggregate counts mismatch")
    _assert(public["plot_rank_rule"] == metadata["plot_rank_rule"], "public plot-rank rule mismatch")
    _assert(public["x_axis_scale"] == "logarithmic actor rank", "public x-axis scale mismatch")
    serialized = json.dumps(public, sort_keys=True)
    for forbidden in ("actor_key", "actor_id", "order_id", "/home/", "run_dir", "raw_events_path"):
        _assert(forbidden not in serialized, f"public metadata contains forbidden token: {forbidden}")


def _validate_upstream_hashes(metadata: dict[str, Any]) -> None:
    for instrument, provenance in metadata["provenance"].items():
        checks = {
            "raw events": (
                _resolve_repo_path(provenance["raw_events_path"]),
                provenance["raw_events_sha256"],
            ),
            "execution metrics": (
                _resolve_repo_path(provenance["execution_metrics_path"]),
                provenance["execution_metrics_sha256"],
            ),
            "run metadata": (
                _resolve_repo_path(provenance["run_dir"]) / "metadata.json",
                provenance["run_metadata_sha256"],
            ),
        }
        for label, (path, expected_hash) in checks.items():
            _assert(path.exists(), f"missing upstream {label} for {instrument}: {path}")
            _assert(
                _sha256(path) == expected_hash,
                f"upstream {label} hash mismatch for {instrument}: {path}",
            )


def _sample_links(links: pl.DataFrame, *, per_anchor: int) -> pl.DataFrame:
    samples: list[pl.DataFrame] = []
    instruments = links.get_column("instrument").unique(maintain_order=True).to_list()
    for instrument in instruments:
        for anchor in ("passive", "aggressive"):
            sample = (
                links.filter(
                    (pl.col("instrument") == instrument)
                    & (pl.col("execution_anchor_mode") == anchor)
                )
                .sort(
                    ["execution_cluster_id", "cancel_event_ts", "cancel_sort_index"],
                )
                .unique("execution_cluster_id", keep="first", maintain_order=True)
                .head(per_anchor)
            )
            _assert(sample.height > 0, f"no qualifying {anchor} sample for {instrument}")
            samples.append(sample)
    return pl.concat(samples, how="vertical_relaxed")


def _audit_raw_source_rows(
    samples: pl.DataFrame,
    metadata: dict[str, Any],
    direct_cancellations: pl.DataFrame,
) -> list[dict[str, Any]]:
    audit_rows: list[dict[str, Any]] = []
    config = LOBConfig(top_n=1, snapshot_mode="none")
    for instrument in metadata["instrument_order"]:
        instrument_samples = samples.filter(pl.col("instrument") == instrument)
        raw_path = _resolve_repo_path(metadata["provenance"][instrument]["raw_events_path"])
        required_indices = set(instrument_samples.get_column("cancel_sort_index").to_list())
        raw_rows = (
            sort_events(pl.read_parquet(raw_path))
            .with_row_index("_source_sort_index", offset=1)
            .filter(pl.col("_source_sort_index").is_in(sorted(required_indices)))
        )
        _assert(
            raw_rows.height == len(required_indices),
            f"could not recover every sampled raw cancellation row for {instrument}",
        )
        normalized_by_index: dict[int, dict[str, Any]] = {}
        for raw_row in raw_rows.iter_rows(named=True):
            sort_index = int(raw_row.pop("_source_sort_index"))
            normalized_by_index[sort_index] = normalize_event(
                raw_row,
                sort_index=sort_index,
                config=config,
            )

        for link in instrument_samples.iter_rows(named=True):
            sort_index = int(link["cancel_sort_index"])
            normalized = normalized_by_index[sort_index]
            _assert(normalized["event_class"] == "cancel", "sampled raw event is not a cancellation")
            _assert(str(normalized["ORDERID"]) == str(link["cancel_order_id"]), "raw order ID mismatch")
            _assert(_partition_id(normalized) == link["partition_id"], "raw trading partition mismatch")

            replay_match = direct_cancellations.filter(
                (pl.col("instrument") == instrument)
                & (pl.col("partition_id") == link["partition_id"])
                & (pl.col("sort_index") == sort_index)
                & (pl.col("ORDERID") == link["cancel_order_id"])
            )
            _assert(replay_match.height == 1, "sampled cancellation lacks one replay record")
            replay = replay_match.row(0, named=True)
            _assert(replay["actor_key"] == link["actor_key"], "active-order actor mismatch")
            _assert(replay["side"] == link["opposite_side"], "active-order side mismatch")
            _assert(replay["event_ts"] == link["cancel_event_ts"], "cancellation timestamp mismatch")
            audit_rows.append(
                {
                    "instrument": instrument,
                    "execution_cluster_id": link["execution_cluster_id"],
                    "actor_key": link["actor_key"],
                    "execution_anchor_mode": link["execution_anchor_mode"],
                    "execution_side": link["execution_side"],
                    "opposite_side": link["opposite_side"],
                    "cluster_end_ts": link["cluster_end_ts"].isoformat(),
                    "cluster_last_sort_index": link["cluster_last_sort_index"],
                    "cancel_order_id": link["cancel_order_id"],
                    "cancel_event_ts": link["cancel_event_ts"].isoformat(),
                    "cancel_sort_index": sort_index,
                    "cancel_delay_seconds": link["cancel_delay_seconds"],
                }
            )
    return audit_rows


def validate(metadata_path: Path, *, per_anchor: int) -> dict[str, Any]:
    metadata, paths = _read_artifacts(metadata_path)
    _validate_public_figure_metadata(metadata, paths)
    _validate_upstream_hashes(metadata)
    actors = pl.read_csv(paths["actor_rates"])
    clusters = pl.read_parquet(paths["cluster_audit"])
    links = pl.read_parquet(paths["links"])
    direct_cancellations = pl.read_parquet(paths["direct_cancellations"])

    _assert(
        clusters.select(["instrument", "execution_cluster_id"]).unique().height == clusters.height,
        "execution cluster IDs are not unique within instrument",
    )
    _assert(
        actors.get_column("execution_cluster_count").sum() == clusters.height,
        "actor denominators do not equal the cluster population",
    )
    _assert(
        actors.get_column("qualifying_execution_cluster_count").sum()
        == clusters.get_column("has_opposite_side_cancellation_within_2s").sum(),
        "actor numerators do not equal the binary cluster outcomes",
    )
    _assert(
        actors.select(
            (
                pl.col("passive_execution_cluster_count")
                + pl.col("aggressive_execution_cluster_count")
                == pl.col("execution_cluster_count")
            ).all()
        ).item(),
        "passive and aggressive denominators do not conserve actor clusters",
    )
    _assert(
        actors.select(
            (
                pl.col("cancellation_rate_2s")
                == pl.col("qualifying_execution_cluster_count")
                / pl.col("execution_cluster_count")
            ).all()
        ).item(),
        "stored actor rates do not equal qualifying clusters divided by all clusters",
    )
    for instrument in metadata["instrument_order"]:
        observed = actors.filter(pl.col("instrument") == instrument).sort("rank_within_instrument")
        expected = actors.filter(pl.col("instrument") == instrument).sort(
            ["cancellation_rate_2s", "execution_cluster_count", "actor_key"],
            descending=[True, True, False],
        )
        _assert(
            observed.get_column("rank_within_instrument").to_list()
            == list(range(1, observed.height + 1)),
            f"ranks are not a complete one-based sequence for {instrument}",
        )
        _assert(
            observed.get_column("actor_key").to_list()
            == expected.get_column("actor_key").to_list(),
            f"rank order does not follow the declared deterministic rule for {instrument}",
        )
    _assert(
        clusters.select(
            (
                pl.col("has_opposite_side_cancellation_within_2s")
                == (pl.col("qualifying_cancellation_count") > 0)
            ).all()
        ).item(),
        "cluster indicator disagrees with its qualifying cancellation count",
    )
    _assert(links.select((pl.col("cancel_event_ts") > pl.col("cluster_end_ts")).all()).item(), "non-positive link delay")
    _assert(links.select((pl.col("cancel_delay_seconds") > 0).all()).item(), "non-positive computed delay")
    _assert(links.select((pl.col("cancel_delay_seconds") <= 2.0).all()).item(), "link exceeds two seconds")
    _assert(
        links.select((pl.col("cancel_sort_index") > pl.col("cluster_last_sort_index")).all()).item(),
        "link violates source order",
    )
    _assert(links.select((pl.col("execution_side") != pl.col("opposite_side")).all()).item(), "link is not opposite-side")

    direct_for_join = direct_cancellations.select(
        "instrument",
        "partition_id",
        pl.col("sort_index").alias("cancel_sort_index"),
        pl.col("ORDERID").alias("cancel_order_id"),
        pl.col("actor_key").alias("cancel_actor_key"),
        pl.col("side").alias("cancel_side"),
        pl.col("event_ts").alias("replayed_cancel_event_ts"),
    )
    links_with_cancellation = links.join(
        direct_for_join,
        on=["instrument", "partition_id", "cancel_sort_index", "cancel_order_id"],
        how="inner",
        validate="m:1",
    )
    _assert(links_with_cancellation.height == links.height, "some links lack one physical cancellation record")
    _assert(
        links_with_cancellation.select(
            (pl.col("actor_key") == pl.col("cancel_actor_key")).all()
        ).item(),
        "a link crosses canonical actors",
    )
    _assert(
        links_with_cancellation.select(
            (pl.col("opposite_side") == pl.col("cancel_side")).all()
        ).item(),
        "a link does not use the cancellation's reconstructed side",
    )
    _assert(
        links_with_cancellation.select(
            (pl.col("cancel_event_ts") == pl.col("replayed_cancel_event_ts")).all()
        ).item(),
        "a link does not use the reconstructed cancellation timestamp",
    )

    cluster_for_join = clusters.select(
        "instrument",
        "execution_cluster_id",
        pl.col("actor_key").alias("cluster_actor_key"),
        pl.col("partition_id").alias("cluster_partition_id"),
        pl.col("execution_side").alias("cluster_execution_side"),
        pl.col("opposite_side").alias("cluster_opposite_side"),
        pl.col("cluster_end_ts").alias("audited_cluster_end_ts"),
        pl.col("cluster_last_sort_index").alias("audited_cluster_last_sort_index"),
    )
    links_with_cluster = links.join(
        cluster_for_join,
        on=["instrument", "execution_cluster_id"],
        how="inner",
        validate="m:1",
    )
    _assert(links_with_cluster.height == links.height, "some links lack one cluster record")
    _assert(
        links_with_cluster.select(
            pl.all_horizontal(
                pl.col("actor_key") == pl.col("cluster_actor_key"),
                pl.col("partition_id") == pl.col("cluster_partition_id"),
                pl.col("execution_side") == pl.col("cluster_execution_side"),
                pl.col("opposite_side") == pl.col("cluster_opposite_side"),
                pl.col("cluster_end_ts") == pl.col("audited_cluster_end_ts"),
                pl.col("cluster_last_sort_index") == pl.col("audited_cluster_last_sort_index"),
            ).all()
        ).item(),
        "a link disagrees with its execution-cluster record",
    )

    link_counts = links.group_by(["instrument", "execution_cluster_id"]).len(name="recounted_links")
    cluster_recount = clusters.join(link_counts, on=["instrument", "execution_cluster_id"], how="left").with_columns(
        pl.col("recounted_links").fill_null(0)
    )
    _assert(
        cluster_recount.select(
            (pl.col("qualifying_cancellation_count") == pl.col("recounted_links")).all()
        ).item(),
        "stored per-cluster cancellation counts do not equal link rows",
    )

    source_conditions = {
        "candidate_age_seconds",
        "withdrawal_to_execution_ratio",
        "favorable_mid_move_pre_fill",
        "post_cancel_mid_reversion",
        "wmsci",
        "msci",
        "spoofing_compatible_sequence",
        "strict_gate",
    }
    _assert(not source_conditions.intersection(actors.columns), "actor artifact contains a later detector condition")
    _assert(not source_conditions.intersection(clusters.columns), "cluster artifact contains a later detector condition")
    _assert(not source_conditions.intersection(links.columns), "link artifact contains a later detector condition")

    samples = _sample_links(links, per_anchor=per_anchor)
    raw_audit = _audit_raw_source_rows(samples, metadata, direct_cancellations)

    summary_rows = (
        actors.group_by("instrument")
        .agg(
            pl.len().alias("actor_count"),
            pl.col("execution_cluster_count").sum().alias("execution_cluster_count"),
            pl.col("qualifying_execution_cluster_count").sum().alias("qualifying_cluster_count"),
            (pl.col("cancellation_rate_2s") > 0).sum().alias("positive_rate_actor_count"),
            pl.col("cancellation_rate_2s").max().alias("maximum_actor_rate"),
        )
        .sort("instrument")
        .to_dicts()
    )
    anchor_rows = (
        clusters.group_by(["instrument", "execution_anchor_mode"])
        .agg(
            pl.len().alias("execution_cluster_count"),
            pl.col("has_opposite_side_cancellation_within_2s").sum().alias("qualifying_cluster_count"),
        )
        .sort(["instrument", "execution_anchor_mode"])
        .to_dicts()
    )
    reused_cancellations = (
        links.group_by(["instrument", "partition_id", "cancel_sort_index", "cancel_order_id"])
        .agg(pl.col("execution_cluster_id").n_unique().alias("linked_cluster_count"))
        .filter(pl.col("linked_cluster_count") > 1)
        .height
    )
    return {
        "schema_version": "actor_cancellation_rank_validation_v1",
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "metadata_path": str(metadata_path),
        "metadata_sha256": _sha256(metadata_path),
        "checks": {
            "artifact_hashes": "passed",
            "public_figure_metadata_integrity_and_privacy": "passed",
            "upstream_input_hashes": "passed",
            "cluster_uniqueness": "passed",
            "actor_cluster_conservation": "passed",
            "actor_rate_and_rank_recomputation": "passed",
            "binary_cluster_numerator": "passed",
            "joint_time_and_source_order": "passed",
            "all_link_actor_side_partition_reconciliation": "passed",
            "later_detector_conditions_absent": "passed",
            "raw_source_sample_audit": "passed",
        },
        "instrument_summary": summary_rows,
        "anchor_summary": anchor_rows,
        "raw_source_audit_sample_count": len(raw_audit),
        "raw_source_audit_samples": raw_audit,
        "physical_cancellations_linked_to_multiple_nearby_clusters": reused_cancellations,
        "interpretive_note": (
            "Cross-cluster reuse follows the stated broad cluster-level definition: each cluster asks whether at least "
            "one cancellation follows it. It is not canonical one-to-one cancellation assignment."
        ),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("paper/generated/actor_cancellation_rank_metadata.json"),
    )
    parser.add_argument("--sample-per-anchor", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.sample_per_anchor <= 0:
        raise ValueError("--sample-per-anchor must be positive")
    result = validate(args.metadata, per_anchor=args.sample_per_anchor)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
