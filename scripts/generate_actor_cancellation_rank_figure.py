#!/usr/bin/env python
"""Generate the motivating actor cancellation-rate artifact and rank figure."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from spoofing_detection.lob.motivating_cancellation_rate import (  # noqa: E402
    aggregate_actor_cancellation_rates,
    match_post_execution_opposite_side_cancellations,
    reconstruct_direct_cancellations,
)

EXPECTED_SCHEMA_VERSION = "actor_execution_anchor_v2"
WINDOW_SECONDS = 2.0
DETERMINISTIC_PDF_DATE = datetime(2000, 1, 1, tzinfo=timezone.utc)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--run must be INSTRUMENT=RUN_DIR")
    instrument, path = value.split("=", 1)
    if not instrument.strip() or not path.strip():
        raise argparse.ArgumentTypeError("--run must be INSTRUMENT=RUN_DIR")
    return instrument.strip(), Path(path)


def _validate_run(run_dir: Path) -> tuple[dict[str, Any], Path, Path]:
    metadata_path = run_dir / "metadata.json"
    execution_path = run_dir / "execution_metrics.parquet"
    if not metadata_path.exists() or not execution_path.exists():
        raise FileNotFoundError(f"run lacks metadata or execution clusters: {run_dir}")
    metadata = json.loads(metadata_path.read_text())
    expected = {
        "output_schema_version": EXPECTED_SCHEMA_VERSION,
        "analytical_unit": "execution_cluster",
        "actor_identity_mode": "client_then_firm",
        "firm_fallback_semantics": "aggregate only when client_original_id is missing",
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"{metadata_path} has {key}={metadata.get(key)!r}; expected {value!r}")
    if set(metadata.get("execution_anchor_modes", [])) != {"passive", "aggressive"}:
        raise ValueError(f"{metadata_path} must select passive and aggressive execution anchors")

    raw_path = Path(str(metadata.get("input") or ""))
    if not raw_path.exists():
        raise FileNotFoundError(f"raw input recorded by {metadata_path} does not exist: {raw_path}")
    expected_raw_hash = metadata.get("input_hashes", {}).get("raw_events_sha256")
    if expected_raw_hash and _sha256(raw_path) != expected_raw_hash:
        raise ValueError(f"raw input hash differs from {metadata_path}: {raw_path}")
    expected_execution_hash = metadata.get("artifact_hashes", {}).get("execution_metrics")
    if expected_execution_hash and _sha256(execution_path) != expected_execution_hash:
        raise ValueError(f"execution cluster hash differs from {metadata_path}: {execution_path}")
    return metadata, raw_path, execution_path


def _marker_area(cluster_counts: np.ndarray, *, max_count: int) -> np.ndarray:
    if max_count <= 0:
        raise ValueError("max_count must be positive")
    return 5.0 + 30.0 * np.log1p(cluster_counts) / np.log1p(max_count)


def plot_rank_figure(
    actor_rates: pl.DataFrame,
    *,
    instrument_order: list[str],
    pdf_path: Path,
    png_path: Path,
) -> None:
    """Plot separate client and firm-fallback rank curves; actor identifiers are never shown."""
    if actor_rates.is_empty():
        raise ValueError("actor rate artifact is empty")
    maximum_count = int(actor_rates.get_column("execution_cluster_count").max())
    fig, axes = plt.subplots(
        1,
        len(instrument_order),
        figsize=(7.25, 2.75),
        sharey=True,
        constrained_layout=True,
    )
    if len(instrument_order) == 1:
        axes = [axes]

    for axis, instrument in zip(axes, instrument_order, strict=True):
        frame = actor_rates.filter(pl.col("instrument") == instrument)
        if frame.is_empty():
            raise ValueError(f"actor rate artifact has no rows for {instrument}")
        for scope, marker, color in (
            ("client-level", "o", "#2878b5"),
            ("firm-fallback", "^", "#d97904"),
        ):
            series = frame.filter(pl.col("identity_scope") == scope).sort(
                ["cancellation_rate_2s", "execution_cluster_count", "actor_key"],
                descending=[True, True, False],
            )
            if series.is_empty():
                continue
            ranks = np.arange(1, series.height + 1)
            rates = series.get_column("cancellation_rate_2s").to_numpy()
            counts = series.get_column("execution_cluster_count").to_numpy()
            axis.plot(
                ranks,
                rates,
                color=color,
                linewidth=1.0,
                alpha=0.9,
                zorder=1,
            )
            axis.scatter(
                ranks,
                rates,
                s=_marker_area(counts, max_count=maximum_count),
                marker=marker,
                color=color,
                edgecolors="white",
                linewidths=0.25,
                alpha=0.75,
                zorder=2,
            )
        axis.set_title(instrument.title(), fontsize=9.5, pad=5)
        axis.set_xlabel("Actor rank (log scale)", fontsize=8.5)
        axis.set_xscale("log")
        axis.set_xlim(0.9, max(float(frame.height) * 1.05, 1.5))
        axis.set_ylim(0.0, 1.0)
        axis.set_yticks(np.linspace(0.0, 1.0, 6))
        axis.grid(axis="y", color="#d9d9d9", linewidth=0.5)
        axis.tick_params(axis="both", labelsize=7.5, length=2.5)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    axes[0].set_ylabel(
        r"$CR_i^{2s}$: opposite-side cancellation within 2 s",
        fontsize=8.5,
    )
    identity_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            linestyle="-",
            linewidth=1.0,
            color=color,
            markerfacecolor=color,
            markeredgecolor="white",
            markersize=5,
            label=label,
        )
        for label, marker, color in (
            ("Client-level", "o", "#2878b5"),
            ("Firm fallback", "^", "#d97904"),
        )
    ]
    axes[0].legend(
        handles=identity_handles,
        title="Identity scope",
        loc="upper right",
        frameon=False,
        fontsize=6.8,
        title_fontsize=7.2,
        handletextpad=0.3,
        borderaxespad=0.2,
    )
    reference_counts = [count for count in (1, 10, 100, 1000) if count <= maximum_count]
    handles = [
        axes[-1].scatter(
            [],
            [],
            s=float(_marker_area(np.array([count]), max_count=maximum_count)[0]),
            color="#2878b5",
            edgecolors="white",
            linewidths=0.25,
            alpha=0.72,
            label=f"{count:,}",
        )
        for count in reference_counts
    ]
    fig.legend(
        handles=handles,
        labels=[f"{count:,}" for count in reference_counts],
        title=r"Executions $N_i^E$",
        loc="outside lower center",
        ncol=len(reference_counts),
        frameon=False,
        fontsize=7.2,
        title_fontsize=7.5,
        handletextpad=0.3,
        columnspacing=0.8,
    )
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        metadata={
            "Creator": "generate_actor_cancellation_rank_figure.py",
            "CreationDate": DETERMINISTIC_PDF_DATE,
            "ModDate": DETERMINISTIC_PDF_DATE,
        },
    )
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate broad two-second post-execution opposite-side cancellation rates."
    )
    parser.add_argument("--run", action="append", type=parse_run, required=True, metavar="INSTRUMENT=RUN_DIR")
    parser.add_argument(
        "--audit-output-dir",
        type=Path,
        default=Path("outputs/actor_cancellation_rank"),
        help="Restricted audit artifacts; contains canonical actor and order identifiers.",
    )
    parser.add_argument(
        "--figure-output-dir",
        type=Path,
        default=Path("paper/generated"),
        help="De-identified publication figure and figure metadata.",
    )
    args = parser.parse_args(argv)

    actor_frames: list[pl.DataFrame] = []
    cluster_frames: list[pl.DataFrame] = []
    link_frames: list[pl.DataFrame] = []
    cancellation_frames: list[pl.DataFrame] = []
    provenance: dict[str, Any] = {}
    instrument_order: list[str] = []

    for instrument, run_dir in args.run:
        if instrument in instrument_order:
            raise ValueError(f"duplicate instrument: {instrument}")
        instrument_order.append(instrument)
        metadata, raw_path, execution_path = _validate_run(run_dir)
        executions = pl.read_parquet(execution_path)
        cancellations = reconstruct_direct_cancellations(pl.read_parquet(raw_path))
        clusters, links = match_post_execution_opposite_side_cancellations(
            executions,
            cancellations,
            window_seconds=WINDOW_SECONDS,
        )
        actors = aggregate_actor_cancellation_rates(clusters, instrument=instrument)
        actor_frames.append(actors)
        cluster_frames.append(clusters.with_columns(pl.lit(instrument).alias("instrument")))
        link_frames.append(links.with_columns(pl.lit(instrument).alias("instrument")))
        cancellation_frames.append(cancellations.with_columns(pl.lit(instrument).alias("instrument")))
        provenance[instrument] = {
            "run_dir": str(run_dir),
            "run_metadata_sha256": _sha256(run_dir / "metadata.json"),
            "execution_metrics_path": str(execution_path),
            "execution_metrics_sha256": _sha256(execution_path),
            "raw_events_path": str(raw_path),
            "raw_events_sha256": _sha256(raw_path),
            "execution_cluster_count": executions.height,
            "direct_cancellation_count": cancellations.height,
            "actor_count": actors.height,
            "qualifying_execution_cluster_count": int(
                actors.get_column("qualifying_execution_cluster_count").sum() or 0
            ),
            "qualifying_link_count": links.height,
            "source_output_schema_version": metadata["output_schema_version"],
        }

    actor_rates = pl.concat(actor_frames, how="vertical_relaxed")
    cluster_audit = pl.concat(cluster_frames, how="vertical_relaxed")
    links = pl.concat(link_frames, how="vertical_relaxed")
    direct_cancellations = pl.concat(cancellation_frames, how="vertical_relaxed")

    if actor_rates.get_column("execution_cluster_count").sum() != cluster_audit.height:
        raise AssertionError("actor denominators do not conserve execution clusters")
    if actor_rates.get_column("qualifying_execution_cluster_count").sum() != cluster_audit.get_column(
        "has_opposite_side_cancellation_within_2s"
    ).sum():
        raise AssertionError("actor numerators do not conserve qualifying clusters")

    args.audit_output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "actor_rates": args.audit_output_dir / "actor_post_execution_cancellation_rates.csv",
        "cluster_audit": args.audit_output_dir / "post_execution_cancellation_cluster_audit.parquet",
        "links": args.audit_output_dir / "post_execution_cancellation_links.parquet",
        "direct_cancellations": args.audit_output_dir / "direct_cancellations_for_rank_analysis.parquet",
        "figure_pdf": args.figure_output_dir / "actor_post_execution_cancellation_rank.pdf",
        "figure_png": args.figure_output_dir / "actor_post_execution_cancellation_rank.png",
        "metadata": args.audit_output_dir / "actor_cancellation_rank_metadata.json",
        "figure_metadata": args.figure_output_dir / "actor_cancellation_rank_figure_metadata.json",
    }
    actor_rates.write_csv(paths["actor_rates"])
    cluster_audit.write_parquet(paths["cluster_audit"])
    links.write_parquet(paths["links"])
    direct_cancellations.write_parquet(paths["direct_cancellations"])

    # The publication figure is deliberately rebuilt from the saved actor table.
    plotted_rates = pl.read_csv(paths["actor_rates"])
    plot_rank_figure(
        plotted_rates,
        instrument_order=instrument_order,
        pdf_path=paths["figure_pdf"],
        png_path=paths["figure_png"],
    )

    metadata = {
        "schema_version": "actor_post_execution_cancellation_rate_v1",
        "distribution_scope": (
            "restricted audit artifact; contains canonical actor identifiers, order identifiers, and local provenance paths"
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator": "scripts/generate_actor_cancellation_rank_figure.py",
        "generator_sha256": _sha256(Path(__file__)),
        "window_seconds": WINDOW_SECONDS,
        "operational_definition": {
            "denominator": "all passive and aggressive execution clusters of the canonical actor",
            "numerator": "execution clusters followed by at least one same-canonical-actor opposite-side cancellation",
            "event_time_rule": "cluster_end_ts < cancel_event_ts <= cluster_end_ts + 2 seconds",
            "source_order_rule": "cancel_sort_index > cluster_last_sort_index",
            "partition_rule": "execution cluster and cancellation share the complete trading partition",
            "cluster_contribution_cap": "one indicator unit irrespective of qualifying cancellation count",
            "cross_cluster_reuse": "a cancellation may qualify for more than one nearby cluster in this deliberately broad statistic",
            "identity_rule": "original client first; firm fallback only when original client is missing",
        },
        "conditions_intentionally_not_applied": [
            "candidate order age",
            "fixed age-conditioned candidate set",
            "top-n candidate rank",
            "withdrawal quantity exceeding execution quantity",
            "favorable price movement",
            "price reversion",
            "WMSCI or MSCI thresholds",
            "strict spoofing-compatible sequence gate",
        ],
        "rank_rule": "descending cancellation_rate_2s, then descending execution_cluster_count, then actor_key",
        "plot_rank_rule": "separate descending rank within instrument and identity scope",
        "support_encoding": "marker area increases with log(1 + execution_cluster_count)",
        "instrument_order": instrument_order,
        "provenance": provenance,
        "row_counts": {
            "actor_rates": actor_rates.height,
            "cluster_audit": cluster_audit.height,
            "qualifying_links": links.height,
            "direct_cancellations": direct_cancellations.height,
        },
        "paths": {key: str(path) for key, path in paths.items()},
    }
    public_summary = {
        "schema_version": "actor_cancellation_rank_figure_v1",
        "generator": "scripts/generate_actor_cancellation_rank_figure.py",
        "generator_sha256": _sha256(Path(__file__)),
        "actor_rate_artifact_sha256": _sha256(paths["actor_rates"]),
        "figure_pdf_sha256": _sha256(paths["figure_pdf"]),
        "figure_png_sha256": _sha256(paths["figure_png"]),
        "window_seconds": WINDOW_SECONDS,
        "instrument_order": instrument_order,
        "instrument_counts": {
            instrument: {
                "actor_count": provenance[instrument]["actor_count"],
                "execution_cluster_count": provenance[instrument]["execution_cluster_count"],
                "qualifying_execution_cluster_count": provenance[instrument][
                    "qualifying_execution_cluster_count"
                ],
            }
            for instrument in instrument_order
        },
        "privacy": "No actor or order identifiers and no local data paths are included in this publication metadata.",
        "support_encoding": "marker area increases with log(1 + execution_cluster_count)",
        "identity_encoding": "separate client-level and firm-fallback curves with distinct colors and markers",
        "plot_rank_rule": "separate descending rank within instrument and identity scope",
        "x_axis_scale": "logarithmic actor rank",
    }
    paths["figure_metadata"].write_text(
        json.dumps(public_summary, indent=2, sort_keys=True) + "\n"
    )
    metadata["artifact_hashes"] = {
        key: _sha256(path) for key, path in paths.items() if key != "metadata"
    }
    metadata["command"] = sys.argv if argv is None else [str(Path(__file__)), *argv]
    paths["metadata"].write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
