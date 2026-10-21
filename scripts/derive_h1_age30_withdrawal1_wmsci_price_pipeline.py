#!/usr/bin/env python3
"""Derive the one-second, single sequential WMSCI→price pipeline from retained H1 bundles.

No raw event replay is performed.  The source bundles retain the candidate-link
clocks, candidate snapshots, execution diagnostics, and replayed state panel
needed to re-run one-second cancellation eligibility, winner assignment,
quantity-only WMSCI, and two-second cancel-anchored reversion on copies.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any, Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

try:  # Supports both `python script.py` and package import in tests.
    from scripts.validate_spoofing_detection_bundle import (
        V3_CLOCK_CONTRACT,
        V3_CLOCK_POLICY_ID,
        V3_OUTPUT_SCHEMA_VERSION,
        validate_bundle as native_validate_bundle,
    )
except ModuleNotFoundError:  # pragma: no cover - exercised by the real CLI
    from validate_spoofing_detection_bundle import (  # type: ignore[no-redef]
        V3_CLOCK_CONTRACT,
        V3_CLOCK_POLICY_ID,
        V3_OUTPUT_SCHEMA_VERSION,
        validate_bundle as native_validate_bundle,
    )
from spoofing_detection.lob.candidate_episodes import build_candidate_episodes
from spoofing_detection.lob.spoofing_metrics import (
    _attach_direct_cancellation_window,
    _parse_ts,
    _weighted_finite_mean,
    assign_cancellations_to_clusters,
)

ROOT = Path(__file__).resolve().parents[1]
INSTRUMENTS = ("FERRARI", "NEXI", "RISANAMENTO")
DEFAULT_BUNDLE_ROOT = ROOT / "outputs" / "spoofing_detection_h1_checkpointed_20261007_231228" / "runs"
DEFAULT_CONFIG = ROOT / "configs" / "h1_age30_withdrawal1_wmsci_price_pipeline.json"
SCHEMA = "h1_age30_withdrawal1_wmsci_price_pipeline_v2_complete_finite_rev"
NS_PER_SECOND = 1_000_000_000
SOURCE_ARTIFACTS = (
    "execution_metrics",
    "execution_cancel_candidates",
    "candidate_deceptive_orders",
    "state_time_series",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quantity_only_wmsci(p: float, q: float, u: float) -> float:
    """Execution-level W=log1p(P/Q)*log1p(U/Q)*(U/P)."""
    if not all(math.isfinite(value) for value in (p, q, u)) or p < 0 or u < 0 or q <= 0:
        raise ValueError("WMSCI requires finite P,U>=0 and Q>0")
    if p == 0 or u == 0:
        return 0.0
    return math.log1p(p / q) * math.log1p(u / q) * (u / p)


def _require_columns(frame: pl.DataFrame, columns: set[str], label: str) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} missing columns: {', '.join(missing)}")


def filter_and_reassign_one_second_links(links: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, int]]:
    """Recalculate every retained candidate-link delay, filter [0,1s), then assign once."""
    required = {
        "partition_id", "cancel_sort_index", "candidate_order_id", "execution_cluster_id",
        "cluster_first_sort_index", "cluster_last_sort_index", "cluster_end_fill_anchor_ns",
        "cancellation_removal_ns",
    }
    _require_columns(links, required, "execution cancellation links")
    if links.is_empty():
        return links, {
            "candidate_link_rows": 0, "eligible_link_rows": 0,
            "eligible_physical_cancellations": 0, "assigned_link_rows": 0,
        }
    clocks_missing = links.filter(
        pl.col("cluster_end_fill_anchor_ns").is_null() | pl.col("cancellation_removal_ns").is_null()
    )
    if clocks_missing.height:
        raise ValueError(f"retained candidate links have {clocks_missing.height} missing v3 eligibility clocks")
    # The retained two-second source contains cancellation-level reversion
    # sufficient statistics.  Keep the original winner flag separately so the
    # derivative can prove that a one-second winner was already computed from
    # the identical state panel and two-second horizon.
    source_assigned = pl.col("assigned_flag") if "assigned_flag" in links.columns else pl.lit(False)
    delayed = links.with_columns(
        source_assigned.alias("_source_assigned_flag"),
        (pl.col("cancellation_removal_ns") - pl.col("cluster_end_fill_anchor_ns")).alias("_recomputed_delay_ns")
    )
    if "cancel_economic_delay_ns" in delayed.columns:
        prior = delayed.filter(pl.col("cancel_economic_delay_ns").is_not_null())
        disagreement = prior.filter(pl.col("cancel_economic_delay_ns") != pl.col("_recomputed_delay_ns"))
        if disagreement.height:
            raise ValueError(f"retained economic-delay disagreement in {disagreement.height} candidate links")
    eligible = delayed.filter(
        (pl.col("_recomputed_delay_ns") >= 0) & (pl.col("_recomputed_delay_ns") < NS_PER_SECOND)
    ).drop("_recomputed_delay_ns")
    assigned = assign_cancellations_to_clusters(eligible, use_v3_clocks=True)
    physical = eligible.select(["partition_id", "cancel_sort_index", "candidate_order_id"]).unique().height
    return assigned, {
        "candidate_link_rows": links.height,
        "eligible_link_rows": eligible.height,
        "eligible_physical_cancellations": physical,
        "assigned_link_rows": int(assigned.get_column("assigned_flag").sum()) if not assigned.is_empty() else 0,
    }


def recompute_event_quantities_and_score(executions: pl.DataFrame, links: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Use the retained core event transform after latest-position reassignment."""
    required = {
        "execution_cluster_id", "partition_id", "actor_key", "deceptive_side",
        "cluster_end_fill_anchor_ns", "cluster_last_sort_index",
        "candidate_deceptive_visible_qty_pre", "execution_quantity",
    }
    _require_columns(executions, required, "execution metrics")
    assigned = assign_cancellations_to_clusters(links, use_v3_clocks=True)
    # Direct-opposite cancellation diagnostics are not a pipeline gate.  The empty
    # raw-cancellation input leaves those descriptive fields at zero while the
    # retained core recomputes matched U/Q, U/P and W from assigned candidate links.
    refreshed = _attach_direct_cancellation_window(
        executions, pl.DataFrame(), window_seconds=1.0, assigned_candidates=assigned, use_v3_clocks=True
    )
    return refreshed, assigned


_REVERSION_RETAINED_FIELDS = (
    "cancel_reversion_target_ts",
    "cancel_reversion_target_removal_ns",
    "cancel_economic_delay_ns",
    "cancel_economic_delay_seconds",
    "cancel_pre_state_sort_index",
    "cancel_post_state_sort_index",
    "cancel_mid_pre",
    "cancel_mid_post_horizon",
    "cancel_microprice_pre",
    "cancel_microprice_post_horizon",
    "post_cancel_mid_reversion",
    "post_cancel_microprice_reversion",
    "cancel_reversion_weight",
    "has_cancel_reversion_state",
)


def reuse_retained_cancel_anchored_reversion(
    executions: pl.DataFrame, candidates: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Reuse validated cancellation-level 2s reversion sufficient statistics.

    The current one-second rule only changes eligibility and winner assignment.
    For every surviving winner, this function proves it was also the source
    two-second winner and that its retained target is precisely cancellation
    removal plus two seconds.  State inputs and the horizon are therefore
    unchanged; the retained per-cancellation observations are identical to a
    fresh core lookup.  Re-aggregation is deliberately performed after the new
    assignment so execution-level means retain core semantics.
    """
    _require_columns(
        candidates,
        {
            "assigned_flag", "_source_assigned_flag", "cancellation_removal_ns",
            "cluster_end_fill_anchor_ns", "attributed_cancel_visible_qty",
            "execution_cluster_id", "cancel_event_ts", "event_ts", *_REVERSION_RETAINED_FIELDS,
        },
        "retained cancellation reversion observations",
    )
    if candidates.filter(pl.col("assigned_flag") & ~pl.col("_source_assigned_flag")).height:
        raise ValueError("one-second winner lacks a retained source reversion observation")
    if candidates.filter(~pl.col("assigned_flag") & pl.col("_source_assigned_flag")).height:
        raise ValueError("one-second reassignment displaced a source winner; retained reversion is insufficient")
    assigned = candidates.filter(pl.col("assigned_flag"))
    if assigned.height:
        expected_target = pl.col("cancellation_removal_ns") + (2 * NS_PER_SECOND)
        expected_delay = pl.col("cancellation_removal_ns") - pl.col("cluster_end_fill_anchor_ns")
        invalid = assigned.filter(
            pl.col("cancel_reversion_target_removal_ns").is_null()
            | (pl.col("cancel_reversion_target_removal_ns") != expected_target)
            | pl.col("cancel_economic_delay_ns").is_null()
            | (pl.col("cancel_economic_delay_ns") != expected_delay)
            | (pl.col("cancel_economic_delay_ns") < 0)
            | (pl.col("cancel_reversion_target_ts").is_not_null())
        )
        if invalid.height:
            raise ValueError("retained reversion target/delay does not prove the unchanged 2s v3 lookup")

    # Core clears every reversion field on an unassigned candidate.  Do the
    # same before writing the audit artifact rather than leaking source flags.
    expressions = []
    for field in _REVERSION_RETAINED_FIELDS:
        if field == "has_cancel_reversion_state":
            replacement = pl.lit(False, dtype=pl.Boolean)
        else:
            replacement = pl.lit(None, dtype=candidates.schema[field])
        expressions.append(
            pl.when(pl.col("assigned_flag")).then(pl.col(field)).otherwise(replacement).alias(field)
        )
    enriched = candidates.with_columns(expressions).drop("_source_assigned_flag")

    assigned_by_cluster: dict[str, list[dict[str, Any]]] = {}
    for candidate in enriched.filter(pl.col("assigned_flag")).iter_rows(named=True):
        assigned_by_cluster.setdefault(str(candidate.get("execution_cluster_id")), []).append(candidate)
    execution_rows: list[dict[str, Any]] = []
    for execution in executions.iter_rows(named=True):
        assigned_rows = assigned_by_cluster.get(str(execution.get("execution_cluster_id")), [])
        cancel_times = [
            timestamp
            for candidate in assigned_rows
            if (timestamp := _parse_ts(candidate.get("cancel_event_ts") or candidate.get("event_ts"))) is not None
        ]
        mid_values = [
            (candidate.get("post_cancel_mid_reversion"), float(candidate.get("cancel_reversion_weight") or 0.0))
            for candidate in assigned_rows
        ]
        microprice_values = [
            (candidate.get("post_cancel_microprice_reversion"), float(candidate.get("cancel_reversion_weight") or 0.0))
            for candidate in assigned_rows
        ]
        execution_rows.append(
            {
                **execution,
                "post_cancel_mid_reversion": _weighted_finite_mean(mid_values),
                "post_cancel_microprice_reversion": _weighted_finite_mean(microprice_values),
                "cancel_reversion_observation_count": sum(
                    1 for value, weight in mid_values if value is not None and weight > 0
                ),
                "has_cancel_reversion_state": any(
                    bool(candidate.get("has_cancel_reversion_state")) for candidate in assigned_rows
                ),
                "first_matched_cancel_ts": min(cancel_times) if cancel_times else None,
                "last_matched_cancel_ts": max(cancel_times) if cancel_times else None,
                "reversion_horizon_seconds": 2.0,
            }
        )
    return pl.DataFrame(execution_rows, infer_schema_length=None), enriched


def enforce_complete_finite_reversion(events: pl.DataFrame, candidates: pl.DataFrame) -> pl.DataFrame:
    """Null screening REV unless every assigned positive-quantity cancel is finite.

    The core weighted mean describes available observations. This derivative's
    strict REV screen requires complete finite coverage, so a partial mean is
    retained only in the explicitly labelled descriptive field.
    """
    required = {
        "execution_cluster_id", "assigned_flag", "attributed_cancel_visible_qty",
        "post_cancel_mid_reversion", "has_cancel_reversion_state",
    }
    _require_columns(candidates, required, "reversion candidates")
    invalid_assigned_quantity = candidates.filter(
        pl.col("assigned_flag")
        & (pl.col("attributed_cancel_visible_qty").is_null()
           | ~pl.col("attributed_cancel_visible_qty").is_finite()
           | (pl.col("attributed_cancel_visible_qty") < 0))
    )
    if invalid_assigned_quantity.height:
        raise ValueError("assigned cancellation has missing, non-finite, or negative attributed quantity")
    available = events.select("execution_cluster_id", "post_cancel_mid_reversion").rename(
        {"post_cancel_mid_reversion": "post_cancel_mid_reversion_available_only"}
    )
    positive_assigned = candidates.filter(
        pl.col("assigned_flag")
        & pl.col("attributed_cancel_visible_qty").is_not_null()
        & pl.col("attributed_cancel_visible_qty").is_finite()
        & (pl.col("attributed_cancel_visible_qty") > 0)
    ).with_columns(
        (pl.col("has_cancel_reversion_state").fill_null(False)
         & pl.col("post_cancel_mid_reversion").is_not_null()
         & pl.col("post_cancel_mid_reversion").is_finite()).alias("_finite_rev")
    )
    aggregate = positive_assigned.group_by("execution_cluster_id").agg(
        pl.len().alias("assigned_positive_qty_cancel_count"),
        pl.col("_finite_rev").sum().cast(pl.Int64).alias("assigned_positive_qty_cancel_reversion_observed_count"),
    ) if not positive_assigned.is_empty() else pl.DataFrame(
        schema={"execution_cluster_id": pl.String, "assigned_positive_qty_cancel_count": pl.Int64,
                "assigned_positive_qty_cancel_reversion_observed_count": pl.Int64}
    )
    screened = available.join(aggregate, on="execution_cluster_id", how="left").with_columns(
        pl.col("assigned_positive_qty_cancel_count").fill_null(0),
        pl.col("assigned_positive_qty_cancel_reversion_observed_count").fill_null(0),
    ).with_columns(
        (pl.col("assigned_positive_qty_cancel_count") - pl.col("assigned_positive_qty_cancel_reversion_observed_count"))
        .alias("assigned_positive_qty_cancel_reversion_missing_count"),
        ((pl.col("assigned_positive_qty_cancel_count") > 0)
         & (pl.col("assigned_positive_qty_cancel_count") == pl.col("assigned_positive_qty_cancel_reversion_observed_count")))
        .alias("post_cancel_mid_reversion_complete"),
    ).with_columns(
        pl.when(
            pl.col("post_cancel_mid_reversion_complete")
            & pl.col("post_cancel_mid_reversion_available_only").is_not_null()
            & pl.col("post_cancel_mid_reversion_available_only").is_finite()
        ).then(pl.col("post_cancel_mid_reversion_available_only")).otherwise(pl.lit(None, dtype=pl.Float64))
        .alias("post_cancel_mid_reversion")
    )
    return events.drop("post_cancel_mid_reversion", strict=False).join(screened, on="execution_cluster_id", how="left")


def sequential_population(events: pl.DataFrame, *, fraction: float) -> tuple[pl.DataFrame, dict[str, int]]:
    """Apply the one execution-level screen with finite checks at every gate."""
    if not (0 < fraction <= 1 and math.isfinite(fraction)):
        raise ValueError("top-W fraction must be finite and in (0,1]")
    required = {
        "execution_cluster_id", "WMSCI_event", "withdrawal_to_execution_ratio",
        "favorable_mid_move_pre_fill", "post_cancel_mid_reversion", "post_cancel_mid_reversion_complete",
    }
    _require_columns(events, required, "event population")
    if events.get_column("execution_cluster_id").null_count() or events.get_column("execution_cluster_id").n_unique() != events.height:
        raise ValueError("execution_cluster_id must be non-null and unique")
    finite_w = events.filter(pl.col("WMSCI_event").is_not_null() & pl.col("WMSCI_event").is_finite())
    positive = finite_w.filter(pl.col("WMSCI_event") > 0)
    k = int(math.ceil(fraction * positive.height)) if positive.height else 0
    top = positive.sort(["WMSCI_event", "execution_cluster_id"], descending=[True, False]).head(k)
    finite_u = top.filter(pl.col("withdrawal_to_execution_ratio").is_not_null() & pl.col("withdrawal_to_execution_ratio").is_finite())
    quantity = finite_u.filter(pl.col("withdrawal_to_execution_ratio") > 1)
    fpm_observed = quantity.filter(pl.col("favorable_mid_move_pre_fill").is_not_null() & pl.col("favorable_mid_move_pre_fill").is_finite())
    fpm = fpm_observed.filter(pl.col("favorable_mid_move_pre_fill") > 0)
    rev_observed = fpm.filter(
        pl.col("post_cancel_mid_reversion_complete").fill_null(False)
        & pl.col("post_cancel_mid_reversion").is_not_null()
        & pl.col("post_cancel_mid_reversion").is_finite()
    )
    final = rev_observed.filter(pl.col("post_cancel_mid_reversion") > 0)
    return final, {
        "all_executions": events.height,
        "finite_W_executions": finite_w.height,
        "positive_W_executions": positive.height,
        "top_W_exact_k": top.height,
        "top_W_finite_U_over_Q": finite_u.height,
        "top_W_after_U_over_Q_gt_1": quantity.height,
        "top_W_after_U_over_Q_gt_1_FPM_observed": fpm_observed.height,
        "top_W_after_U_over_Q_gt_1_FPM_gt_0": fpm.height,
        "top_W_after_FPM_gt_0_REV_observed": rev_observed.height,
        "final_WMSCI_to_FPM_to_REV_executions": final.height,
    }


def _artifact_path(bundle: Path, metadata: Mapping[str, Any], name: str) -> Path:
    value = metadata.get("paths", {}).get(name)
    if not isinstance(value, str):
        raise ValueError(f"missing source path in metadata: {name}")
    path = Path(value)
    path = path if path.is_absolute() else bundle / path
    path = path.resolve()
    try:
        path.relative_to(bundle.resolve())
    except ValueError as exc:
        raise ValueError(f"source artifact escapes bundle: {name}") from exc
    if not path.is_file():
        raise ValueError(f"missing source artifact: {name}")
    return path


def _load_contract(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text())
    expected = {
        "schema": SCHEMA,
        "source_output_schema_version": V3_OUTPUT_SCHEMA_VERSION,
        "source_clock_policy_id": V3_CLOCK_POLICY_ID,
        "source_clock_contract": V3_CLOCK_CONTRACT,
        "profile_window_seconds": 1.0,
        "candidate_age_seconds": 30.0,
        "withdrawal_window_seconds": 1.0,
        "reversion_horizon_seconds": 2.0,
        "top_positive_w_fraction": 0.05,
        "quantity_gate": "execution_level_U_over_Q_gt_1",
        "assignment_rule": "latest_prior_cluster_sort_index",
        "selection_order": ["positive_W_exact_top_5pct", "U_over_Q_gt_1", "FPM_gt_0", "REV_gt_0"],
        "episode_role": "recomputed_shared_order_reporting_only_not_a_selection_gate",
        "reversion_coverage": "all_assigned_positive_quantity_cancellations_must_have_finite_REV",
    }
    if config != expected:
        raise ValueError("analysis config contract mismatch")
    return config


def _validate_bundle_metadata(metadata: Mapping[str, Any]) -> None:
    expected = {
        "window_seconds": 1.0,
        "max_deceptive_order_age_seconds": 30.0,
        "withdrawal_window_seconds": 2.0,
        "reversion_horizon_seconds": 2.0,
        "output_schema_version": V3_OUTPUT_SCHEMA_VERSION,
        "clock_policy_id": V3_CLOCK_POLICY_ID,
        "clock_contract": V3_CLOCK_CONTRACT,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"source metadata mismatch: {key}")


def _validate_source_positions(executions: pl.DataFrame, links: pl.DataFrame, candidates: pl.DataFrame) -> None:
    """Reject stale link rows before any reassignment, including equal-time ties."""
    if executions.get_column("execution_cluster_id").n_unique() != executions.height:
        raise ValueError("source execution_cluster_id is not unique")
    execution_positions = executions.select(
        "execution_cluster_id", "partition_id", "actor_key", "cluster_first_sort_index", "cluster_last_sort_index", "cluster_end_fill_anchor_ns"
    )
    checked = links.join(execution_positions, on="execution_cluster_id", how="left", suffix="_source")
    if checked.get_column("partition_id_source").null_count() or checked.filter(
        (pl.col("partition_id") != pl.col("partition_id_source"))
        | (pl.col("actor_key") != pl.col("actor_key_source"))
        | (pl.col("cluster_first_sort_index") != pl.col("cluster_first_sort_index_source"))
        | (pl.col("cluster_last_sort_index") != pl.col("cluster_last_sort_index_source"))
        | (pl.col("cluster_end_fill_anchor_ns") != pl.col("cluster_end_fill_anchor_ns_source"))
        | (pl.col("cluster_last_sort_index") >= pl.col("cancel_sort_index"))
    ).height:
        raise ValueError("candidate links disagree with execution source position or causal order")
    source_orders = candidates.select(
        pl.col("partition_id"), pl.col("actor_key"), pl.col("execution_cluster_id"),
        pl.col("deceptive_order_id").alias("candidate_order_id"),
    ).unique()
    unresolved = links.join(source_orders, on=["partition_id", "actor_key", "execution_cluster_id", "candidate_order_id"], how="anti")
    if unresolved.height:
        raise ValueError("candidate link does not match retained candidate-order identity")


def _safe_execution_diagnostics(frame: pl.DataFrame, instrument: str) -> pl.DataFrame:
    """Export only stable inputs and recomputed outputs; never inherited strict flags."""
    columns = [
        "partition_id", "execution_cluster_id", "actor_key", "actor_id", "identity_level", "identity_source",
        "execution_anchor_mode", "execution_side", "deceptive_side", "cluster_first_sort_index", "cluster_last_sort_index",
        "cluster_end_fill_anchor_ns", "candidate_deceptive_visible_qty_pre", "execution_quantity",
        "matched_deceptive_cancel_count_window", "matched_deceptive_cancel_visible_qty_window",
        "matched_deceptive_cancel_fraction_window", "withdrawal_to_execution_ratio", "WMSCI_event",
        "MSCI", "favorable_mid_move_pre_fill", "execution_price_advantage_vs_posture_mid",
        "post_cancel_mid_reversion_available_only", "post_cancel_mid_reversion",
        "post_cancel_mid_reversion_complete", "assigned_positive_qty_cancel_count",
        "assigned_positive_qty_cancel_reversion_observed_count", "assigned_positive_qty_cancel_reversion_missing_count",
        "episode_id",
    ]
    _require_columns(frame, set(columns), "refreshed diagnostics")
    return frame.select(columns).with_columns(pl.lit(instrument).alias("instrument")).select("instrument", pl.all().exclude("instrument"))


def _step_membership(frame: pl.DataFrame, fraction: float, instrument: str) -> pl.DataFrame:
    finite_w = pl.col("WMSCI_event").is_not_null() & pl.col("WMSCI_event").is_finite()
    positive = frame.filter(finite_w & (pl.col("WMSCI_event") > 0)).sort(
        ["WMSCI_event", "execution_cluster_id"], descending=[True, False]
    )
    top_ids = set(positive.head(int(math.ceil(fraction * positive.height))).get_column("execution_cluster_id").to_list())
    base = frame.with_columns(
        pl.lit(instrument).alias("instrument"),
        finite_w.alias("finite_W"),
        (finite_w & (pl.col("WMSCI_event") > 0)).alias("positive_W"),
        pl.col("execution_cluster_id").is_in(top_ids).alias("exact_top_positive_W"),
        (pl.col("withdrawal_to_execution_ratio").is_not_null() & pl.col("withdrawal_to_execution_ratio").is_finite()
         & (pl.col("withdrawal_to_execution_ratio") > 1)).alias("U_over_Q_gt_1_finite"),
        (pl.col("favorable_mid_move_pre_fill").is_not_null() & pl.col("favorable_mid_move_pre_fill").is_finite()
         & (pl.col("favorable_mid_move_pre_fill") > 0)).alias("FPM_gt_0_finite"),
        (pl.col("post_cancel_mid_reversion_complete") & pl.col("post_cancel_mid_reversion").is_not_null()
         & pl.col("post_cancel_mid_reversion").is_finite() & (pl.col("post_cancel_mid_reversion") > 0)).alias("REV_gt_0_complete_finite"),
    ).select(
        "instrument", "execution_cluster_id", "finite_W", "positive_W", "exact_top_positive_W",
        "U_over_Q_gt_1_finite", "FPM_gt_0_finite", "REV_gt_0_complete_finite",
    )
    return base.with_columns(
        (pl.col("exact_top_positive_W") & pl.col("U_over_Q_gt_1_finite")).alias("after_W_then_U_over_Q_gt_1"),
    ).with_columns(
        (pl.col("after_W_then_U_over_Q_gt_1") & pl.col("FPM_gt_0_finite")).alias("after_W_then_U_over_Q_then_FPM_gt_0"),
    ).with_columns(
        (pl.col("after_W_then_U_over_Q_then_FPM_gt_0") & pl.col("REV_gt_0_complete_finite")).alias("selected_final")
    ).select("instrument", pl.all().exclude("instrument")).sort("execution_cluster_id")


def _derive_instrument(instrument: str, bundle: Path, fraction: float) -> dict[str, Any]:
    native = native_validate_bundle(bundle)
    metadata = json.loads((bundle / "metadata.json").read_text())
    _validate_bundle_metadata(metadata)
    paths = {name: _artifact_path(bundle, metadata, name) for name in SOURCE_ARTIFACTS}
    executions = pl.read_parquet(paths["execution_metrics"])
    links = pl.read_parquet(paths["execution_cancel_candidates"])
    candidates = pl.read_parquet(paths["candidate_deceptive_orders"])
    _validate_source_positions(executions, links, candidates)
    assigned, link_counts = filter_and_reassign_one_second_links(links)
    refreshed, assigned = recompute_event_quantities_and_score(executions, assigned)
    refreshed, enriched_links = reuse_retained_cancel_anchored_reversion(refreshed, assigned)
    refreshed = enforce_complete_finite_reversion(refreshed, enriched_links)
    episodes = build_candidate_episodes(refreshed, candidates, enriched_links, require_v3_clocks=True)
    membership = episodes.members.select("execution_cluster_id", "episode_id")
    if membership.get_column("execution_cluster_id").n_unique() != membership.height:
        raise ValueError("recomputed episode membership is not one-to-one")
    refreshed = refreshed.drop("episode_id", strict=False).join(membership, on="execution_cluster_id", how="left")
    selected, counts = sequential_population(refreshed, fraction=fraction)
    diagnostics = _safe_execution_diagnostics(refreshed, instrument)
    selected = diagnostics.join(selected.select("execution_cluster_id"), on="execution_cluster_id", how="semi")
    grouped = selected.filter(pl.col("episode_id").is_not_null()).group_by(["instrument", "episode_id"]).agg(
        pl.len().alias("selected_execution_count"),
        pl.col("execution_cluster_id").n_unique().alias("distinct_selected_execution_count"),
        pl.col("execution_quantity").sum().alias("selected_execution_Q_total"),
        pl.col("matched_deceptive_cancel_visible_qty_window").sum().alias("selected_assigned_U_total"),
    ).sort(["instrument", "episode_id"])
    counts.update({
        "selected_executions_with_shared_order_episode": int(selected.get_column("episode_id").is_not_null().sum()),
        "shared_order_episodes_reporting_only": grouped.height,
        "assigned_positive_qty_cancel_reversion_missing": int(diagnostics.get_column("assigned_positive_qty_cancel_reversion_missing_count").sum()),
    })
    provenance = {
        "bundle": str(bundle.resolve()), "metadata_sha256": sha256_file(bundle / "metadata.json"),
        "native_bundle_artifact_count": native["artifact_count"],
        "tick_size": metadata.get("tick_size"),
        "source_artifact_sha256": {name: sha256_file(path) for name, path in paths.items()},
        "candidate_link_delay_recomputed_from_v3_clocks": True,
        "source_position_and_candidate_identity_validated": True,
        "score_max_abs_identity_error": float(refreshed.select((pl.col("WMSCI_event") - pl.struct([
            pl.col("candidate_deceptive_visible_qty_pre").alias("p"), pl.col("execution_quantity").alias("q"),
            pl.col("matched_deceptive_cancel_visible_qty_window").alias("u"),
        ]).map_elements(lambda r: quantity_only_wmsci(r["p"], r["q"], r["u"]), return_dtype=pl.Float64)).abs().max()).item() or 0.0),
        **link_counts,
    }
    if provenance["score_max_abs_identity_error"] > 1e-12:
        raise ValueError(f"{instrument} WMSCI identity mismatch")
    if not isinstance(provenance["tick_size"], (int, float)) or not math.isfinite(float(provenance["tick_size"])) or provenance["tick_size"] <= 0:
        raise ValueError(f"{instrument} source tick_size is invalid")
    return {"selected": selected, "grouped": grouped, "counts": counts, "provenance": provenance,
            "diagnostics": diagnostics, "cancellations": enriched_links.with_columns(pl.lit(instrument).alias("instrument")),
            "membership": _step_membership(refreshed, fraction, instrument)}


def _csv(frame: pl.DataFrame) -> str:
    return frame.write_csv()


def _tex_escape(value: object) -> str:
    substitutions = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(substitutions.get(char, char) for char in str(value))


def _display(value: object) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return format(value, ".5g") if math.isfinite(value) else "NA"
    return str(value)


def _table_tex(frame: pl.DataFrame, caption: str, label: str) -> str:
    if frame.is_empty():
        raise ValueError(f"cannot render empty table: {label}")
    headers = [column.replace("_", " ") for column in frame.columns]
    line_break = " " + "\\" * 2
    header = " & ".join(_tex_escape(column) for column in headers) + line_break
    rows = [" & ".join(_tex_escape(_display(value)) for value in row) + line_break for row in frame.iter_rows()]
    alignment = "l" * len(frame.columns)
    return "\n".join([
        r"\begin{table}[htbp]", r"\centering", f"\\caption{{{_tex_escape(caption)}}}", f"\\label{{{label}}}",
        r"\small", f"\\begin{{tabular}}{{{alignment}}}", r"\toprule", header, r"\midrule", *rows,
        r"\bottomrule", r"\end{tabular}", r"\end{table}", "",
    ])


def _spearman(left: np.ndarray, right: np.ndarray) -> tuple[int, float | None]:
    """Spearman rho with average ranks, without adding a scipy dependency."""
    valid = np.isfinite(left) & np.isfinite(right)
    x, y = left[valid], right[valid]
    if len(x) < 2 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return int(len(x)), None

    def rank(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="stable")
        sorted_values = values[order]
        ranks = np.empty(len(values), dtype=float)
        start = 0
        while start < len(values):
            end = start + 1
            while end < len(values) and sorted_values[end] == sorted_values[start]:
                end += 1
            ranks[order[start:end]] = (start + 1 + end) / 2.0
            start = end
        return ranks

    rho = float(np.corrcoef(rank(x), rank(y))[0, 1])
    return int(len(x)), rho if math.isfinite(rho) else None


def _score_price_associations(diagnostics: pl.DataFrame) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for instrument in INSTRUMENTS:
        frame = diagnostics.filter((pl.col("instrument") == instrument) & pl.col("WMSCI_event").is_finite() & (pl.col("WMSCI_event") > 0))
        score = frame.get_column("WMSCI_event").to_numpy().astype(float)
        for name in ("favorable_mid_move_pre_fill", "post_cancel_mid_reversion", "execution_price_advantage_vs_posture_mid"):
            values = frame.get_column(name).to_numpy().astype(float)
            n, rho = _spearman(score, values)
            rows.append({"instrument": instrument, "price_diagnostic": name, "positive_W_finite_pairs": n, "spearman_W_price": rho})
    return pl.DataFrame(rows).sort(["instrument", "price_diagnostic"])


def _role_summary(diagnostics: pl.DataFrame, membership: pl.DataFrame) -> pl.DataFrame:
    combined = diagnostics.join(membership.select("instrument", "execution_cluster_id", "selected_final"), on=["instrument", "execution_cluster_id"], how="left").with_columns(
        pl.col("selected_final").fill_null(False)
    )
    return combined.group_by(["instrument", "identity_level", "execution_anchor_mode", "selected_final"]).agg(
        pl.len().alias("execution_count"),
        pl.col("execution_quantity").sum().alias("execution_Q_total"),
        pl.col("matched_deceptive_cancel_visible_qty_window").sum().alias("assigned_U_total"),
    ).sort(["instrument", "identity_level", "execution_anchor_mode", "selected_final"])


def _save_figures(diagnostics: pl.DataFrame, output: Path) -> None:
    """Plot only current stable/recomputed fields; no inherited strict flags are used."""
    fig, axes = plt.subplots(len(INSTRUMENTS), 1, figsize=(7.2, 8.4), sharex=True, layout="constrained")
    for ax, instrument in zip(axes, INSTRUMENTS):
        frame = diagnostics.filter((pl.col("instrument") == instrument) & pl.col("MSCI").is_not_null() & pl.col("MSCI").is_finite() & pl.col("WMSCI_event").is_finite())
        ax.scatter(frame.get_column("MSCI").to_numpy(), np.log1p(frame.get_column("WMSCI_event").to_numpy()), s=5, alpha=0.28, rasterized=True)
        ax.set(title=instrument, ylabel="log(1 + WMSCI)")
    axes[-1].set_xlabel("MSCI")
    fig.savefig(output / "msci_vs_wmsci.pdf", bbox_inches="tight")
    fig.savefig(output / "msci_vs_wmsci.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fields = (("FPM", "favorable_mid_move_pre_fill"), ("REV", "post_cancel_mid_reversion"), ("ADV", "execution_price_advantage_vs_posture_mid"))
    fig, axes = plt.subplots(3, len(INSTRUMENTS), figsize=(10.5, 8.5), layout="constrained")
    for column, instrument in enumerate(INSTRUMENTS):
        base = diagnostics.filter((pl.col("instrument") == instrument) & pl.col("WMSCI_event").is_finite() & (pl.col("WMSCI_event") > 0))
        for row, (label, field) in enumerate(fields):
            ax = axes[row, column]
            frame = base.filter(pl.col(field).is_not_null() & pl.col(field).is_finite())
            ax.scatter(np.log1p(frame.get_column("WMSCI_event").to_numpy()), frame.get_column(field).to_numpy(), s=5, alpha=0.28, rasterized=True)
            ax.axhline(0, color="0.5", linewidth=0.7, linestyle=":")
            ax.set(title=instrument if row == 0 else "", ylabel=f"{label} (source price units)")
            if row == 2:
                ax.set_xlabel("log(1 + WMSCI)")
    fig.savefig(output / "wmsci_price_diagnostics.pdf", bbox_inches="tight")
    fig.savefig(output / "wmsci_price_diagnostics.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _expected_files() -> set[str]:
    reports = {
        "sequential_counts.csv", "selected_execution_diagnostics.csv", "refreshed_execution_diagnostics.csv",
        "recomputed_cancellation_links.csv", "step_memberships.csv", "shared_order_episodes_reporting_only.csv",
        "publication_funnel.csv", "publication_roles.csv", "publication_score_price_associations.csv",
    }
    tex = {f"{name[:-4]}.tex" for name in ("publication_funnel.csv", "publication_roles.csv", "publication_score_price_associations.csv")}
    figures = {f"{stem}.{suffix}" for stem in ("msci_vs_wmsci", "wmsci_price_diagnostics") for suffix in ("pdf", "png")}
    return reports | tex | figures | {"metadata.json"}


def build_output(output: Path, bundle_root: Path, config_path: Path) -> dict[str, Any]:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    config = _load_contract(config_path)
    all_selected: list[pl.DataFrame] = []
    all_grouped: list[pl.DataFrame] = []
    all_diagnostics: list[pl.DataFrame] = []
    all_cancellations: list[pl.DataFrame] = []
    all_memberships: list[pl.DataFrame] = []
    count_rows: list[dict[str, Any]] = []
    provenance: dict[str, Any] = {}
    for instrument in INSTRUMENTS:
        derived = _derive_instrument(instrument, bundle_root / instrument, config["top_positive_w_fraction"])
        all_selected.append(derived["selected"])
        all_grouped.append(derived["grouped"])
        all_diagnostics.append(derived["diagnostics"])
        all_cancellations.append(derived["cancellations"])
        all_memberships.append(derived["membership"])
        count_rows.append({"instrument": instrument, **derived["counts"]})
        provenance[instrument] = derived["provenance"]
    selected = pl.concat(all_selected).sort(["instrument", "execution_cluster_id"])
    grouped = pl.concat(all_grouped).sort(["instrument", "episode_id"])
    diagnostics = pl.concat(all_diagnostics).sort(["instrument", "execution_cluster_id"])
    cancellations = pl.concat(all_cancellations).sort(["instrument", "partition_id", "cancel_sort_index", "candidate_order_id", "execution_cluster_id"])
    memberships = pl.concat(all_memberships).sort(["instrument", "execution_cluster_id"])
    counts = pl.DataFrame(count_rows).sort("instrument")
    roles = _role_summary(diagnostics, memberships)
    associations = _score_price_associations(diagnostics)
    output.mkdir(parents=True)
    try:
        (output / "sequential_counts.csv").write_text(_csv(counts))
        (output / "selected_execution_diagnostics.csv").write_text(_csv(selected))
        (output / "refreshed_execution_diagnostics.csv").write_text(_csv(diagnostics))
        (output / "recomputed_cancellation_links.csv").write_text(_csv(cancellations))
        (output / "step_memberships.csv").write_text(_csv(memberships))
        (output / "shared_order_episodes_reporting_only.csv").write_text(_csv(grouped))
        (output / "publication_funnel.csv").write_text(_csv(counts))
        (output / "publication_roles.csv").write_text(_csv(roles))
        (output / "publication_score_price_associations.csv").write_text(_csv(associations))
        (output / "publication_funnel.tex").write_text(_table_tex(counts, "Sequential execution-cluster funnel with deterministic exact top-five-percent WMSCI selection.", "tab:h1-age30-withdrawal1-funnel"))
        (output / "publication_roles.tex").write_text(_table_tex(roles, "Execution-anchor and attribution composition. Episode membership is reporting-only and not a selection gate.", "tab:h1-age30-withdrawal1-roles"))
        (output / "publication_score_price_associations.tex").write_text(_table_tex(associations, "Spearman associations of refreshed WMSCI with observed price diagnostics; descriptive pairs only.", "tab:h1-age30-withdrawal1-score-price"))
        _save_figures(diagnostics, output)
        metadata = {
            "schema": SCHEMA,
            "config": str(config_path.resolve()),
            "config_sha256": sha256_file(config_path),
            "contract": config,
            "source_provenance": provenance,
            "generator_sha256": sha256_file(Path(__file__)),
        }
        metadata["outputs"] = {
            path.name: sha256_file(path) for path in sorted(output.iterdir()) if path.is_file()
        }
        if set(metadata["outputs"]) | {"metadata.json"} != _expected_files():
            raise ValueError("output inventory mismatch")
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        return metadata
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise


def verify_output(output: Path, bundle_root: Path, config_path: Path) -> dict[str, Any]:
    output = output.resolve()
    if not output.is_dir() or any(path.is_dir() or path.is_symlink() for path in output.iterdir()) or {path.name for path in output.iterdir()} != _expected_files():
        raise ValueError("output inventory mismatch")
    metadata = json.loads((output / "metadata.json").read_text())
    if metadata.get("schema") != SCHEMA or metadata.get("generator_sha256") != sha256_file(Path(__file__)):
        raise ValueError("output metadata mismatch")
    config = _load_contract(config_path)
    if metadata.get("contract") != config or metadata.get("config_sha256") != sha256_file(config_path):
        raise ValueError("config provenance mismatch")
    expected_hashes = {path.name: sha256_file(path) for path in output.iterdir() if path.name != "metadata.json"}
    if metadata.get("outputs") != expected_hashes:
        raise ValueError("output hash mismatch")
    # Re-derive all data and compare the exact byte representation of every report.
    selected_parts: list[pl.DataFrame] = []
    group_parts: list[pl.DataFrame] = []
    diagnostic_parts: list[pl.DataFrame] = []
    cancellation_parts: list[pl.DataFrame] = []
    membership_parts: list[pl.DataFrame] = []
    count_rows: list[dict[str, Any]] = []
    provenance: dict[str, Any] = {}
    for instrument in INSTRUMENTS:
        derived = _derive_instrument(instrument, bundle_root / instrument, config["top_positive_w_fraction"])
        selected_parts.append(derived["selected"])
        group_parts.append(derived["grouped"])
        diagnostic_parts.append(derived["diagnostics"])
        cancellation_parts.append(derived["cancellations"])
        membership_parts.append(derived["membership"])
        count_rows.append({"instrument": instrument, **derived["counts"]})
        provenance[instrument] = derived["provenance"]
    counts = pl.DataFrame(count_rows).sort("instrument")
    diagnostics = pl.concat(diagnostic_parts).sort(["instrument", "execution_cluster_id"])
    cancellations = pl.concat(cancellation_parts).sort(["instrument", "partition_id", "cancel_sort_index", "candidate_order_id", "execution_cluster_id"])
    memberships = pl.concat(membership_parts).sort(["instrument", "execution_cluster_id"])
    expected = {
        "sequential_counts.csv": _csv(counts),
        "selected_execution_diagnostics.csv": _csv(pl.concat(selected_parts).sort(["instrument", "execution_cluster_id"])),
        "refreshed_execution_diagnostics.csv": _csv(diagnostics),
        "recomputed_cancellation_links.csv": _csv(cancellations),
        "step_memberships.csv": _csv(memberships),
        "shared_order_episodes_reporting_only.csv": _csv(pl.concat(group_parts).sort(["instrument", "episode_id"])),
        "publication_funnel.csv": _csv(counts),
        "publication_roles.csv": _csv(_role_summary(diagnostics, memberships)),
        "publication_score_price_associations.csv": _csv(_score_price_associations(diagnostics)),
    }
    for name, text in expected.items():
        if (output / name).read_text() != text:
            raise ValueError(f"re-derived output mismatch: {name}")
    expected_tex = {
        "publication_funnel.tex": _table_tex(counts, "Sequential execution-cluster funnel with deterministic exact top-five-percent WMSCI selection.", "tab:h1-age30-withdrawal1-funnel"),
        "publication_roles.tex": _table_tex(_role_summary(diagnostics, memberships), "Execution-anchor and attribution composition. Episode membership is reporting-only and not a selection gate.", "tab:h1-age30-withdrawal1-roles"),
        "publication_score_price_associations.tex": _table_tex(_score_price_associations(diagnostics), "Spearman associations of refreshed WMSCI with observed price diagnostics; descriptive pairs only.", "tab:h1-age30-withdrawal1-score-price"),
    }
    for name, text in expected_tex.items():
        if (output / name).read_text() != text:
            raise ValueError(f"re-derived output mismatch: {name}")
    if metadata.get("source_provenance") != provenance:
        raise ValueError("source provenance mismatch")
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, default=DEFAULT_BUNDLE_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--verify-output", action="store_true")
    args = parser.parse_args(argv)
    if args.verify_output:
        verify_output(args.output_dir, args.bundle_root, args.config)
        print("H1_AGE30_WITHDRAWAL1_WMSCI_PRICE_PIPELINE_VERIFIED")
    else:
        build_output(args.output_dir, args.bundle_root, args.config)
        verify_output(args.output_dir, args.bundle_root, args.config)
        print("H1_AGE30_WITHDRAWAL1_WMSCI_PRICE_PIPELINE_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
