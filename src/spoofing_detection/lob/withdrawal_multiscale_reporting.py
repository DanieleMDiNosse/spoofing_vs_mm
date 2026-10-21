"""Private descriptive tables; no publication labels or filtering by priority."""
from __future__ import annotations

import polars as pl

KEYS = ["instrument", "partition_id", "actor_key", "execution_cluster_id"]
CASE = ["instrument", "horizon_ns", "age_policy"]
FLAGS = ["priority_top_w_self", "priority_top_w_market", "priority_top_j_given_self", "priority_top_j_given_market"]


def build_tables(population: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Describe full cells and support on a frozen union of raw candidates."""
    for flag in ["is_candidate", "complete_grid", "market_score_supported"] + FLAGS:
        if population.schema.get(flag) != pl.Boolean or population[flag].null_count():
            raise ValueError(f"{flag} must be a non-null Boolean")
    full = KEYS + ["horizon_ns", "age_policy"]
    if population.select(full).unique().height != population.height:
        raise ValueError("duplicate report cell identity")
    counts = population.group_by(CASE).agg(
        pl.len().alias("total_execution_rows"),
        pl.col("is_candidate").sum().alias("raw_candidates"),
        (pl.col("WMSCI_self").is_finite() & (pl.col("WMSCI_self") > 0)).sum().alias("positive_finite_self"),
        (pl.col("WMSCI_market").is_finite() & (pl.col("WMSCI_market") > 0)).sum().alias("positive_finite_market"),
        (pl.col("is_candidate") & pl.col("complete_grid") & pl.col("J").is_finite()).sum().alias("complete_J_candidates"),
        (pl.col("is_candidate") & (~pl.col("complete_grid") | ~pl.col("J").is_finite().fill_null(False))).sum().alias("incomplete_J_candidates"),
        (pl.col("is_candidate") & (pl.col("age_weight_underflow_count") > 0)).sum().alias("age_underflow_candidates"),
        (~pl.col("market_score_supported")).sum().alias("unsupported_market"),
    ).sort(CASE)
    priorities = population.group_by(CASE).agg(
        [pl.col(name).sum().alias(name) for name in FLAGS] + [
            (pl.col("priority_top_w_self") & pl.col("priority_top_w_market")).sum().alias("top_w_overlap"),
            (pl.col("is_candidate") & pl.col("WMSCI_self").is_finite() & (pl.col("WMSCI_self") > 0)).sum().alias("self_ranking_denominator"),
            (pl.col("is_candidate") & pl.col("WMSCI_market").is_finite() & (pl.col("WMSCI_market") > 0)).sum().alias("market_ranking_denominator"),
            (pl.col("priority_top_w_self") & pl.col("complete_grid") & pl.col("J").is_finite()).sum().alias("conditional_j_self_denominator"),
            (pl.col("priority_top_w_market") & pl.col("complete_grid") & pl.col("J").is_finite()).sum().alias("conditional_j_market_denominator"),
        ]
    ).sort(CASE)
    support = []
    for group in population.partition_by(["instrument", "age_policy"], maintain_order=True):
        instrument, policy = group.select("instrument", "age_policy").row(0)
        horizons = sorted(group["horizon_ns"].unique().to_list())
        union = group.filter(pl.col("is_candidate")).select(KEYS).unique()
        tracked = group.join(union, on=KEYS, how="semi")
        complete = tracked.filter(pl.col("complete_grid") & pl.col("J").is_finite())
        common = complete.group_by(KEYS).agg(pl.col("horizon_ns").n_unique().alias("supported_horizons"))
        common_count = common.filter(pl.col("supported_horizons") == len(horizons)).height
        for horizon in horizons:
            support.append({"instrument": instrument, "age_policy": policy, "horizon_ns": horizon,
                "union_candidates": union.height,
                "complete_support_at_horizon": complete.filter(pl.col("horizon_ns") == horizon).height,
                "common_complete_support": common_count})
    support_frame = pl.DataFrame(support, schema={"instrument": pl.String, "age_policy": pl.String,
        "horizon_ns": pl.Int64, "union_candidates": pl.Int64,
        "complete_support_at_horizon": pl.Int64, "common_complete_support": pl.Int64}).sort(CASE)
    candidates = population.filter(pl.col("is_candidate")).sort(full)
    fields = full + ["execution_quantity", "opposite_market_visible_qty_pre_topN", "candidate_qty_raw",
        "withdrawal_qty_raw", "candidate_qty_age_weighted", "withdrawal_qty_age_weighted",
        "removal_share_age_weighted", "candidate_market_share_age_weighted", "WMSCI_self",
        "WMSCI_market", "J", "complete_grid", "profile_status", "age_weight_underflow_count"]
    return {"population_counts": counts, "priority_counts": priorities,
        "common_support_counts": support_frame, "score_components": candidates.select(fields),
        "review": candidates}
