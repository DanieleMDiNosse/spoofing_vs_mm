#!/usr/bin/env python3
"""Read-only comparison of isolated NEXI detector bundles at matching cluster IDs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import polars as pl


TABLE_KEYS = {
    "execution_metrics": "execution_cluster_id",
    "candidate_episodes": "episode_id",
    "spoofing_compatible_events": "execution_cluster_id",
    "episode_cluster_members": "execution_cluster_id",
}
COMPARE_FIELDS = (
    "actor_key", "execution_anchor_mode", "episode_id", "candidate_deceptive_order_count_pre",
    "has_matched_deceptive_cancel_window", "spoofing_compatible_sequence",
    "episode_strict_detection", "gate_rapid_matched_withdrawal",
    "gate_small_fill_relative_to_withdrawal", "gate_favorable_pre_fill_move",
    "gate_cancel_anchored_reversion",
)


def audit(legacy: Path, v3: Path) -> dict:
    old = json.loads((legacy / "metadata.json").read_text())
    new = json.loads((v3 / "metadata.json").read_text())
    if old["input_hashes"]["raw_events_sha256"] != new["input_hashes"]["raw_events_sha256"]:
        raise ValueError("Raw input hashes differ")
    if old["input_hashes"]["quote_panel_sha256"] != new["input_hashes"]["quote_panel_sha256"]:
        raise ValueError("Quote panel hashes differ")
    if old["input_hashes"]["empirical_depth_kernel_sha256"] != new["input_hashes"]["empirical_depth_kernel_sha256"]:
        raise ValueError("Depth kernel hashes differ")
    if old["analysis_semantics_version"] != new["analysis_semantics_version"]:
        raise ValueError("Analysis semantics versions differ")
    if new.get("clock_policy_id") != "spoofing_detection_sequence_replay_trade_fill_sequence_cancel_v3_utc_provisional":
        raise ValueError("Unexpected v3 clock policy")
    configs = []
    for metadata in (old, new):
        path = Path(metadata["config"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["config_sha256"]:
            raise ValueError("Copied detector config hash mismatch")
        configs.append(json.loads(path.read_text()))
    removed_policy = configs[1].pop("spoofing_detection_clock_policy_v3", None)
    if removed_policy is None or configs[0] != configs[1]:
        raise ValueError("Detector configs differ beyond the v3 clock policy")
    code_old, code_new = old["scientific_source_hashes"], new["scientific_source_hashes"]
    changed_sources = sorted(k for k in code_old.keys() | code_new.keys() if code_old.get(k) != code_new.get(k))

    result = {
        "legacy": str(legacy), "v3": str(v3),
        "raw_events_sha256": old["input_hashes"]["raw_events_sha256"],
        "analysis_semantics_version": old["analysis_semantics_version"],
        "clock_policy_id_v3": new["clock_policy_id"],
        "config_sha256_legacy": old["config_sha256"],
        "config_sha256_v3": new["config_sha256"],
        "config_difference": "v3 clock policy object only (checked against copied config hashes)",
        "generator_sha256": {"legacy": old["generator_sha256"], "v3": new["generator_sha256"]},
        "output_schema_version": {"legacy": old["output_schema_version"], "v3": new["output_schema_version"]},
        "changed_or_added_scientific_sources": changed_sources,
        "row_counts": {name: {"legacy": old["row_counts"].get(name), "v3": count,
                              "delta": count - old["row_counts"].get(name, 0)}
                       for name, count in new["row_counts"].items()},
        "table_id_overlap": {}, "execution_field_changes_on_common_ids": {},
    }
    for table, key in TABLE_KEYS.items():
        a = pl.scan_parquet(legacy / (table + ".parquet"))
        b = pl.scan_parquet(v3 / (table + ".parquet"))
        # Tables may have repeated cluster IDs when a cluster is present in
        # multiple episode/member rows. Compare sets here, not row counts.
        ak, bk = a.select(key).unique(), b.select(key).unique()
        result["table_id_overlap"][table] = {
            "legacy_unique": ak.select(pl.len()).collect().item(),
            "v3_unique": bk.select(pl.len()).collect().item(),
            "both": ak.join(bk, on=key).select(pl.len()).collect().item(),
            "legacy_only": ak.join(bk, on=key, how="anti").select(pl.len()).collect().item(),
            "v3_only": bk.join(ak, on=key, how="anti").select(pl.len()).collect().item(),
        }
        if table == "episode_cluster_members":
            pair = ["episode_id", "execution_cluster_id"]
            ap, bp = a.select(pair).unique(), b.select(pair).unique()
            result["episode_membership_pairs"] = {
                "both": ap.join(bp, on=pair).select(pl.len()).collect().item(),
                "legacy_only": ap.join(bp, on=pair, how="anti").select(pl.len()).collect().item(),
                "v3_only": bp.join(ap, on=pair, how="anti").select(pl.len()).collect().item(),
            }
    a = pl.scan_parquet(legacy / "execution_metrics.parquet")
    b = pl.scan_parquet(v3 / "execution_metrics.parquet")
    key = "execution_cluster_id"
    for label, frame, metadata in (("legacy", a, old), ("v3", b, new)):
        unique, total, missing = frame.select(pl.col(key).n_unique().alias("unique"), pl.len().alias("total"), pl.col(key).null_count().alias("missing")).collect().row(0)
        if missing or unique != total or total != metadata["row_counts"]["execution_metrics"]:
            raise ValueError(f"{label} execution IDs are null, duplicated or count-mismatched")
    fields = [f for f in COMPARE_FIELDS if f in a.collect_schema() and f in b.collect_schema()]
    shared = a.select([key, *fields]).join(b.select([key, *fields, "cluster_end_replay_time_ns", "cluster_end_fill_anchor_ns"]), on=key, suffix="_v3", validate="1:1")
    result["execution_field_changes_on_common_ids"] = {
        f: shared.filter(~pl.col(f).eq_missing(pl.col(f + "_v3"))).select(pl.len()).collect().item()
        for f in fields
    }
    for flag in ("spoofing_compatible_sequence", "episode_strict_detection"):
        transitions = shared.group_by(flag, flag + "_v3").agg(pl.len().alias("clusters")).collect()
        result[flag + "_cluster_diagnostic_transitions"] = transitions.sort([flag, flag + "_v3"]).to_dicts()
    episodes_a = pl.scan_parquet(legacy / "candidate_episodes.parquet")
    episodes_b = pl.scan_parquet(v3 / "candidate_episodes.parquet")
    for label, frame in (("legacy", episodes_a), ("v3", episodes_b)):
        unique, total, missing = frame.select(pl.col("episode_id").n_unique().alias("unique"), pl.len().alias("total"), pl.col("episode_id").null_count().alias("missing")).collect().row(0)
        if missing or unique != total:
            raise ValueError(f"{label} candidate episode IDs are null or duplicated")
    episode_common = episodes_a.select("episode_id", "spoofing_compatible_episode").join(
        episodes_b.select("episode_id", "spoofing_compatible_episode"), on="episode_id", suffix="_v3", validate="1:1"
    )
    result["common_episode_id_strict_transitions"] = episode_common.group_by(
        "spoofing_compatible_episode", "spoofing_compatible_episode_v3"
    ).agg(pl.len().alias("episodes")).collect().sort(
        "spoofing_compatible_episode", "spoofing_compatible_episode_v3"
    ).to_dicts()
    delta = shared.select((pl.col("cluster_end_replay_time_ns") - pl.col("cluster_end_fill_anchor_ns")).alias("delta_ns"))
    result["cluster_end_replay_minus_fill_ns"] = {
        "min": delta.select(pl.col("delta_ns").min()).collect().item(),
        "median": delta.select(pl.col("delta_ns").median()).collect().item(),
        "max": delta.select(pl.col("delta_ns").max()).collect().item(),
    }
    result["unresolved_caveat"] = (
        "Matched IDs and identical raw/panel/kernel do not isolate clock semantics: "
        "source code and output schema differ; episode membership and eligibility can change. "
        "All field/count changes are confounded and unresolved. No clock-only causal attribution "
        "or eligibility-matched stability rate is asserted. Cluster flags are diagnostics, "
        "not independent episode detections."
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("legacy", type=Path)
    parser.add_argument("v3", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.legacy, args.v3), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
