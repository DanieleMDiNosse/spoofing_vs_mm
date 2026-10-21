"""Verified, chronological W association stage; never applies detector settings."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import polars as pl

from .temporal_calibration_controls import (
    CONTRIBUTION_SCHEMA, W_CURVE_SCHEMA, PROPOSAL_SCHEMA,
    run_shifted_control_w, _curve_from_contributions, propose_w_association,
)
from .temporal_calibration_report import (
    _verified_source, _stage_identity, _publish_stage, _atomic_csv, _atomic_json,
    _load_json, _sha256, _split,
)


def run_w_association(output_root: Path | str, design: Mapping[str, Any], instrument: str,
                      *, dates: list[str] | None = None) -> dict[str, Any]:
    """Replay canonical fills, then sum common-support counts by calendar split.

    Raw inputs are never actor-filtered. Previously completed reference extraction
    remains immutable; W artifacts live in a separate atomic stage. Held-out
    curves are descriptive only and cannot select proposals.
    """
    from scripts.calibrate_temporal_parameters import _load_partition, _OriginalIndices

    root = Path(output_root).resolve()
    manifest, source_hash = _verified_source(root, design, instrument, dates=dates)
    provenance = _load_json(root / "private" / "provenance.json")
    raw_path = Path(provenance["observed"]["observed_inputs"]["raw_events"]["path"])
    dependencies = {name: _sha256(Path(__file__).with_name(name)) for name in (
        "temporal_calibration_w_stage.py", "temporal_calibration_controls.py",
        "empirical_controls_v2.py", "execution_clusters.py",
    )}
    identity = _stage_identity("w_association", source_hash, design, instrument, dependencies)

    def build(stage: Path) -> None:
        by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
        partition_diagnostics = []
        for item in manifest["partitions"]:
            partition = item["identity"]["partition"]
            raw = _load_partition(raw_path, partition)
            indices = _OriginalIndices(int(partition["rows"]), int(partition["index_offset"]))
            from .empirical_controls_v2_io import scan_checkpoint
            from .withdrawal_risk_v2 import CompactRiskResult, SCHEMAS
            folder = root / item["path"]
            reference = _load_json(folder / "reference_manifest.json")
            tables = scan_checkpoint(folder / "risk", cache_key=reference["risk_cache_key"],
                                     schemas=SCHEMAS, provenance=reference["risk_provenance"])
            risk = CompactRiskResult(**{name: frame.collect() for name, frame in tables.items()}, counters={})
            cap = _load_json(folder / "reference_diagnostics.json")["age_cap_seconds"]
            result = run_shifted_control_w(raw, instrument=instrument, original_sort_indices=indices,
                                          windows_seconds=design["withdrawal_windows_seconds"],
                                          _prepared_risk=(risk, cap))
            split = _split(instrument, item["event_date"])
            by_split[split].extend(result.contributions.to_dicts())
            partition_diagnostics.append({"event_date": item["event_date"], "split": split,
                                          "cluster_count": result.clusters.height,
                                          "provenance": result.provenance})
        curves, proposals, contributions = [], [], []
        expected_splits = ("calibration", "retrospective_validation") if instrument == "RISANAMENTO" else ("exploratory_single_day",)
        windows = design["withdrawal_windows_seconds"]
        for split in expected_splits:
            rows = by_split.get(split, [])
            # The same groups must contribute at every search horizon. Changing
            # the supported population cannot be mistaken for a timing curve.
            keys = ("instrument", "partition_id", "event_date", "actor_key", "side", "time_band_id", "execution_anchor_mode")
            supported: dict[tuple[Any, ...], set[float]] = defaultdict(set)
            for row in rows:
                if row["contribution_status"] == "included":
                    supported[tuple(row[key] for key in keys)].add(row["window_seconds"])
            for row in rows:
                key = tuple(row[name] for name in keys)
                if row["contribution_status"] == "included" and supported[key] != set(windows):
                    row = {**row, "contribution_status": "excluded_common_support",
                           "exclusion_reason": "group_not_supported_at_all_search_windows",
                           "expected_background_count": None, "excess_count": None}
                contributions.append({**row, "split": split})
            frame = pl.DataFrame([row for row in contributions if row["split"] == split],
                                 schema={**CONTRIBUTION_SCHEMA, "split": pl.String}).drop("split")
            curve = _curve_from_contributions(frame) if frame.height else pl.DataFrame(schema=W_CURVE_SCHEMA)
            # Explicit empty client/firm and role strata are not borrowed.
            present = {(row["identity_level"], row["execution_anchor_mode"]) for row in curve.to_dicts()}
            empty = [{"instrument": instrument, "identity_level": level, "execution_anchor_mode": role,
                      "window_seconds": float(window), "excess_count": None, "included_group_count": 0,
                      "excluded_group_count": 0, "supported_risk_seconds": 0.0,
                      "aggregation": "sum_group_excess_counts"}
                     for level in ("client_original", "firm") for role in ("passive", "aggressive")
                     if (level, role) not in present for window in windows]
            if empty:
                curve = pl.concat([curve, pl.DataFrame(empty, schema=W_CURVE_SCHEMA)])
            curves.extend({**row, "split": split} for row in curve.to_dicts())
            if split != "retrospective_validation":
                proposals.extend({**row, "split": split} for row in propose_w_association(curve).to_dicts())
        _atomic_csv(stage / "curve.csv", curves, {**W_CURVE_SCHEMA, "split": pl.String})
        _atomic_csv(stage / "proposals.csv", proposals, {**PROPOSAL_SCHEMA, "split": pl.String})
        pl.DataFrame(contributions, schema={**CONTRIBUTION_SCHEMA, "split": pl.String}).write_parquet(stage / "private_contributions.parquet")
        _atomic_json(stage / "diagnostics.json", {"proposal_only": True,
            "population": "all_canonical_executions_100ms", "partitions": partition_diagnostics,
            "proposal_selection": "calibration_only_fixed_plateau_rule",
            "holdout": "descriptive_curve_no_refit", "search_window_support": "same_groups_at_all_windows"})
        _, final_hash = _verified_source(root, design, instrument, dates=dates)
        if final_hash != source_hash:
            raise ValueError("reference source changed during W association stage")

    resumed = _publish_stage(root, "w_association", identity, build)
    return {"status": "w_association_complete", "resume_verified": resumed}
