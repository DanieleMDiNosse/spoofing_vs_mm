"""Verified multi-day W uncertainty and broad physical-delay tau proposals.

The stage is proposal-only.  It resamples complete available-day contribution rows,
recomputes W on every replicate, describes the frozen RISANAMENTO holdout only at
an actually evaluated grid point, and derives tau only from a broad unselected
physical-cancellation reference when the corresponding W association is identified.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import polars as pl

from .temporal_calibration import tau_from_reference
from .temporal_calibration_controls import CONTRIBUTION_SCHEMA, PROPOSAL_SCHEMA, W_CURVE_SCHEMA, _curve_from_contributions, propose_w_association
from .temporal_calibration_inputs import frozen_design, validate_design

_STAGE = "uncertainty_diagnostics"
_SCHEMA_VERSION = "temporal_calibration_uncertainty_diagnostics_v1"
_KEYS = ("instrument", "identity_level", "execution_anchor_mode")
_PUBLIC_ALIAS_PREFIX = "actor_"

_BOOTSTRAP_SCHEMA = {
    "instrument": pl.String, "identity_level": pl.String, "execution_anchor_mode": pl.String,
    "available_day_count": pl.Int64, "supported_day_count": pl.Int64, "calendar_day_count": pl.Int64,
    "replicates": pl.Int64, "identified_replicates": pl.Int64,
    "identified_fraction": pl.Float64, "bootstrap_w_ci_lower_seconds": pl.Float64,
    "bootstrap_w_ci_upper_seconds": pl.Float64, "status": pl.String, "inference_note": pl.String,
}
_SENSITIVITY_SCHEMA = {
    "instrument": pl.String, "identity_level": pl.String, "execution_anchor_mode": pl.String,
    "omitted_actor_alias": pl.String, "retained_day_count": pl.Int64,
    "proposed_window_seconds": pl.Float64, "proposal_status": pl.String,
    "search_cap_binding": pl.Boolean, "status": pl.String,
}
_VALIDATION_SCHEMA = {
    "instrument": pl.String, "identity_level": pl.String, "execution_anchor_mode": pl.String,
    "chosen_window_seconds": pl.Float64, "calibration_proposal_status": pl.String,
    "heldout_excess_count": pl.Float64, "heldout_included_group_count": pl.Int64,
    "heldout_supported_risk_seconds": pl.Float64, "status": pl.String, "interpolation": pl.String,
}
_DECAY_SCHEMA = {
    "instrument": pl.String, "identity_level": pl.String, "execution_anchor_mode": pl.String,
    "direction": pl.String, "rho": pl.Float64, "reference_q90_delay_ns": pl.Int64,
    "tau_seconds": pl.Float64, "reference_pair_count": pl.Int64, "physical_cancellation_count": pl.Int64,
    "association_window_seconds": pl.Float64, "status": pl.String, "model": pl.String,
}


def _require_finite_rows(rows: Iterable[Mapping[str, Any]]) -> None:
    for row in rows:
        if any(isinstance(value, float) and not math.isfinite(value) for value in row.values()):
            raise ValueError("non-finite numeric data prohibited")


def _supported_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep common-supported groups evaluated at every window in the stratum."""
    _require_finite_rows(rows)
    windows = {row["window_seconds"] for row in rows}
    grain = ("partition_id", "event_date", "actor_key", "side", "time_band_id")
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if (row.get("contribution_status") == "included"
                and row.get("proposal_eligible") is True and not row.get("support_anomaly")
                and row.get("excess_count") is not None
                and row.get("expected_background_count") is not None
                and (row.get("observed_risk_seconds") or 0) > 0):
            groups[tuple(row[name] for name in grain)].append(row)
    return [row for group in groups.values()
            if {row["window_seconds"] for row in group} == windows for row in group]


def _frame(rows: list[dict[str, Any]], schema: Mapping[str, pl.DataType]) -> pl.DataFrame:
    _require_finite_rows(rows)
    if not rows:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(rows, infer_schema_length=None).select(
        pl.col(name).cast(dtype, strict=False) for name, dtype in schema.items()
    )


def _day(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return date.fromisoformat(value)
    raise ValueError("event_date must be a calendar date")


def _higher(values: Iterable[float], q: float) -> float | None:
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        return None
    return ordered[math.ceil((len(ordered) - 1) * q)]


def _strata(rows: Iterable[Mapping[str, Any]]) -> list[tuple[str, str, str]]:
    return sorted({tuple(str(row[name]) for name in _KEYS) for row in rows})


def _proposal_for(rows: list[dict[str, Any]], key: tuple[str, str, str]) -> dict[str, Any]:
    selected = [row for row in rows if tuple(str(row[name]) for name in _KEYS) == key]
    if not selected:
        return {"proposed_window_seconds": None, "proposal_status": "insufficient_support", "search_cap_binding": False}
    curve = _curve_from_contributions(_frame(selected, CONTRIBUTION_SCHEMA))
    proposed = propose_w_association(curve)
    found = proposed.filter(
        (pl.col("instrument") == key[0]) & (pl.col("identity_level") == key[1])
        & (pl.col("execution_anchor_mode") == key[2])
    )
    return found.row(0, named=True) if found.height else {"proposed_window_seconds": None, "proposal_status": "insufficient_support", "search_cap_binding": False}


def bootstrap_w_summary(contributions: pl.DataFrame, *, block_days: int, replicates: int, seed: int,
                        min_days: int) -> pl.DataFrame:
    """Moving-block descriptive uncertainty from complete contribution rows.

    The available-day calendar is the sorted observed calendar: blocks are
    contiguous in that list, not in civil-day time.  Each selected date contributes
    all of its original rows, including duplicate draws in a replicate.
    """
    from .temporal_calibration import moving_day_replicates
    if not isinstance(contributions, pl.DataFrame):
        raise ValueError("contributions must be a Polars DataFrame")
    rows = contributions.to_dicts()
    output: list[dict[str, Any]] = []
    for key in _strata(rows):
        selected = [row for row in rows if tuple(str(row[name]) for name in _KEYS) == key]
        dates = sorted({_day(row["event_date"]) for row in selected})
        supported_days = len({_day(row["event_date"]) for row in _supported_rows(selected)})
        base = {"instrument": key[0], "identity_level": key[1], "execution_anchor_mode": key[2],
                "available_day_count": supported_days, "supported_day_count": supported_days,
                "calendar_day_count": len(dates), "replicates": 0, "identified_replicates": 0,
                "identified_fraction": None, "bootstrap_w_ci_lower_seconds": None,
                "bootstrap_w_ci_upper_seconds": None,
                "inference_note": "moving_available_day_resampling_descriptive_not_iid"}
        if supported_days < min_days:
            output.append(base | {"status": "insufficient_supported_days"})
            continue
        by_day: dict[date, list[dict[str, Any]]] = defaultdict(list)
        for row in selected:
            by_day[_day(row["event_date"])].append(row)
        windows: list[float] = []
        identified = 0
        for indices in moving_day_replicates(dates, block_days=block_days, replicates=replicates, seed=seed):
            sampled = [row for index in indices for row in by_day[dates[index]]]
            proposal = _proposal_for(sampled, key)
            if proposal["proposal_status"] == "exploratory_association_scale":
                identified += 1
                windows.append(float(proposal["proposed_window_seconds"]))
        output.append(base | {"replicates": replicates, "identified_replicates": identified,
                              "identified_fraction": identified / replicates,
                              "bootstrap_w_ci_lower_seconds": _higher(windows, .025),
                              "bootstrap_w_ci_upper_seconds": _higher(windows, .975),
                              "status": "estimated" if identified else "not_identified"})
    return _frame(output, _BOOTSTRAP_SCHEMA).sort(list(_KEYS))


def _alias(actor: str, salt: str) -> str:
    digest = hashlib.sha256(f"{salt}\0{actor}".encode()).hexdigest()[:16]
    return f"{_PUBLIC_ALIAS_PREFIX}{digest}"


def leave_one_actor_out_w(contributions: pl.DataFrame, *, minimum_days: int, alias_salt: str) -> pl.DataFrame:
    """Recompute sums and plateau proposal after each actor omission; never emit IDs."""
    rows = contributions.to_dicts()
    output: list[dict[str, Any]] = []
    for key in _strata(rows):
        selected = [row for row in rows if tuple(str(row[name]) for name in _KEYS) == key]
        for actor in sorted({str(row["actor_key"]) for row in selected if row.get("actor_key") is not None}):
            retained = [row for row in selected if row.get("actor_key") != actor]
            retained_days = len({_day(row["event_date"]) for row in _supported_rows(retained)})
            proposal = _proposal_for(retained, key) if retained_days >= minimum_days else {
                "proposed_window_seconds": None, "proposal_status": "insufficient_available_days", "search_cap_binding": False,
            }
            output.append({"instrument": key[0], "identity_level": key[1], "execution_anchor_mode": key[2],
                           "omitted_actor_alias": _alias(actor, alias_salt), "retained_day_count": retained_days,
                           "proposed_window_seconds": proposal["proposed_window_seconds"],
                           "proposal_status": proposal["proposal_status"],
                           "search_cap_binding": proposal["search_cap_binding"],
                           "status": "descriptive_leave_one_actor_out"})
    return _frame(output, _SENSITIVITY_SCHEMA).sort(list(_KEYS) + ["omitted_actor_alias"])


def frozen_w_validation(proposals: pl.DataFrame, heldout_curve: pl.DataFrame) -> pl.DataFrame:
    """Evaluate holdout at selected W only if that exact grid point was emitted."""
    _require_finite_rows(proposals.to_dicts())
    _require_finite_rows(heldout_curve.to_dicts())
    output: list[dict[str, Any]] = []
    for proposal in proposals.to_dicts():
        if proposal.get("split") != "calibration":
            continue
        key = tuple(str(proposal[name]) for name in _KEYS)
        window = proposal.get("proposed_window_seconds")
        base = {"instrument": key[0], "identity_level": key[1], "execution_anchor_mode": key[2],
                "chosen_window_seconds": window, "calibration_proposal_status": proposal.get("proposal_status"),
                "heldout_excess_count": None, "heldout_included_group_count": None,
                "heldout_supported_risk_seconds": None, "interpolation": "prohibited"}
        if proposal.get("proposal_status") != "exploratory_association_scale" or not isinstance(window, (int, float)):
            output.append(base | {"status": "calibration_w_not_identified"})
            continue
        matches = [row for row in heldout_curve.to_dicts() if row.get("split") == "retrospective_validation"
                   and tuple(str(row[name]) for name in _KEYS) == key and row.get("window_seconds") == float(window)]
        if len(matches) != 1:
            output.append(base | {"status": "heldout_grid_point_unavailable"})
            continue
        row = matches[0]
        output.append(base | {"heldout_excess_count": row.get("excess_count"),
                              "heldout_included_group_count": row.get("included_group_count"),
                              "heldout_supported_risk_seconds": row.get("supported_risk_seconds"),
                              "status": "descriptive_holdout_actual_grid"})
    return _frame(output, _VALIDATION_SCHEMA).sort(list(_KEYS))


def _control_gate(rows: list[dict[str, Any]], window: float) -> str | None:
    at_window = [row for row in rows if row.get("window_seconds") == float(window)]
    if not at_window or any(row.get("contribution_status") != "included" for row in at_window):
        return "control_domain_tail_unidentified"
    expected = [row.get("expected_background_count") for row in at_window]
    observed = [row.get("observed_count") for row in at_window]
    if any(not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in expected + observed):
        return "control_domain_tail_unidentified"
    if sum(float(value) for value in expected) >= sum(float(value) for value in observed):
        return "control_background_dominant"
    return None


def decay_proposals(reference: pl.DataFrame, proposals: pl.DataFrame, contributions: pl.DataFrame,
                    *, tau_weights: Sequence[float]) -> pl.DataFrame:
    """Calculate q90-derived tau only behind W/control-domain identification gates."""
    ref_rows, proposal_rows, control_rows = reference.to_dicts(), proposals.to_dicts(), contributions.to_dicts()
    _require_finite_rows(ref_rows + proposal_rows + control_rows)
    ref_rows = [row for row in ref_rows if row.get("split") != "retrospective_validation"]
    proposal_by_key = {tuple(str(row[name]) for name in _KEYS): row for row in proposal_rows
                       if row.get("split", "calibration") == "calibration"}
    keys = sorted(set(proposal_by_key) | set(_strata(ref_rows)))
    output: list[dict[str, Any]] = []
    for key in keys:
        proposal = proposal_by_key.get(key, {})
        window = proposal.get("proposed_window_seconds")
        status = None
        if proposal.get("proposal_status") != "exploratory_association_scale" or not isinstance(window, (int, float)):
            status = "w_not_identified"
        selected_controls = [row for row in control_rows if tuple(str(row[name]) for name in _KEYS) == key]
        if status is None:
            status = _control_gate(selected_controls, float(window))
        selected_reference = [row for row in ref_rows if tuple(str(row[name]) for name in _KEYS) == key]
        directions = sorted({str(row.get("direction")) for row in selected_reference}) or ["own_side", "opposite_side"]
        for direction in directions:
            values = [int(row["delay_ns"]) for row in selected_reference if row.get("direction") == direction
                      and type(row.get("delay_ns")) is int and 0 < int(row["delay_ns"]) <= 20_000_000_000]
            q90 = _higher(values, .9)
            physical = len({(row.get("physical_order_key"), row.get("withdrawal_event_id"))
                            for row in selected_reference if row.get("direction") == direction})
            capped = any(type(row.get("delay_ns")) is int and row["delay_ns"] >= 20_000_000_000
                         for row in selected_reference if row.get("direction") == direction)
            local_status = status or ("reference_search_cap_binding" if capped else
                                      "no_decay_reference" if q90 is None else "tau_proposed_descriptive")
            for rho in tau_weights:
                tau = tau_from_reference(int(q90), rho) if local_status == "tau_proposed_descriptive" else None
                output.append({"instrument": key[0], "identity_level": key[1], "execution_anchor_mode": key[2],
                               "direction": direction, "rho": float(rho), "reference_q90_delay_ns": q90,
                               "tau_seconds": tau, "reference_pair_count": len(values),
                               "physical_cancellation_count": physical, "association_window_seconds": window,
                               "status": local_status, "model": "q90_reference_weight_no_distribution_fit"})
            output.append({"instrument": key[0], "identity_level": key[1], "execution_anchor_mode": key[2],
                           "direction": direction, "rho": 1.0, "reference_q90_delay_ns": q90,
                           "tau_seconds": None, "reference_pair_count": len(values),
                           "physical_cancellation_count": physical, "association_window_seconds": window,
                           "status": "no_decay_design_reference" if local_status == "tau_proposed_descriptive" else local_status,
                           "model": "no_decay"})
    return _frame(output, _DECAY_SCHEMA).sort(list(_KEYS) + ["direction", "rho"])


def _atomic_csv(path: Path, frame: pl.DataFrame) -> None:
    _require_finite_rows(frame.to_dicts())
    temporary = path.with_name(f".{path.name}.tmp")
    frame.write_csv(temporary)
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1_048_576), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def stage_identity(source_manifest_sha256: str, design: Mapping[str, Any], instrument: str,
                   w_stage_manifest_sha256: str) -> dict[str, Any]:
    return {"schema_version": _SCHEMA_VERSION, "stage": _STAGE,
            "source_manifest_sha256": source_manifest_sha256, "instrument": instrument,
            "design_sha256": _json_hash(dict(design)), "uncertainty_module_sha256": _sha256(Path(__file__)),
            "w_stage_manifest_sha256": w_stage_manifest_sha256}


def verify_uncertainty_stage(path: Path | str, identity: Mapping[str, Any]) -> dict[str, Any]:
    from .temporal_calibration_report import _safe_stage_artifact
    target = Path(path)
    manifest_path = _safe_stage_artifact(target, "stage_manifest.json")
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("uncertainty diagnostics stage manifest required") from exc
    if (set(manifest) != {"schema_version", "complete", "identity", "artifacts"}
            or manifest["schema_version"] != _SCHEMA_VERSION or manifest["complete"] is not True
            or manifest["identity"] != dict(identity) or not isinstance(manifest["artifacts"], dict)):
        raise ValueError("uncertainty diagnostics stage identity mismatch")
    required = {"bootstrap_summary.csv", "actor_sensitivity.csv", "frozen_validation.csv", "decay_proposals.csv"}
    if set(manifest["artifacts"]) != required:
        raise ValueError("uncertainty diagnostics artifact set mismatch")
    for name, digest in manifest["artifacts"].items():
        artifact = _safe_stage_artifact(target, name)
        if not isinstance(digest, str) or not artifact.is_file() or _sha256(artifact) != digest:
            raise ValueError("uncertainty diagnostics artifact hash mismatch")
    return manifest


# Parent integration uses the generic stage verifier naming convention.
_stage_identity = stage_identity
_verify_stage = verify_uncertainty_stage


def _publish(root: Path, identity: Mapping[str, Any], build) -> bool:
    destination = root / _STAGE
    if destination.exists():
        verify_uncertainty_stage(destination, identity)
        return True
    temporary = Path(tempfile.mkdtemp(prefix=f".{_STAGE}.incomplete-", dir=root))
    try:
        build(temporary)
        artifacts = {path.relative_to(temporary).as_posix(): _sha256(path)
                     for path in sorted(temporary.iterdir()) if path.is_file()}
        if set(artifacts) != {"bootstrap_summary.csv", "actor_sensitivity.csv", "frozen_validation.csv", "decay_proposals.csv"}:
            raise ValueError("uncertainty diagnostics did not produce required artifacts")
        (temporary / "stage_manifest.json").write_text(json.dumps({"schema_version": _SCHEMA_VERSION, "complete": True,
            "identity": dict(identity), "artifacts": artifacts}, sort_keys=True, indent=2, allow_nan=False) + "\n")
        verify_uncertainty_stage(temporary, identity)
        os.rename(temporary, destination)
        return False
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _reference_delays(root: Path, manifest: Mapping[str, Any], instrument: str, raw_path: Path) -> pl.DataFrame:
    """Build broad, unselected execution-to-physical-cancellation delay pairs.

    Every canonical 100-ms cluster role is retained; no candidate gate or nearest
    cancellation/winner selection is used.  A pair must remain inside the same
    compact-risk coverage epoch, preserving the physical cancellation provenance.
    """
    from scripts.calibrate_temporal_parameters import _OriginalIndices, _load_partition
    from .empirical_controls_v2_io import scan_checkpoint
    from .temporal_calibration_controls import _collect_canonical_clusters
    from .withdrawal_risk_v2 import CompactRiskResult, SCHEMAS
    rows: list[dict[str, Any]] = []
    for item in manifest["partitions"]:
        partition = item["identity"]["partition"]
        raw = _load_partition(raw_path, partition)
        indices = _OriginalIndices(int(partition["rows"]), int(partition["index_offset"]))
        folder = root / item["path"]
        reference = json.loads((folder / "reference_manifest.json").read_text())
        tables = scan_checkpoint(folder / "risk", cache_key=reference["risk_cache_key"], schemas=SCHEMAS,
                                 provenance=reference["risk_provenance"])
        risk = CompactRiskResult(**{name: frame.collect() for name, frame in tables.items()}, counters={})
        clusters = _collect_canonical_clusters(raw, instrument=instrument, original_sort_indices=indices).to_dicts()
        withdrawals = [row for row in risk.withdrawal_events.to_dicts() if row.get("transition_reason") == "cancellation"]
        epochs = risk.coverage_epochs.to_dicts()
        for withdrawal in withdrawals:
            from .temporal_calibration_report import _split
            split = _split(instrument, withdrawal["event_date"])
            if split == "retrospective_validation":
                continue
            event_ns = withdrawal.get("event_ns")
            if type(event_ns) is not int:
                continue
            for cluster in clusters:
                if any(cluster.get(name) != withdrawal.get(name) for name in ("instrument", "partition_id", "event_date", "actor_key")):
                    continue
                fill_ns = cluster.get("fill_anchor_ns")
                if type(fill_ns) is not int or not 0 < event_ns - fill_ns <= 20_000_000_000:
                    continue
                epoch = next((row for row in epochs if row.get("coverage_epoch_id") == withdrawal.get("coverage_epoch_id")
                              and row.get("partition_id") == withdrawal.get("partition_id")
                              and row.get("event_date") == withdrawal.get("event_date")
                              and row.get("instrument") == withdrawal.get("instrument")
                              and type(row.get("start_ns")) is int and type(row.get("end_ns")) is int
                              and row["start_ns"] <= fill_ns <= event_ns < row["end_ns"]), None)
                if epoch is None:
                    continue
                direction = "own_side" if cluster.get("event_side") == withdrawal.get("side") else "opposite_side"
                rows.append({"instrument": instrument, "identity_level": cluster["identity_level"],
                             "event_date": withdrawal["event_date"], "split": split,
                             "execution_anchor_mode": cluster["execution_anchor_mode"], "direction": direction,
                             "delay_ns": event_ns - fill_ns, "physical_order_key": withdrawal.get("physical_order_key"),
                             "withdrawal_event_id": withdrawal.get("withdrawal_event_id")})
    return pl.DataFrame(rows) if rows else pl.DataFrame(schema={"instrument": pl.String, "identity_level": pl.String,
        "execution_anchor_mode": pl.String, "direction": pl.String, "delay_ns": pl.Int64,
        "event_date": pl.Date, "split": pl.String,
        "physical_order_key": pl.String, "withdrawal_event_id": pl.String})


def run_uncertainty_diagnostics(output_root: Path | str, design: Mapping[str, Any], instrument: str,
                                *, dates: list[str] | None = None) -> dict[str, Any]:
    """Publish verified W uncertainty, frozen holdout, and association-gated tau proposals."""
    validate_design(design)
    if _json_hash(dict(design)) != _json_hash(frozen_design()):
        raise ValueError("design differs from frozen temporal calibration design")
    root = Path(output_root).resolve()
    from .temporal_calibration_report import _load_json, _stage_identity as report_stage_identity, _verified_source, _verify_stage as verify_report_stage
    manifest, source_hash = _verified_source(root, design, instrument, dates=dates)
    dependencies = {name: _sha256(Path(__file__).with_name(name)) for name in (
        "temporal_calibration_w_stage.py", "temporal_calibration_controls.py", "empirical_controls_v2.py", "execution_clusters.py",
    )}
    w_identity = report_stage_identity("w_association", source_hash, design, instrument, dependencies)
    verify_report_stage(root / "w_association", w_identity)
    w_digest = _sha256(root / "w_association" / "stage_manifest.json")
    identity = stage_identity(source_hash, design, instrument, w_digest)

    def build(stage: Path) -> None:
        contributions = pl.read_parquet(root / "w_association" / "private_contributions.parquet")
        curves = pl.read_csv(root / "w_association" / "curve.csv", try_parse_dates=True)
        proposals = pl.read_csv(root / "w_association" / "proposals.csv", try_parse_dates=True)
        calibration = contributions.filter(pl.col("split") == "calibration")
        bootstrap = bootstrap_w_summary(calibration, block_days=int(design["bootstrap_block_days"]),
                                        replicates=int(design["bootstrap_replicates"]), seed=int(design["bootstrap_seed"]),
                                        min_days=int(design["bootstrap_min_supported_days"]))
        sensitivity = leave_one_actor_out_w(calibration, minimum_days=int(design["bootstrap_min_supported_days"]),
                                            alias_salt=source_hash)
        validation = frozen_w_validation(proposals, curves)
        provenance = _load_json(root / "private" / "provenance.json")
        raw_path = Path(provenance["observed"]["observed_inputs"]["raw_events"]["path"])
        reference = _reference_delays(root, manifest, instrument, raw_path)
        decay = decay_proposals(reference, proposals, calibration, tau_weights=tuple(design["tau_reference_weights"]))
        _atomic_csv(stage / "bootstrap_summary.csv", bootstrap)
        _atomic_csv(stage / "actor_sensitivity.csv", sensitivity)
        _atomic_csv(stage / "frozen_validation.csv", validation)
        _atomic_csv(stage / "decay_proposals.csv", decay)
        _, final_source_hash = _verified_source(root, design, instrument, dates=dates)
        if final_source_hash != source_hash:
            raise ValueError("reference source changed during uncertainty diagnostics")

    resumed = _publish(root, identity, build)
    return {"status": "uncertainty_diagnostics_complete", "resume_verified": resumed}
