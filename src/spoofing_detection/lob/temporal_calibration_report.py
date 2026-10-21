"""Verified, proposal-only summaries for temporal calibration reference extracts.

This module consumes the immutable reference partitions produced by the extraction
stage.  It never replays raw events, changes the extraction manifest, or promotes
parameters into detector configuration.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import date, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable, Mapping

import polars as pl

from .temporal_calibration import age_time_quantile_ns, higher_quantile_ns, moving_day_replicates, survival_quantile_ns
from .temporal_calibration_inputs import frozen_design, validate_design

_LEVELS = ("client_original", "firm")
_ROLES = ("passive", "aggressive")
_QUANTILES = (0.5, 0.9, 0.95, 0.99)
_STAGE_SCHEMA = "temporal_calibration_summary_stage_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1_048_576), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    def reject(value: str) -> None:
        raise ValueError(f"non-finite JSON constant forbidden: {value}")
    try:
        value = json.loads(path.read_text(), parse_constant=reject,
                           object_pairs_hook=lambda pairs: _unique_object(pairs, path))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _unique_object(pairs: list[tuple[str, Any]], path: Path) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key in {path}")
        value[key] = item
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _atomic_csv(path: Path, rows: list[dict[str, Any]], schema: Mapping[str, pl.DataType]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    pl.DataFrame(rows, schema=schema, strict=False).write_csv(temporary)
    os.replace(temporary, path)


def _strict_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return date.fromisoformat(value)
    raise ValueError("event_date must be an ISO calendar date")


def _split(instrument: str, value: Any) -> str:
    day = _strict_date(value)
    if instrument == "RISANAMENTO":
        return "calibration" if day <= date(2024, 9, 30) else "retrospective_validation"
    return "exploratory_single_day"


def _verified_source(output_root: Path, design: Mapping[str, Any], instrument: str,
                     *, dates: list[str] | None = None) -> tuple[dict[str, Any], str]:
    """Verify the extract graph through its owning verifier, imported only here."""
    validate_design(design)
    if _json_hash(dict(design)) != _json_hash(frozen_design()):
        raise ValueError("design differs from frozen temporal calibration design")
    manifest_path = output_root / "manifest.json"
    manifest = _load_json(manifest_path)
    identity = manifest.get("identity")
    if not isinstance(identity, Mapping) or identity.get("instrument") != instrument:
        raise ValueError("extract manifest instrument identity mismatch")
    # Rebuild the expected identity from live certified inputs, never from the
    # manifest being checked. Partial pilots require caller-selected dates.
    from scripts.calibrate_temporal_parameters import (
        verify_complete_checkpoint, preflight, checkpoint_identity, extraction_identity,
    )
    project_root = next((parent for parent in output_root.parents
                         if (parent / "configs" / "spoofing_empirical_controls.json").is_file()),
                        Path(__file__).resolve().parents[3])
    observed = preflight(design, instrument, output_root, dates=dates,
                         allow_existing=True, project_root=project_root)
    expected = extraction_identity(checkpoint_identity(design, observed), observed, is_pilot=dates is not None)
    verified = verify_complete_checkpoint(output_root, expected)
    return verified, _sha256(manifest_path)


def _stage_identity(stage: str, source_hash: str, design: Mapping[str, Any], instrument: str,
                    dependencies: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Bind reusable stages to the estimator source and immutable predecessors."""
    return {"schema_version": _STAGE_SCHEMA, "stage": stage, "source_manifest_sha256": source_hash,
            "design_sha256": _json_hash(dict(design)), "instrument": instrument,
            "report_module_sha256": _sha256(Path(__file__)),
            "dependency_stage_manifest_sha256": dict(sorted((dependencies or {}).items()))}


def _safe_stage_artifact(stage: Path, name: str) -> Path:
    """Require a regular, contained file with no symlink at any stage component."""
    if not isinstance(name, str) or not name:
        raise ValueError("invalid stage artifact path")
    relative = Path(name)
    if stage.is_symlink() or relative.is_absolute() or ".." in relative.parts:
        raise ValueError("stage artifact symlink or invalid path")
    candidate = stage / relative
    current = candidate
    while current != stage:
        if current.is_symlink():
            raise ValueError("stage artifact symlink prohibited")
        current = current.parent
    if stage.resolve() not in candidate.resolve().parents or not candidate.is_file():
        raise ValueError("stage artifact must be a regular contained file")
    return candidate


def _verify_stage(path: Path, identity: Mapping[str, Any]) -> dict[str, Any]:
    manifest = _load_json(_safe_stage_artifact(path, "stage_manifest.json"))
    if set(manifest) != {"schema_version", "complete", "identity", "artifacts"}:
        raise ValueError("incompatible stage manifest")
    if manifest["schema_version"] != _STAGE_SCHEMA or manifest["complete"] is not True or manifest["identity"] != dict(identity):
        raise ValueError("stage completion identity mismatch")
    artifacts = manifest["artifacts"]
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("stage artifacts required")
    for name, digest in artifacts.items():
        target = _safe_stage_artifact(path, name)
        if (not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts or not isinstance(digest, str)
                or not target.is_file() or _sha256(target) != digest):
            raise ValueError("stage artifact hash mismatch")
    return manifest


def _publish_stage(root: Path, stage: str, identity: Mapping[str, Any], build) -> bool:
    destination = root / stage
    if destination.exists():
        _verify_stage(destination, identity)
        return True
    temporary = Path(tempfile.mkdtemp(prefix=f".{stage}.incomplete-", dir=root))
    try:
        build(temporary)
        artifacts = {path.relative_to(temporary).as_posix(): _sha256(path)
                     for path in sorted(temporary.rglob("*")) if path.is_file()}
        if not artifacts:
            raise ValueError("stage produced no artifacts")
        _atomic_json(temporary / "stage_manifest.json", {"schema_version": _STAGE_SCHEMA, "complete": True,
                                                          "identity": dict(identity), "artifacts": artifacts})
        _verify_stage(temporary, identity)
        os.rename(temporary, destination)
        return False
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _read_reference(root: Path, manifest: Mapping[str, Any], name: str, required: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in manifest.get("partitions", []):
        if not isinstance(item, Mapping) or not isinstance(item.get("path"), str):
            raise ValueError("extract partition reference invalid")
        path = root / item["path"] / name
        if not path.is_file():
            raise ValueError(f"reference artifact missing: {name}")
        frame = pl.read_parquet(path)
        if not required <= set(frame.columns):
            raise ValueError(f"reference schema missing required columns: {name}")
        rows.extend(frame.iter_rows(named=True))
    return rows


def _support(rows: Iterable[dict[str, Any]], *, duration: bool = False) -> tuple[int, int, int, float]:
    values = list(rows)
    actors = {str(row["actor_key"]) for row in values if row.get("actor_key") is not None}
    days = {_strict_date(row["event_date"]) for row in values if row.get("event_date") is not None}
    supported_ns = sum(int(row.get("duration_ns", 0)) for row in values) if duration else 0
    return len(values), len(actors), len(days), supported_ns / 1_000_000_000


def _base_row(*, parameter: str, population: str, level: str, role: str, split: str, quantile: float | None,
              value_ns: int | None, rows: list[dict[str, Any]], duration: bool, status: str, reason: str | None,
              design_hash: str, source_hash: str) -> dict[str, Any]:
    n_obs, n_actors, n_days, seconds = _support(rows, duration=duration)
    return {"parameter": parameter, "population": population, "identity_level": level,
            "weighting": "physical_order_time" if duration else "event_pairs", "role": role, "split": split,
            "quantile": quantile, "value_ns": value_ns, "n_obs": n_obs, "n_actors": n_actors, "n_days": n_days,
            "supported_seconds": seconds, "status": status, "reason": reason,
            "design_sha256": design_hash, "source_manifest_sha256": source_hash}


def _quantile_rows(gaps: list[dict[str, Any]], level: str, role: str, split: str, design_hash: str, source_hash: str) -> list[dict[str, Any]]:
    selected = [row for row in gaps if row.get("identity_level") == level and row.get("execution_anchor_mode") == role
                and _split(str(row["instrument"]), row["event_date"]) == split]
    values = [int(row["gap_ns"]) for row in selected if type(row.get("gap_ns")) is int and int(row["gap_ns"]) >= 0]
    output: list[dict[str, Any]] = []
    for population, selected_values in (("all_gaps", values), ("positive_gaps", [value for value in values if value > 0])):
        for q in _QUANTILES:
            value = higher_quantile_ns(selected_values, q) if selected_values else None
            output.append(_base_row(parameter="gap_quantile", population=population, level=level, role=role, split=split,
                                    quantile=q, value_ns=value, rows=selected, duration=False,
                                    status="estimated" if value is not None else "no_support",
                                    reason=None if value is not None else "no observed gaps in stratum",
                                    design_hash=design_hash, source_hash=source_hash))
    for q, name in ((0.9, "gap_ceiling_p90"), (0.95, "gap_ceiling_p95")):
        value = higher_quantile_ns(values, q) if values else None
        output.append(_base_row(parameter=name, population="all_gaps", level=level, role=role, split=split, quantile=q,
                                value_ns=value, rows=selected, duration=False, status="estimated" if value is not None else "no_support",
                                reason=None if value is not None else "no observed gaps in stratum", design_hash=design_hash,
                                source_hash=source_hash))
        floor = None if value is None else max(value, 1_000_000)
        output.append(_base_row(parameter=f"{name}_floor_1ms", population="all_gaps", level=level, role=role, split=split,
                                quantile=q, value_ns=floor, rows=selected, duration=False,
                                status="estimated" if floor is not None else "no_support",
                                reason=None if floor is not None else "no observed gaps in stratum", design_hash=design_hash,
                                source_hash=source_hash))
    return output


def _age_rows(exposure: list[dict[str, Any]], spells: list[dict[str, Any]], level: str, split: str,
              design_hash: str, source_hash: str) -> list[dict[str, Any]]:
    selected = [row for row in exposure if row.get("identity_level") == level and _split(str(row["instrument"]), row["event_date"]) == split]
    selected_spells = [row for row in spells if row.get("identity_level") == level and _split(str(row["instrument"]), row["event_date"]) == split]
    intervals = [(int(row["age_start_ns"]), int(row["age_end_ns"])) for row in selected
                 if type(row.get("age_start_ns")) is int and type(row.get("age_end_ns")) is int and row["age_end_ns"] > row["age_start_ns"]]
    km = [(int(row["duration_ns"]), not bool(row.get("right_censored"))) for row in selected_spells
          if type(row.get("duration_ns")) is int and int(row["duration_ns"]) >= 0]
    output: list[dict[str, Any]] = []
    for q in _QUANTILES:
        value = age_time_quantile_ns(intervals, q) if intervals else None
        output.append(_base_row(parameter="age_order_time_quantile", population="visible_order_time", level=level,
                                role="not_applicable", split=split, quantile=q, value_ns=value, rows=selected, duration=True,
                                status="estimated" if value is not None else "no_support",
                                reason=None if value is not None else "no observed order-time intervals", design_hash=design_hash,
                                source_hash=source_hash))
        survival = survival_quantile_ns(km, q) if km else None
        output.append(_base_row(parameter="age_spell_km_quantile", population="visible_spells", level=level,
                                role="not_applicable", split=split, quantile=q, value_ns=survival, rows=selected_spells,
                                duration=True, status="diagnostic" if survival is not None else "not_identified",
                                reason=None if survival is not None else "no observed spell-ending event", design_hash=design_hash,
                                source_hash=source_hash))
    for q, name in ((0.9, "age_ceiling_p90"), (0.95, "age_ceiling_p95")):
        value = age_time_quantile_ns(intervals, q) if intervals else None
        output.append(_base_row(parameter=name, population="visible_order_time", level=level, role="not_applicable", split=split,
                                quantile=q, value_ns=value, rows=selected, duration=True,
                                status="estimated" if value is not None else "no_support",
                                reason=None if value is not None else "no observed order-time intervals", design_hash=design_hash,
                                source_hash=source_hash))
    return output


def _gap_quantile(rows: Iterable[dict[str, Any]], quantile: float) -> int | None:
    values = [int(row["gap_ns"]) for row in rows if type(row.get("gap_ns")) is int and row["gap_ns"] >= 0]
    return higher_quantile_ns(values, quantile) if values else None


def _age_quantile(rows: Iterable[dict[str, Any]], quantile: float) -> int | None:
    intervals = [(int(row["age_start_ns"]), int(row["age_end_ns"])) for row in rows
                 if type(row.get("age_start_ns")) is int and type(row.get("age_end_ns")) is int
                 and row["age_end_ns"] > row["age_start_ns"]]
    return age_time_quantile_ns(intervals, quantile) if intervals else None


def _diagnostic_estimates(kind: str, rows: Iterable[dict[str, Any]]) -> tuple[int | None, int | None]:
    estimator = _gap_quantile if kind == "gaps" else _age_quantile
    return estimator(rows, 0.9), estimator(rows, 0.95)


def _bootstrap_diagnostics(gaps: list[dict[str, Any]], exposure: list[dict[str, Any]], design: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Summarize role-preserving moving-day bootstrap and private in-memory LOO.

    Each selected day contributes all of its original rows whenever sampled.  The
    interval endpoints are exact discrete ``higher`` quantiles of replicate
    estimates; no float conversion or daily-preaggregation is used.
    """
    diagnostics: list[dict[str, Any]] = []
    minimum_days = int(design["bootstrap_min_supported_days"])
    for kind, rows in (("gaps", gaps), ("age", exposure)):
        roles = _ROLES if kind == "gaps" else ("not_applicable",)
        for level in _LEVELS:
            for role in roles:
                selected = [r for r in rows if r.get("identity_level") == level
                            and (kind != "gaps" or r.get("execution_anchor_mode") == role)]
                days = sorted({_strict_date(r["event_date"]) for r in selected if r.get("event_date") is not None})
                actors = {str(r["actor_key"]) for r in selected if r.get("actor_key") is not None}
                result: dict[str, Any] = {
                    "population": kind, "identity_level": level, "role": role,
                    "available_days": len(days), "actors": len(actors), "replicates": 0,
                    "bootstrap_status": "no_support" if not days else "insufficient_supported_days",
                    "identified_fraction": None,
                    "bootstrap_p90_ci_lower_ns": None, "bootstrap_p90_ci_upper_ns": None,
                    "bootstrap_p95_ci_lower_ns": None, "bootstrap_p95_ci_upper_ns": None,
                    "loo_replicates": 0, "leave_one_actor_out_identified_fraction": None,
                    "loo_p90_min_value_ns": None, "loo_p90_max_value_ns": None,
                    "loo_p95_min_value_ns": None, "loo_p95_max_value_ns": None,
                }
                if len(days) >= minimum_days:
                    by_day: dict[date, list[dict[str, Any]]] = defaultdict(list)
                    for row in selected:
                        by_day[_strict_date(row["event_date"])].append(row)
                    p90_values: list[int] = []
                    p95_values: list[int] = []
                    for indices in moving_day_replicates(days, block_days=int(design["bootstrap_block_days"]),
                                                         replicates=int(design["bootstrap_replicates"]), seed=int(design["bootstrap_seed"])):
                        sampled = [row for index in indices for row in by_day[days[index]]]
                        p90, p95 = _diagnostic_estimates(kind, sampled)
                        if p90 is not None:
                            p90_values.append(p90)
                        if p95 is not None:
                            p95_values.append(p95)
                    result["replicates"] = int(design["bootstrap_replicates"])
                    result["bootstrap_status"] = "estimated"
                    result["identified_fraction"] = len(p95_values) / result["replicates"]
                    for quantile, values in (("p90", p90_values), ("p95", p95_values)):
                        result[f"bootstrap_{quantile}_ci_lower_ns"] = higher_quantile_ns(values, 0.025) if values else None
                        result[f"bootstrap_{quantile}_ci_upper_ns"] = higher_quantile_ns(values, 0.975) if values else None
                loo_p90: list[int] = []
                loo_p95: list[int] = []
                for actor in actors:
                    retained = [row for row in selected if row.get("actor_key") != actor]
                    retained_days = {_strict_date(row["event_date"]) for row in retained if row.get("event_date") is not None}
                    if len(retained_days) < minimum_days:
                        continue
                    p90, p95 = _diagnostic_estimates(kind, retained)
                    if p90 is not None:
                        loo_p90.append(p90)
                    if p95 is not None:
                        loo_p95.append(p95)
                result["loo_replicates"] = len(loo_p95)
                if actors and loo_p95:
                    result["leave_one_actor_out_identified_fraction"] = len(loo_p95) / len(actors)
                    result["loo_p90_min_value_ns"] = min(loo_p90) if loo_p90 else None
                    result["loo_p90_max_value_ns"] = max(loo_p90) if loo_p90 else None
                    result["loo_p95_min_value_ns"] = min(loo_p95)
                    result["loo_p95_max_value_ns"] = max(loo_p95)
                diagnostics.append(result)
    return diagnostics


_ESTIMATE_SCHEMA = {"instrument": pl.String, "parameter": pl.String, "population": pl.String, "identity_level": pl.String, "weighting": pl.String,
                    "role": pl.String, "split": pl.String, "quantile": pl.Float64, "value_ns": pl.Int64,
                    "n_obs": pl.Int64, "n_actors": pl.Int64, "n_days": pl.Int64, "supported_seconds": pl.Float64,
                    "status": pl.String, "reason": pl.String, "design_sha256": pl.String, "source_manifest_sha256": pl.String}


def run_estimate(output_root: Path | str, design: Mapping[str, Any], instrument: str,
                 *, dates: list[str] | None = None) -> dict[str, Any]:
    root = Path(output_root).resolve(); manifest, source_hash = _verified_source(root, design, instrument, dates=dates)
    identity = _stage_identity("estimate", source_hash, design, instrument)
    result: dict[str, Any] = {"status": "estimate_complete", "resume_verified": False}
    def build(stage: Path) -> None:
        gaps = [r for r in _read_reference(root, manifest, "gap_pairs.parquet", {"instrument", "identity_level", "actor_key", "execution_anchor_mode", "event_date", "gap_ns"}) if r["instrument"] == instrument]
        exposure = [r for r in _read_reference(root, manifest, "age_exposure.parquet", {"instrument", "identity_level", "actor_key", "event_date", "age_start_ns", "age_end_ns", "duration_ns"}) if r["instrument"] == instrument]
        spells = [r for r in _read_reference(root, manifest, "age_spells.parquet", {"instrument", "identity_level", "actor_key", "event_date", "duration_ns", "right_censored"}) if r["instrument"] == instrument]
        split = "calibration" if instrument == "RISANAMENTO" else "exploratory_single_day"
        rows = [row for level in _LEVELS for role in _ROLES for row in _quantile_rows(gaps, level, role, split, identity["design_sha256"], source_hash)]
        rows += [row for level in _LEVELS for row in _age_rows(exposure, spells, level, split, identity["design_sha256"], source_hash)]
        rows = [{"instrument": instrument, **row} for row in rows]
        _atomic_csv(stage / "estimates.csv", rows, _ESTIMATE_SCHEMA)
        proposal_rows = [{"parameter": row["parameter"], "population": row["population"], "identity_level": row["identity_level"],
                          "role": row["role"], "split": row["split"], "value_ns": row["value_ns"], "status": row["status"],
                          "reason": row["reason"]} for row in rows if row["parameter"] in {"gap_ceiling_p90", "gap_ceiling_p95", "age_ceiling_p90", "age_ceiling_p95"}]
        proposal_rows += [{"parameter": name, "population": "not_run", "identity_level": "not_applicable", "role": "not_applicable",
                           "split": split, "value_ns": None, "status": "not_run", "reason": "future estimator block not run"}
                          for name in ("W", "REV", "SCI", "tau")]
        _atomic_json(stage / "proposals.json", {"schema_version": "temporal_calibration_proposals_v1", "proposal_only": True,
                                                  "source_manifest_sha256": source_hash, "proposals": proposal_rows})
        _atomic_json(stage / "diagnostics.json", {"instrument": instrument, "bootstrap": _bootstrap_diagnostics(gaps, exposure, design),
                                                    "note": "moving blocks resample complete observation rows; no detector parameter was applied"})
    result["resume_verified"] = _publish_stage(root, "estimate", identity, build)
    return result


def _frozen_gap_coverage(rows: Iterable[dict[str, Any]], reference_ns: int | None) -> float | None:
    values = [int(row["gap_ns"]) for row in rows if type(row.get("gap_ns")) is int and row["gap_ns"] >= 0]
    return None if reference_ns is None or not values else sum(value <= reference_ns for value in values) / len(values)


def _frozen_age_coverage(rows: Iterable[dict[str, Any]], reference_ns: int | None) -> float | None:
    intervals = [(int(row["age_start_ns"]), int(row["age_end_ns"])) for row in rows
                 if type(row.get("age_start_ns")) is int and type(row.get("age_end_ns")) is int
                 and row["age_end_ns"] > row["age_start_ns"]]
    if reference_ns is None or not intervals:
        return None
    total = sum(end - start for start, end in intervals)
    below = sum(max(0, min(end, reference_ns) - start) for start, end in intervals)
    return below / total


def _validation_rows(gaps: list[dict[str, Any]], exposure: list[dict[str, Any]],
                     references: Mapping[tuple[str, str, str], Any], instrument: str) -> list[dict[str, Any]]:
    """Describe frozen-reference holdout coverage; this function never selects a reference."""
    rows: list[dict[str, Any]] = []
    instrument_has_frozen_holdout = instrument == "RISANAMENTO"
    for level in _LEVELS:
        for role in _ROLES:
            selected = [r for r in gaps if r.get("identity_level") == level and r.get("execution_anchor_mode") == role]
            ref = references.get(("gap_ceiling_p95", level, role))
            reference = ref if type(ref) is int and ref >= 0 else None
            n, actors, days, seconds = _support(selected)
            descriptive = _gap_quantile(selected, 0.95)
            status = ("not_applicable" if not instrument_has_frozen_holdout else
                      "no_support" if not selected else
                      "no_frozen_reference" if reference is None else "descriptive_holdout")
            rows.append({"instrument": instrument, "parameter": "gap_ceiling_p95", "population": "all_gaps",
                         "identity_level": level, "role": role, "split": "retrospective_validation",
                         "reference_value_ns": reference, "heldout_descriptive_p95_ns": descriptive,
                         "frozen_reference_coverage": _frozen_gap_coverage(selected, reference), "n_obs": n,
                         "n_actors": actors, "n_days": days, "supported_seconds": seconds, "status": status,
                         "reason": "instrument has no predeclared frozen holdout" if status == "not_applicable" else
                                   "no holdout observations" if status == "no_support" else
                                   "frozen estimate is unavailable" if status == "no_frozen_reference" else None})
        selected = [r for r in exposure if r.get("identity_level") == level]
        ref = references.get(("age_ceiling_p95", level, "not_applicable"))
        reference = ref if type(ref) is int and ref >= 0 else None
        n, actors, days, seconds = _support(selected, duration=True)
        descriptive = _age_quantile(selected, 0.95)
        status = ("not_applicable" if not instrument_has_frozen_holdout else
                  "no_support" if not selected else
                  "no_frozen_reference" if reference is None else "descriptive_holdout")
        rows.append({"instrument": instrument, "parameter": "age_ceiling_p95", "population": "visible_order_time",
                     "identity_level": level, "role": "not_applicable", "split": "retrospective_validation",
                     "reference_value_ns": reference, "heldout_descriptive_p95_ns": descriptive,
                     "frozen_reference_coverage": _frozen_age_coverage(selected, reference), "n_obs": n,
                     "n_actors": actors, "n_days": days, "supported_seconds": seconds, "status": status,
                     "reason": "instrument has no predeclared frozen holdout" if status == "not_applicable" else
                               "no holdout observations" if status == "no_support" else
                               "frozen estimate is unavailable" if status == "no_frozen_reference" else None})
    return rows


_VALIDATION_SCHEMA = {"instrument": pl.String, "parameter": pl.String, "population": pl.String, "identity_level": pl.String,
                      "role": pl.String, "split": pl.String, "reference_value_ns": pl.Int64,
                      "heldout_descriptive_p95_ns": pl.Int64, "frozen_reference_coverage": pl.Float64,
                      "n_obs": pl.Int64, "n_actors": pl.Int64, "n_days": pl.Int64,
                      "supported_seconds": pl.Float64, "status": pl.String, "reason": pl.String}


def run_validate(output_root: Path | str, design: Mapping[str, Any], instrument: str,
                 *, dates: list[str] | None = None) -> dict[str, Any]:
    root = Path(output_root).resolve(); manifest, source_hash = _verified_source(root, design, instrument, dates=dates)
    estimate_identity = _stage_identity("estimate", source_hash, design, instrument)
    _verify_stage(root / "estimate", estimate_identity)
    estimate_digest = _sha256(root / "estimate" / "stage_manifest.json")
    identity = _stage_identity("validate", source_hash, design, instrument, {"estimate": estimate_digest})
    def build(stage: Path) -> None:
        proposals = _load_json(root / "estimate" / "proposals.json").get("proposals")
        if not isinstance(proposals, list):
            raise ValueError("frozen estimate proposals required")
        references = {(r["parameter"], r["identity_level"], r["role"]): r.get("value_ns") for r in proposals
                      if isinstance(r, dict) and r.get("parameter") in {"gap_ceiling_p90", "gap_ceiling_p95", "age_ceiling_p90", "age_ceiling_p95"}}
        gaps = [r for r in _read_reference(root, manifest, "gap_pairs.parquet", {"instrument", "identity_level", "actor_key", "execution_anchor_mode", "event_date", "gap_ns"})
                if r["instrument"] == instrument and _split(instrument, r["event_date"]) == "retrospective_validation"]
        exposure = [r for r in _read_reference(root, manifest, "age_exposure.parquet", {"instrument", "identity_level", "actor_key", "event_date", "age_start_ns", "age_end_ns", "duration_ns"})
                    if r["instrument"] == instrument and _split(instrument, r["event_date"]) == "retrospective_validation"]
        _atomic_csv(stage / "validation.csv", _validation_rows(gaps, exposure, references, instrument), _VALIDATION_SCHEMA)
    resumed = _publish_stage(root, "validate", identity, build)
    return {"status": "validate_complete", "resume_verified": resumed}


def run_report(output_root: Path | str, design: Mapping[str, Any], instrument: str,
               *, dates: list[str] | None = None) -> dict[str, Any]:
    root = Path(output_root).resolve(); _, source_hash = _verified_source(root, design, instrument, dates=dates)
    _verify_stage(root / "estimate", _stage_identity("estimate", source_hash, design, instrument))
    estimate_digest = _sha256(root / "estimate" / "stage_manifest.json")
    _verify_stage(root / "validate", _stage_identity("validate", source_hash, design, instrument,
                                                    {"estimate": estimate_digest}))
    identity = _stage_identity("report", source_hash, design, instrument,
                               {"estimate": estimate_digest,
                                "validate": _sha256(root / "validate" / "stage_manifest.json")})
    def build(stage: Path) -> None:
        estimates = pl.read_csv(root / "estimate" / "estimates.csv").to_dicts()
        validation = pl.read_csv(root / "validate" / "validation.csv").to_dicts()
        for level, filename in (("client_original", "client_original.csv"), ("firm", "firm_fallback.csv")):
            selected = [r for r in estimates if r["identity_level"] == level]
            # Public aggregate deliberately has no actor/order/partition/path columns.
            public = [{key: row.get(key) for key in ("instrument", "parameter", "population", "weighting", "role", "split", "quantile", "value_ns",
                                                      "n_obs", "n_actors", "n_days", "supported_seconds", "status", "reason")}
                      for row in selected]
            schema = {key: _ESTIMATE_SCHEMA[key] for key in ("instrument", "parameter", "population", "weighting", "role", "split", "quantile", "value_ns",
                                                               "n_obs", "n_actors", "n_days", "supported_seconds", "status", "reason")}
            _atomic_csv(stage / filename, public, schema)
        status_counts = Counter(str(row["status"]) for row in estimates)
        proposals = _load_json(root / "estimate" / "proposals.json")["proposals"]
        status_counts.update(str(row["status"]) for row in proposals if row.get("status") == "not_run")
        _atomic_json(stage / "summary.json", {"instrument": instrument, "source_manifest_sha256": source_hash,
                                                "estimated_rows": sum(r["status"] == "estimated" for r in estimates),
                                                "validation_rows": len(validation), "proposal_only": True,
                                                "parameter_status_counts": dict(sorted(status_counts.items())),
                                                "not_run_parameters": [r["parameter"] for r in proposals if r.get("status") == "not_run"],
                                                "scope": "gap and age only; W/REV/SCI/tau remain NOT_RUN"})
    resumed = _publish_stage(root, "report", identity, build)
    return {"status": "report_complete", "resume_verified": resumed}
