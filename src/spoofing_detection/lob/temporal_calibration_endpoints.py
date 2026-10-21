"""Verified market-only paths after every eligible physical cancellation.

This independent, proposal-only stage consumes typed compact-risk withdrawals and
coverage plus extracted market states.  It does not construct executions, select a
horizon, or represent a native episode REV measure.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable, Mapping

import polars as pl

from .empirical_controls_v2_io import file_hash, scan_checkpoint
from .temporal_calibration import physical_cancel_paths
from .temporal_calibration_inputs import preflight, validate_design
from .withdrawal_risk_v2 import SCHEMAS as RISK_SCHEMAS

_STAGE = "endpoint_diagnostics"
_STAGE_SCHEMA = "temporal_calibration_endpoint_diagnostics_v1"
_POPULATION = "all_eligible_physical_cancellations"
_PATH_SCHEMA: dict[str, pl.DataType] = {
    "instrument": pl.String, "partition_id": pl.String, "event_date": pl.Date,
    "coverage_epoch_id": pl.String, "withdrawal_event_id": pl.String,
    "physical_order_key": pl.String, "order_id": pl.String, "first_seen_sort_index": pl.Int64,
    "sort_index": pl.Int64, "event_ns": pl.Int64, "actor_key": pl.String, "actor_id": pl.String,
    "identity_level": pl.String, "identity_source": pl.String, "identity_fallback_flag": pl.Boolean,
    "side": pl.String, "visible_qty_removed": pl.Float64, "transition_reason": pl.String,
    "split": pl.String, "population": pl.String, "endpoint_kind": pl.String,
    "horizon_ns": pl.Int64, "horizon_seconds": pl.Float64, "target_ns": pl.Int64,
    "pre_source_index": pl.Int64, "pre_source_time_ns": pl.Int64,
    "post_source_index": pl.Int64, "post_source_time_ns": pl.Int64,
    "endpoint_time_ns": pl.Int64, "state_target_distance_ns": pl.Int64,
    "raw_midprice_delta_ticks": pl.Float64, "side_directional_delta_ticks": pl.Float64,
    "status": pl.String, "null_reason": pl.String,
}
_AGGREGATE_SCHEMA: dict[str, pl.DataType] = {
    "instrument": pl.String, "identity_level": pl.String, "split": pl.String,
    "population": pl.String, "endpoint_kind": pl.String, "row_scope": pl.String,
    "null_reason": pl.String, "horizon_ns": pl.Int64, "horizon_seconds": pl.Float64,
    "n_cases": pl.Int64, "n_observed": pl.Int64, "n_missing": pl.Int64,
    "n_negative_raw": pl.Int64, "n_zero_raw": pl.Int64, "n_positive_raw": pl.Int64,
    "n_negative_directional": pl.Int64, "n_zero_directional": pl.Int64, "n_positive_directional": pl.Int64,
    "n_recency": pl.Int64, "raw_delta_median_ticks": pl.Float64,
    "raw_delta_p90_ticks": pl.Float64, "raw_delta_p95_ticks": pl.Float64,
    "directional_delta_median_ticks": pl.Float64, "directional_delta_p90_ticks": pl.Float64,
    "directional_delta_p95_ticks": pl.Float64, "recency_median_ns": pl.Int64,
    "recency_p90_ns": pl.Int64, "recency_p95_ns": pl.Int64,
}


def _json_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _sha256(path: Path) -> str:
    return file_hash(path)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _atomic_parquet(path: Path, rows: list[dict[str, Any]], schema: Mapping[str, pl.DataType]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    pl.DataFrame(rows, schema=schema, strict=False).write_parquet(temporary)
    os.replace(temporary, path)


def _atomic_csv(path: Path, rows: list[dict[str, Any]], schema: Mapping[str, pl.DataType]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    pl.DataFrame(rows, schema=schema, strict=False).write_csv(temporary)
    os.replace(temporary, path)


def _project_root(output_root: Path) -> Path:
    return next((parent for parent in output_root.parents
                 if (parent / "configs" / "spoofing_empirical_controls.json").is_file()),
                Path(__file__).resolve().parents[3])


def _effective_tick(output_root: Path, design: Mapping[str, Any], instrument: str,
                    *, dates: list[str] | None) -> Decimal | None:
    """Recover the live, source-verified v3 tick rather than a guessed design tick."""
    observed = preflight(design, instrument, output_root, dates=dates, allow_existing=True,
                         project_root=_project_root(output_root))
    raw = observed.get("baseline", {}).get("tick_size") if isinstance(observed.get("baseline"), Mapping) else None
    if type(raw) not in (int, float) or isinstance(raw, bool):
        return None
    try:
        tick = Decimal(str(raw))
    except (InvalidOperation, ValueError):
        return None
    return tick if tick.is_finite() and tick > 0 else None


def _horizons(design: Mapping[str, Any]) -> list[tuple[int, float]]:
    horizons: list[tuple[int, float]] = []
    for value in design["endpoint_horizons_seconds"]:
        try:
            ns = Decimal(str(value)) * Decimal(1_000_000_000)
        except (InvalidOperation, ValueError) as exc:
            raise ValueError("endpoint horizon must be exact integral nanoseconds") from exc
        if not ns.is_finite() or ns <= 0 or ns != ns.to_integral_value():
            raise ValueError("endpoint horizon must be exact integral nanoseconds")
        horizons.append((int(ns), float(value)))
    return horizons


def _strict_day(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return date.fromisoformat(value)
    raise ValueError("event_date must be an ISO calendar date")


def _split(instrument: str, value: Any) -> str:
    day = _strict_day(value)
    if instrument == "RISANAMENTO":
        return "calibration" if day <= date(2024, 9, 30) else "retrospective_validation"
    return "exploratory_single_day"


def _stage_identity(source_hash: str, design: Mapping[str, Any], instrument: str) -> dict[str, Any]:
    here = Path(__file__)
    return {
        "schema_version": _STAGE_SCHEMA,
        "stage": _STAGE,
        "source_manifest_sha256": source_hash,
        "design_sha256": _json_hash(dict(design)),
        "instrument": instrument,
        "endpoint_module_sha256": _sha256(here),
        "physical_path_module_sha256": _sha256(here.with_name("temporal_calibration.py")),
        "risk_io_module_sha256": _sha256(here.with_name("empirical_controls_v2_io.py")),
    }


def _verify_stage(path: Path, identity: Mapping[str, Any]) -> dict[str, Any]:
    from .temporal_calibration_report import _safe_stage_artifact
    manifest_path = _safe_stage_artifact(path, "stage_manifest.json")
    if not manifest_path.is_file():
        raise ValueError("endpoint diagnostics stage manifest required")
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("invalid endpoint diagnostics stage manifest") from exc
    if (set(manifest) != {"schema_version", "complete", "identity", "artifacts"}
            or manifest["schema_version"] != _STAGE_SCHEMA or manifest["complete"] is not True
            or manifest["identity"] != dict(identity) or not isinstance(manifest["artifacts"], dict)):
        raise ValueError("endpoint diagnostics stage identity mismatch")
    for name, digest in manifest["artifacts"].items():
        target = _safe_stage_artifact(path, name)
        if (not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts
                or not isinstance(digest, str) or not target.is_file() or _sha256(target) != digest):
            raise ValueError("endpoint diagnostics stage artifact hash mismatch")
    return manifest


def _publish_stage(root: Path, identity: Mapping[str, Any], build) -> bool:
    destination = root / _STAGE
    if destination.exists():
        _verify_stage(destination, identity)
        return True
    temporary = Path(tempfile.mkdtemp(prefix=f".{_STAGE}.incomplete-", dir=root))
    try:
        build(temporary)
        artifacts = {path.relative_to(temporary).as_posix(): _sha256(path)
                     for path in sorted(temporary.rglob("*")) if path.is_file()}
        if not artifacts:
            raise ValueError("endpoint diagnostics produced no artifacts")
        _atomic_json(temporary / "stage_manifest.json", {"schema_version": _STAGE_SCHEMA, "complete": True,
                                                          "identity": dict(identity), "artifacts": artifacts})
        _verify_stage(temporary, identity)
        os.rename(temporary, destination)
        return False
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _path_base(withdrawal: Mapping[str, Any], *, horizon_ns: int, horizon_seconds: float) -> dict[str, Any]:
    return {key: withdrawal.get(key) for key in (
        "instrument", "partition_id", "event_date", "coverage_epoch_id", "withdrawal_event_id",
        "physical_order_key", "order_id", "first_seen_sort_index", "sort_index", "event_ns", "actor_key",
        "actor_id", "identity_level", "identity_source", "identity_fallback_flag", "side",
        "visible_qty_removed", "transition_reason",
    )} | {"split": _split(str(withdrawal["instrument"]), withdrawal["event_date"]),
         "population": _POPULATION, "endpoint_kind": "market_only_physical_cancel_path",
         "horizon_ns": horizon_ns, "horizon_seconds": horizon_seconds,
         "target_ns": int(withdrawal["event_ns"]) + horizon_ns}


def _null_path(base: Mapping[str, Any], *, status: str, null_reason: str,
               pre_index: int | None = None, pre_time: int | None = None) -> dict[str, Any]:
    return dict(base) | {"pre_source_index": pre_index, "pre_source_time_ns": pre_time,
                         "post_source_index": None, "post_source_time_ns": None,
                         "endpoint_time_ns": None, "state_target_distance_ns": None,
                         "raw_midprice_delta_ticks": None, "side_directional_delta_ticks": None,
                         "status": status, "null_reason": null_reason}


def _coverage_reaches(coverage: Iterable[Mapping[str, Any]], withdrawal: Mapping[str, Any], target: int) -> bool:
    key = tuple(withdrawal.get(name) for name in ("instrument", "partition_id", "event_date", "coverage_epoch_id"))
    event_ns = withdrawal.get("event_ns")
    if type(event_ns) is not int:
        raise ValueError("typed withdrawal event_ns must be integer nanoseconds")
    for row in coverage:
        if tuple(row.get(name) for name in ("instrument", "partition_id", "event_date", "coverage_epoch_id")) != key:
            continue
        start, end = row.get("start_ns"), row.get("end_ns")
        if type(start) is int and type(end) is int and start <= event_ns and end > target:
            return True
    return False


def _same_group_states(states: Iterable[Mapping[str, Any]], withdrawal: Mapping[str, Any]) -> list[dict[str, Any]]:
    key = tuple(withdrawal.get(name) for name in ("instrument", "partition_id", "event_date", "coverage_epoch_id"))
    return [dict(row) for row in states if tuple(row.get(name) for name in
            ("instrument", "partition_id", "event_date", "coverage_epoch_id")) == key]


def _pre_state(states: Iterable[Mapping[str, Any]], withdrawal: Mapping[str, Any]) -> tuple[int | None, int | None]:
    event_ns, sort_index = withdrawal.get("event_ns"), withdrawal.get("sort_index")
    if type(event_ns) is not int or type(sort_index) is not int:
        raise ValueError("typed withdrawal source time/index required")
    candidates = [row for row in states if type(row.get("source_index")) is int and type(row.get("replay_time_ns")) is int
                  and row["source_index"] < sort_index and row["replay_time_ns"] <= event_ns]
    if not candidates:
        return None, None
    selected = max(candidates, key=lambda row: (row["replay_time_ns"], row["source_index"]))
    return selected["source_index"], selected["replay_time_ns"]


def _null_reason(status: str) -> str | None:
    return None if status == "observed" else {
        "no_pre_cancel_state": "no_pre_cancel_state",
        "uncovered_horizon": "no_market_state_after_target",
        "missing_pre_midprice": "missing_pre_midprice",
        "missing_endpoint_midprice": "missing_endpoint_midprice",
    }.get(status, "invalid_market_path")


def _group_paths(withdrawals: list[dict[str, Any]], states: list[dict[str, Any]], coverage: list[dict[str, Any]],
                 *, horizons: list[tuple[int, float]], tick: Decimal) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for withdrawal in withdrawals:
        local_states = _same_group_states(states, withdrawal)
        group = (withdrawal["instrument"], withdrawal["partition_id"], withdrawal["event_date"], withdrawal["coverage_epoch_id"])
        pre_index, pre_time = _pre_state(local_states, withdrawal)
        covered: list[tuple[int, float]] = []
        for horizon_ns, horizon_seconds in horizons:
            base = _path_base(withdrawal, horizon_ns=horizon_ns, horizon_seconds=horizon_seconds)
            if _coverage_reaches(coverage, withdrawal, base["target_ns"]):
                covered.append((horizon_ns, horizon_seconds))
            else:
                result.append(_null_path(base, status="uncovered_horizon", null_reason="coverage_not_after_target",
                                         pre_index=pre_index, pre_time=pre_time))
        if not covered:
            continue
        pure = physical_cancel_paths([withdrawal], local_states, group=group,
                                     horizons_ns=[ns for ns, _ in covered], tick_size=tick, directional=True)
        seconds = {ns: value for ns, value in covered}
        for item in pure:
            horizon_ns = int(item["horizon_ns"])
            base = _path_base(withdrawal, horizon_ns=horizon_ns, horizon_seconds=seconds[horizon_ns])
            delta = item.get("midprice_change_ticks")
            directional = item.get("side_directional_delta_ticks")
            endpoint_time = item.get("endpoint_time_ns")
            result.append(base | {
                "pre_source_index": item.get("pre_source_index"), "pre_source_time_ns": item.get("pre_source_time_ns"),
                "post_source_index": item.get("post_source_index"), "post_source_time_ns": item.get("post_source_time_ns"),
                "endpoint_time_ns": endpoint_time,
                "state_target_distance_ns": None if endpoint_time is None else int(base["target_ns"]) - int(endpoint_time),
                "raw_midprice_delta_ticks": None if delta is None else float(delta),
                "side_directional_delta_ticks": None if directional is None else float(directional),
                "status": item["status"], "null_reason": _null_reason(str(item["status"])),
            })
    return result


def _quantile(values: Iterable[float | int], q: float) -> float | int | None:
    ordered = sorted(values)
    if not ordered:
        return None
    return ordered[max(0, math.ceil(len(ordered) * q) - 1)]


def _aggregate_one(rows: list[dict[str, Any]], *, row_scope: str, null_reason: str) -> dict[str, Any]:
    first = rows[0]
    raw = [float(row["raw_midprice_delta_ticks"]) for row in rows if row.get("raw_midprice_delta_ticks") is not None]
    directional = [float(row["side_directional_delta_ticks"]) for row in rows if row.get("side_directional_delta_ticks") is not None]
    recency = [int(row["state_target_distance_ns"]) for row in rows if row.get("state_target_distance_ns") is not None]
    signs = lambda values: (sum(value < 0 for value in values), sum(value == 0 for value in values), sum(value > 0 for value in values))
    neg, zero, pos = signs(raw); dneg, dzero, dpos = signs(directional)
    return {"instrument": first["instrument"], "identity_level": first["identity_level"], "split": first["split"],
            "population": _POPULATION, "endpoint_kind": "market_only_physical_cancel_path", "row_scope": row_scope,
            "null_reason": null_reason, "horizon_ns": first["horizon_ns"], "horizon_seconds": first["horizon_seconds"],
            "n_cases": len(rows), "n_observed": len(raw), "n_missing": len(rows) - len(raw),
            "n_negative_raw": neg, "n_zero_raw": zero, "n_positive_raw": pos,
            "n_negative_directional": dneg, "n_zero_directional": dzero, "n_positive_directional": dpos,
            "n_recency": len(recency), "raw_delta_median_ticks": _quantile(raw, .5),
            "raw_delta_p90_ticks": _quantile(raw, .9), "raw_delta_p95_ticks": _quantile(raw, .95),
            "directional_delta_median_ticks": _quantile(directional, .5),
            "directional_delta_p90_ticks": _quantile(directional, .9),
            "directional_delta_p95_ticks": _quantile(directional, .95),
            "recency_median_ns": _quantile(recency, .5), "recency_p90_ns": _quantile(recency, .9),
            "recency_p95_ns": _quantile(recency, .95)}


def _aggregate(paths: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in paths:
        grouped[(row["instrument"], row["identity_level"], row["split"], row["horizon_ns"])].append(row)
    output: list[dict[str, Any]] = []
    for rows in grouped.values():
        output.append(_aggregate_one(rows, row_scope="all_cases", null_reason="all_cases"))
        by_reason: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            if row["null_reason"] is not None:
                by_reason[str(row["null_reason"])].append(row)
        output.extend(_aggregate_one(selected, row_scope="null_reason", null_reason=reason)
                      for reason, selected in sorted(by_reason.items()))
    return sorted(output, key=lambda row: (row["instrument"], row["identity_level"], row["split"], row["horizon_ns"],
                                           row["row_scope"], row["null_reason"]))


def _partition_rows(root: Path, manifest: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    withdrawals: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    states: list[dict[str, Any]] = []
    for item in manifest.get("partitions", []):
        if not isinstance(item, Mapping) or not isinstance(item.get("path"), str):
            raise ValueError("verified reference partition required")
        folder = root / item["path"]
        reference = json.loads((folder / "reference_manifest.json").read_text())
        cache_key, provenance = reference.get("risk_cache_key"), reference.get("risk_provenance")
        if not isinstance(cache_key, str) or not isinstance(provenance, Mapping):
            raise ValueError("typed risk cache identity/provenance required")
        risk = scan_checkpoint(folder / "risk", cache_key=cache_key, schemas=RISK_SCHEMAS, provenance=provenance)
        withdrawals.extend(risk["withdrawal_events"].collect().iter_rows(named=True))
        coverage.extend(risk["coverage_epochs"].collect().iter_rows(named=True))
        market = pl.read_parquet(folder / "market_states.parquet")
        required = {"instrument", "partition_id", "event_date", "coverage_epoch_id", "source_index", "replay_time_ns", "market_mid"}
        if not required <= set(market.columns):
            raise ValueError("reference market_states coverage/source-time schema required")
        states.extend(market.iter_rows(named=True))
    return withdrawals, coverage, states


def run_endpoint_diagnostics(output_root: Path | str, design: Mapping[str, Any], instrument: str,
                             *, dates: list[str] | None = None) -> dict[str, Any]:
    """Publish all physical-cancellation market paths in a separately verified stage."""
    validate_design(design)
    root = Path(output_root).resolve()
    # Keep source verification owned by the reference/report graph; this stage
    # does not replay, alter extraction artifacts, or infer an execution link.
    from .temporal_calibration_report import _verified_source
    manifest, source_hash = _verified_source(root, design, instrument, dates=dates)
    tick = _effective_tick(root, design, instrument, dates=dates)
    if tick is None:
        raise ValueError("effective preflight tick is required")
    identity = _stage_identity(source_hash, design, instrument)
    horizons = _horizons(design)

    def build(stage: Path) -> None:
        withdrawals, coverage, states = _partition_rows(root, manifest)
        selected = [dict(row) for row in withdrawals if row.get("instrument") == instrument]
        for row in selected:
            required = ("instrument", "partition_id", "event_date", "coverage_epoch_id", "physical_order_key",
                        "actor_key", "identity_level", "side", "sort_index", "event_ns")
            if any(row.get(name) is None for name in required):
                raise ValueError("typed physical withdrawal provenance required")
        paths = _group_paths(selected, states, coverage, horizons=horizons, tick=tick)
        aggregates = _aggregate(paths)
        _atomic_parquet(stage / "physical_cancel_paths.parquet", paths, _PATH_SCHEMA)
        _atomic_csv(stage / "aggregate.csv", aggregates, _AGGREGATE_SCHEMA)
        status_counts = Counter(str(row["status"]) for row in paths)
        reason_counts = Counter(str(row["null_reason"]) for row in paths if row["null_reason"] is not None)
        _atomic_json(stage / "diagnostics.json", {
            "instrument": instrument, "source_manifest_sha256": source_hash, "effective_tick_size": str(tick),
            "population": _POPULATION, "endpoint_kind": "market_only_physical_cancel_path",
            "native_episode_rev": False, "horizon_selection": "none_all_frozen_horizons_emitted",
            "path_count": len(paths), "status_counts": dict(sorted(status_counts.items())),
            "null_reason_counts": dict(sorted(reason_counts.items())),
            "note": "Raw midprice delta and side-directional delta are distinct; neither is labeled favorable.",
        })

    resumed = _publish_stage(root, identity, build)
    return {"status": "endpoint_diagnostics_complete", "resume_verified": resumed}
