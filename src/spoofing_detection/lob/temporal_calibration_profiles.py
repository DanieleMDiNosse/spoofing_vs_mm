"""Native, multi-horizon SCI/MSCI diagnostics on unselected executions.

This stage is deliberately diagnostic: it runs the canonical exploratory metric
producer on every attributable passive/aggressive execution in each complete raw
partition.  It does not consume strict detections, W-positive links, or outcome
signs.  A RISANAMENTO kernel is fitted only on the frozen calibration dates and
is reused unchanged for retrospective validation; a missing train kernel remains
an explicit null-status result rather than a full-sample fallback.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import hashlib
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Mapping

import polars as pl

from .depth_kernel_calibration import calibrate_empirical_depth_kernel
from .spoofing_metrics import compute_exploratory_metrics
from .temporal_calibration_inputs import _partition_span_seconds
from .temporal_calibration_report import (
    _atomic_csv,
    _atomic_json,
    _publish_stage,
    _read_reference,
    _sha256,
    _split,
    _stage_identity,
    _verified_source,
)

_LEVELS = ("client_original", "firm")
_ROLES = ("passive", "aggressive")
_HORIZONS = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0)

_PRIVATE_SCHEMA = {
    "instrument": pl.String, "partition_id": pl.String, "event_date": pl.Date,
    "split": pl.String, "identity_level": pl.String, "actor_key": pl.String,
    "execution_anchor_mode": pl.String, "execution_cluster_id": pl.String,
    "horizon_seconds": pl.Float64, "horizon_ns": pl.Int64, "SCI": pl.Float64,
    "MSCI": pl.Float64, "post_target_replay_ns": pl.Int64, "post_state_replay_ns": pl.Int64,
    "recency_ns": pl.Int64, "status": pl.String, "null_reason": pl.String,
    "episode_strict_detection": pl.Boolean, "max_deceptive_order_age_seconds": pl.Float64,
}
_PUBLIC_SCHEMA = {
    "instrument": pl.String, "identity_level": pl.String, "execution_anchor_mode": pl.String,
    "role": pl.String, "split": pl.String, "horizon_seconds": pl.Float64, "horizon_ns": pl.Int64,
    "n_cases": pl.Int64, "n_observed": pl.Int64, "n_missing": pl.Int64,
    "sci_median": pl.Float64, "msci_median": pl.Float64, "recency_median_ns": pl.Int64,
    "recency_p90_ns": pl.Int64, "status": pl.String, "null_reason": pl.String,
}


@dataclass(frozen=True)
class KernelTraining:
    status: str
    weights: dict[str, dict[int, float]] | None
    kernel: pl.DataFrame
    reason: str | None = None


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _horizon_ns(seconds: float) -> int:
    if type(seconds) not in (int, float) or isinstance(seconds, bool) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("horizon_seconds must be positive finite numeric, never boolean")
    value = int(round(seconds * 1_000_000_000))
    if value <= 0 or not math.isclose(seconds * 1_000_000_000, value, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("frozen horizon must be positive exact integer nanoseconds")
    return value


def _kernel_weights(kernel: pl.DataFrame, *, top_n: int = 10) -> dict[str, dict[int, float]] | None:
    required = {"side", "rank", "kernel_weight"}
    if kernel.is_empty() or not required <= set(kernel.columns):
        return None
    weights: dict[str, dict[int, float]] = {}
    for side in ("bid", "ask"):
        rows = kernel.filter(pl.col("side") == side)
        raw: dict[int, float] = {}
        for row in rows.iter_rows(named=True):
            rank, value = row.get("rank"), _finite(row.get("kernel_weight"))
            if type(rank) is not int or not 1 <= rank <= top_n or value is None or value < 0 or rank in raw:
                return None
            raw[rank] = value
        if set(raw) != set(range(1, top_n + 1)):
            return None
        total = sum(raw.values())
        if total <= 0 or not math.isfinite(total):
            return None
        weights[side] = {rank: value / total for rank, value in raw.items()}
    return weights


def train_kernel(raw_events: pl.DataFrame, *, instrument: str, tick_size: float) -> KernelTraining:
    """Calibrate a top-10 native kernel from exactly the supplied train population."""
    if raw_events.is_empty():
        return KernelTraining("kernel_not_identified", None, pl.DataFrame(), "no calibration raw events")
    try:
        kernel = calibrate_empirical_depth_kernel(raw_events, instrument_id=instrument, top_n=10,
                                                   tick_size=tick_size, horizon_seconds=10.0)
    except (ValueError, ArithmeticError) as exc:
        return KernelTraining("kernel_not_identified", None, pl.DataFrame(), str(exc))
    weights = _kernel_weights(kernel)
    if weights is None:
        return KernelTraining("kernel_not_identified", None, kernel,
                              "train-only kernel lacks finite positive bid/ask ranks 1..10")
    return KernelTraining("trained", weights, kernel, None)


def _state_time_lookup(states: pl.DataFrame) -> dict[tuple[str, str, int], int]:
    result: dict[tuple[str, str, int], int] = {}
    needed = {"partition_id", "actor_key", "sort_index", "replay_time_ns"}
    if not needed <= set(states.columns):
        return result
    for row in states.iter_rows(named=True):
        value = row.get("replay_time_ns")
        if type(value) is int:
            result[(str(row["partition_id"]), str(row["actor_key"]), int(row["sort_index"]))] = value
    return result


def native_profile_rows(raw_events: pl.DataFrame, *, instrument: str, event_date: date,
                        split: str, tick_size: float, kernel_weights: Mapping[str, Mapping[int, float]],
                        horizon_seconds: float) -> list[dict[str, Any]]:
    """Run the native producer once for one complete partition/horizon.

    ``execution_metrics`` is retained before any strict/episode filtering.  The
    function records native nulls and lookup recency exactly as emitted by the
    producer rather than imputing a score or endpoint.
    """
    horizon_ns = _horizon_ns(horizon_seconds)
    cap = max(120.0, _partition_span_seconds(raw_events) + 1.0)
    # The frozen adapter policy is reproduced explicitly, changing only the
    # native SCI/MSCI observation horizon.  In particular, the pre-trigger
    # profile remains the producer's canonical top-10 pre-state; it is not a
    # horizon-specific redefinition of the resting-profile estimand.
    result = compute_exploratory_metrics(
        raw_events, top_n=10, tick_size=float(tick_size), window_seconds=float(horizon_seconds),
        withdrawal_window_seconds=20, reversion_horizon_seconds=20, max_rows=None,
        include_level_columns=True, max_deceptive_order_age_seconds=float(cap),
        execution_cluster_max_gap_ms=100, state_actor_keys=None,
        execution_anchor_modes=("passive", "aggressive"), empirical_kernel_weights=kernel_weights,
        use_v3_clocks=True,
    )
    state_times = _state_time_lookup(result.state_time_series)
    rows: list[dict[str, Any]] = []
    for execution in result.execution_metrics.iter_rows(named=True):
        level, role = execution.get("identity_level"), execution.get("execution_anchor_mode")
        if level not in _LEVELS or role not in _ROLES:
            continue
        target = execution.get("post_target_replay_ns")
        post_index = execution.get("post_state_sort_index")
        post_ns = (state_times.get((str(execution.get("partition_id")), str(execution.get("actor_key")), int(post_index)))
                   if type(post_index) is int else None)
        sci, msci = _finite(execution.get("SCI")), _finite(execution.get("MSCI_resting_profile"))
        if msci is None:
            msci = _finite(execution.get("MSCI"))
        observed = sci is not None and msci is not None
        reason = None
        if not observed:
            reason = "native_sci_or_msci_missing"
        elif type(target) is not int or post_ns is None:
            reason = "native_endpoint_recency_missing"
        recency = int(target) - post_ns if type(target) is int and post_ns is not None else None
        if recency is not None and recency < 0:
            raise ValueError("native post state cannot be later than its target")
        rows.append({
            "instrument": instrument, "partition_id": str(execution.get("partition_id")), "event_date": event_date,
            "split": split, "identity_level": level, "actor_key": str(execution.get("actor_key")),
            "execution_anchor_mode": role, "execution_cluster_id": str(execution.get("execution_cluster_id")),
            "horizon_seconds": float(horizon_seconds), "horizon_ns": horizon_ns, "SCI": sci, "MSCI": msci,
            "post_target_replay_ns": target if type(target) is int else None, "post_state_replay_ns": post_ns,
            "recency_ns": recency, "status": "observed" if observed else "missing_native_endpoint",
            "null_reason": reason, "episode_strict_detection": bool(execution.get("episode_strict_detection", False)),
            "max_deceptive_order_age_seconds": cap,
        })
    return rows


def _q90_integer(values: list[int]) -> int | None:
    if not values:
        return None
    return sorted(values)[math.ceil(0.9 * len(values)) - 1]


def _aggregate(paths: list[dict[str, Any]], instrument: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    splits = ("calibration", "retrospective_validation") if instrument == "RISANAMENTO" else ("exploratory_single_day",)
    for level in _LEVELS:
        for role in _ROLES:
            for split in splits:
                for horizon in _HORIZONS:
                    selected = [row for row in paths if row["identity_level"] == level
                                and row["execution_anchor_mode"] == role and row["split"] == split
                                and row["horizon_ns"] == _horizon_ns(horizon)]
                    observed = [row for row in selected if row["SCI"] is not None and row["MSCI"] is not None]
                    recencies = [int(row["recency_ns"]) for row in observed if type(row.get("recency_ns")) is int]
                    status = "observed" if observed else "no_support" if not selected else "missing_native_endpoint"
                    rows.append({"instrument": instrument, "identity_level": level, "execution_anchor_mode": role,
                                 "role": role, "split": split, "horizon_seconds": horizon,
                                 "horizon_ns": _horizon_ns(horizon), "n_cases": len(selected),
                                 "n_observed": len(observed), "n_missing": len(selected) - len(observed),
                                 "sci_median": median([float(row["SCI"]) for row in observed]) if observed else None,
                                 "msci_median": median([float(row["MSCI"]) for row in observed]) if observed else None,
                                 "recency_median_ns": int(median(recencies)) if recencies else None,
                                 "recency_p90_ns": _q90_integer(recencies), "status": status,
                                 "null_reason": None if observed else ("no attributable executions" if not selected else
                                                                        "native SCI/MSCI unavailable")})
    return rows


def _raw_for_partition(raw_path: Path, partition: Mapping[str, Any]) -> pl.DataFrame:
    # Reuse the extraction owner's canonical complete-partition loader.
    from scripts.calibrate_temporal_parameters import _load_partition
    return _load_partition(raw_path, partition)


def _behavior_closure_identity() -> dict[str, str]:
    """Hash the runner/loader plus every module defining native profile values."""
    # The extraction runner owns partition loading and its existing closure
    # covers replay, identity, state, and empirical-clock behavior.  Native
    # SCI/MSCI additionally executes the formula/episode modules below.
    from scripts.calibrate_temporal_parameters import behavior_closure_identity

    root = Path(__file__).resolve().parents[3]
    closure = dict(behavior_closure_identity())
    native_files = (
        "src/spoofing_detection/lob/temporal_calibration_profiles.py",
        "src/spoofing_detection/lob/depth_kernel_calibration.py",
        "src/spoofing_detection/lob/spoofing_metrics.py",
        "src/spoofing_detection/lob/behavioral_gate.py",
        "src/spoofing_detection/lob/candidate_episodes.py",
        "src/spoofing_detection/lob/replay_observation.py",
        "src/spoofing_detection/lob/spoofing_clock.py",
        "src/spoofing_detection/lob/models.py",
        "src/spoofing_detection/lob/enums.py",
    )
    for relative in native_files:
        path = root / relative
        if not path.is_file():
            raise ValueError(f"profile behavior closure source missing: {path}")
        closure[relative] = _sha256(path)
    return dict(sorted(closure.items()))


def _behavior_closure_hash() -> str:
    """Produce the stage-identity digest from the complete named closure."""
    encoded = json.dumps(_behavior_closure_identity(), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def run_profile_diagnostics(output_root: Path | str, design: Mapping[str, Any], instrument: str,
                            *, dates: list[str] | None = None) -> dict[str, Any]:
    """Publish verified private native paths and identity-separated public summaries."""
    root = Path(output_root).resolve()
    manifest, source_hash = _verified_source(root, design, instrument, dates=dates)
    provenance = root / "private" / "provenance.json"
    if not provenance.is_file():
        raise ValueError("verified extract raw provenance required")
    observed = __import__("json").loads(provenance.read_text()).get("observed")
    if not isinstance(observed, Mapping):
        raise ValueError("verified extract observed provenance required")
    raw_info = observed.get("observed_inputs", {}).get("raw_events") if isinstance(observed.get("observed_inputs"), Mapping) else None
    tick = observed.get("baseline", {}).get("tick_size") if isinstance(observed.get("baseline"), Mapping) else None
    if not isinstance(raw_info, Mapping) or not isinstance(raw_info.get("path"), str) or _finite(tick) is None or float(tick) <= 0:
        raise ValueError("actual preflight raw provenance and tick size required")
    raw_path = Path(raw_info["path"])
    if not raw_path.is_file() or _sha256(raw_path) != raw_info.get("sha256"):
        raise ValueError("raw provenance changed after extraction")
    dependencies = {"profile_diagnostics_behavior_closure_sha256": _behavior_closure_hash()}
    identity = _stage_identity("profile_diagnostics", source_hash, design, instrument, dependencies)

    def build(stage: Path) -> None:
        selected = [item for item in manifest.get("partitions", []) if isinstance(item, Mapping)]
        if not selected:
            raise ValueError("canonical selected partitions required")
        raw_by_partition: list[tuple[Mapping[str, Any], pl.DataFrame, str]] = []
        for item in selected:
            partition = item.get("identity", {}).get("partition") if isinstance(item.get("identity"), Mapping) else None
            if not isinstance(partition, Mapping):
                raise ValueError("canonical selected partition identity required")
            day = date.fromisoformat(str(item.get("event_date")))
            raw_by_partition.append((partition, _raw_for_partition(raw_path, partition), _split(instrument, day)))
        train_frames = [raw for _, raw, split in raw_by_partition if split != "retrospective_validation"]
        train_raw = pl.concat(train_frames, how="diagonal_relaxed") if train_frames else pl.DataFrame()
        kernel = train_kernel(train_raw, instrument=instrument, tick_size=float(tick))
        private = stage / "private"; private.mkdir()
        kernel.kernel.write_parquet(private / "train_kernel.parquet")
        _atomic_json(private / "provenance.json", {"instrument": instrument, "source_manifest_sha256": source_hash,
                     "raw_sha256": raw_info["sha256"], "raw_path": str(raw_path),
                     "profile_diagnostics_behavior_closure": _behavior_closure_identity(),
                     "kernel_training_split": "calibration" if instrument == "RISANAMENTO" else "exploratory_single_day",
                     "kernel_train_dates": sorted({str(item.get("event_date")) for item, _, split in raw_by_partition
                                                  if split != "retrospective_validation"}),
                     "kernel_status": kernel.status, "kernel_reason": kernel.reason,
                     "selected_partitions": [{"partition_id": str(item.get("partition_id")), "event_date": str(item.get("event_date"))}
                                             for item, _, _ in raw_by_partition]})
        paths: list[dict[str, Any]] = []
        if kernel.weights is not None:
            for item, raw, split in raw_by_partition:
                day = date.fromisoformat(str(item.get("event_date")))
                for horizon in _HORIZONS:
                    paths.extend(native_profile_rows(raw, instrument=instrument, event_date=day, split=split,
                                                     tick_size=float(tick), kernel_weights=kernel.weights,
                                                     horizon_seconds=horizon))
        _atomic_csv(private / "native_paths.parquet.csv", paths, _PRIVATE_SCHEMA)
        # Parquet is typed private evidence; write after CSV helper's atomic validation.
        pl.DataFrame(paths, schema=_PRIVATE_SCHEMA, strict=False).write_parquet(private / "native_paths.parquet")
        (private / "native_paths.parquet.csv").unlink()
        aggregate = _aggregate(paths, instrument)
        if kernel.weights is None:
            for row in aggregate:
                row["status"] = "kernel_not_identified"
                row["null_reason"] = kernel.reason
        _atomic_csv(stage / "aggregate.csv", aggregate, _PUBLIC_SCHEMA)
        _atomic_json(stage / "summary.json", {"instrument": instrument, "source_manifest_sha256": source_hash,
                     "kernel_status": kernel.status, "kernel_reason": kernel.reason,
                     "kernel_train_dates": sorted({str(item.get("event_date")) for item, _, split in raw_by_partition
                                                  if split != "retrospective_validation"}),
                     "horizons_seconds": list(_HORIZONS), "proposal_only": True,
                     "population": "all attributable native executions; no strict/W/REV/sign gate"})
        if _sha256(raw_path) != raw_info["sha256"]:
            raise ValueError("raw source changed during profile diagnostics")

    resumed = _publish_stage(root, "profile_diagnostics", identity, build)
    return {"status": "profile_diagnostics_complete", "resume_verified": resumed}
