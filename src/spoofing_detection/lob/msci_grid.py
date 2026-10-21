"""Pure MSCI grid-study configuration, sampling, and numerical kernels."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import polars as pl

from spoofing_detection.lob.spoofing_metrics import _collapse, _finite_signed_msci

_SCHEMA = "msci_grid_study_v1"
_SOURCE_CLOCK_POLICY = "spoofing_detection_sequence_replay_trade_fill_sequence_cancel_v3_utc_provisional"
_GRID_POLICY = "fixed_pre_exact_post_execution_closed_grid_v1"
_COHORT = "exact_top_positive_W_5pct_then_U_over_Q_gt_1_before_prices"
_REQUIRED_CONFIG_KEYS = frozenset(
    {
        "schema",
        "source_clock_policy",
        "grid_policy",
        "grid_step_ns",
        "lag_ns",
        "horizon_ns",
        "complete_grid_required",
        "cohort",
        "grid_sensitivity_ns",
        "lag_sensitivity_ns",
        "lag_sensitivity_grid_ns",
        "selection_threshold",
    }
)


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_grid_case(*, step_ns: Any, lag_ns: Any, horizon_ns: int, label: str) -> None:
    step = _positive_integer(step_ns, name=f"{label} grid step")
    lag = _positive_integer(lag_ns, name=f"{label} lag")
    if lag > horizon_ns:
        raise ValueError(f"{label} lag must not exceed horizon_ns")
    if horizon_ns % step != 0:
        raise ValueError(f"{label} horizon_ns must be divisible by grid_step_ns")
    if lag % step != 0:
        raise ValueError(f"{label} lag_ns must be divisible by grid_step_ns")


def _sensitivity_values(value: Any, *, name: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    parsed = [_positive_integer(item, name=name) for item in value]
    if len(set(parsed)) != len(parsed):
        raise ValueError(f"{name} must not contain duplicates")
    return parsed


def validate_grid_config(config: Mapping[str, Any]) -> None:
    """Validate the closed-grid study's frozen scientific contract.

    The primary grid must use the specified source clock, observation policy,
    cohort and complete-grid rule. Sensitivity cases are validated independently
    against the same fixed horizon.
    """

    if not isinstance(config, Mapping):
        raise ValueError("MSCI grid config must be an object")
    actual_keys = set(config)
    missing = sorted(_REQUIRED_CONFIG_KEYS - actual_keys)
    unknown = sorted(actual_keys - _REQUIRED_CONFIG_KEYS)
    if missing:
        raise ValueError(f"missing required MSCI grid config fields: {missing}")
    if unknown:
        raise ValueError(f"unknown MSCI grid config fields: {unknown}")
    if config["schema"] != _SCHEMA:
        raise ValueError(f"schema must be {_SCHEMA!r}")
    if config["source_clock_policy"] != _SOURCE_CLOCK_POLICY:
        raise ValueError("incompatible source_clock_policy")
    if config["grid_policy"] != _GRID_POLICY:
        raise ValueError("incompatible grid_policy")
    if config["complete_grid_required"] is not True:
        raise ValueError("complete_grid_required must be true")
    if config["cohort"] != _COHORT:
        raise ValueError("incompatible cohort")
    if config["selection_threshold"] is not None:
        raise ValueError("selection_threshold must be null")

    horizon_ns = _positive_integer(config["horizon_ns"], name="horizon_ns")
    _validate_grid_case(
        step_ns=config["grid_step_ns"],
        lag_ns=config["lag_ns"],
        horizon_ns=horizon_ns,
        label="primary",
    )

    grid_sensitivity = _sensitivity_values(config["grid_sensitivity_ns"], name="grid_sensitivity_ns")
    for step_ns in grid_sensitivity:
        _validate_grid_case(
            step_ns=step_ns,
            lag_ns=config["lag_ns"],
            horizon_ns=horizon_ns,
            label="grid sensitivity",
        )

    sensitivity_grid_ns = _positive_integer(
        config["lag_sensitivity_grid_ns"], name="lag_sensitivity_grid_ns"
    )
    lag_sensitivity = _sensitivity_values(config["lag_sensitivity_ns"], name="lag_sensitivity_ns")
    for lag_ns in lag_sensitivity:
        _validate_grid_case(
            step_ns=sensitivity_grid_ns,
            lag_ns=lag_ns,
            horizon_ns=horizon_ns,
            label="lag sensitivity",
        )


def _finite_float(value: Any, *, lower: float, upper: float) -> float | None:
    if value is None or isinstance(value, (bool, np.bool_)):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) and lower <= parsed <= upper else None


def profile_components(
    pre: Mapping[str, Any], post: Mapping[str, Any], deceptive_side: str
) -> dict[str, float | None]:
    """Compute pointwise signed MSCI components from two resting profiles."""

    if deceptive_side not in {"bid", "ask"}:
        raise ValueError("invalid deceptive_side")

    pre_dwi = _finite_float(pre.get("DWI"), lower=-1.0, upper=1.0)
    post_dwi = _finite_float(post.get("DWI"), lower=-1.0, upper=1.0)
    sci = abs(pre_dwi - post_dwi) if pre_dwi is not None and post_dwi is not None else None

    bid = _collapse(
        _finite_float(pre.get("L_bid_topN"), lower=0.0, upper=1.0),
        _finite_float(post.get("L_bid_topN"), lower=0.0, upper=1.0),
    )
    ask = _collapse(
        _finite_float(pre.get("L_ask_topN"), lower=0.0, upper=1.0),
        _finite_float(post.get("L_ask_topN"), lower=0.0, upper=1.0),
    )
    c_opposite, c_same = (bid, ask) if deceptive_side == "bid" else (ask, bid)
    return {
        "SCI": sci,
        "C_opposite": c_opposite,
        "C_same": c_same,
        "MSCI": _finite_signed_msci(sci, c_opposite, c_same),
    }


def summarize_increment(
    offset_ns: Any, msci: Any, valid: Any, *, lag_ns: int
) -> dict[str, float | int | bool | None]:
    """Summarize the positive lagged MSCI increment on a regular closed grid."""

    offsets = np.asarray(offset_ns)
    raw_values = np.asarray(msci, dtype=object)
    validity = np.asarray(valid, dtype=bool)
    if (
        offsets.ndim != 1
        or raw_values.ndim != 1
        or validity.ndim != 1
        or raw_values.shape != offsets.shape
        or validity.shape != offsets.shape
        or len(offsets) < 2
    ):
        raise ValueError("incompatible grid arrays")
    if offsets.dtype.kind not in "iu" or int(offsets[0]) != 0:
        raise ValueError("integer offsets from zero required")

    step_ns = int(offsets[1]) - int(offsets[0])
    if step_ns <= 0 or any(
        int(offsets[index]) - int(offsets[index - 1]) != step_ns
        for index in range(1, len(offsets))
    ):
        raise ValueError("regular increasing grid required")
    if isinstance(lag_ns, bool) or not isinstance(lag_ns, int) or lag_ns <= 0 or lag_ns % step_ns:
        raise ValueError("lag must be a positive integer grid multiple")
    lag_steps = lag_ns // step_ns
    if lag_steps >= len(offsets):
        raise ValueError("lag exceeds horizon")

    values = np.full(raw_values.shape, np.nan, dtype=float)
    finite_msci = np.zeros(raw_values.shape, dtype=bool)
    for index, value in enumerate(raw_values):
        parsed = _finite_float(value, lower=-1.0, upper=2.0)
        if parsed is not None:
            values[index] = parsed
            finite_msci[index] = True
    finite_valid = validity & finite_msci
    pair_valid = finite_valid[lag_steps:] & finite_valid[:-lag_steps]
    increments = np.maximum(values[lag_steps:] - values[:-lag_steps], 0.0)
    valid_indices = np.flatnonzero(pair_valid)
    partial = float(increments[valid_indices].max()) if len(valid_indices) else None
    complete_grid = bool(finite_valid.all())
    primary = partial if complete_grid else None

    end_offset_ns: int | None = None
    if primary is not None and primary > 0:
        start_index = int(valid_indices[np.argmax(increments[valid_indices])])
        end_offset_ns = int(offsets[start_index + lag_steps])
    return {
        "J": primary,
        "J_observed_partial": partial,
        "argmax_end_offset_ns": end_offset_ns,
        "argmax_start_offset_ns": end_offset_ns - lag_ns if end_offset_ns is not None else None,
        "complete_grid": complete_grid,
        "valid_pair_count": int(pair_valid.sum()),
    }


_EXECUTION_KEYS = frozenset(
    {
        "partition_id",
        "actor_key",
        "execution_cluster_id",
        "deceptive_side",
        "cluster_first_sort_index",
        "cluster_last_sort_index",
        "cluster_end_replay_time_ns",
    }
)
_STATE_COLUMNS = (
    "partition_id",
    "actor_key",
    "sort_index",
    "replay_time_ns",
    "DWI",
    "L_bid_topN",
    "L_ask_topN",
)


def _grid_schema(execution: Mapping[str, Any]) -> dict[str, pl.DataType]:
    schema: dict[str, pl.DataType] = {
        "partition_id": pl.String,
        "actor_key": pl.String,
        "execution_cluster_id": pl.String,
        "grid_offset_ns": pl.Int64,
        "target_replay_ns": pl.Int64,
        "selected_sort_index": pl.Int64,
        "selected_replay_ns": pl.Int64,
        "baseline_sort_index": pl.Int64,
        "DWI": pl.Float64,
        "L_bid_topN": pl.Float64,
        "L_ask_topN": pl.Float64,
        "SCI": pl.Float64,
        "C_opposite": pl.Float64,
        "C_same": pl.Float64,
        "MSCI": pl.Float64,
        "valid": pl.Boolean,
        "reason": pl.String,
    }
    if "instrument" in execution:
        schema = {"instrument": pl.String, **schema}
    return schema


def _grid_dataframe(rows: list[dict[str, Any]], execution: Mapping[str, Any]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=_grid_schema(execution), strict=False)


def _integer_ns(value: Any) -> int | None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        return None
    return int(value)


def _empty_grid_rows(execution: Mapping[str, Any], config: Mapping[str, Any], reason: str) -> list[dict[str, Any]]:
    step_ns = int(config["grid_step_ns"])
    horizon_ns = int(config["horizon_ns"])
    end_ns = _integer_ns(execution["cluster_end_replay_time_ns"])
    assert end_ns is not None
    rows: list[dict[str, Any]] = []
    for offset_ns in range(0, horizon_ns + step_ns, step_ns):
        row = {
            "partition_id": execution["partition_id"],
            "actor_key": execution["actor_key"],
            "execution_cluster_id": execution["execution_cluster_id"],
            "grid_offset_ns": offset_ns,
            "target_replay_ns": end_ns + offset_ns,
            "selected_sort_index": None,
            "selected_replay_ns": None,
            "baseline_sort_index": None,
            "DWI": None,
            "L_bid_topN": None,
            "L_ask_topN": None,
            "SCI": None,
            "C_opposite": None,
            "C_same": None,
            "MSCI": None,
            "valid": False,
            "reason": reason,
        }
        if "instrument" in execution:
            row = {"instrument": execution["instrument"], **row}
        rows.append(row)
    return rows


def _coverage_for_partition(coverage: Any, partition_id: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not isinstance(coverage, Mapping):
        return [], []
    raw_segments = coverage.get("segments")
    raw_invalid = coverage.get("invalid_intervals", [])
    if not isinstance(raw_segments, list) or not isinstance(raw_invalid, list):
        raise ValueError("coverage segments and invalid_intervals must be lists")
    segments: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    for segment in raw_segments:
        if not isinstance(segment, Mapping):
            raise ValueError("coverage segment must be an object")
        if segment.get("partition_id") != partition_id:
            continue
        start = _integer_ns(segment.get("start_ns"))
        end = _integer_ns(segment.get("end_ns"))
        if start is None or end is None or end < start:
            raise ValueError("coverage segment bounds must be increasing integer ns")
        if segment.get("attested") is True and isinstance(segment.get("source"), str) and segment["source"]:
            segments.append({"start_ns": start, "end_ns": end})
    for interval in raw_invalid:
        if not isinstance(interval, Mapping):
            raise ValueError("coverage invalid interval must be an object")
        if interval.get("partition_id") != partition_id:
            continue
        start = _integer_ns(interval.get("start_ns"))
        end = _integer_ns(interval.get("end_ns"))
        if start is None or end is None or end < start:
            raise ValueError("coverage invalid interval bounds must be increasing integer ns")
        invalid.append({"start_ns": start, "end_ns": end})
    return segments, invalid


def _support_reason(
    segments: list[dict[str, Any]], invalid: list[dict[str, Any]], start_ns: int, end_ns: int
) -> str | None:
    if not segments:
        return "coverage_unverified"
    if any(item["start_ns"] <= end_ns and start_ns <= item["end_ns"] for item in invalid):
        return "known_coverage_gap"
    covered_through = start_ns
    for segment in sorted(segments, key=lambda item: (item["start_ns"], item["end_ns"])):
        if segment["end_ns"] < covered_through:
            continue
        if segment["start_ns"] > covered_through:
            return "outside_coverage"
        covered_through = max(covered_through, segment["end_ns"])
        if covered_through >= end_ns:
            return None
    return "outside_coverage"


def _profile_valid(profile: Mapping[str, Any]) -> bool:
    return all(
        _finite_float(profile.get(name), lower=lower, upper=upper) is not None
        for name, lower, upper in (
            ("DWI", -1.0, 1.0),
            ("L_bid_topN", 0.0, 1.0),
            ("L_ask_topN", 0.0, 1.0),
        )
    )


def sample_execution_grid(
    execution: Mapping[str, Any], states: pl.DataFrame, coverage: Mapping[str, Any] | None, config: Mapping[str, Any]
) -> pl.DataFrame:
    """Sample one execution on its fixed baseline and closed, attested post grid.

    ``coverage`` is deliberately narrow: mapping fields ``segments`` and optional
    ``invalid_intervals`` hold partition-specific inclusive integer-ns intervals.
    A usable segment has ``attested: true`` and a non-empty evidence ``source``.
    Missing attestation, unsupported intervals, and known gaps produce explicit
    invalid rows; this function does not infer coverage from replay pauses.
    """

    validate_grid_config(config)
    if not isinstance(execution, Mapping):
        raise ValueError("execution must be an object")
    missing = sorted(_EXECUTION_KEYS - set(execution))
    if missing:
        raise ValueError(f"missing execution fields: {missing}")
    first_index = _integer_ns(execution["cluster_first_sort_index"])
    last_index = _integer_ns(execution["cluster_last_sort_index"])
    end_ns = _integer_ns(execution["cluster_end_replay_time_ns"])
    if first_index is None or last_index is None or end_ns is None or first_index > last_index:
        raise ValueError("execution indexes and end clock must be integer ns")
    if execution["deceptive_side"] not in {"bid", "ask"}:
        raise ValueError("invalid deceptive_side")
    if not isinstance(states, pl.DataFrame):
        raise ValueError("states must be a polars DataFrame")
    missing_state_columns = [column for column in _STATE_COLUMNS if column not in states.columns]
    if missing_state_columns:
        raise ValueError(f"missing state columns: {missing_state_columns}")

    local = (
        states.filter(
            (pl.col("partition_id") == execution["partition_id"])
            & (pl.col("actor_key") == execution["actor_key"])
        )
        .select(_STATE_COLUMNS)
        .to_dicts()
    )
    if any(_integer_ns(row["sort_index"]) is None for row in local):
        return _grid_dataframe(_empty_grid_rows(execution, config, "invalid_state_clock"), execution)
    local.sort(key=lambda row: int(row["sort_index"]))
    if any(int(local[index - 1]["sort_index"]) >= int(local[index]["sort_index"]) for index in range(1, len(local))):
        return _grid_dataframe(_empty_grid_rows(execution, config, "invalid_canonical_sort_index"), execution)

    baseline_position = next(
        (index for index in range(len(local) - 1, -1, -1) if int(local[index]["sort_index"]) < first_index),
        None,
    )
    if baseline_position is None:
        return _grid_dataframe(_empty_grid_rows(execution, config, "missing_pre_execution_state"), execution)
    t0_position = next(
        (index for index, row in enumerate(local) if int(row["sort_index"]) == last_index), None
    )
    if t0_position is None:
        return _grid_dataframe(_empty_grid_rows(execution, config, "missing_post_execution_state"), execution)
    baseline = local[baseline_position]
    t0 = local[t0_position]
    if _integer_ns(baseline["replay_time_ns"]) is None or _integer_ns(t0["replay_time_ns"]) is None:
        return _grid_dataframe(_empty_grid_rows(execution, config, "invalid_state_clock"), execution)
    if int(t0["replay_time_ns"]) != end_ns:
        return _grid_dataframe(_empty_grid_rows(execution, config, "missing_post_execution_state"), execution)
    if not _profile_valid(baseline):
        return _grid_dataframe(_empty_grid_rows(execution, config, "invalid_baseline_profile"), execution)

    horizon_end_ns = end_ns + int(config["horizon_ns"])
    previous_clock: int | None = None
    boundary_position: int | None = None
    for index in range(baseline_position, len(local)):
        clock = _integer_ns(local[index]["replay_time_ns"])
        if clock is None:
            return _grid_dataframe(_empty_grid_rows(execution, config, "invalid_state_clock"), execution)
        if previous_clock is not None and clock < previous_clock:
            return _grid_dataframe(_empty_grid_rows(execution, config, "replay_time_regression"), execution)
        previous_clock = clock
        if index >= t0_position and clock > horizon_end_ns:
            boundary_position = index
            break

    segments, invalid = _coverage_for_partition(coverage, execution["partition_id"])
    post_end = len(local) if boundary_position is None else boundary_position + 1
    post = local[t0_position:post_end]
    post_clocks = np.asarray([int(row["replay_time_ns"]) for row in post], dtype=np.int64)
    step_ns = int(config["grid_step_ns"])
    horizon_ns = int(config["horizon_ns"])
    rows: list[dict[str, Any]] = []
    for offset_ns in range(0, horizon_ns + step_ns, step_ns):
        target_ns = end_ns + offset_ns
        if offset_ns == 0:
            selected = t0
        else:
            position = int(np.searchsorted(post_clocks, target_ns, side="right"))
            selected = post[position - 1] if position else None
        row = {
            "partition_id": execution["partition_id"],
            "actor_key": execution["actor_key"],
            "execution_cluster_id": execution["execution_cluster_id"],
            "grid_offset_ns": offset_ns,
            "target_replay_ns": target_ns,
            "selected_sort_index": None if selected is None else int(selected["sort_index"]),
            "selected_replay_ns": None if selected is None else int(selected["replay_time_ns"]),
            "baseline_sort_index": int(baseline["sort_index"]),
            "DWI": None if selected is None else selected["DWI"],
            "L_bid_topN": None if selected is None else selected["L_bid_topN"],
            "L_ask_topN": None if selected is None else selected["L_ask_topN"],
            "SCI": None,
            "C_opposite": None,
            "C_same": None,
            "MSCI": None,
            "valid": False,
            "reason": None,
        }
        if "instrument" in execution:
            row = {"instrument": execution["instrument"], **row}
        if selected is None:
            row["reason"] = "missing_post_execution_state"
        else:
            support = _support_reason(segments, invalid, int(baseline["replay_time_ns"]), target_ns)
            if support is not None:
                row["reason"] = support
            elif not _profile_valid(selected):
                row["reason"] = "invalid_post_execution_profile"
            else:
                components = profile_components(baseline, selected, execution["deceptive_side"])
                row.update(components)
                row["valid"] = components["MSCI"] is not None
                if not row["valid"]:
                    row["reason"] = "invalid_profile_components"
        rows.append(row)
    return _grid_dataframe(rows, execution)
