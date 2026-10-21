"""Pure partition-level adapter for the fixed MSCI trajectory grid.

This module deliberately has no filesystem or CLI responsibilities.  It groups
state rows once by local partition/actor, computes every requested execution's
closed grid, and emits the schemas consumed by ``msci_grid_store``.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from collections.abc import Mapping
from typing import Any

import numpy as np
import polars as pl

from spoofing_detection.lob.msci_grid import (
    _coverage_for_partition,
    _integer_ns,
    _profile_valid,
    _support_reason,
    profile_components,
    summarize_increment,
    validate_grid_config,
)
from spoofing_detection.lob.msci_grid_store import SUMMARY_SCHEMA, TRAJECTORY_SCHEMA


_EXECUTION_COLUMNS = (
    "instrument",
    "partition_id",
    "actor_key",
    "execution_cluster_id",
    "deceptive_side",
    "cluster_first_sort_index",
    "cluster_last_sort_index",
    "cluster_end_replay_time_ns",
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
_KEY_COLUMNS = ("instrument", "partition_id", "actor_key", "execution_cluster_id")


def _required_columns(frame: Any, columns: tuple[str, ...], *, label: str) -> pl.DataFrame:
    if not isinstance(frame, pl.DataFrame):
        raise ValueError(f"{label} must be a polars DataFrame")
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing {label} columns: {missing}")
    return frame


def _identity_text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _execution_identity(execution: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return tuple(_identity_text(execution[column], label=column) for column in _KEY_COLUMNS)  # type: ignore[return-value]


def _validate_execution(execution: Mapping[str, Any]) -> None:
    _execution_identity(execution)
    if execution["deceptive_side"] not in {"bid", "ask"}:
        raise ValueError("invalid deceptive_side")
    first = _integer_ns(execution["cluster_first_sort_index"])
    last = _integer_ns(execution["cluster_last_sort_index"])
    # Source cluster_end_replay_time_ns is mandatory.  In particular, no fill
    # anchor field is inspected or used as a fallback.
    end = _integer_ns(execution["cluster_end_replay_time_ns"])
    if first is None or last is None or end is None or first > last:
        raise ValueError("execution indexes and cluster_end_replay_time_ns must be integer ns")


def _state_groups(states: pl.DataFrame) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Build each local actor partition once; never materialize all states at once."""

    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for local_frame in states.select(_STATE_COLUMNS).partition_by(
        ["partition_id", "actor_key"], maintain_order=True
    ):
        if local_frame.height == 0:
            continue
        partition_id = _identity_text(local_frame.item(0, "partition_id"), label="state partition_id")
        actor_key = _identity_text(local_frame.item(0, "actor_key"), label="state actor_key")
        # This is deliberately once per local actor partition, not once per
        # execution or grid node (the production state table can be very large).
        groups[(partition_id, actor_key)] = local_frame.to_dicts()
    return groups


def _invalid_rows(execution: Mapping[str, Any], config: Mapping[str, Any], reason: str) -> list[dict[str, Any]]:
    end_ns = _integer_ns(execution["cluster_end_replay_time_ns"])
    assert end_ns is not None
    rows: list[dict[str, Any]] = []
    for offset_ns in range(0, int(config["horizon_ns"]) + int(config["grid_step_ns"]), int(config["grid_step_ns"])):
        rows.append(
            {
                "instrument": execution["instrument"], "partition_id": execution["partition_id"],
                "actor_key": execution["actor_key"], "execution_cluster_id": execution["execution_cluster_id"],
                "grid_offset_ns": offset_ns, "target_replay_ns": end_ns + offset_ns,
                "selected_sort_index": None, "selected_replay_ns": None, "baseline_sort_index": None,
                "DWI": None, "L_bid_topN": None, "L_ask_topN": None, "SCI": None,
                "C_opposite": None, "C_same": None, "MSCI": None, "valid": False, "reason": reason,
            }
        )
    return rows


def _prepared_local(local: list[dict[str, Any]]) -> tuple[list[dict[str, Any]] | None, str | None]:
    if any(_integer_ns(row["sort_index"]) is None for row in local):
        return None, "invalid_state_clock"
    ordered = sorted(local, key=lambda row: int(row["sort_index"]))
    if any(int(ordered[index - 1]["sort_index"]) >= int(ordered[index]["sort_index"]) for index in range(1, len(ordered))):
        return None, "invalid_canonical_sort_index"
    return ordered, None


def _sample_local_execution(
    execution: Mapping[str, Any],
    local: list[dict[str, Any]],
    coverage: Mapping[str, Any] | None,
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Sampler-equivalent local operation using pre-grouped state records."""

    ordered, preparation_reason = _prepared_local(local)
    if preparation_reason is not None:
        return _invalid_rows(execution, config, preparation_reason)
    assert ordered is not None
    first = int(execution["cluster_first_sort_index"])
    last = int(execution["cluster_last_sort_index"])
    end_ns = int(execution["cluster_end_replay_time_ns"])
    baseline_position = next(
        (index for index in range(len(ordered) - 1, -1, -1) if int(ordered[index]["sort_index"]) < first),
        None,
    )
    if baseline_position is None:
        return _invalid_rows(execution, config, "missing_pre_execution_state")
    t0_position = next(
        (index for index, row in enumerate(ordered) if int(row["sort_index"]) == last), None
    )
    if t0_position is None:
        return _invalid_rows(execution, config, "missing_post_execution_state")
    baseline, t0 = ordered[baseline_position], ordered[t0_position]
    if _integer_ns(baseline["replay_time_ns"]) is None or _integer_ns(t0["replay_time_ns"]) is None:
        return _invalid_rows(execution, config, "invalid_state_clock")
    if int(t0["replay_time_ns"]) != end_ns:
        return _invalid_rows(execution, config, "missing_post_execution_state")
    if not _profile_valid(baseline):
        return _invalid_rows(execution, config, "invalid_baseline_profile")

    horizon_end = end_ns + int(config["horizon_ns"])
    previous_clock: int | None = None
    boundary_position: int | None = None
    for index in range(baseline_position, len(ordered)):
        clock = _integer_ns(ordered[index]["replay_time_ns"])
        if clock is None:
            return _invalid_rows(execution, config, "invalid_state_clock")
        if previous_clock is not None and clock < previous_clock:
            return _invalid_rows(execution, config, "replay_time_regression")
        previous_clock = clock
        if index >= t0_position and clock > horizon_end:
            boundary_position = index
            break

    post_end = len(ordered) if boundary_position is None else boundary_position + 1
    post = ordered[t0_position:post_end]
    post_clocks = [int(row["replay_time_ns"]) for row in post]
    segments, invalid = _coverage_for_partition(coverage, execution["partition_id"])
    rows: list[dict[str, Any]] = []
    step_ns, horizon_ns = int(config["grid_step_ns"]), int(config["horizon_ns"])
    for offset_ns in range(0, horizon_ns + step_ns, step_ns):
        target_ns = end_ns + offset_ns
        selected = t0 if offset_ns == 0 else (post[bisect_right(post_clocks, target_ns) - 1] if bisect_right(post_clocks, target_ns) else None)
        row: dict[str, Any] = {
            "instrument": execution["instrument"], "partition_id": execution["partition_id"],
            "actor_key": execution["actor_key"], "execution_cluster_id": execution["execution_cluster_id"],
            "grid_offset_ns": offset_ns, "target_replay_ns": target_ns,
            "selected_sort_index": None if selected is None else int(selected["sort_index"]),
            "selected_replay_ns": None if selected is None else int(selected["replay_time_ns"]),
            "baseline_sort_index": int(baseline["sort_index"]),
            "DWI": None if selected is None else selected["DWI"],
            "L_bid_topN": None if selected is None else selected["L_bid_topN"],
            "L_ask_topN": None if selected is None else selected["L_ask_topN"],
            "SCI": None, "C_opposite": None, "C_same": None, "MSCI": None,
            "valid": False, "reason": None,
        }
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
    return rows


def _finite_msci(value: Any) -> float | None:
    if value is None or isinstance(value, (bool, np.bool_)):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) and -1.0 <= parsed <= 2.0 else None


def summarize_grid(trajectory: pl.DataFrame, *, lag_ns: int) -> dict[str, float | int | bool | None]:
    """Summarize one execution and retain rich endpoints outside store schema."""

    _required_columns(trajectory, ("grid_offset_ns", "MSCI", "valid"), label="trajectory")
    if trajectory.height < 2:
        raise ValueError("trajectory must contain at least two grid nodes")
    ordered = trajectory.sort("grid_offset_ns")
    core = summarize_increment(
        ordered["grid_offset_ns"].to_numpy(),
        ordered["MSCI"].to_numpy(),
        ordered["valid"].to_numpy(),
        lag_ns=lag_ns,
    )
    values = [_finite_msci(value) if valid is True else None for value, valid in zip(ordered["MSCI"].to_list(), ordered["valid"].to_list(), strict=True)]
    valid_values = [(index, value) for index, value in enumerate(values) if value is not None]
    maximum = max((value for _, value in valid_values), default=None)
    max_offset = next((int(ordered[index, "grid_offset_ns"]) for index, value in valid_values if value == maximum), None)
    return {
        **core,
        "MSCI_t0": values[0],
        "MSCI_horizon": values[-1],
        "MSCI_max": maximum,
        "MSCI_max_offset_ns": max_offset,
    }


def compute_partition_grids(
    executions: pl.DataFrame,
    states: pl.DataFrame,
    coverage: Mapping[str, Any] | None,
    config: Mapping[str, Any],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Compute closed trajectories and core summaries for one supplied partition unit.

    No coverage is manufactured from state timestamps.  Each supplied execution
    receives all grid nodes even when it is invalid; malformed source execution
    identities/clocks instead fail before an unverifiable grid can be addressed.
    """

    validate_grid_config(config)
    _required_columns(executions, _EXECUTION_COLUMNS, label="executions")
    _required_columns(states, _STATE_COLUMNS, label="states")
    execution_rows = executions.select(_EXECUTION_COLUMNS).to_dicts()
    identities = [_execution_identity(row) for row in execution_rows]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate execution identity")
    for execution in execution_rows:
        _validate_execution(execution)

    groups = _state_groups(states)
    trajectory_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for execution in execution_rows:
        local = groups.get((execution["partition_id"], execution["actor_key"]), [])
        rows = _sample_local_execution(execution, local, coverage, config)
        trajectory_rows.extend(rows)
        rich = summarize_grid(pl.DataFrame(rows, schema=TRAJECTORY_SCHEMA, strict=False), lag_ns=int(config["lag_ns"]))
        summary_rows.append({
            "instrument": execution["instrument"], "partition_id": execution["partition_id"],
            "actor_key": execution["actor_key"], "execution_cluster_id": execution["execution_cluster_id"],
            **{name: rich[name] for name in SUMMARY_SCHEMA if name not in _KEY_COLUMNS},
        })
    return (
        pl.DataFrame(trajectory_rows, schema=TRAJECTORY_SCHEMA, strict=False),
        pl.DataFrame(summary_rows, schema=SUMMARY_SCHEMA, strict=False),
    )


def _same_value(observed: Any, expected: Any) -> bool:
    if observed is None or expected is None:
        return observed is expected
    if isinstance(observed, bool) or isinstance(expected, bool):
        return observed == expected
    if isinstance(observed, (int, np.integer)) and isinstance(expected, (int, np.integer)):
        return int(observed) == int(expected)
    if isinstance(observed, (float, np.floating)) and isinstance(expected, (float, np.floating)):
        return math.isclose(float(observed), float(expected), rel_tol=1e-12, abs_tol=1e-12)
    return observed == expected


def _reconstruct_increment(rows: list[dict[str, Any]], *, lag_ns: int) -> dict[str, float | int | bool | None]:
    """Independent direct-loop reconstruction for verifier use only."""

    offsets = [int(row["grid_offset_ns"]) for row in rows]
    step = offsets[1] - offsets[0]
    lag_steps = lag_ns // step
    values = [_finite_msci(row["MSCI"]) if row["valid"] is True else None for row in rows]
    complete = all(value is not None for value in values)
    pairs: list[tuple[int, float]] = []
    for start in range(len(rows) - lag_steps):
        start_value, end_value = values[start], values[start + lag_steps]
        if start_value is not None and end_value is not None:
            pairs.append((start, max(end_value - start_value, 0.0)))
    partial = max((increment for _, increment in pairs), default=None)
    primary = partial if complete else None
    arg_start = arg_end = None
    if primary is not None and primary > 0:
        index = next(start for start, increment in pairs if increment == primary)
        arg_start, arg_end = offsets[index], offsets[index + lag_steps]
    return {
        "J": primary, "J_observed_partial": partial,
        "argmax_end_offset_ns": arg_end, "argmax_start_offset_ns": arg_start,
        "complete_grid": complete, "valid_pair_count": len(pairs),
    }


def _expected_invalid_rows(
    execution: Mapping[str, Any], config: Mapping[str, Any], reason: str
) -> list[dict[str, Any]]:
    """Independent expected representation for terminal source-evidence failure."""

    end_ns = int(execution["cluster_end_replay_time_ns"])
    rows: list[dict[str, Any]] = []
    for offset_ns in range(0, int(config["horizon_ns"]) + int(config["grid_step_ns"]), int(config["grid_step_ns"])):
        rows.append({
            "instrument": execution["instrument"], "partition_id": execution["partition_id"],
            "actor_key": execution["actor_key"], "execution_cluster_id": execution["execution_cluster_id"],
            "grid_offset_ns": offset_ns, "target_replay_ns": end_ns + offset_ns,
            "selected_sort_index": None, "selected_replay_ns": None, "baseline_sort_index": None,
            "DWI": None, "L_bid_topN": None, "L_ask_topN": None, "SCI": None,
            "C_opposite": None, "C_same": None, "MSCI": None, "valid": False, "reason": reason,
        })
    return rows


def _expected_rows_with_states(
    execution: Mapping[str, Any],
    local: list[dict[str, Any]],
    coverage: Mapping[str, Any] | None,
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Reconstruct the bounded canonical state/coverage contract without sampling output.

    The verifier deliberately does not invoke ``_sample_local_execution`` or
    ``compute_partition_grids``.  It derives terminal invalidity first, then
    uses a bounded canonical segment and an explicit per-target as-of loop.
    """

    prepared, reason = _prepared_local(local)
    if reason is not None or prepared is None:
        return _expected_invalid_rows(execution, config, reason or "invalid_state_clock")
    first, last = int(execution["cluster_first_sort_index"]), int(execution["cluster_last_sort_index"])
    baseline_position = next(
        (index for index in range(len(prepared) - 1, -1, -1) if int(prepared[index]["sort_index"]) < first),
        None,
    )
    if baseline_position is None:
        return _expected_invalid_rows(execution, config, "missing_pre_execution_state")
    t0_position = next((index for index, row in enumerate(prepared) if int(row["sort_index"]) == last), None)
    if t0_position is None:
        return _expected_invalid_rows(execution, config, "missing_post_execution_state")
    baseline, t0 = prepared[baseline_position], prepared[t0_position]
    baseline_clock, t0_clock = _integer_ns(baseline["replay_time_ns"]), _integer_ns(t0["replay_time_ns"])
    if baseline_clock is None or t0_clock is None:
        return _expected_invalid_rows(execution, config, "invalid_state_clock")
    end_ns = int(execution["cluster_end_replay_time_ns"])
    if t0_clock != end_ns:
        return _expected_invalid_rows(execution, config, "missing_post_execution_state")
    if not _profile_valid(baseline):
        return _expected_invalid_rows(execution, config, "invalid_baseline_profile")

    horizon_end = end_ns + int(config["horizon_ns"])
    bounded: list[dict[str, Any]] = []
    previous_clock: int | None = None
    for index in range(baseline_position, len(prepared)):
        candidate = prepared[index]
        clock = _integer_ns(candidate["replay_time_ns"])
        if clock is None:
            return _expected_invalid_rows(execution, config, "invalid_state_clock")
        if previous_clock is not None and clock < previous_clock:
            return _expected_invalid_rows(execution, config, "replay_time_regression")
        previous_clock = clock
        bounded.append(candidate)
        if index >= t0_position and clock > horizon_end:
            break

    post = [row for row in bounded if int(row["sort_index"]) >= last]
    segments, invalid = _coverage_for_partition(coverage, execution["partition_id"])
    expected_rows: list[dict[str, Any]] = []
    for offset_ns in range(0, int(config["horizon_ns"]) + int(config["grid_step_ns"]), int(config["grid_step_ns"])):
        target_ns = end_ns + offset_ns
        selected = t0 if offset_ns == 0 else None
        if offset_ns:
            for candidate in post:
                if int(candidate["replay_time_ns"]) <= target_ns:
                    selected = candidate
                else:
                    break
        row: dict[str, Any] = {
            "instrument": execution["instrument"], "partition_id": execution["partition_id"],
            "actor_key": execution["actor_key"], "execution_cluster_id": execution["execution_cluster_id"],
            "grid_offset_ns": offset_ns, "target_replay_ns": target_ns,
            "selected_sort_index": None if selected is None else int(selected["sort_index"]),
            "selected_replay_ns": None if selected is None else int(selected["replay_time_ns"]),
            "baseline_sort_index": int(baseline["sort_index"]),
            "DWI": None if selected is None else selected["DWI"],
            "L_bid_topN": None if selected is None else selected["L_bid_topN"],
            "L_ask_topN": None if selected is None else selected["L_ask_topN"],
            "SCI": None, "C_opposite": None, "C_same": None, "MSCI": None,
            "valid": False, "reason": None,
        }
        if selected is None:
            row["reason"] = "missing_post_execution_state"
        else:
            support = _support_reason(segments, invalid, baseline_clock, target_ns)
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
        expected_rows.append(row)
    return expected_rows


def _verify_rows_with_states(
    execution: Mapping[str, Any],
    rows: list[dict[str, Any]],
    local: list[dict[str, Any]],
    coverage: Mapping[str, Any] | None,
    config: Mapping[str, Any],
) -> None:
    expected_rows = _expected_rows_with_states(execution, local, coverage, config)
    terminal_reason = expected_rows[0]["reason"] if all(
        expected["baseline_sort_index"] is None for expected in expected_rows
    ) else None
    if terminal_reason is not None and any(
        observed["valid"] is not False or observed["reason"] != terminal_reason
        for observed in rows
    ):
        raise ValueError(f"trajectory must be the exact terminal invalid grid: {terminal_reason}")
    for observed, expected in zip(rows, expected_rows, strict=True):
        # Check the scientific status first.  This makes absent coverage and
        # source-evidence failures explicit rather than reporting a derivative
        # component mismatch.
        for field in ("valid", "reason"):
            if not _same_value(observed[field], expected[field]):
                raise ValueError(
                    f"trajectory {field} disagrees with state and coverage evidence; "
                    f"expected reason {expected['reason']!r}"
                )
        for field, expected_value in expected.items():
            if field in {"valid", "reason"} or _same_value(observed[field], expected_value):
                continue
            if field == "baseline_sort_index":
                raise ValueError("trajectory baseline does not agree with supplied local states")
            if field in {"selected_sort_index", "selected_replay_ns"}:
                raise ValueError("trajectory selected state violates canonical as-of rule")
            raise ValueError(f"trajectory {field} disagrees with supplied state and coverage evidence")


def verify_partition_grids(
    executions: pl.DataFrame,
    trajectory: pl.DataFrame,
    summary: pl.DataFrame,
    config: Mapping[str, Any],
    *,
    states: pl.DataFrame | None = None,
    coverage: Mapping[str, Any] | None = None,
) -> None:
    """Validate stored-compatible outputs; state-free use is structural/arithmetic only.

    Supplying ``states`` makes this a state-aware verification.  In that mode
    ``coverage=None`` is explicit unverified coverage, so no node can be
    accepted as scientifically valid.  Without states the function deliberately
    makes no scientific-completeness claim.
    """

    validate_grid_config(config)
    _required_columns(executions, _EXECUTION_COLUMNS, label="executions")
    if trajectory.schema != TRAJECTORY_SCHEMA or summary.schema != SUMMARY_SCHEMA:
        raise ValueError("trajectory or summary schema mismatch")
    execution_rows = executions.select(_EXECUTION_COLUMNS).to_dicts()
    execution_by_key = {_execution_identity(row): row for row in execution_rows}
    if len(execution_by_key) != len(execution_rows):
        raise ValueError("duplicate execution identity")
    trajectory_groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in trajectory.to_dicts():
        key = tuple(row[column] for column in _KEY_COLUMNS)
        trajectory_groups.setdefault(key, []).append(row)
    summary_by_key = {tuple(row[column] for column in _KEY_COLUMNS): row for row in summary.to_dicts()}
    if set(trajectory_groups) != set(execution_by_key) or set(summary_by_key) != set(execution_by_key):
        raise ValueError("trajectory or summary execution membership mismatch")
    if len(summary_by_key) != summary.height:
        raise ValueError("duplicate summary execution identity")

    state_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    if states is not None:
        _required_columns(states, _STATE_COLUMNS, label="states")
        state_groups = _state_groups(states)
    step, horizon, lag = int(config["grid_step_ns"]), int(config["horizon_ns"]), int(config["lag_ns"])
    expected_offsets = list(range(0, horizon + step, step))
    for key, execution in execution_by_key.items():
        rows = sorted(trajectory_groups[key], key=lambda row: int(row["grid_offset_ns"]))
        offsets = [row["grid_offset_ns"] for row in rows]
        if offsets != expected_offsets or len(rows) != len(expected_offsets):
            raise ValueError("trajectory does not contain exactly the closed grid nodes")
        end = int(execution["cluster_end_replay_time_ns"])
        if any(row["target_replay_ns"] != end + int(row["grid_offset_ns"]) for row in rows):
            raise ValueError("trajectory target replay clocks do not use canonical cluster end")
        if any(row["selected_replay_ns"] is not None and int(row["selected_replay_ns"]) > int(row["target_replay_ns"]) for row in rows):
            raise ValueError("trajectory selects a future clock")
        reconstructed = _reconstruct_increment(rows, lag_ns=lag)
        for field, expected in reconstructed.items():
            if not _same_value(summary_by_key[key][field], expected):
                raise ValueError(f"summary {field} disagrees with independently reconstructed trajectory")
        if states is not None:
            _verify_rows_with_states(
                execution,
                rows,
                state_groups.get((execution["partition_id"], execution["actor_key"]), []),
                coverage,
                config,
            )
