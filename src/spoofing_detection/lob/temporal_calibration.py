"""Pure, proposal-only temporal calibration estimators.

The functions here deliberately consume primitive observations rather than replay
book state.  They return descriptive calibration inputs, never detector labels
or claims of exponential dynamics.
"""
from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from datetime import date, datetime
from decimal import Decimal
from fractions import Fraction
import math
import random
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

LEVELS = ("client_original", "firm")
_NANOSECONDS_PER_SECOND = 1_000_000_000


def _integer_ns(value: Any, *, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be nonnegative integer nanoseconds")
    return value


def _fraction(value: Any, *, name: str) -> Fraction:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite numeric value")
    if isinstance(value, Fraction):
        result = value
    elif isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError(f"{name} must be a finite numeric value")
        result = Fraction(value)
    elif isinstance(value, int):
        result = Fraction(value)
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be a finite numeric value")
        result = Fraction(str(value))
    else:
        raise ValueError(f"{name} must be a finite numeric value")
    return result


def _quantile(value: Any) -> Fraction:
    result = _fraction(value, name="quantile")
    if not 0 <= result <= 1:
        raise ValueError("quantile must lie in [0, 1]")
    return result


def _strict_quantile(value: Any) -> Fraction:
    result = _quantile(value)
    if not 0 < result < 1:
        raise ValueError("quantile must lie strictly between zero and one")
    return result


def higher_quantile_ns(values: Iterable[int], quantile: Any) -> int | None:
    """Return the exact higher order statistic at a quantile, or ``None`` empty."""
    q = _quantile(quantile)
    ordered = sorted(_integer_ns(value, name="values") for value in values)
    if not ordered:
        return None
    index = -(-((len(ordered) - 1) * q.numerator) // q.denominator)
    return ordered[index]


def age_time_quantile_ns(intervals: Iterable[tuple[int, int]], quantile: Any) -> int | None:
    """Quantile of uniform order-time over positive half-open age intervals."""
    q = _quantile(quantile)
    normalized: list[tuple[int, int]] = []
    total = 0
    for interval in intervals:
        if not isinstance(interval, tuple) or len(interval) != 2:
            raise ValueError("intervals must be (start_ns, end_ns) pairs")
        start = _integer_ns(interval[0], name="interval start")
        end = _integer_ns(interval[1], name="interval end")
        if end <= start:
            raise ValueError("intervals must be positive integer nanosecond intervals")
        normalized.append((start, end))
        total += end - start
    if not normalized:
        return None
    # Find the least integer age whose CDF reaches q, without float conversion.
    low = min(start for start, _ in normalized)
    high = max(end for _, end in normalized)
    target = q * total
    while low < high:
        middle = (low + high) // 2
        mass = sum(max(0, min(end, middle) - start) for start, end in normalized)
        if Fraction(mass) >= target:
            high = middle
        else:
            low = middle + 1
    return low


def survival_quantile_ns(spells: Iterable[tuple[int, bool]], quantile: Any) -> int | None:
    """Kaplan--Meier duration quantile with event/censor ties sharing risk set."""
    q = _strict_quantile(quantile)
    rows: list[tuple[int, bool]] = []
    for spell in spells:
        if not isinstance(spell, tuple) or len(spell) != 2:
            raise ValueError("spells must be (duration_ns, observed_event) pairs")
        duration = _integer_ns(spell[0], name="duration")
        if type(spell[1]) is not bool:
            raise ValueError("observed_event must be bool")
        rows.append((duration, spell[1]))
    if not rows or not any(observed for _, observed in rows):
        return None
    rows.sort()
    risk = len(rows)
    survival = Fraction(1)
    index = 0
    target_survival = 1 - q
    while index < len(rows):
        duration = rows[index][0]
        events = censored = 0
        while index < len(rows) and rows[index][0] == duration:
            if rows[index][1]:
                events += 1
            else:
                censored += 1
            index += 1
        if events:
            survival *= Fraction(risk - events, risk)
            if survival <= target_survival:
                return duration
        risk -= events + censored
    return None


def split_label(instrument: str, event_day: date | datetime) -> str:
    """Label the predeclared RISANAMENTO temporal split; all others stay exploratory."""
    if not isinstance(instrument, str) or not instrument:
        raise ValueError("instrument must be nonempty")
    if isinstance(event_day, datetime):
        event_day = event_day.date()
    if not isinstance(event_day, date):
        raise ValueError("event_day must be a date")
    if instrument == "RISANAMENTO":
        return "calibration" if event_day <= date(2024, 9, 30) else "retrospective_validation"
    return "exploratory_single_day"


def stratified_groups(frame: pd.DataFrame, *, extra_dimensions: Sequence[str] = ()) -> dict[tuple[Any, ...], pd.DataFrame]:
    """Split pandas rows by mandatory instrument/identity level and requested strata."""
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("frame must be a pandas DataFrame")
    dimensions = ("instrument", "identity_level", *extra_dimensions)
    if len(set(dimensions)) != len(dimensions):
        raise ValueError("strata dimensions must be unique")
    missing = [name for name in dimensions if name not in frame.columns]
    if missing:
        raise ValueError(f"missing strata columns: {', '.join(missing)}")
    if frame["identity_level"].isnull().any():
        raise ValueError("identity_level cannot contain null values")
    if frame.loc[:, dimensions].isnull().any().any():
        raise ValueError("strata columns cannot contain null values")
    levels = set(frame["identity_level"])
    unknown = levels - set(LEVELS)
    if unknown:
        raise ValueError(f"unknown identity_level: {sorted(unknown)!r}")
    return {key if isinstance(key, tuple) else (key,): group.copy() for key, group in frame.groupby(list(dimensions), sort=True, dropna=False)}


def proposed_window(window_seconds: Sequence[Any], excess: Sequence[Any], *, search_cap_seconds: Any) -> dict[str, Any]:
    """Return a plateau-derived exploratory association scale, never a detector rule."""
    if len(window_seconds) != len(excess) or len(window_seconds) < 3:
        raise ValueError("window_seconds and excess must have matching length of at least 3")
    cap = _fraction(search_cap_seconds, name="search_cap_seconds")
    if cap <= 0:
        raise ValueError("search_cap_seconds must be positive")
    windows = [_fraction(value, name="window_seconds") for value in window_seconds]
    values = [_fraction(value, name="excess") for value in excess]
    if any(value <= 0 for value in windows) or any(right <= left for left, right in zip(windows, windows[1:])):
        raise ValueError("window_seconds must be finite, positive, and strictly increasing")
    if windows[-1] > cap:
        raise ValueError("window exceeds declared search cap")
    if any(right < left for left, right in zip(values, values[1:])) or values[-1] <= 0:
        return {"window_seconds": None, "proposal_status": "not_identified", "search_cap_binding": False}
    last_three = values[-3:]
    plateau = last_three[0] * 10 >= last_three[-1] * 9 and max(last_three) - min(last_three) <= last_three[-1] / 10
    if not plateau:
        return {"window_seconds": None, "proposal_status": "search_cap_binding", "search_cap_binding": True}
    first = next(i for i, value in enumerate(values) if value >= Fraction(9, 10) * values[-1])
    if first == len(windows) - 1:
        return {"window_seconds": None, "proposal_status": "search_cap_binding", "search_cap_binding": True}
    return {"window_seconds": float(windows[first]), "proposal_status": "exploratory_association_scale", "search_cap_binding": False}


def tau_from_reference(q_ns: int | None, rho: Any) -> float | None:
    """Convert a valid reference quantile and reference weight into descriptive tau."""
    value = _fraction(rho, name="rho")
    if not 0 < value < 1:
        raise ValueError("rho must be finite and lie strictly between zero and one")
    if q_ns is None:
        return None
    q = _integer_ns(q_ns, name="q_ns")
    if q <= 0:
        return None
    return -(q / _NANOSECONDS_PER_SECOND) / math.log(float(value))


def moving_day_indices(available_dates: Sequence[date], *, n_blocks: int, seed: int, block_size: int = 5) -> tuple[tuple[int, ...], ...]:
    """Sample complete blocks from the ordered available-day calendar."""
    if type(n_blocks) is not int or n_blocks < 0:
        raise ValueError("n_blocks must be a nonnegative integer")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if type(block_size) is not int or block_size <= 0:
        raise ValueError("block_size must be a positive integer")
    if any(not isinstance(value, date) or isinstance(value, datetime) for value in available_dates):
        raise ValueError("available_dates must contain dates")
    if list(available_dates) != sorted(set(available_dates)):
        raise ValueError("available_dates must be sorted and unique")
    starts = list(range(len(available_dates) - block_size + 1))
    if n_blocks and not starts:
        raise ValueError("no full available date block")
    rng = random.Random(seed)
    return tuple(tuple(range(start, start + block_size)) for start in (rng.choice(starts) for _ in range(n_blocks)))


def endpoint_index(points: Sequence[tuple[int, int]], *, target_ns: int, cancel_index: int) -> tuple[int, int] | None:
    """Last lexicographic state at/before target, constrained to post-cancel index."""
    target = _integer_ns(target_ns, name="target_ns")
    if type(cancel_index) is not int:
        raise ValueError("cancel_index must be an integer")
    normalized = []
    for point in points:
        if not isinstance(point, tuple) or len(point) != 2:
            raise ValueError("points must be (time_ns, index) tuples")
        time_ns = _integer_ns(point[0], name="point time")
        if type(point[1]) is not int:
            raise ValueError("point index must be an integer")
        normalized.append((time_ns, point[1]))
    if not any(time >= target and index >= cancel_index for time, index in normalized):
        return None
    eligible = [point for point in normalized if point[0] <= target and point[1] >= cancel_index]
    return max(eligible) if eligible else None


_GROUP_ALIASES = {
    "instrument": ("instrument",),
    "partition": ("partition", "partition_id"),
    "day": ("day", "event_date"),
    "coverage_epoch": ("coverage_epoch", "coverage_epoch_id"),
}
_STATE_ALIASES = {
    **_GROUP_ALIASES,
    "time_ns": ("time_ns", "replay_time_ns"),
    "index": ("index", "sort_index", "source_index"),
    "midprice": ("midprice", "market_mid"),
}
_CANCEL_ALIASES = {
    **_GROUP_ALIASES,
    "time_ns": ("time_ns", "event_ns"),
    "index": ("index", "sort_index", "source_index"),
}


def _canonicalize_row(row: Mapping[str, Any], *, aliases: Mapping[str, tuple[str, ...]], row_name: str) -> tuple[dict[str, Any], bool]:
    if not isinstance(row, Mapping):
        raise ValueError(f"{row_name} must be a mapping")
    original = dict(row)
    normalized = dict(original)
    canonical = False
    for field, names in aliases.items():
        present = [(name, original[name]) for name in names if name in original]
        if not present:
            continue
        value = present[0][1]
        if any(candidate != value for _, candidate in present[1:]):
            raise ValueError(f"conflicting aliases for {field}")
        normalized[field] = value
        canonical = canonical or any(name != field for name, _ in present)
    return normalized, canonical


def _canonical_path_fields(*, pre: Mapping[str, Any] | None, endpoint: Mapping[str, Any] | None, delta: Decimal | None) -> dict[str, Any]:
    return {
        "pre_source_index": None if pre is None else pre["index"],
        "post_source_index": None if endpoint is None else endpoint["index"],
        "pre_source_time_ns": None if pre is None else pre["time_ns"],
        "post_source_time_ns": None if endpoint is None else endpoint["time_ns"],
        "endpoint_time_ns": None if endpoint is None else endpoint["time_ns"],
        "midprice_change_ticks": delta,
    }


def physical_cancel_paths(
    cancels: Iterable[Mapping[str, Any]], states: Iterable[Mapping[str, Any]], *, group: tuple[Any, Any, Any, Any],
    horizons_ns: Iterable[Decimal | int], tick_size: Decimal | int, directional: bool = False,
) -> list[dict[str, Any]]:
    """Measure observed same-coverage midprice paths after physical cancellations.

    Legacy rows use ``partition/day/coverage_epoch/time_ns/index/midprice``.
    Canonical adapter rows use ``partition_id/event_date/coverage_epoch_id``;
    states use ``replay_time_ns``, ``sort_index`` (or ``source_index``), and
    ``market_mid``; cancellation messages use ``event_ns`` and ``sort_index``.
    Mixed aliases are accepted only when their values agree.
    """
    if not isinstance(group, tuple) or len(group) != 4:
        raise ValueError("group must be (instrument, partition, day, coverage_epoch)")
    tick = Decimal(tick_size)
    if not tick.is_finite() or tick <= 0:
        raise ValueError("tick_size must be positive")
    horizons = []
    for horizon in horizons_ns:
        value = Decimal(horizon)
        if not value.is_finite() or value <= 0 or value != value.to_integral_value():
            raise ValueError("horizons_ns must be positive integral Decimal nanoseconds")
        horizons.append(int(value))
    if not horizons:
        raise ValueError("horizons_ns cannot be empty")
    fields = ("instrument", "partition", "day", "coverage_epoch")
    state_rows = [_canonicalize_row(state, aliases=_STATE_ALIASES, row_name="state")[0] for state in states]
    epochs = {row.get("coverage_epoch") for row in state_rows if tuple(row.get(field) for field in fields[:3]) == group[:3]}
    if epochs - {group[3]}:
        raise ValueError("mixed coverage epochs")
    selected = [row for row in state_rows if tuple(row.get(field) for field in fields) == group]
    selected.sort(key=lambda row: (_integer_ns(row.get("time_ns"), name="state time"), row.get("index")))
    output: list[dict[str, Any]] = []
    for cancel_input in cancels:
        cancel, canonical_cancel = _canonicalize_row(cancel_input, aliases=_CANCEL_ALIASES, row_name="cancel")
        if tuple(cancel.get(field) for field in fields) != group:
            raise ValueError("cancel group must match caller group")
        cancel_time = _integer_ns(cancel.get("time_ns"), name="cancel time")
        cancel_index = cancel.get("index")
        if type(cancel_index) is not int:
            raise ValueError("cancel index must be an integer")
        before = [row for row in selected if row["index"] < cancel_index and row["time_ns"] <= cancel_time]
        for horizon in horizons:
            target = cancel_time + horizon
            base = {"cancel_index": cancel_index, "horizon_ns": horizon}
            if canonical_cancel:
                base = {**dict(cancel_input), "cancel_provenance": dict(cancel_input), **base, "target_ns": target}
            if not before:
                row = {**base, "pre_index": None, "endpoint_index": None, "status": "no_pre_cancel_state", "raw_midprice_delta_ticks": None, "side_directional_delta_ticks": None, "recency_ns": None}
                if canonical_cancel:
                    row.update(_canonical_path_fields(pre=None, endpoint=None, delta=None))
                output.append(row)
                continue
            pre = max(before, key=lambda row: (row["time_ns"], row["index"]))
            endpoint_key = endpoint_index([(row["time_ns"], row["index"]) for row in selected], target_ns=target, cancel_index=cancel_index)
            coverage_reaches_target = any(row["time_ns"] >= target for row in selected)
            if endpoint_key is None or not coverage_reaches_target:
                row = {**base, "pre_index": pre["index"], "endpoint_index": None, "status": "uncovered_horizon", "raw_midprice_delta_ticks": None, "side_directional_delta_ticks": None, "recency_ns": None}
                if canonical_cancel:
                    row.update(_canonical_path_fields(pre=pre, endpoint=None, delta=None))
                output.append(row)
                continue
            endpoint = next(row for row in selected if (row["time_ns"], row["index"]) == endpoint_key)
            pre_mid = pre.get("midprice")
            post_mid = endpoint.get("midprice")
            if pre_mid is None or not Decimal(str(pre_mid)).is_finite():
                status = "missing_pre_midprice"
            elif post_mid is None or not Decimal(str(post_mid)).is_finite():
                status = "missing_endpoint_midprice"
            else:
                status = "observed"
            delta = None if status != "observed" else (Decimal(str(post_mid)) - Decimal(str(pre_mid))) / tick
            side = cancel.get("side")
            if delta is None:
                directional_delta = None
            elif side == "bid":
                directional_delta = delta
            elif side == "ask":
                directional_delta = -delta
            else:
                directional_delta = None
            row = {**base, "pre_index": pre["index"], "endpoint_index": endpoint["index"], "status": status,
                   "raw_midprice_delta_ticks": delta, "side_directional_delta_ticks": directional_delta if directional else None,
                   "recency_ns": target - endpoint["time_ns"]}
            if canonical_cancel:
                row.update(_canonical_path_fields(pre=pre, endpoint=endpoint, delta=delta))
            output.append(row)
    return output


def moving_day_replicates(available_dates, *, block_days, replicates, seed):
    """Bootstrap full blocks in the ordered available-day calendar, not clock days.

    Weekends and venue holidays do not break adjacent observations. Duplicate
    sampled dates are retained; callers must assign replicate-day IDs.
    """
    import numpy as np
    dates = list(available_dates)
    if dates != sorted(set(dates)):
        raise ValueError("available dates must be sorted and unique")
    if any(type(v) is not int or v <= 0 for v in (block_days, replicates)) or block_days > len(dates):
        raise ValueError("invalid bootstrap design")
    if type(seed) is not int or seed < 0:
        raise ValueError("invalid seed")
    rng = np.random.default_rng(seed)
    n = len(dates)
    for _ in range(replicates):
        starts = rng.integers(0, n - block_days + 1, size=math.ceil(n / block_days))
        yield [int(i) for start in starts for i in range(start, start + block_days)][:n]
