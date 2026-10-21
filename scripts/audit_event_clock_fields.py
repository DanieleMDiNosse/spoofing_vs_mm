#!/usr/bin/env python3
"""Read-only audit of raw event timestamp fields at nanosecond precision.

The audit deliberately measures each source timestamp independently.  It never
constructs a fallback event clock and never changes replay inputs or outputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from collections import defaultdict
from itertools import permutations
from pathlib import Path
from typing import Any, Iterable

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from spoofing_detection.lob.enums import EVENT_LABEL_BY_CODE, normalize_enum_code
from spoofing_detection.lob.execution_clusters import classify_execution_anchor


CLOCK_FIELDS = ("BOOKIN", "BOOKOUTTIME", "TRADETIME", "SEQUENCETIME")
PARTITION_FIELDS = ("TRADEDATE", "MIC", "MARKETCODE", "SYMBOLINDEX", "EMM (*)")
LIFECYCLE_KEY_FIELDS = ("instrument", *PARTITION_FIELDS, "ORDERID")
BASE_COLUMNS = (
    *PARTITION_FIELDS,
    "ORDERID",
    "ORDEREVENTTYPE (*)",
    "PASSIVEORDER",
    "AGGRESSIVEORDER",
    "HDR_APPLKEYSEQUENCENUMBER",
    "HDR_HWMSEQUENCENUMBER",
    "HDR_OFFSETID",
    "ROW_NUMBER",
    *CLOCK_FIELDS,
)


def parse_clock_series(series: pl.Series) -> tuple[pl.Series, dict[str, int]]:
    """Return nullable ns timestamps and mutually exclusive availability counts.

    ``SEQUENCETIME``, ``BOOKIN`` and ``BOOKOUTTIME`` are documented UTC
    epoch-nanosecond representations. ``TRADETIME`` is a Matching Engine IN
    time in ns; this audit reports its numeric alignment to sequence time rather
    than asserting undocumented timezone metadata. Native ``Datetime('ns')`` is
    cast directly so sub-microsecond precision is retained.
    """
    if isinstance(series.dtype, pl.datatypes.Datetime):
        parsed = series.dt.cast_time_unit("ns").cast(pl.Int64, strict=False)
        missing = series.is_null()
    else:
        text = series.cast(pl.String, strict=False)
        stripped = text.str.strip_chars()
        missing = series.is_null() | stripped.is_null() | stripped.str.to_lowercase().is_in(["", "none", "null", "nan"])
        parsed_datetime = stripped.str.strptime(
            pl.Datetime("ns"), format="%Y%m%d %H:%M:%S%.f", strict=False
        )
        # The raw extract convention above is primary; explicit ISO alternatives
        # avoid Polars' inference failure when one malformed value is present.
        parsed_datetime = parsed_datetime.fill_null(
            stripped.str.strptime(pl.Datetime("ns"), format="%Y-%m-%d %H:%M:%S%.f", strict=False)
        ).fill_null(
            stripped.str.strptime(pl.Datetime("ns"), format="%Y-%m-%dT%H:%M:%S%.f", strict=False)
        )
        parsed = parsed_datetime.cast(pl.Int64, strict=False).set(missing, None)
    malformed = (~missing) & parsed.is_null()
    present = parsed.is_not_null()
    counts = {
        "present": int(present.sum()),
        "missing": int(missing.sum()),
        "malformed": int(malformed.sum()),
    }
    return parsed, counts


def parse_clock_columns(frame: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    """Add nullable ``*_ns`` columns without mutating source clock columns."""
    expressions = []
    for name in columns:
        if name not in frame.columns:
            raise KeyError(f"clock column not found: {name}")
        parsed, _ = parse_clock_series(frame.get_column(name))
        expressions.append(parsed.alias(f"{name}_ns"))
    return frame.with_columns(expressions)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_path(value: str | Path, config_path: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidates = (config_path.parent / path, ROOT / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return ROOT / path


def _clock_status(raw: Any, parsed_ns: int | None) -> str:
    if raw is None:
        return "missing"
    if isinstance(raw, float) and math.isnan(raw):
        return "missing"
    if isinstance(raw, str) and raw.strip().lower() in {"", "none", "null", "nan"}:
        return "missing"
    return "present" if parsed_ns is not None else "malformed"


def _event_label(value: Any) -> str:
    code = normalize_enum_code(value)
    return EVENT_LABEL_BY_CODE.get(code, "Unknown")


def _fill_role(row: dict[str, Any], event_label: str) -> str:
    if event_label != "Fill":
        return "not_fill"
    return classify_execution_anchor(row) or "ambiguous_or_unknown"


def _sortable(value: Any) -> tuple[int, Any]:
    if value is None:
        return (1, "")
    if isinstance(value, (int, float, str)):
        return (0, value)
    return (0, str(value))


def _canonical_key(row: dict[str, Any], stable_sort_columns: Iterable[str]) -> tuple[tuple[int, Any], ...]:
    return tuple(_sortable(row.get(f"{field}_ns") if field in CLOCK_FIELDS else row.get(field)) for field in stable_sort_columns)


def _group_sort_key(item: tuple[tuple[Any, ...], Any]) -> tuple[tuple[int, Any], ...]:
    """Return a deterministic null-safe key for grouped audit rows."""
    return tuple(_sortable(value) for value in item[0])


def _empty_csv(path: Path, schema: dict[str, pl.DataType]) -> None:
    pl.DataFrame(schema=schema).write_csv(path)


def _write_csv(rows: list[dict[str, Any]], path: Path, schema: dict[str, pl.DataType]) -> None:
    if rows:
        pl.DataFrame(rows, schema=schema, strict=False).write_csv(path)
    else:
        _empty_csv(path, schema)


def _availability_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str], dict[str, int]] = defaultdict(lambda: {"total_rows": 0, "present": 0, "missing": 0, "malformed": 0})
    for row in rows:
        for field in CLOCK_FIELDS:
            target = grouped[(row["instrument"], str(row["TRADEDATE"]), row["event_label"], row["fill_role"], field)]
            target["total_rows"] += 1
            target[_clock_status(row[field], row[f"{field}_ns"])] += 1
    return [
        {"instrument": instrument, "event_date": event_date, "event_label": label, "fill_role": role, "field": field, **counts}
        for (instrument, event_date, label, role, field), counts in sorted(grouped.items())
    ]


def _pairwise_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["instrument"], str(row["TRADEDATE"]), row["event_label"], row["fill_role"])].append(row)
    result: list[dict[str, Any]] = []
    for (instrument, event_date, label, role), group in sorted(groups.items()):
        for left, right in permutations(CLOCK_FIELDS, 2):
            deltas = [row[f"{left}_ns"] - row[f"{right}_ns"] for row in group if row[f"{left}_ns"] is not None and row[f"{right}_ns"] is not None]
            result.append({
                "instrument": instrument,
                "event_date": event_date,
                "event_label": label,
                "fill_role": role,
                "left_field": left,
                "right_field": right,
                "comparable_rows": len(deltas),
                "equal": sum(delta == 0 for delta in deltas),
                "later": sum(delta > 0 for delta in deltas),
                "earlier": sum(delta < 0 for delta in deltas),
                "median_delta_ns": float(pl.Series(deltas).median()) if deltas else None,
                "p99_delta_ns": float(pl.Series(deltas).quantile(0.99)) if deltas else None,
            })
    return result


def _ordering_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, tuple[Any, ...]], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["instrument"], tuple(row.get(field) for field in PARTITION_FIELDS))].append(row)
    result: list[dict[str, Any]] = []
    for (instrument, partition), partition_rows in sorted(grouped.items(), key=_group_sort_key):
        ordered = sorted(partition_rows, key=lambda row: row["_canonical_key"])
        for field in CLOCK_FIELDS:
            observed = [row[f"{field}_ns"] for row in ordered if row[f"{field}_ns"] is not None]
            adjacent = sum(current < previous for previous, current in zip(observed, observed[1:]))
            high_watermark = 0
            high_regressions = 0
            seen = False
            for value in observed:
                if seen and value < high_watermark:
                    high_regressions += 1
                high_watermark = value if not seen else max(high_watermark, value)
                seen = True
            for comparison, regressions in (("adjacent_regressions", adjacent), ("high_watermark_regressions", high_regressions)):
                result.append({
                    "instrument": instrument,
                    **dict(zip(PARTITION_FIELDS, partition, strict=True)),
                    "field": field,
                    "comparison": comparison,
                    "observation_count": len(observed),
                    "comparable_pairs": max(0, len(observed) - 1),
                    "regressions": regressions,
                })
    return result


def _lifecycle_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    new_indices: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if row["event_label"] == "New" and all(row.get(field) is not None for field in LIFECYCLE_KEY_FIELDS):
            new_indices[tuple(row[field] for field in LIFECYCLE_KEY_FIELDS)].append(index)
    result: dict[tuple[str, str, str, str, str], dict[str, int]] = defaultdict(lambda: {"event_rows": 0, "matched_rows": 0, "excluded_missing_lifecycle_key": 0, "excluded_no_prior_new": 0, "excluded_duplicate_new": 0})
    for index, row in enumerate(rows):
        if row["event_label"] not in {"Modify", "Fill", "Cancel"}:
            continue
        key_complete = all(row.get(field) is not None for field in LIFECYCLE_KEY_FIELDS)
        cancellation_class = "unknown" if row["event_label"] == "Cancel" else "not_applicable"
        target = result[(row["instrument"], str(row["TRADEDATE"]), row["event_label"], row["fill_role"], cancellation_class)]
        target["event_rows"] += 1
        if not key_complete:
            target["excluded_missing_lifecycle_key"] += 1
            continue
        candidates = new_indices[tuple(row[field] for field in LIFECYCLE_KEY_FIELDS)]
        if len(candidates) == 1 and candidates[0] < index:
            target["matched_rows"] += 1
        elif len(candidates) == 0 or candidates[0] >= index:
            target["excluded_no_prior_new"] += 1
        else:
            target["excluded_duplicate_new"] += 1
    return [
        {"instrument": instrument, "event_date": event_date, "event_label": label, "fill_role": role, "cancellation_class": cancellation_class, "relation": f"new_to_{label.lower()}", **counts}
        for (instrument, event_date, label, role, cancellation_class), counts in sorted(result.items())
    ]


def _examples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return anonymous temporal examples for each observed ordering anomaly."""
    grouped: dict[tuple[str, tuple[Any, ...]], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["instrument"], tuple(row.get(field) for field in PARTITION_FIELDS))].append(row)
    examples: list[dict[str, Any]] = []
    for (_instrument, _partition), partition_rows in sorted(grouped.items(), key=_group_sort_key):
        ordered = sorted(partition_rows, key=lambda row: row["_canonical_key"])
        for field in CLOCK_FIELDS:
            previous: int | None = None
            high_watermark: int | None = None
            for row in ordered:
                value = row[f"{field}_ns"]
                if value is None:
                    continue
                base = {
                    "instrument": row["instrument"],
                    "event_date": str(row["TRADEDATE"]),
                    "event_label": row["event_label"],
                    "fill_role": row["fill_role"],
                    "field": field,
                    **{f"{clock}_ns": row[f"{clock}_ns"] for clock in CLOCK_FIELDS},
                }
                if previous is not None and value < previous:
                    examples.append({**base, "anomaly": "adjacent_regression"})
                if high_watermark is not None and value < high_watermark:
                    examples.append({**base, "anomaly": "high_watermark_regression"})
                previous = value
                high_watermark = value if high_watermark is None else max(high_watermark, value)
    return examples


def _load_instrument_rows(instrument: str, raw_path: Path, selected_dates: set[str], stable_sort_columns: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    schema = pl.scan_parquet(raw_path).collect_schema()
    missing = sorted(set(BASE_COLUMNS) - set(schema.names()))
    if missing:
        raise ValueError(f"{instrument}: raw events missing required columns: {', '.join(missing)}")
    df = pl.scan_parquet(raw_path).select(list(BASE_COLUMNS)).collect(engine="streaming")
    if selected_dates:
        trade_date = pl.col("TRADEDATE")
        if isinstance(df.schema["TRADEDATE"], (pl.Date, pl.Datetime)):
            trade_date = trade_date.dt.strftime("%Y-%m-%d")
        else:
            trade_date = trade_date.cast(pl.String).str.slice(0, 10)
        df = df.filter(trade_date.is_in(selected_dates))
    if isinstance(df.schema["TRADEDATE"], (pl.Date, pl.Datetime)):
        df = df.with_columns(pl.col("TRADEDATE").dt.strftime("%Y-%m-%d"))
    else:
        df = df.with_columns(pl.col("TRADEDATE").cast(pl.String).str.slice(0, 10))
    parsed: dict[str, list[int | None]] = {}
    parse_counts: dict[str, dict[str, int]] = {}
    for field in CLOCK_FIELDS:
        values, counts = parse_clock_series(df.get_column(field))
        parsed[field] = values.to_list()
        parse_counts[field] = counts
    rows = df.to_dicts()
    for index, row in enumerate(rows):
        row["instrument"] = instrument
        row["event_label"] = _event_label(row["ORDEREVENTTYPE (*)"])
        row["fill_role"] = _fill_role(row, row["event_label"])
        for field in CLOCK_FIELDS:
            row[f"{field}_ns"] = parsed[field][index]
        row["_canonical_key"] = _canonical_key(row, stable_sort_columns)
    return rows, {"source": str(raw_path), "source_sha256": _sha256(raw_path), "selected_rows": len(rows), "parse_counts": parse_counts}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="empirical-controls JSON configuration")
    parser.add_argument("--output", required=True, type=Path, help="new output directory")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"output directory already exists: {args.output}")
    config = json.loads(args.config.read_text())
    stable_sort_columns = config.get("clock", {}).get("stable_sort_columns")
    if not stable_sort_columns:
        raise ValueError("config.clock.stable_sort_columns is required")
    runs = config.get("baseline_runs", {})
    if not runs:
        raise ValueError("config.baseline_runs is required")

    all_rows: list[dict[str, Any]] = []
    sources: dict[str, Any] = {}
    for instrument, run in runs.items():
        raw_path = _resolve_path(run["source_paths"]["raw_events"], args.config)
        rows, source = _load_instrument_rows(instrument, raw_path, set(run.get("selected_event_dates", [])), stable_sort_columns)
        all_rows.extend(rows)
        sources[instrument] = source
    all_rows.sort(key=lambda row: (row["instrument"], row["_canonical_key"]))

    availability = _availability_rows(all_rows)
    pairwise = _pairwise_rows(all_rows)
    ordering = _ordering_rows(all_rows)
    lifecycle = _lifecycle_rows(all_rows)
    examples = _examples(all_rows)
    args.output.mkdir(parents=True, exist_ok=False)

    _write_csv(availability, args.output / "availability.csv", {
        "instrument": pl.String, "event_date": pl.String, "event_label": pl.String, "fill_role": pl.String, "field": pl.String,
        "total_rows": pl.Int64, "present": pl.Int64, "missing": pl.Int64, "malformed": pl.Int64,
    })
    _write_csv(pairwise, args.output / "pairwise.csv", {
        "instrument": pl.String, "event_date": pl.String, "event_label": pl.String, "fill_role": pl.String, "left_field": pl.String, "right_field": pl.String,
        "comparable_rows": pl.Int64, "equal": pl.Int64, "later": pl.Int64, "earlier": pl.Int64, "median_delta_ns": pl.Float64, "p99_delta_ns": pl.Float64,
    })
    _write_csv(ordering, args.output / "ordering.csv", {
        "instrument": pl.String, "TRADEDATE": pl.String, "MIC": pl.String, "MARKETCODE": pl.String, "SYMBOLINDEX": pl.Int64, "EMM (*)": pl.Int64,
        "field": pl.String, "comparison": pl.String, "observation_count": pl.Int64, "comparable_pairs": pl.Int64, "regressions": pl.Int64,
    })
    _write_csv(lifecycle, args.output / "lifecycle.csv", {
        "instrument": pl.String, "event_date": pl.String, "event_label": pl.String, "fill_role": pl.String, "cancellation_class": pl.String, "relation": pl.String,
        "event_rows": pl.Int64, "matched_rows": pl.Int64, "excluded_missing_lifecycle_key": pl.Int64, "excluded_no_prior_new": pl.Int64, "excluded_duplicate_new": pl.Int64,
    })
    _write_csv(examples, args.output / "examples.csv", {
        "instrument": pl.String, "event_date": pl.String, "event_label": pl.String, "fill_role": pl.String, "anomaly": pl.String, "field": pl.String,
        "BOOKIN_ns": pl.Int64, "BOOKOUTTIME_ns": pl.Int64, "TRADETIME_ns": pl.Int64, "SEQUENCETIME_ns": pl.Int64,
    })
    provenance = {
        "script_sha256": _sha256(Path(__file__)),
        "config_sha256": _sha256(args.config),
        "sources": sources,
        "selected_event_dates": {instrument: run.get("selected_event_dates", []) for instrument, run in runs.items()},
        "timestamp_unit": "ns",
        "timestamp_representation": (
            "SEQUENCETIME/BOOKIN/BOOKOUTTIME: documented UTC epoch nanoseconds; "
            "TRADETIME: Matching Engine IN time in ns; pairwise alignment audited"
        ),
        "clock_fields": list(CLOCK_FIELDS),
        "event_type_decoding": "spoofing_detection.lob.enums.normalize_enum_code and EVENT_LABEL_BY_CODE",
        "fill_role_decoding": "spoofing_detection.lob.execution_clusters.classify_execution_anchor",
        "canonical_order": list(stable_sort_columns),
        "ordering_policy": "individual source fields only; no mixed fallback clock",
        "lifecycle_key_fields": list(LIFECYCLE_KEY_FIELDS),
        "cancellation_classification": "unknown; automatic-versus-client cancellation is not inferred",
        "polars": pl.__version__,
        "python": platform.python_version(),
        "command": f"{Path(__file__).name} --config {args.config} --output {args.output}",
    }
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    total_rows = len(all_rows)
    report = [
        "# Event-clock field audit",
        "",
        f"Rows audited: {total_rows}",
        f"Instruments: {len(sources)}",
        "",
        "- SEQUENCETIME, BOOKIN and BOOKOUTTIME are retained as documented UTC epoch nanoseconds; TRADETIME is retained as Matching Engine IN time in ns and its pairwise alignment is reported without an undocumented timezone assertion.",
        "- BOOKIN, BOOKOUTTIME, TRADETIME, and SEQUENCETIME are audited independently; no fallback or composite event clock is constructed.",
        "- Event types use project enum decoding and fill roles use the project exclusive-role classifier.",
        "- Lifecycle joins require TRADEDATE, MIC, MARKETCODE, SYMBOLINDEX, EMM (*), and ORDERID. Exclusions are counted in `lifecycle.csv`.",
        "- Cancellation automatic-versus-client classification remains unknown.",
        "- This audit describes source-field availability and ordering only; it does not establish an economic event-time interpretation.",
        "",
        "## Output accounting",
        "",
        "| artifact | rows |",
        "|---|---:|",
        f"| availability.csv | {len(availability)} |",
        f"| pairwise.csv | {len(pairwise)} |",
        f"| ordering.csv | {len(ordering)} |",
        f"| lifecycle.csv | {len(lifecycle)} |",
        f"| examples.csv | {len(examples)} |",
        "",
        "`examples.csv` contains no actor, client, or order identifiers.",
    ]
    (args.output / "report.md").write_text("\n".join(report) + "\n")


if __name__ == "__main__":
    main()
