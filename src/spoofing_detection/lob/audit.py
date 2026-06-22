from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import polars as pl

from .enums import (
    ACCOUNT_TYPE_INTERNAL_BY_CODE,
    ACK_PHASE_BY_CODE,
    ACK_TYPE_BY_CODE,
    EVENT_LABEL_BY_CODE,
    EXECUTION_PHASE_BY_CODE,
    KILL_REASON_BY_CODE,
    LP_ROLE_BY_CODE,
    ORDER_SIDE_BY_CODE,
    ORDER_TYPE_BY_CODE,
    TIME_IN_FORCE_BY_CODE,
    TRADE_TYPE_BY_CODE,
    TRADING_CAPACITY_BY_CODE,
    normalize_enum_code,
)

REQUIRED_COLUMNS: tuple[str, ...] = (
    "TRADEDATE",
    "MIC",
    "MARKETCODE",
    "SYMBOLINDEX",
    "EMM (*)",
    "ISIN",
    "SEQUENCETIME",
    "BOOKIN",
    "BOOKOUTTIME",
    "TRADETIME",
    "HDR_APPLKEYSEQUENCENUMBER",
    "HDR_HWMSEQUENCENUMBER",
    "HDR_OFFSETID",
    "ROW_NUMBER",
    "EVENTID",
    "ORDEREVENTTYPE (*)",
    "ORDERID",
    "ORDERPRIORITY",
    "ORDERSIDE (*)",
    "ORDERPX",
    "ORDERQTY",
    "DISPLAYEDQTY",
    "LEAVESQTY",
    "LASTSHARES",
    "LASTTRADEDPX",
    "ORDERTYPE (*)",
    "TIMEINFORCE (*)",
    "KILLREASON (*)",
    "ORDERSTATUS",
    "PASSIVEORDER",
    "AGGRESSIVEORDER",
    "EXECUTIONID",
    "TRADEUNIQUEIDENTIFIER",
    "FIRMID",
    "NMSC_ORIGINALCLIENTIDSHORTCODE",
    "MSC_EVENTCLIENTIDSHORTCODE",
    "NMSC_ORIGINALEXECWFIRMSHORTCODE",
    "MSC_EVENTEXECWFIRMSHORTCODE",
    "NMSC_ORIGINALINVESTDECISWFIRMSHORTCODE",
    "NMSC_ORIGINALNONEXECBROKERSHORTCODE",
    "ACCOUNTTYPEINTERNAL (*)",
    "LPROLE (*)",
    "ORDER_TRADINGCAPACITY (*)",
    "DEAINDICATOR",
    "INVESTMENTALGOINDICATOR",
    "EXECUTIONALGOINDICATOR",
)

PARTITION_COLUMNS: tuple[str, ...] = ("TRADEDATE", "MIC", "MARKETCODE", "SYMBOLINDEX", "EMM (*)")
SORT_COLUMNS: tuple[str, ...] = (
    *PARTITION_COLUMNS,
    "SEQUENCETIME",
    "HDR_APPLKEYSEQUENCENUMBER",
    "HDR_HWMSEQUENCENUMBER",
    "HDR_OFFSETID",
    "BOOKIN",
    "BOOKOUTTIME",
    "TRADETIME",
    "ROW_NUMBER",
)
IDENTITY_COLUMNS: tuple[str, ...] = ("FIRMID", "NMSC_ORIGINALCLIENTIDSHORTCODE")
PASSIVE_AGGRESSIVE_COLUMNS: tuple[str, ...] = ("PASSIVEORDER", "AGGRESSIVEORDER")

ENUM_SPECS: dict[str, Mapping[int, str]] = {
    "ORDEREVENTTYPE (*)": EVENT_LABEL_BY_CODE,
    "ORDERSIDE (*)": ORDER_SIDE_BY_CODE,
    "ORDERTYPE (*)": ORDER_TYPE_BY_CODE,
    "TIMEINFORCE (*)": TIME_IN_FORCE_BY_CODE,
    "KILLREASON (*)": KILL_REASON_BY_CODE,
    "ACCOUNTTYPEINTERNAL (*)": ACCOUNT_TYPE_INTERNAL_BY_CODE,
    "LPROLE (*)": LP_ROLE_BY_CODE,
    "ORDER_TRADINGCAPACITY (*)": TRADING_CAPACITY_BY_CODE,
    "TRADETYPE (*)": TRADE_TYPE_BY_CODE,
    "EXECUTIONPHASE (*)": EXECUTION_PHASE_BY_CODE,
    "ACKTYPE (*)": ACK_TYPE_BY_CODE,
    "ACKPHASE (*)": ACK_PHASE_BY_CODE,
}


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and value != value:
        return True
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null", "nan"}:
        return True
    return False


def _json_counts(counter: Counter[Any]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items(), key=lambda item: str(item[0]))}


def _observed_values(df: pl.DataFrame, column: str) -> list[Any]:
    if column not in df.columns:
        return []
    return df.get_column(column).to_list()


def _audit_enum_field(df: pl.DataFrame, field: str, mapping: Mapping[int, str]) -> dict[str, Any]:
    if field not in df.columns:
        return {
            "present": False,
            "observed_codes": [],
            "unknown_codes": [],
            "null_count": None,
            "parse_error_count": None,
            "observed_code_counts": {},
        }

    code_counts: Counter[int] = Counter()
    parse_errors: Counter[str] = Counter()
    null_count = 0
    for value in _observed_values(df, field):
        if _is_missing(value):
            null_count += 1
            continue
        try:
            code = normalize_enum_code(value)
        except ValueError:
            parse_errors[str(value)] += 1
            continue
        if code is None:
            null_count += 1
        else:
            code_counts[code] += 1

    observed_codes = sorted(code_counts)
    unknown_codes = [code for code in observed_codes if code not in mapping]
    return {
        "present": True,
        "observed_codes": observed_codes,
        "unknown_codes": unknown_codes,
        "null_count": null_count,
        "parse_error_count": sum(parse_errors.values()),
        "parse_errors": _json_counts(parse_errors),
        "observed_code_counts": _json_counts(code_counts),
        "mapped_labels": {str(code): mapping[code] for code in observed_codes if code in mapping},
    }


def _audit_identity_field(df: pl.DataFrame, field: str) -> dict[str, Any]:
    if field not in df.columns:
        return {"present": False, "missing_count": None, "distinct_non_null_count": None}
    values = _observed_values(df, field)
    non_null = [str(value) for value in values if not _is_missing(value)]
    return {
        "present": True,
        "missing_count": len(values) - len(non_null),
        "distinct_non_null_count": len(set(non_null)),
    }


def _audit_passive_aggressive_field(df: pl.DataFrame, field: str) -> dict[str, Any]:
    if field not in df.columns:
        return {"present": False, "observed_values": [], "value_counts": {}}
    counter: Counter[str] = Counter()
    for value in _observed_values(df, field):
        if _is_missing(value):
            counter["<missing>"] += 1
        else:
            counter[str(value)] += 1
    return {
        "present": True,
        "observed_values": sorted(counter),
        "value_counts": _json_counts(counter),
    }


def _ordering_key_duplicate_count(df: pl.DataFrame) -> int | None:
    available = [column for column in SORT_COLUMNS if column in df.columns]
    if len(available) != len(SORT_COLUMNS):
        return None
    return int(df.select(available).is_duplicated().sum())


def _first_event_counts(df: pl.DataFrame) -> dict[str, dict[str, int]]:
    required = [*SORT_COLUMNS, "ORDEREVENTTYPE (*)", "ORDERID"]
    if any(column not in df.columns for column in required):
        return {"partition_first_event_code_counts": {}, "order_first_event_code_counts": {}}

    sorted_df = df.sort(list(SORT_COLUMNS), nulls_last=True)
    partition_seen: set[tuple[Any, ...]] = set()
    order_seen: set[tuple[Any, ...]] = set()
    partition_counter: Counter[int | None] = Counter()
    order_counter: Counter[int | None] = Counter()

    for row in sorted_df.iter_rows(named=True):
        partition_key = tuple(row.get(column) for column in PARTITION_COLUMNS)
        order_key = (*partition_key, row.get("ORDERID"))
        event_code = normalize_enum_code(row.get("ORDEREVENTTYPE (*)"))
        if partition_key not in partition_seen:
            partition_seen.add(partition_key)
            partition_counter[event_code] += 1
        if order_key not in order_seen:
            order_seen.add(order_key)
            order_counter[event_code] += 1

    return {
        "partition_first_event_code_counts": _json_counts(partition_counter),
        "order_first_event_code_counts": _json_counts(order_counter),
    }


def audit_dataframe(df: pl.DataFrame, *, source_name: str) -> dict[str, Any]:
    """Audit one event table schema and provisional enum coverage without mutating LOB state."""

    missing_required = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    enum_fields = {field: _audit_enum_field(df, field, mapping) for field, mapping in ENUM_SPECS.items()}
    identity_fields = {field: _audit_identity_field(df, field) for field in IDENTITY_COLUMNS}
    passive_aggressive_fields = {
        field: _audit_passive_aggressive_field(df, field) for field in PASSIVE_AGGRESSIVE_COLUMNS
    }

    return {
        "source_name": source_name,
        "row_count": df.height,
        "column_count": len(df.columns),
        "missing_required_columns": missing_required,
        "extra_columns_count": len([column for column in df.columns if column not in REQUIRED_COLUMNS]),
        "ordering_key_duplicate_count": _ordering_key_duplicate_count(df),
        "enum_fields": enum_fields,
        "identity_fields": identity_fields,
        "passive_aggressive_fields": passive_aggressive_fields,
        **_first_event_counts(df),
    }


def read_event_table(path: str | Path) -> pl.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pl.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pl.read_csv(path, infer_schema_length=10000)
    raise ValueError(f"unsupported input suffix {path.suffix!r}; expected .parquet or .csv")


def _aggregate_file_audits(file_audits: list[dict[str, Any]]) -> dict[str, Any]:
    missing_required_union = sorted(
        {column for audit in file_audits for column in audit["missing_required_columns"]}
    )
    identity_missing_counts = {
        field: sum(
            audit["identity_fields"][field]["missing_count"] or 0
            for audit in file_audits
            if field in audit["identity_fields"]
        )
        for field in IDENTITY_COLUMNS
    }
    unknown_enum_codes = {
        field: sorted(
            {
                code
                for audit in file_audits
                for code in audit["enum_fields"].get(field, {}).get("unknown_codes", [])
            }
        )
        for field in ENUM_SPECS
    }
    passive_aggressive_values = {
        field: sorted(
            {
                value
                for audit in file_audits
                for value in audit["passive_aggressive_fields"].get(field, {}).get("observed_values", [])
            }
        )
        for field in PASSIVE_AGGRESSIVE_COLUMNS
    }

    partition_first_counts: Counter[str] = Counter()
    order_first_counts: Counter[str] = Counter()
    for audit in file_audits:
        partition_first_counts.update(audit["partition_first_event_code_counts"])
        order_first_counts.update(audit["order_first_event_code_counts"])

    return {
        "row_count": sum(audit["row_count"] for audit in file_audits),
        "missing_required_columns_union": missing_required_union,
        "all_required_columns_present": not missing_required_union,
        "identity_missing_counts": identity_missing_counts,
        "unknown_enum_codes": unknown_enum_codes,
        "passive_aggressive_observed_values": passive_aggressive_values,
        "ordering_key_duplicate_count": sum(
            audit["ordering_key_duplicate_count"] or 0 for audit in file_audits
        ),
        "partition_first_event_code_counts": _json_counts(partition_first_counts),
        "order_first_event_code_counts": _json_counts(order_first_counts),
    }


def audit_paths(paths: Sequence[str | Path]) -> dict[str, Any]:
    file_audits = []
    for path_like in paths:
        path = Path(path_like)
        file_audits.append(audit_dataframe(read_event_table(path), source_name=str(path)))
    return {
        "files_audited": len(file_audits),
        "files": file_audits,
        "aggregate": _aggregate_file_audits(file_audits),
    }


def write_audit_report(audit: dict[str, Any], output_dir: str | Path) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "schema_audit.json"
    markdown_path = output_dir / "schema_audit.md"

    import json

    json_path.write_text(json.dumps(audit, indent=2, sort_keys=True))

    aggregate = audit["aggregate"]
    lines = [
        "# LOB Schema and Enum Audit",
        "",
        f"- files_audited: {audit['files_audited']}",
        f"- total_rows: {aggregate['row_count']}",
        f"- all_required_columns_present: {aggregate['all_required_columns_present']}",
        f"- ordering_key_duplicate_count: {aggregate['ordering_key_duplicate_count']}",
        "",
        "## Missing required columns",
    ]
    if aggregate["missing_required_columns_union"]:
        for column in aggregate["missing_required_columns_union"]:
            lines.append(f"- {column}")
    else:
        lines.append("- none")

    lines.extend(["", "## Unknown enum codes"])
    any_unknown = False
    for field, codes in aggregate["unknown_enum_codes"].items():
        if codes:
            any_unknown = True
            lines.append(f"- {field}: {', '.join(str(code) for code in codes)}")
    if not any_unknown:
        lines.append("- none")

    lines.extend(["", "## Identity missing counts"])
    for field, count in aggregate["identity_missing_counts"].items():
        lines.append(f"- {field}: {count}")

    lines.extend(["", "## PASSIVEORDER / AGGRESSIVEORDER observed values"])
    for field, values in aggregate["passive_aggressive_observed_values"].items():
        lines.append(f"- {field}: {', '.join(values) if values else 'not present'}")

    lines.extend(["", "## First event code counts"])
    lines.append("### By partition")
    for code, count in aggregate["partition_first_event_code_counts"].items():
        label = EVENT_LABEL_BY_CODE.get(int(code), "unknown") if code != "None" else "None"
        lines.append(f"- {code} ({label}): {count}")
    lines.append("### By partition/order")
    for code, count in aggregate["order_first_event_code_counts"].items():
        label = EVENT_LABEL_BY_CODE.get(int(code), "unknown") if code != "None" else "None"
        lines.append(f"- {code} ({label}): {count}")

    lines.extend(["", "## Files"])
    for file_audit in audit["files"]:
        lines.append(f"### {file_audit['source_name']}")
        lines.append(f"- rows: {file_audit['row_count']}")
        lines.append(f"- columns: {file_audit['column_count']}")
        lines.append(f"- ordering_key_duplicate_count: {file_audit['ordering_key_duplicate_count']}")
        if file_audit["missing_required_columns"]:
            missing = ", ".join(file_audit["missing_required_columns"])
            lines.append(f"- missing_required_columns: {missing}")
        else:
            lines.append("- missing_required_columns: none")

    markdown_path.write_text("\n".join(lines) + "\n")
    return {"json": json_path, "markdown": markdown_path}
