#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

UNKNOWN_CLIENT_IDS = {"", "0", "null", "none", "nan"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_attributable_client(column: str = "client_id") -> pl.Expr:
    normalized = pl.col(column).cast(pl.Utf8).fill_null("").str.to_lowercase()
    return ~normalized.is_in(sorted(UNKNOWN_CLIENT_IDS))


def _matched_expr(columns: list[str]) -> pl.Expr:
    if "has_matched_deceptive_cancel_window" in columns:
        return pl.col("has_matched_deceptive_cancel_window").fill_null(False)
    if "assigned_cancel_count" in columns:
        return pl.col("assigned_cancel_count").fill_null(0) > 0
    return pl.lit(False)


def _assigned_cancellation_count(run_dir: Path) -> int:
    candidates_path = run_dir / "execution_cancel_candidates.parquet"
    if not candidates_path.exists():
        raise FileNotFoundError(f"required canonical cancellation artifact not found: {candidates_path}")
    candidates = pl.read_parquet(candidates_path)
    if "assigned_flag" not in candidates.columns:
        raise ValueError(f"{candidates_path} lacks assigned_flag")
    assigned = candidates.filter(pl.col("assigned_flag").fill_null(False))
    id_column = next(
        (name for name in ("candidate_order_id", "cancel_order_id", "matched_deceptive_order_id") if name in assigned.columns),
        None,
    )
    if id_column is None:
        raise ValueError(f"{candidates_path} lacks a cancellation order identifier")
    physical_key = ["partition_id", "cancel_sort_index", id_column]
    missing = [column for column in physical_key if column not in assigned.columns]
    if missing:
        raise ValueError(f"{candidates_path} lacks physical cancellation key columns: {missing}")
    return assigned.select(physical_key).unique().height


def _positive_share(frame: pl.DataFrame, column: str) -> float | None:
    if column not in frame.columns:
        return None
    finite = frame.filter(pl.col(column).is_not_null() & pl.col(column).is_finite())
    if finite.is_empty():
        return None
    return float(finite.select((pl.col(column) > 0).mean()).item())


def summarize_run(instrument: str, run_dir: Path) -> tuple[dict[str, object], pl.DataFrame, dict[str, str]]:
    execution_path = run_dir / "execution_metrics.parquet"
    if not execution_path.exists():
        raise FileNotFoundError(execution_path)
    executions = pl.read_parquet(execution_path)
    if "execution_cluster_id" not in executions.columns:
        raise ValueError(f"{execution_path} is not cluster-aware")
    if executions.get_column("execution_cluster_id").is_duplicated().any():
        raise ValueError(f"duplicate execution_cluster_id in {execution_path}")

    matched = executions.filter(_matched_expr(executions.columns))
    child_expr = pl.col("child_fill_count").fill_null(1) if "child_fill_count" in executions.columns else pl.lit(1)
    raw_fill_count = int(executions.select(child_expr.sum()).item() or 0)
    attributable = executions.filter(_is_attributable_client()) if "client_id" in executions.columns else executions.head(0)
    matched_attributable = matched.filter(_is_attributable_client()) if "client_id" in matched.columns else matched.head(0)
    unknown_matched_count = matched.height - matched_attributable.height

    summary: dict[str, object] = {
        "instrument": instrument,
        "execution_cluster_count": executions.height,
        "raw_fill_message_count": raw_fill_count,
        "matched_cluster_count": matched.height,
        "assigned_cancellation_count": _assigned_cancellation_count(run_dir),
        "attributable_client_count": attributable.select(pl.col("client_id").cast(pl.Utf8).n_unique()).item()
        if "client_id" in attributable.columns and not attributable.is_empty()
        else 0,
        "matched_attributable_client_count": matched_attributable.select(pl.col("client_id").cast(pl.Utf8).n_unique()).item()
        if "client_id" in matched_attributable.columns and not matched_attributable.is_empty()
        else 0,
        "unknown_client_matched_cluster_count": unknown_matched_count,
        "positive_fpm_mid_share": _positive_share(matched, "favorable_mid_move_pre_fill"),
        "positive_reversion_mid_share": _positive_share(matched, "post_cancel_mid_reversion"),
    }

    if matched_attributable.is_empty():
        top_clients = pl.DataFrame(
            schema={
                "instrument": pl.Utf8,
                "client_id": pl.Utf8,
                "matched_cluster_count": pl.UInt32,
                "max_withdrawal_to_fill_ratio": pl.Float64,
                "positive_fpm_mid_share": pl.Float64,
                "positive_reversion_mid_share": pl.Float64,
            }
        )
    else:
        prepared = matched_attributable.with_columns(pl.col("client_id").cast(pl.Utf8))
        optional = {
            "withdrawal_to_fill_ratio": None,
            "favorable_mid_move_pre_fill": None,
            "post_cancel_mid_reversion": None,
        }
        for column, default in optional.items():
            if column not in prepared.columns:
                prepared = prepared.with_columns(pl.lit(default, dtype=pl.Float64).alias(column))
        top_clients = (
            prepared.group_by("client_id")
            .agg(
                pl.len().cast(pl.UInt32).alias("matched_cluster_count"),
                pl.col("withdrawal_to_fill_ratio").max().alias("max_withdrawal_to_fill_ratio"),
                (pl.col("favorable_mid_move_pre_fill") > 0).mean().alias("positive_fpm_mid_share"),
                (pl.col("post_cancel_mid_reversion") > 0).mean().alias("positive_reversion_mid_share"),
            )
            .sort(["matched_cluster_count", "client_id"], descending=[True, False])
            .head(3)
            .with_columns(pl.lit(instrument).alias("instrument"))
            .select(
                "instrument",
                "client_id",
                "matched_cluster_count",
                "max_withdrawal_to_fill_ratio",
                "positive_fpm_mid_share",
                "positive_reversion_mid_share",
            )
        )

    hashes = {"execution_metrics": _sha256(execution_path)}
    candidates_path = run_dir / "execution_cancel_candidates.parquet"
    if candidates_path.exists():
        hashes["execution_cancel_candidates"] = _sha256(candidates_path)
    metadata_path = run_dir / "metadata.json"
    if metadata_path.exists():
        hashes["run_metadata"] = _sha256(metadata_path)
    return summary, top_clients, hashes


def _fmt_int(value: object) -> str:
    if value is None:
        return "--"
    return f"{int(value):,}"


def _fmt_float(value: object, digits: int = 2) -> str:
    return "--" if value is None else f"{float(value):.{digits}f}"


def _fmt_pct(value: object) -> str:
    return "--" if value is None else f"{100.0 * float(value):.1f}\\%"


def _macro_name(instrument: str, suffix: str) -> str:
    stem = "".join(ch for ch in instrument.title() if ch.isalnum())
    return f"{stem}{suffix}"


def render_tex(summary: pl.DataFrame, top_clients: pl.DataFrame) -> tuple[str, str]:
    macros: list[str] = ["% Generated by scripts/generate_spoofing_paper_tables.py; do not edit manually."]
    for row in summary.iter_rows(named=True):
        name = str(row["instrument"])
        values = {
            "Clusters": _fmt_int(row["execution_cluster_count"]),
            "RawFills": _fmt_int(row["raw_fill_message_count"]),
            "MatchedClusters": _fmt_int(row["matched_cluster_count"]),
            "AssignedCancellations": _fmt_int(row["assigned_cancellation_count"]),
            "FPMShare": _fmt_pct(row["positive_fpm_mid_share"]),
            "REVShare": _fmt_pct(row["positive_reversion_mid_share"]),
        }
        for suffix, value in values.items():
            macros.append(f"\\newcommand{{\\{_macro_name(name, suffix)}}}{{{value}}}")

    lines = [
        "% Generated by scripts/generate_spoofing_paper_tables.py; do not edit manually.",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Cluster-level exploratory attributed-withdrawal diagnostics. Raw fills remain audit objects; clusters are the analytical unit. Assigned cancellations are counted once by the canonical assignment rule.}",
        "\\label{tab:empirical_spoofing_diagnostics}",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{2.5pt}",
        "\\begin{tabular}{@{}lrrrrrr@{}}",
        "\\toprule",
        "Instrument & Clusters & Raw fills & Matched & Assigned cancels & FPM$>0$ & REV$>0$ \\\\",
        "\\midrule",
    ]
    for row in summary.iter_rows(named=True):
        lines.append(
            f"{row['instrument']} & {_fmt_int(row['execution_cluster_count'])} & {_fmt_int(row['raw_fill_message_count'])} & "
            f"{_fmt_int(row['matched_cluster_count'])} & {_fmt_int(row['assigned_cancellation_count'])} & "
            f"{_fmt_pct(row['positive_fpm_mid_share'])} & {_fmt_pct(row['positive_reversion_mid_share'])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])

    lines.extend(
        [
            "\\begin{table}[htbp]",
            "\\centering",
            "\\caption{Largest attributable client-level rapid-withdrawal groups. Unknown or missing client identifiers are excluded from this tabulation and reported separately in the generated CSV.}",
            "\\label{tab:top_client_results}",
            "\\footnotesize",
            "\\setlength{\\tabcolsep}{2.5pt}",
            "\\begin{tabular}{@{}llrrr@{}}",
            "\\toprule",
            "Instrument & Client & Clusters & W/F max & FPM$>0$ / REV$>0$ \\\\",
            "\\midrule",
        ]
    )
    for row in top_clients.iter_rows(named=True):
        lines.append(
            f"{row['instrument']} & {row['client_id']} & {_fmt_int(row['matched_cluster_count'])} & "
            f"{_fmt_float(row['max_withdrawal_to_fill_ratio'], 1)} & "
            f"{_fmt_pct(row['positive_fpm_mid_share'])} / {_fmt_pct(row['positive_reversion_mid_share'])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(macros) + "\n", "\n".join(lines)


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--run must be INSTRUMENT=RUN_DIR")
    instrument, path = value.split("=", 1)
    if not instrument.strip() or not path.strip():
        raise argparse.ArgumentTypeError("--run must be INSTRUMENT=RUN_DIR")
    return instrument.strip(), Path(path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate manuscript tables from cluster-level spoofing outputs.")
    parser.add_argument("--run", action="append", type=parse_run, required=True, metavar="INSTRUMENT=RUN_DIR")
    parser.add_argument("--output-dir", type=Path, default=Path("paper/generated"))
    args = parser.parse_args(argv)

    summaries: list[dict[str, object]] = []
    client_frames: list[pl.DataFrame] = []
    provenance: dict[str, object] = {}
    for instrument, run_dir in args.run:
        summary, top_clients, hashes = summarize_run(instrument, run_dir)
        summaries.append(summary)
        client_frames.append(top_clients)
        provenance[instrument] = {"run_dir": str(run_dir), "input_hashes": hashes}

    summary_frame = pl.DataFrame(summaries)
    clients_frame = pl.concat(client_frames, how="vertical_relaxed") if client_frames else pl.DataFrame()
    macros, tables = render_tex(summary_frame, clients_frame)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_frame.write_csv(args.output_dir / "cluster_summary.csv")
    clients_frame.write_csv(args.output_dir / "top_clients.csv")
    (args.output_dir / "spoofing_empirical_macros.tex").write_text(macros)
    (args.output_dir / "spoofing_empirical_tables.tex").write_text(tables)
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "analytical_unit": "execution_cluster",
        "generator": "scripts/generate_spoofing_paper_tables.py",
        "runs": provenance,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
