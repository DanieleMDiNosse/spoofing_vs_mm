#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

EXPECTED_SCHEMA_VERSION = "actor_execution_anchor_v2"
EXPECTED_ANCHOR_MODES = frozenset({"passive", "aggressive"})
EXPECTED_IDENTITY_LEVELS = frozenset({"client_original", "firm"})
MISSING_IDENTIFIER_TOKENS = frozenset({"", "null", "none", "nan"})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_identifier(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text.lower() not in MISSING_IDENTIFIER_TOKENS else None


def _require_columns(frame: pl.DataFrame, required: set[str], *, path: Path) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} lacks required actor/anchor columns: {missing}")


def _load_contract_metadata(run_dir: Path) -> tuple[dict[str, Any], Path]:
    metadata_path = run_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"required run metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    expected = {
        "output_schema_version": EXPECTED_SCHEMA_VERSION,
        "analytical_unit": "execution_cluster",
        "actor_identity_mode": "client_then_firm",
        "firm_fallback_semantics": "aggregate only when client_original_id is missing",
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"{metadata_path} has {key}={metadata.get(key)!r}; expected {value!r}")
    modes = {str(mode).strip().lower() for mode in metadata.get("execution_anchor_modes", [])}
    if modes != EXPECTED_ANCHOR_MODES:
        raise ValueError(
            f"{metadata_path} must select passive and aggressive execution anchors; observed {sorted(modes)}"
        )
    return metadata, metadata_path


def _validate_actor_contract(executions: pl.DataFrame, *, path: Path) -> None:
    observed_anchors = set(executions.get_column("execution_anchor_mode").drop_nulls().to_list())
    if observed_anchors != EXPECTED_ANCHOR_MODES:
        raise ValueError(f"{path} must contain both execution anchors; observed {sorted(observed_anchors)}")
    observed_levels = set(executions.get_column("identity_level").drop_nulls().to_list())
    if not observed_levels.issubset(EXPECTED_IDENTITY_LEVELS):
        raise ValueError(f"{path} contains unsupported identity levels: {sorted(observed_levels)}")

    identity_columns = [
        "actor_key",
        "actor_id",
        "identity_level",
        "identity_fallback_flag",
        "client_original_id",
        "firm_id",
    ]
    for row in executions.select(identity_columns).iter_rows(named=True):
        level = row["identity_level"]
        actor_key = _normalized_identifier(row["actor_key"])
        actor_id = _normalized_identifier(row["actor_id"])
        client_id = _normalized_identifier(row["client_original_id"])
        firm_id = _normalized_identifier(row["firm_id"])
        fallback = row["identity_fallback_flag"]
        if level == "client_original":
            if client_id is None:
                raise ValueError(f"{path}: client identity row lacks a client identifier")
            if actor_key != f"client_original:{client_id}" or actor_id != client_id or fallback is not False:
                raise ValueError(f"{path}: inconsistent client actor identity fields")
        elif level == "firm":
            if client_id is not None:
                raise ValueError(f"{path}: firm fallback row has a client identifier")
            if firm_id is None:
                raise ValueError(f"{path}: firm fallback row lacks a firm identifier")
            if actor_key != f"firm:{firm_id}" or actor_id != firm_id or fallback is not True:
                raise ValueError(f"{path}: inconsistent firm fallback identity fields")
        else:
            raise ValueError(f"{path}: missing or unsupported identity level {level!r}")


def _matched_expr() -> pl.Expr:
    return pl.col("has_matched_deceptive_cancel_window").fill_null(False)


def _assigned_candidates(run_dir: Path) -> tuple[pl.DataFrame, Path]:
    candidates_path = run_dir / "execution_cancel_candidates.parquet"
    if not candidates_path.exists():
        raise FileNotFoundError(f"required canonical cancellation artifact not found: {candidates_path}")
    candidates = pl.read_parquet(candidates_path)
    required = {"assigned_flag", "partition_id", "cancel_sort_index", "candidate_order_id"}
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise ValueError(f"{candidates_path} lacks physical cancellation key columns: {missing}")
    assigned = (
        candidates.filter(pl.col("assigned_flag").fill_null(False))
        .unique(subset=["partition_id", "cancel_sort_index", "candidate_order_id"])
    )
    return assigned, candidates_path


def _assigned_cancellation_count(run_dir: Path) -> int:
    assigned, _ = _assigned_candidates(run_dir)
    return assigned.height


def _positive_counts(frame: pl.DataFrame, column: str) -> tuple[int, int]:
    observed = frame.filter(pl.col(column).is_not_null() & pl.col(column).is_finite())
    if observed.is_empty():
        return 0, 0
    positive = int(observed.select((pl.col(column) > 0).sum()).item() or 0)
    return positive, observed.height


def _positive_group_share(column: str, alias: str) -> pl.Expr:
    observed = pl.col(column).is_not_null() & pl.col(column).is_finite()
    return pl.when(observed).then(pl.col(column) > 0).otherwise(None).mean().alias(alias)


def _empty_top_actor_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "instrument": pl.String,
            "actor_key": pl.String,
            "actor_id": pl.String,
            "identity_level": pl.String,
            "execution_anchor_mode": pl.String,
            "matched_cluster_count": pl.UInt32,
            "max_withdrawal_to_execution_ratio": pl.Float64,
            "positive_fpm_mid_share": pl.Float64,
            "positive_reversion_mid_share": pl.Float64,
        }
    )


def summarize_run(instrument: str, run_dir: Path) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, str]]:
    _, metadata_path = _load_contract_metadata(run_dir)
    execution_path = run_dir / "execution_metrics.parquet"
    if not execution_path.exists():
        raise FileNotFoundError(execution_path)
    executions = pl.read_parquet(execution_path)
    if "execution_cluster_id" not in executions.columns:
        raise ValueError(f"{execution_path} is not cluster-aware")
    if executions.get_column("execution_cluster_id").is_duplicated().any():
        raise ValueError(f"duplicate execution_cluster_id in {execution_path}")
    required_execution_columns = {
        "execution_cluster_id",
        "actor_key",
        "actor_id",
        "identity_level",
        "identity_fallback_flag",
        "client_original_id",
        "firm_id",
        "execution_anchor_mode",
        "child_fill_count",
        "has_matched_deceptive_cancel_window",
        "spoofing_compatible_sequence",
        "withdrawal_to_execution_ratio",
        "favorable_mid_move_pre_fill",
        "post_cancel_mid_reversion",
    }
    _require_columns(executions, required_execution_columns, path=execution_path)
    _validate_actor_contract(executions, path=execution_path)

    assigned, candidates_path = _assigned_candidates(run_dir)
    if not assigned.is_empty() and "execution_anchor_mode" not in assigned.columns:
        raise ValueError(f"{candidates_path} lacks execution_anchor_mode")

    summary_rows: list[dict[str, object]] = []
    for anchor in sorted(EXPECTED_ANCHOR_MODES):
        anchor_frame = executions.filter(pl.col("execution_anchor_mode") == anchor)
        matched = anchor_frame.filter(_matched_expr())
        assigned_count = (
            assigned.filter(pl.col("execution_anchor_mode") == anchor).height if not assigned.is_empty() else 0
        )
        fpm_positive, fpm_observed = _positive_counts(matched, "favorable_mid_move_pre_fill")
        reversion_positive, reversion_observed = _positive_counts(matched, "post_cancel_mid_reversion")
        summary_rows.append(
            {
                "instrument": instrument,
                "execution_anchor_mode": anchor,
                "execution_cluster_count": anchor_frame.height,
                "raw_fill_message_count": int(anchor_frame.get_column("child_fill_count").sum() or 0),
                "matched_cluster_count": matched.height,
                "firm_fallback_cluster_count": anchor_frame.filter(pl.col("identity_level") == "firm").height,
                "firm_fallback_matched_cluster_count": matched.filter(pl.col("identity_level") == "firm").height,
                "assigned_cancellation_count": assigned_count,
                "compatible_sequence_count": int(
                    anchor_frame.get_column("spoofing_compatible_sequence").fill_null(False).sum() or 0
                ),
                "fpm_positive_count": fpm_positive,
                "fpm_observed_count": fpm_observed,
                "positive_fpm_mid_share": fpm_positive / fpm_observed if fpm_observed else None,
                "reversion_positive_count": reversion_positive,
                "reversion_observed_count": reversion_observed,
                "positive_reversion_mid_share": (
                    reversion_positive / reversion_observed if reversion_observed else None
                ),
            }
        )
    summary = pl.DataFrame(summary_rows)

    matched = executions.filter(_matched_expr())
    if matched.is_empty():
        top_actors = _empty_top_actor_frame()
    else:
        top_actors = (
            matched.group_by(["actor_key", "actor_id", "identity_level", "execution_anchor_mode"])
            .agg(
                pl.len().cast(pl.UInt32).alias("matched_cluster_count"),
                pl.col("withdrawal_to_execution_ratio")
                .max()
                .alias("max_withdrawal_to_execution_ratio"),
                _positive_group_share("favorable_mid_move_pre_fill", "positive_fpm_mid_share"),
                _positive_group_share("post_cancel_mid_reversion", "positive_reversion_mid_share"),
            )
            .sort(
                ["matched_cluster_count", "actor_key", "execution_anchor_mode"],
                descending=[True, False, False],
            )
            .head(3)
            .with_columns(pl.lit(instrument).alias("instrument"))
            .select(_empty_top_actor_frame().columns)
        )

    hashes = {
        "execution_metrics": _sha256(execution_path),
        "execution_cancel_candidates": _sha256(candidates_path),
        "run_metadata": _sha256(metadata_path),
    }
    return summary, top_actors, hashes


def _fmt_int(value: object) -> str:
    return "--" if value is None else f"{int(value):,}"


def _fmt_float(value: object, digits: int = 2) -> str:
    return "--" if value is None else f"{float(value):.{digits}f}"


def _fmt_pct(value: object) -> str:
    return "--" if value is None else f"{100.0 * float(value):.1f}\\%"


def _macro_name(instrument: str, suffix: str) -> str:
    stem = "".join(ch for ch in instrument.title() if ch.isalnum())
    return f"{stem}{suffix}"


def _tex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(char, char) for char in text)


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _instrument_totals(frame: pl.DataFrame) -> dict[str, object]:
    rows = frame.iter_rows(named=True)
    anchor_rows = {str(row["execution_anchor_mode"]): row for row in rows}
    totals: dict[str, object] = {}
    for column in (
        "execution_cluster_count",
        "raw_fill_message_count",
        "matched_cluster_count",
        "firm_fallback_cluster_count",
        "firm_fallback_matched_cluster_count",
        "assigned_cancellation_count",
        "compatible_sequence_count",
        "fpm_positive_count",
        "fpm_observed_count",
        "reversion_positive_count",
        "reversion_observed_count",
    ):
        totals[column] = sum(int(row[column]) for row in anchor_rows.values())
    totals["positive_fpm_mid_share"] = _ratio(
        int(totals["fpm_positive_count"]), int(totals["fpm_observed_count"])
    )
    totals["positive_reversion_mid_share"] = _ratio(
        int(totals["reversion_positive_count"]), int(totals["reversion_observed_count"])
    )
    for anchor in EXPECTED_ANCHOR_MODES:
        totals[anchor] = anchor_rows[anchor]
    return totals


def render_tex(summary: pl.DataFrame, top_actors: pl.DataFrame) -> tuple[str, str, str]:
    macros: list[str] = ["% Generated by scripts/generate_spoofing_paper_tables.py; do not edit manually."]
    for instrument in summary.get_column("instrument").unique(maintain_order=True):
        name = str(instrument)
        totals = _instrument_totals(summary.filter(pl.col("instrument") == instrument))
        passive = totals["passive"]
        aggressive = totals["aggressive"]
        values = {
            "Clusters": _fmt_int(totals["execution_cluster_count"]),
            "RawFills": _fmt_int(totals["raw_fill_message_count"]),
            "MatchedClusters": _fmt_int(totals["matched_cluster_count"]),
            "FirmFallbackClusters": _fmt_int(totals["firm_fallback_cluster_count"]),
            "FirmFallbackMatchedClusters": _fmt_int(totals["firm_fallback_matched_cluster_count"]),
            "AssignedCancellations": _fmt_int(totals["assigned_cancellation_count"]),
            "CompatibleSequences": _fmt_int(totals["compatible_sequence_count"]),
            "FPMShare": _fmt_pct(totals["positive_fpm_mid_share"]),
            "REVShare": _fmt_pct(totals["positive_reversion_mid_share"]),
            "PassiveClusters": _fmt_int(passive["execution_cluster_count"]),
            "PassiveMatchedClusters": _fmt_int(passive["matched_cluster_count"]),
            "AggressiveClusters": _fmt_int(aggressive["execution_cluster_count"]),
            "AggressiveMatchedClusters": _fmt_int(aggressive["matched_cluster_count"]),
        }
        for suffix, value in values.items():
            macros.append(f"\\newcommand{{\\{_macro_name(name, suffix)}}}{{{value}}}")

    lines = [
        "% Generated by scripts/generate_spoofing_paper_tables.py; do not edit manually.",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Actor-attributed execution-cluster populations and reconstructed withdrawals by execution anchor. Raw fills are provenance messages; clusters are the analytical unit. Firm fallback counts matched clusters for which the client code is missing and the firm identifier defines the actor. Assigned cancellations are counted once. Strict is the four-condition spoofing-compatible sequence gate.}",
        "\\label{tab:empirical_spoofing_diagnostics}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{2.1pt}",
        "\\begin{tabular}{@{}llrrrrrr@{}}",
        "\\toprule",
        "Instrument & Anchor & Clusters & Raw fills & Matched & Firm fallback & Assigned cancels & Strict \\\\",
        "\\midrule",
    ]
    for row in summary.iter_rows(named=True):
        anchor = str(row["execution_anchor_mode"]).title()
        lines.append(
            f"{_tex_escape(row['instrument'])} & {anchor} & {_fmt_int(row['execution_cluster_count'])} & "
            f"{_fmt_int(row['raw_fill_message_count'])} & {_fmt_int(row['matched_cluster_count'])} & "
            f"{_fmt_int(row['firm_fallback_matched_cluster_count'])} & "
            f"{_fmt_int(row['assigned_cancellation_count'])} & {_fmt_int(row['compatible_sequence_count'])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    summary_table = "\n".join(lines)

    lines = [
        "% Generated by scripts/generate_spoofing_paper_tables.py; do not edit manually.",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Largest actor--anchor rapid-withdrawal groups by matched-cluster count. Client denotes the original client code. Firm fallback denotes aggregation by firm only when that client code is missing. W/E max is the largest assigned withdrawal-to-execution ratio in the group. FPM$+$ and REV$+$ are descriptive shares among matched clusters with finite mid-price diagnostics.}",
        "\\label{tab:top_actor_results}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{2.0pt}",
        "\\begin{tabular}{@{}llllrrrr@{}}",
        "\\toprule",
        "Instrument & Actor & Identity & Anchor & Matched & W/E max & FPM$+$ & REV$+$ \\\\",
        "\\midrule",
    ]
    identity_labels = {"client_original": "Client", "firm": "Firm fallback"}
    for row in top_actors.iter_rows(named=True):
        lines.append(
            f"{_tex_escape(row['instrument'])} & {_tex_escape(row['actor_id'])} & "
            f"{identity_labels[str(row['identity_level'])]} & "
            f"{str(row['execution_anchor_mode']).title()} & {_fmt_int(row['matched_cluster_count'])} & "
            f"{_fmt_float(row['max_withdrawal_to_execution_ratio'], 1)} & "
            f"{_fmt_pct(row['positive_fpm_mid_share'])} & "
            f"{_fmt_pct(row['positive_reversion_mid_share'])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(macros) + "\n", summary_table, "\n".join(lines)


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--run must be INSTRUMENT=RUN_DIR")
    instrument, path = value.split("=", 1)
    if not instrument.strip() or not path.strip():
        raise argparse.ArgumentTypeError("--run must be INSTRUMENT=RUN_DIR")
    return instrument.strip(), Path(path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate actor- and execution-anchor-aware manuscript tables from validated cluster outputs."
    )
    parser.add_argument("--run", action="append", type=parse_run, required=True, metavar="INSTRUMENT=RUN_DIR")
    parser.add_argument("--output-dir", type=Path, default=Path("paper/generated"))
    args = parser.parse_args(argv)

    summary_frames: list[pl.DataFrame] = []
    actor_frames: list[pl.DataFrame] = []
    provenance: dict[str, object] = {}
    for instrument, run_dir in args.run:
        summary, top_actors, hashes = summarize_run(instrument, run_dir)
        summary_frames.append(summary)
        actor_frames.append(top_actors)
        provenance[instrument] = {"run_dir": str(run_dir), "input_hashes": hashes}

    summary_frame = pl.concat(summary_frames, how="vertical_relaxed")
    actors_frame = pl.concat(actor_frames, how="vertical_relaxed") if actor_frames else _empty_top_actor_frame()
    macros, summary_table, actor_table = render_tex(summary_frame, actors_frame)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_frame.write_csv(args.output_dir / "cluster_summary.csv")
    actors_frame.write_csv(args.output_dir / "top_actor_groups.csv")
    stale_clients_path = args.output_dir / "top_clients.csv"
    if stale_clients_path.exists():
        stale_clients_path.unlink()
    (args.output_dir / "spoofing_empirical_macros.tex").write_text(macros)
    (args.output_dir / "spoofing_empirical_tables.tex").write_text(summary_table)
    (args.output_dir / "spoofing_actor_table.tex").write_text(actor_table)
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "analytical_unit": "execution_cluster",
        "actor_identity_mode": "client_then_firm",
        "execution_anchor_modes": ["passive", "aggressive"],
        "firm_fallback_semantics": "aggregate only when client_original_id is missing",
        "generator": "scripts/generate_spoofing_paper_tables.py",
        "runs": provenance,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()