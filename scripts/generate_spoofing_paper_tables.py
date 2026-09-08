#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from spoofing_detection.lob.actor_identity import (  # noqa: E402
    normalize_client_original_identity_value,
)

EXPECTED_SCHEMA_VERSION = "actor_execution_anchor_v2"
EXPECTED_ANCHOR_MODES = frozenset({"passive", "aggressive"})
EXTERNAL_ALERT_COUNT_KEYS = (
    "source_external_periods",
    "source_timed_periods",
    "source_date_only_periods",
    "source_periods_with_raw_subject_event",
    "union_timed_periods",
    "union_periods_with_raw_subject_event",
    "union_periods_with_recovered_execution",
    "union_periods_with_matched_withdrawal",
    "union_periods_with_strict_subject_scope_detection",
    "identity_aligned_union_timed_periods",
    "identity_aligned_union_periods_with_strict_detection",
    "identity_unaligned_union_timed_periods",
    "identity_unaligned_union_periods_with_strict_subject_scope_detection",
)
PUBLIC_SUBJECT_FIELDS = (
    "detector_actor_key_count_in_dataset",
    "identity_granularity_aligned",
    "identity_namespace",
    "source_external_periods",
    "source_timed_periods",
    "source_date_only_periods",
    "union_timed_periods",
    "union_periods_with_recovered_execution",
    "union_periods_with_matched_withdrawal",
    "union_periods_with_strict_subject_scope_detection",
    "identity_aligned_union_periods_with_strict_detection",
)
PUBLIC_PERIOD_SCALAR_FIELDS = (
    "recovered_child_fill_rows",
    "recovered_execution_clusters",
    "clusters_with_matched_withdrawal",
    "clusters_with_strict_detection",
    "date_level_recovered_child_fill_rows",
    "date_level_recovered_execution_clusters",
    "date_level_clusters_with_matched_withdrawal",
    "date_level_clusters_with_strict_detection",
)
PUBLIC_PERIOD_COUNT_MAP_FIELDS = (
    "recovered_child_fill_rows_by_anchor",
    "recovered_execution_clusters_by_anchor",
    "date_level_recovered_child_fill_rows_by_anchor",
    "date_level_recovered_execution_clusters_by_anchor",
)
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
            if normalize_client_original_identity_value(client_id) is None:
                raise ValueError(f"{path}: zero client sentinel cannot define an actor")
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
        "\\caption{Actor-attributed execution populations and reconstructed withdrawals by execution anchor. Individual execution messages provide provenance; executions are the analytical unit. Firm fallback counts matched executions for which the client code is missing and the firm identifier defines the actor. Assigned cancellations are counted once. Strict is the four-condition spoofing-compatible sequence gate.}",
        "\\label{tab:empirical_spoofing_diagnostics}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{2.1pt}",
        "\\begin{tabular}{@{}llrrrrrr@{}}",
        "\\toprule",
        "Instrument & Anchor & Executions & Execution messages & Matched & Firm fallback & Assigned cancels & Strict \\\\",
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
        "\\caption{Largest actor--anchor rapid-withdrawal groups by matched-execution count. Client denotes the original client code. Firm fallback denotes aggregation by firm only when that client code is missing. W/E max is the largest assigned withdrawal-to-execution ratio in the group. FPM$+$ and REV$+$ are descriptive shares among matched executions with finite midprice diagnostics.}",
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


def _public_period_totals(rows: object) -> dict[str, object]:
    if not isinstance(rows, list):
        raise ValueError("external alert audit source_period_results must be a list")
    totals: dict[str, object] = {}
    for field in PUBLIC_PERIOD_SCALAR_FIELDS:
        values: list[int] = []
        for row in rows:
            value = row.get(field) if isinstance(row, dict) else None
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"external alert audit field {field} must contain nonnegative integers")
            values.append(value)
        if values:
            totals[field] = sum(values)
    for field in PUBLIC_PERIOD_COUNT_MAP_FIELDS:
        aggregate: dict[str, int] = {}
        for row in rows:
            mapping = row.get(field) if isinstance(row, dict) else None
            if mapping is None:
                continue
            if not isinstance(mapping, dict):
                raise ValueError(f"external alert audit field {field} must contain count mappings")
            for key, value in mapping.items():
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError(f"external alert audit field {field} must contain nonnegative integers")
                aggregate[str(key)] = aggregate.get(str(key), 0) + value
        if aggregate:
            totals[field] = dict(sorted(aggregate.items()))
    return totals


def external_alert_artifacts(audit_path: Path) -> tuple[dict[str, object], dict[str, object]]:
    payload = audit_path.read_bytes()
    audit = json.loads(payload)
    source = audit.get("external_source")
    if not isinstance(source, dict):
        raise ValueError("external alert audit is missing external_source")
    source_hash = source.get("sha256")
    if (
        not isinstance(source_hash, str)
        or len(source_hash) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in source_hash)
    ):
        raise ValueError("external alert audit source hash must be a 64-character hexadecimal string")
    if source.get("local_path_disclosed") is not False:
        raise ValueError("external alert audit must not disclose the primary source path")

    overall = audit.get("overall")
    if not isinstance(overall, dict):
        raise ValueError("external alert audit is missing overall counts")
    missing = [key for key in EXTERNAL_ALERT_COUNT_KEYS if key not in overall]
    if missing:
        raise ValueError(f"external alert audit is missing overall counts: {', '.join(missing)}")
    overall_counts: dict[str, int] = {}
    for key, value in overall.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"external alert audit count {key} must be a nonnegative integer")
        overall_counts[str(key)] = value

    datasets = audit.get("datasets")
    if not isinstance(datasets, dict):
        raise ValueError("external alert audit is missing dataset summaries")
    public_datasets: dict[str, object] = {}
    for dataset_name, dataset in sorted(datasets.items()):
        if not isinstance(dataset, dict):
            raise ValueError(f"external alert audit dataset {dataset_name} must be a mapping")
        dataset_counts = {
            key: dataset[key]
            for key in (*EXTERNAL_ALERT_COUNT_KEYS, "distinct_external_actors")
            if key in dataset
        }
        actors = dataset.get("per_pseudonymized_actor", {})
        if not isinstance(actors, dict):
            raise ValueError(f"external alert audit dataset {dataset_name} has invalid actor summaries")
        ordered_actors = sorted(
            actors.values(),
            key=lambda actor: (
                int(actor.get("union_timed_periods", 0)) if isinstance(actor, dict) else 0,
                int(actor.get("source_external_periods", 0)) if isinstance(actor, dict) else 0,
            ),
            reverse=True,
        )
        subject_summaries = []
        for subject_rank, actor in enumerate(ordered_actors, start=1):
            if not isinstance(actor, dict):
                raise ValueError(f"external alert audit dataset {dataset_name} has invalid actor summary")
            subject_summaries.append(
                {
                    "subject_rank": subject_rank,
                    **{field: actor[field] for field in PUBLIC_SUBJECT_FIELDS if field in actor},
                }
            )
        public_datasets[str(dataset_name)] = {
            "aggregate_counts": dataset_counts,
            "subject_summaries": subject_summaries,
            "source_record_totals": _public_period_totals(dataset.get("source_period_results")),
        }

    provenance = {
        "audit_path": str(audit_path),
        "audit_sha256": hashlib.sha256(payload).hexdigest(),
        "primary_source": source,
        "verification_tier": audit.get("verification_tier"),
        "recall_unit": audit.get("recall_unit"),
        "overall_counts": overall_counts,
    }
    public_summary = {
        "schema_version": "spoofing_external_alert_audit_summary_v1",
        "audit_input_sha256": provenance["audit_sha256"],
        "primary_source": source,
        "verification_tier": audit.get("verification_tier"),
        "detector_recall_evaluated": audit.get("detector_recall_evaluated"),
        "recall_unit": audit.get("recall_unit"),
        "timestamp_precedence": audit.get("timestamp_precedence"),
        "identifier_policy": audit.get("identifier_policy"),
        "overall_counts": overall_counts,
        "datasets": public_datasets,
        "limitations": audit.get("limitations"),
    }
    return provenance, public_summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate actor- and execution-anchor-aware manuscript tables from validated execution outputs."
    )
    parser.add_argument("--run", action="append", type=parse_run, required=True, metavar="INSTRUMENT=RUN_DIR")
    parser.add_argument(
        "--external-alert-audit",
        type=Path,
        required=True,
        help="De-identified external-alert audit required by the manuscript",
    )
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
    external_provenance, external_public_summary = external_alert_artifacts(args.external_alert_audit)
    public_summary_path = args.output_dir / "external_alert_audit_summary.json"
    public_summary_payload = json.dumps(external_public_summary, indent=2, sort_keys=True) + "\n"
    public_summary_path.write_text(public_summary_payload)
    external_provenance["public_summary"] = {
        "path": str(public_summary_path),
        "sha256": hashlib.sha256(public_summary_payload.encode()).hexdigest(),
    }
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "analytical_unit": "execution_cluster",
        "actor_identity_mode": "client_then_firm",
        "execution_anchor_modes": ["passive", "aggressive"],
        "firm_fallback_semantics": "aggregate only when client_original_id is missing",
        "generator": "scripts/generate_spoofing_paper_tables.py",
        "runs": provenance,
        "external_alert_audit": external_provenance,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()