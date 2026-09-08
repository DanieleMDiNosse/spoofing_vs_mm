#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from spoofing_detection.lob.spoofing_config import (
    DEFAULT_SPOOFING_CONFIG_PATH,
    load_spoofing_config_defaults,
    reject_parameter_overrides,
    spoofing_config_provenance,
)


_CONFIGURABLE_DEFAULT_KEYS = {"gap_ms"}
_CONFIG_PARAMETER_OPTIONS = {
    "--gap-ms",
    "--empirical-kernel",
    "--tick-size",
    "--compact-state",
    "--no-compact-state",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_output(instrument: str, gap_ms: int, output_dir: Path) -> dict[str, object]:
    execution_path = output_dir / "execution_metrics.parquet"
    members_path = output_dir / "execution_cluster_members.parquet"
    candidates_path = output_dir / "execution_cancel_candidates.parquet"
    for required in (execution_path, members_path, candidates_path):
        if not required.exists():
            raise FileNotFoundError(required)
    executions = pl.read_parquet(execution_path)
    members = pl.read_parquet(members_path)
    candidates = pl.read_parquet(candidates_path)
    if executions.get_column("execution_cluster_id").is_duplicated().any():
        raise ValueError(f"duplicate clusters at gap={gap_ms}")
    member_key = [column for column in ("partition_id", "child_sort_index") if column in members.columns]
    if members.select(member_key).is_duplicated().any():
        raise ValueError(f"raw child fill assigned more than once at gap={gap_ms}")
    causal_columns = {"cancel_sort_index", "cluster_last_sort_index"}
    missing_causal_columns = causal_columns - set(candidates.columns)
    if missing_causal_columns:
        raise ValueError(
            f"missing cancellation-causality columns at gap={gap_ms}: "
            f"{sorted(missing_causal_columns)}"
        )
    noncausal = candidates.filter(
        pl.col("cancel_sort_index") <= pl.col("cluster_last_sort_index")
    )
    if not noncausal.is_empty():
        raise ValueError(f"non-causal cancellation candidate at gap={gap_ms}")
    assigned = candidates.filter(pl.col("assigned_flag").fill_null(False))
    cancel_key = [column for column in ("partition_id", "cancel_sort_index", "candidate_order_id") if column in assigned.columns]
    if cancel_key and assigned.select(cancel_key).is_duplicated().any():
        raise ValueError(f"cancellation assigned more than once at gap={gap_ms}")
    matched_column = "has_matched_deceptive_cancel_window"
    matched = executions.filter(pl.col(matched_column).fill_null(False)) if matched_column in executions.columns else executions.head(0)
    matched_ids = set(matched.get_column("execution_cluster_id").to_list())
    assigned_ids = set(assigned.get_column("execution_cluster_id").to_list())
    if matched_ids != assigned_ids:
        raise ValueError(f"matched-cluster flags disagree with assigned cancellations at gap={gap_ms}")
    scale_column = (
        "withdrawal_profile_scale_event"
        if "withdrawal_profile_scale_event" in matched.columns
        else "withdrawal_to_fill_ratio"
    )
    branch_summaries: dict[str, object] = {}
    for anchor_mode, canonical_metric in (
        ("passive", "WMSCI_passive"),
        ("aggressive", "WMSCI_aggressive"),
    ):
        branch = matched.filter(pl.col("execution_anchor_mode") == anchor_mode)
        product_column = canonical_metric if canonical_metric in branch.columns else "WMSCI_event"
        branch_summaries[f"matched_cluster_count_{anchor_mode}"] = branch.height
        branch_summaries[f"max_withdrawal_profile_scale_event_{anchor_mode}"] = (
            branch.select(pl.col(scale_column).max()).item() if not branch.is_empty() else None
        )
        branch_summaries[f"median_withdrawal_profile_scale_event_{anchor_mode}"] = (
            branch.select(pl.col(scale_column).median()).item() if not branch.is_empty() else None
        )
        branch_summaries[f"max_{canonical_metric}"] = (
            branch.select(pl.col(product_column).max()).item() if not branch.is_empty() else None
        )
        branch_summaries[f"median_{canonical_metric}"] = (
            branch.select(pl.col(product_column).median()).item() if not branch.is_empty() else None
        )
    return {
        "instrument": instrument,
        "execution_cluster_max_gap_ms": gap_ms,
        "execution_cluster_count": executions.height,
        "raw_fill_message_count": members.height,
        "matched_cluster_count": matched.height,
        "assigned_candidate_count": assigned.height,
        "candidate_link_count": candidates.height,
        **branch_summaries,
        "execution_metrics_sha256": _sha256(execution_path),
        "members_sha256": _sha256(members_path),
        "candidates_sha256": _sha256(candidates_path),
    }


def _write_effective_config(source: Path, *, gap_ms: int, destination: Path) -> None:
    payload = json.loads(source.read_text())
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("configuration section 'metrics' must be a JSON object")
    metrics["execution_cluster_max_gap_ms"] = gap_ms
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _without_comment_keys(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: _without_comment_keys(item)
            for key, item in value.items()
            if not str(key).startswith("_comment_")
        }
    if isinstance(value, list):
        return [_without_comment_keys(item) for item in value]
    return value


def _effective_metrics(config_path: Path) -> dict[str, object]:
    payload = json.loads(config_path.read_text())
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("configuration section 'metrics' must be a JSON object")
    return {
        str(key): _without_comment_keys(value)
        for key, value in metrics.items()
        if not str(key).startswith("_comment_")
    }


def _can_reuse_output(
    output_dir: Path,
    *,
    effective_config: Path,
    gap_ms: int,
    input_sha256: str,
    quote_panel_sha256: str | None,
    empirical_depth_kernel_sha256: str | None,
) -> bool:
    required_artifacts = {
        "execution_metrics": output_dir / "execution_metrics.parquet",
        "execution_cluster_members": output_dir / "execution_cluster_members.parquet",
        "execution_cancel_candidates": output_dir / "execution_cancel_candidates.parquet",
    }
    metadata_path = output_dir / "metadata.json"
    if not metadata_path.exists() or any(not path.exists() for path in required_artifacts.values()):
        return False
    try:
        metadata = json.loads(metadata_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    expected = {
        "parameter_source": "json_config_only",
        "config_section": "metrics",
        "config_sha256": _sha256(effective_config),
        "execution_cluster_max_gap_ms": gap_ms,
    }
    expected_input_hashes = {
        "raw_events_sha256": input_sha256,
        "quote_panel_sha256": quote_panel_sha256,
        "empirical_depth_kernel_sha256": empirical_depth_kernel_sha256,
    }
    input_hashes = metadata.get("input_hashes")
    artifact_hashes = metadata.get("artifact_hashes")
    return (
        all(metadata.get(key) == value for key, value in expected.items())
        and isinstance(input_hashes, dict)
        and all(input_hashes.get(key) == value for key, value in expected_input_hashes.items())
        and isinstance(artifact_hashes, dict)
        and all(artifact_hashes.get(key) == _sha256(path) for key, path in required_artifacts.items())
    )


def build_command(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    effective_config: Path,
) -> list[str]:
    command = [
        args.python,
        str(args.compute_script),
        "--input",
        str(args.input),
        "--output-dir",
        str(output_dir),
        "--config",
        str(effective_config),
    ]
    if args.quote_panel is not None:
        command.extend(["--quote-panel", str(args.quote_panel)])
    return command


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    config_parser.add_argument("--config", type=Path, default=DEFAULT_SPOOFING_CONFIG_PATH)
    config_args, _ = config_parser.parse_known_args(argv)
    try:
        config_defaults = load_spoofing_config_defaults(
            config_path=config_args.config,
            section="cluster_sensitivity",
            allowed_keys=_CONFIGURABLE_DEFAULT_KEYS,
        )
    except (OSError, ValueError) as exc:
        config_parser.error(str(exc))

    parser = argparse.ArgumentParser(
        description="Rerun cluster-aware spoofing metrics across inter-fill-gap thresholds.",
        allow_abbrev=False,
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--instrument", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--quote-panel", type=Path)
    parser.add_argument("--config", type=Path, default=config_args.config)
    parser.add_argument("--gap-ms", type=int, nargs="+")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--compute-script", type=Path, default=Path("scripts/compute_spoofing_metrics.py"))
    parser.add_argument("--reuse-existing", action="store_true")
    parser.set_defaults(**config_defaults)
    reject_parameter_overrides(
        parser,
        argv,
        parameter_options=_CONFIG_PARAMETER_OPTIONS,
    )
    args = parser.parse_args(argv)
    if any(gap < 0 for gap in args.gap_ms):
        parser.error("--gap-ms values must be non-negative")
    if len(set(args.gap_ms)) != len(args.gap_ms):
        parser.error("--gap-ms values must be unique")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    source_metrics = _effective_metrics(args.config)
    if source_metrics.get("tick_size") is None and args.quote_panel is None:
        raise ValueError("--quote-panel is required when metrics.tick_size is null")
    input_sha256 = _sha256(args.input)
    quote_panel_sha256 = _sha256(args.quote_panel) if args.quote_panel is not None else None
    kernel_path = source_metrics.get("empirical_depth_kernel")
    empirical_depth_kernel_sha256 = _sha256(Path(str(kernel_path))) if kernel_path is not None else None
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    commands: list[list[str]] = []
    effective_configs: list[dict[str, object]] = []
    for gap_ms in sorted(args.gap_ms):
        output_dir = args.output_root / f"{args.instrument}_cluster_gap_{gap_ms}ms"
        effective_config = args.output_root / f"effective_config_gap_{gap_ms}ms.json"
        _write_effective_config(args.config, gap_ms=gap_ms, destination=effective_config)
        command = build_command(
            args,
            output_dir=output_dir,
            effective_config=effective_config,
        )
        reuse_output = args.reuse_existing and _can_reuse_output(
            output_dir,
            effective_config=effective_config,
            gap_ms=gap_ms,
            input_sha256=input_sha256,
            quote_panel_sha256=quote_panel_sha256,
            empirical_depth_kernel_sha256=empirical_depth_kernel_sha256,
        )
        commands.append(command)
        effective_configs.append(
            {
                "execution_cluster_max_gap_ms": gap_ms,
                "path": str(effective_config),
                "sha256": _sha256(effective_config),
                "effective_metrics": _effective_metrics(effective_config),
                "reused_output": reuse_output,
            }
        )
        if not reuse_output:
            if output_dir.exists() and any(output_dir.iterdir()):
                raise FileExistsError(
                    "refusing to reuse or overwrite non-empty output directory without "
                    f"matching strict-config provenance: {output_dir}"
                )
            subprocess.run(command, check=True)
        rows.append(summarize_output(args.instrument, gap_ms, output_dir))

    summary = pl.DataFrame(rows).sort("execution_cluster_max_gap_ms")
    if summary.get_column("raw_fill_message_count").n_unique() != 1:
        raise ValueError("raw child-fill population changed across sensitivity runs")
    summary.write_csv(args.output_root / "cluster_sensitivity.csv")
    summary.write_parquet(args.output_root / "cluster_sensitivity.parquet")
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        **spoofing_config_provenance(args.config, section="cluster_sensitivity"),
        "analytical_unit": "execution_cluster",
        "input": str(args.input),
        "input_sha256": input_sha256,
        "quote_panel": str(args.quote_panel) if args.quote_panel is not None else None,
        "quote_panel_sha256": quote_panel_sha256,
        "empirical_depth_kernel_sha256": empirical_depth_kernel_sha256,
        "gap_ms": sorted(args.gap_ms),
        "effective_configs": effective_configs,
        "commands": commands,
    }
    (args.output_root / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
