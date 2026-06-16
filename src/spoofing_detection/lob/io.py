from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from .config import LOBConfig
from .models import OutputPaths
from .panel import reconstruct_dataframe


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_events(path: str | Path) -> pl.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pl.read_parquet(path)
    if suffix == ".csv":
        return pl.read_csv(path, infer_schema_length=10000)
    raise ValueError(f"unsupported input suffix {suffix!r}; expected .parquet or .csv")


def reconstruct_file(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    config: LOBConfig | None = None,
    max_rows: int | None = None,
) -> OutputPaths:
    config = config or LOBConfig()
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = read_events(input_path)
    result = reconstruct_dataframe(df, config=config, max_rows=max_rows)

    panel_path = output_dir / "lob_event_state_panel.parquet"
    normalized_path = output_dir / "normalized_events.parquet"
    agent_panel_path = output_dir / "agent_event_state_panel.parquet"
    active_orders_path = output_dir / "active_order_snapshots.parquet"
    price_level_depth_path = output_dir / "price_level_depth_snapshots.parquet"
    metadata_path = output_dir / "metadata.json"
    validation_path = output_dir / "validation_report.md"

    result.panel.write_parquet(panel_path)
    result.normalized_events.write_parquet(normalized_path)
    result.agent_event_state_panel.write_parquet(agent_panel_path)
    result.active_order_snapshots.write_parquet(active_orders_path)
    result.price_level_depth_snapshots.write_parquet(price_level_depth_path)

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_file": str(input_path),
        "source_file_size_bytes": input_path.stat().st_size,
        "source_file_sha256": _sha256(input_path),
        "max_rows": max_rows,
        "config": config.to_dict(),
        "row_counts": {
            "input": df.height,
            "panel": result.panel.height,
            "normalized_events": result.normalized_events.height,
            "agent_event_state_panel": result.agent_event_state_panel.height,
            "active_order_snapshots": result.active_order_snapshots.height,
            "price_level_depth_snapshots": result.price_level_depth_snapshots.height,
        },
        "validation": result.validation,
        "outputs": {
            "panel": str(panel_path),
            "normalized_events": str(normalized_path),
            "agent_event_state_panel": str(agent_panel_path),
            "active_order_snapshots": str(active_orders_path),
            "price_level_depth_snapshots": str(price_level_depth_path),
            "validation_report": str(validation_path),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))

    issue_lines = ["# LOB Reconstruction Validation Report", ""]
    issue_lines.append(f"- source_file: `{input_path}`")
    issue_lines.append(f"- input_rows: {df.height}")
    issue_lines.append(f"- panel_rows: {result.panel.height}")
    issue_lines.append(f"- normalized_rows: {result.normalized_events.height}")
    issue_lines.append(f"- agent_event_state_rows: {result.agent_event_state_panel.height}")
    issue_lines.append(f"- active_order_snapshot_rows: {result.active_order_snapshots.height}")
    issue_lines.append(f"- price_level_depth_snapshot_rows: {result.price_level_depth_snapshots.height}")
    issue_lines.append(f"- active_orders_end: {result.validation['active_orders_end']}")
    issue_lines.append(f"- partitions_processed: {result.validation['partitions_processed']}")
    issue_lines.append(f"- top_n: {config.top_n}")
    issue_lines.append(f"- snapshot_mode: {config.snapshot_mode}")
    issue_lines.append(f"- agent_dimensions: {', '.join(config.agent_dimensions)}")
    issue_lines.append("")
    issue_lines.append("## Issue counts")
    if result.validation["issue_counts"]:
        for key, value in sorted(result.validation["issue_counts"].items()):
            issue_lines.append(f"- {key}: {value}")
    else:
        issue_lines.append("- none")
    validation_path.write_text("\n".join(issue_lines) + "\n")

    return OutputPaths(
        panel_path=panel_path,
        normalized_path=normalized_path,
        agent_panel_path=agent_panel_path,
        active_orders_path=active_orders_path,
        price_level_depth_path=price_level_depth_path,
        metadata_path=metadata_path,
        validation_path=validation_path,
    )
