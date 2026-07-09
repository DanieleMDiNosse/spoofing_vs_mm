#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from spoofing_detection.lob.alert_objects import build_client_session_alerts
from spoofing_detection.lob.client_session_features import compute_client_session_features
from spoofing_detection.lob.legitimacy_features import compute_legitimacy_features
from spoofing_detection.lob.spoofing_config import DEFAULT_SPOOFING_CONFIG_PATH, load_spoofing_config_defaults


_CONFIGURABLE_DEFAULT_KEYS = {"msci_threshold", "min_events", "min_mcps"}


def run_pipeline(
    *,
    execution_metrics_path: Path,
    event_log_path: Path,
    output_dir: Path,
    msci_threshold: float,
    min_events: int,
    min_mcps: float,
) -> dict[str, Path]:
    executions = pl.read_parquet(execution_metrics_path)
    event_log = pl.read_parquet(event_log_path)
    risk = compute_client_session_features(executions, msci_threshold=msci_threshold)
    legitimacy = compute_legitimacy_features(event_log)
    alerts = build_client_session_alerts(risk, legitimacy, min_events=min_events, min_mcps=min_mcps)
    output_dir.mkdir(parents=True, exist_ok=True)
    risk_path = output_dir / "client_session_risk_features.parquet"
    legitimacy_path = output_dir / "client_legitimacy_features.parquet"
    alerts_path = output_dir / "client_session_alerts.parquet"
    metadata_path = output_dir / "metadata.json"
    risk.write_parquet(risk_path)
    legitimacy.write_parquet(legitimacy_path)
    alerts.write_parquet(alerts_path)
    metadata_path.write_text(json.dumps({"created_at_utc": datetime.now(timezone.utc).isoformat(), "execution_metrics_path": str(execution_metrics_path), "event_log_path": str(event_log_path), "msci_threshold": msci_threshold, "min_events": min_events, "min_mcps": min_mcps, "alert_count": alerts.height}, indent=2, sort_keys=True))
    return {"risk": risk_path, "legitimacy": legitimacy_path, "alerts": alerts_path, "metadata": metadata_path}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, default=DEFAULT_SPOOFING_CONFIG_PATH)
    config_args, _ = config_parser.parse_known_args(argv)
    config_defaults = load_spoofing_config_defaults(
        config_path=config_args.config,
        section="production_readiness",
        allowed_keys=_CONFIGURABLE_DEFAULT_KEYS,
    )

    parser = argparse.ArgumentParser(description="Run production-readiness spoofing surveillance layer.")
    parser.add_argument(
        "--config",
        type=Path,
        default=config_args.config,
        help="JSON config file containing spoofing parameter defaults",
    )
    parser.add_argument("--execution-metrics", type=Path, required=True)
    parser.add_argument("--event-log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--msci-threshold", type=float, default=0.5)
    parser.add_argument("--min-events", type=int, default=3, help="minimum repeated matched-withdrawal events required")
    parser.add_argument(
        "--min-mcps",
        type=float,
        default=0.0,
        help="optional minimum matched-event share floor; 0 disables the share floor",
    )
    parser.set_defaults(**config_defaults)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    outputs = run_pipeline(execution_metrics_path=args.execution_metrics, event_log_path=args.event_log, output_dir=args.output_dir, msci_threshold=args.msci_threshold, min_events=args.min_events, min_mcps=args.min_mcps)
    print(outputs["alerts"])


if __name__ == "__main__":
    main()
