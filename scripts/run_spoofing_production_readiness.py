#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from spoofing_detection.lob.alert_objects import build_actor_session_alerts
from spoofing_detection.lob.client_session_features import (
    compute_actor_session_features,
    filter_attributable_actor_rows,
)
from spoofing_detection.lob.legitimacy_features import compute_actor_legitimacy_features
from spoofing_detection.lob.spoofing_config import (
    DEFAULT_SPOOFING_CONFIG_PATH,
    assert_config_parameters_match,
    load_spoofing_config_defaults,
    parse_execution_anchor_modes,
    parse_msci_threshold_by_anchor,
    reject_parameter_overrides,
    spoofing_config_provenance,
    validate_actor_identity_mode,
)


_CONFIGURABLE_DEFAULT_KEYS = {
    "msci_threshold_by_anchor",
    "min_events",
    "min_mcps",
    "actor_identity_mode",
    "execution_anchor_modes",
}

_CONFIG_PARAMETER_OPTIONS = {
    "--msci-threshold",
    "--msci-threshold-by-anchor",
    "--min-events",
    "--min-mcps",
    "--actor-identity-mode",
    "--execution-anchor-modes",
}


def _market_observation(
    configured_anchor_modes: tuple[str, ...],
    observed_anchor_modes: list[str],
) -> str:
    configured = "_and_".join(configured_anchor_modes)
    configured_text = f"configured_{configured}_execution_branch"
    if len(configured_anchor_modes) != 1:
        configured_text += "es"
    if not observed_anchor_modes:
        return f"{configured_text}; no_execution_branches_observed"
    observed = "_and_".join(observed_anchor_modes)
    observed_text = f"observed_{observed}_execution_branch"
    if len(observed_anchor_modes) != 1:
        observed_text += "es"
    return f"{configured_text}; {observed_text}_only"


def run_pipeline(
    *,
    execution_metrics_path: Path,
    event_log_path: Path,
    output_dir: Path,
    msci_threshold: float | Mapping[str, float | None],
    min_events: int,
    min_mcps: float,
    actor_identity_mode: str = "client_then_firm",
    execution_anchor_modes: tuple[str, ...] = ("passive",),
    config_path: Path | None = None,
) -> dict[str, Path]:
    actor_identity_mode = validate_actor_identity_mode(actor_identity_mode)
    execution_anchor_modes = parse_execution_anchor_modes(execution_anchor_modes)
    effective_thresholds = (
        parse_msci_threshold_by_anchor(msci_threshold)
        if isinstance(msci_threshold, Mapping)
        else {anchor: float(msci_threshold) for anchor in execution_anchor_modes}
    )
    if config_path is not None:
        assert_config_parameters_match(
            config_path=config_path,
            section="production_readiness",
            allowed_keys=_CONFIGURABLE_DEFAULT_KEYS,
            effective_parameters={
                "msci_threshold_by_anchor": effective_thresholds,
                "min_events": min_events,
                "min_mcps": min_mcps,
                "actor_identity_mode": actor_identity_mode,
                "execution_anchor_modes": execution_anchor_modes,
            },
            normalizers={
                "msci_threshold_by_anchor": parse_msci_threshold_by_anchor,
                "actor_identity_mode": validate_actor_identity_mode,
                "execution_anchor_modes": parse_execution_anchor_modes,
            },
        )
    executions = pl.read_parquet(execution_metrics_path)
    observed_anchor_values = (
        executions["execution_anchor_mode"].drop_nulls().unique().to_list()
        if "execution_anchor_mode" in executions.columns
        else []
    )
    observed_anchor_modes = (
        list(parse_execution_anchor_modes(observed_anchor_values)) if observed_anchor_values else []
    )
    unconfigured_anchor_modes = sorted(set(observed_anchor_modes) - set(execution_anchor_modes))
    if unconfigured_anchor_modes:
        raise ValueError(f"input contains unconfigured execution anchor mode(s): {', '.join(unconfigured_anchor_modes)}")
    event_log = pl.read_parquet(event_log_path)
    attributable_executions = filter_attributable_actor_rows(executions)
    attributable_event_log = filter_attributable_actor_rows(event_log)
    excluded_unattributable_execution_rows = executions.height - attributable_executions.height
    excluded_unattributable_event_rows = event_log.height - attributable_event_log.height
    risk = compute_actor_session_features(attributable_executions, msci_threshold=msci_threshold)
    legitimacy = compute_actor_legitimacy_features(attributable_event_log)
    alerts = build_actor_session_alerts(risk, legitimacy, min_events=min_events, min_mcps=min_mcps)
    output_dir.mkdir(parents=True, exist_ok=True)
    risk_path = output_dir / "actor_session_risk_features.parquet"
    legitimacy_path = output_dir / "actor_legitimacy_features.parquet"
    alerts_path = output_dir / "actor_session_alerts.parquet"
    metadata_path = output_dir / "metadata.json"
    risk.write_parquet(risk_path)
    legitimacy.write_parquet(legitimacy_path)
    alerts.write_parquet(alerts_path)
    metadata_path.write_text(
        json.dumps(
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                **(
                    spoofing_config_provenance(config_path, section="production_readiness")
                    if config_path is not None
                    else {"parameter_source": "programmatic"}
                ),
                "output_schema_version": "actor_execution_anchor_v2",
                "actor_identity_mode": actor_identity_mode,
                "execution_anchor_modes": list(execution_anchor_modes),
                "observed_execution_anchor_modes": observed_anchor_modes,
                "identity": "actor_key with client_original preferred and firm fallback",
                "firm_fallback_semantics": "aggregate only when client_original_id is missing",
                "market_observation": _market_observation(execution_anchor_modes, observed_anchor_modes),
                "score_grouping": ["actor_key", "execution_anchor_mode"],
                "actor_feature_population": "attributable_execution_and_event_rows_only",
                "excluded_unattributable_execution_rows": excluded_unattributable_execution_rows,
                "excluded_unattributable_event_rows": excluded_unattributable_event_rows,
                "event_selection": "execution clusters stratified by actor identity and execution anchor",
                "execution_metrics_path": str(execution_metrics_path),
                "event_log_path": str(event_log_path),
                "msci_threshold": (
                    None if isinstance(msci_threshold, Mapping) else float(msci_threshold)
                ),
                "msci_threshold_by_anchor": (
                    dict(msci_threshold)
                    if isinstance(msci_threshold, Mapping)
                    else {anchor: float(msci_threshold) for anchor in execution_anchor_modes}
                ),
                "min_events": min_events,
                "min_mcps": min_mcps,
                "alert_count": alerts.height,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return {"risk": risk_path, "legitimacy": legitimacy_path, "alerts": alerts_path, "metadata": metadata_path}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    config_parser.add_argument("--config", type=Path, default=DEFAULT_SPOOFING_CONFIG_PATH)
    config_args, _ = config_parser.parse_known_args(argv)
    try:
        config_defaults = load_spoofing_config_defaults(
            config_path=config_args.config,
            section="production_readiness",
            allowed_keys=_CONFIGURABLE_DEFAULT_KEYS,
        )
    except (OSError, ValueError) as exc:
        config_parser.error(str(exc))

    parser = argparse.ArgumentParser(
        description="Run production-readiness spoofing surveillance layer.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=config_args.config,
        help="Authoritative JSON file containing all production-readiness parameters",
    )
    parser.add_argument("--execution-metrics", type=Path, required=True)
    parser.add_argument("--event-log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--msci-threshold", type=float)
    parser.add_argument(
        "--msci-threshold-by-anchor",
        type=parse_msci_threshold_by_anchor,
        help='JSON object, e.g. {"passive": 0.1, "aggressive": null}',
    )
    parser.add_argument("--min-events", type=int, help="minimum repeated matched-withdrawal events required")
    parser.add_argument(
        "--min-mcps",
        type=float,
        help="optional minimum matched-event share floor; 0 disables the share floor",
    )
    parser.add_argument("--actor-identity-mode", choices=("client_then_firm",))
    parser.add_argument("--execution-anchor-modes")
    parser.set_defaults(**config_defaults)
    reject_parameter_overrides(
        parser,
        argv,
        parameter_options=_CONFIG_PARAMETER_OPTIONS,
    )
    args = parser.parse_args(argv)
    try:
        args.actor_identity_mode = validate_actor_identity_mode(args.actor_identity_mode)
        args.execution_anchor_modes = parse_execution_anchor_modes(args.execution_anchor_modes)
        if args.msci_threshold_by_anchor is not None:
            args.msci_threshold_by_anchor = parse_msci_threshold_by_anchor(
                args.msci_threshold_by_anchor
            )
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    outputs = run_pipeline(
        execution_metrics_path=args.execution_metrics,
        event_log_path=args.event_log,
        output_dir=args.output_dir,
        msci_threshold=(
            args.msci_threshold_by_anchor
            if args.msci_threshold_by_anchor is not None
            else args.msci_threshold
        ),
        min_events=args.min_events,
        min_mcps=args.min_mcps,
        actor_identity_mode=args.actor_identity_mode,
        execution_anchor_modes=args.execution_anchor_modes,
        config_path=args.config,
    )
    print(outputs["alerts"])


if __name__ == "__main__":
    main()
