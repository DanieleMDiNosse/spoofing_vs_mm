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

from spoofing_detection.lob.client_session_features import (
    compute_actor_session_features,
    filter_attributable_actor_rows,
)
from spoofing_detection.lob.spoofing_config import (
    DEFAULT_SPOOFING_CONFIG_PATH,
    load_spoofing_config_defaults,
    parse_execution_anchor_modes,
    parse_msci_threshold_by_anchor,
    validate_actor_identity_mode,
)


_CONFIGURABLE_DEFAULT_KEYS = {
    "msci_threshold",
    "msci_threshold_by_anchor",
    "actor_identity_mode",
    "execution_anchor_modes",
}


def _option_is_explicit(argv: list[str] | None, option: str) -> bool:
    tokens = sys.argv[1:] if argv is None else argv
    return any(token == option or token.startswith(f"{option}=") for token in tokens)


def compute_and_write(
    *,
    input_path: Path,
    output_dir: Path,
    msci_threshold: float | Mapping[str, float | None],
    actor_identity_mode: str = "client_then_firm",
    execution_anchor_modes: tuple[str, ...] = ("passive",),
) -> dict[str, Path]:
    actor_identity_mode = validate_actor_identity_mode(actor_identity_mode)
    execution_anchor_modes = parse_execution_anchor_modes(execution_anchor_modes)
    executions = pl.read_parquet(input_path)
    attributable_executions = filter_attributable_actor_rows(executions)
    excluded_unattributable_rows = executions.height - attributable_executions.height
    observed_anchor_values = (
        attributable_executions["execution_anchor_mode"].drop_nulls().unique().to_list()
        if "execution_anchor_mode" in attributable_executions.columns
        else []
    )
    observed_anchor_modes = (
        list(parse_execution_anchor_modes(observed_anchor_values)) if observed_anchor_values else []
    )
    unconfigured_anchor_modes = sorted(set(observed_anchor_modes) - set(execution_anchor_modes))
    if unconfigured_anchor_modes:
        raise ValueError(f"input contains unconfigured execution anchor mode(s): {', '.join(unconfigured_anchor_modes)}")
    features = compute_actor_session_features(attributable_executions, msci_threshold=msci_threshold)
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / "actor_session_features.parquet"
    csv_path = output_dir / "actor_session_features.csv"
    metadata_path = output_dir / "metadata.json"
    features.write_parquet(parquet_path)
    features.write_csv(csv_path)
    metadata_path.write_text(
        json.dumps(
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "output_schema_version": "actor_execution_anchor_v2",
                "actor_identity_mode": actor_identity_mode,
                "execution_anchor_modes": list(execution_anchor_modes),
                "observed_execution_anchor_modes": observed_anchor_modes,
                "identity": "actor_key with client_original preferred and firm fallback",
                "firm_fallback_semantics": "aggregate only when client_original_id is missing",
                "score_grouping": ["actor_key", "execution_anchor_mode"],
                "actor_feature_population": "attributable_execution_rows_only",
                "excluded_unattributable_execution_rows": excluded_unattributable_rows,
                "input_path": str(input_path),
                "msci_threshold": (
                    None if isinstance(msci_threshold, Mapping) else float(msci_threshold)
                ),
                "msci_threshold_by_anchor": (
                    dict(msci_threshold)
                    if isinstance(msci_threshold, Mapping)
                    else {anchor: float(msci_threshold) for anchor in execution_anchor_modes}
                ),
                "rows": features.height,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return {"parquet": parquet_path, "csv": csv_path, "metadata": metadata_path}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, default=DEFAULT_SPOOFING_CONFIG_PATH)
    config_args, _ = config_parser.parse_known_args(argv)
    config_defaults = load_spoofing_config_defaults(
        config_path=config_args.config,
        section="session_features",
        allowed_keys=_CONFIGURABLE_DEFAULT_KEYS,
    )

    parser = argparse.ArgumentParser(description="Compute actor-session spoofing surveillance features.")
    parser.add_argument(
        "--config",
        type=Path,
        default=config_args.config,
        help="JSON config file containing spoofing parameter defaults",
    )
    parser.add_argument("--execution-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--msci-threshold", type=float, default=0.1)
    parser.add_argument(
        "--msci-threshold-by-anchor",
        type=parse_msci_threshold_by_anchor,
        default=None,
        help='JSON object, e.g. {"passive": 0.1, "aggressive": null}',
    )
    parser.add_argument("--actor-identity-mode", choices=("client_then_firm",), default="client_then_firm")
    parser.add_argument("--execution-anchor-modes", default="passive")
    parser.set_defaults(**config_defaults)
    args = parser.parse_args(argv)
    try:
        args.actor_identity_mode = validate_actor_identity_mode(args.actor_identity_mode)
        args.execution_anchor_modes = parse_execution_anchor_modes(args.execution_anchor_modes)
        if args.msci_threshold_by_anchor is not None:
            args.msci_threshold_by_anchor = parse_msci_threshold_by_anchor(
                args.msci_threshold_by_anchor
            )
        if (
            _option_is_explicit(argv, "--msci-threshold")
            and not _option_is_explicit(argv, "--msci-threshold-by-anchor")
            and args.msci_threshold_by_anchor is not None
        ):
            args.msci_threshold_by_anchor = {
                **args.msci_threshold_by_anchor,
                "passive": args.msci_threshold,
            }
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    outputs = compute_and_write(
        input_path=args.execution_metrics,
        output_dir=args.output_dir,
        msci_threshold=(
            args.msci_threshold_by_anchor
            if args.msci_threshold_by_anchor is not None
            else args.msci_threshold
        ),
        actor_identity_mode=args.actor_identity_mode,
        execution_anchor_modes=args.execution_anchor_modes,
    )
    print(outputs["parquet"])


if __name__ == "__main__":
    main()
