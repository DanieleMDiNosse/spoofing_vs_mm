from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import polars as pl


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "compute_spoofing_metrics.py"


def load_module():
    spec = importlib.util.spec_from_file_location("compute_spoofing_metrics", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_infer_state_client_ids_from_passive_limit_fills_only():
    module = load_module()
    raw = pl.DataFrame(
        {
            "ORDEREVENTTYPE (*)": [3, 3, 3, 1],
            "PASSIVEORDER": ["Y", "Y", None, None],
            "AGGRESSIVEORDER": ["N", "Y", "N", None],
            "ORDERTYPE (*)": [2, 2, 2, 2],
            "NMSC_ORIGINALCLIENTIDSHORTCODE": ["C1", "C2", "C3", "C4"],
        }
    )

    assert module._infer_state_client_ids(raw, mode="passive-fill-clients") == {"C1"}
    assert module._infer_state_client_ids(raw, mode="all") is None


def test_parse_args_supports_compact_memory_options(tmp_path: Path):
    module = load_module()

    args = module.parse_args(
        [
            "--input",
            str(tmp_path / "input.parquet"),
            "--quote-panel",
            str(tmp_path / "quotes.parquet"),
            "--output-dir",
            str(tmp_path / "out"),
            "--state-client-mode",
            "passive-fill-clients",
            "--compact-state",
        ]
    )

    assert args.state_client_mode == "passive-fill-clients"
    assert args.compact_state is True


def test_parse_args_loads_spoofing_metric_parameters_from_config_with_cli_overrides(tmp_path: Path):
    module = load_module()
    config_path = tmp_path / "spoofing_parameters.json"
    config_path.write_text(
        json.dumps(
            {
                "metrics": {
                    "top_n": 5,
                    "kappa": 2.0,
                    "lambda": 0.5,
                    "window_seconds": 30.0,
                    "max_deceptive_order_age_seconds": 120.0,
                    "gamma_grid": [0.001, 0.01],
                    "state_client_mode": "passive-fill-clients",
                    "compact_state": True,
                    "empirical_depth_kernel": str(tmp_path / "kernel.parquet"),
                }
            }
        )
    )

    args = module.parse_args(
        [
            "--config",
            str(config_path),
            "--input",
            str(tmp_path / "input.parquet"),
            "--quote-panel",
            str(tmp_path / "quotes.parquet"),
            "--output-dir",
            str(tmp_path / "out"),
            "--top-n",
            "7",
        ]
    )

    assert args.config == config_path
    assert args.top_n == 7
    assert args.kappa == 2.0
    assert args.lambda_ == 0.5
    assert args.window_seconds == 30.0
    assert args.max_deceptive_order_age_seconds == 120.0
    assert args.gamma_grid == "0.001,0.01"
    assert args.state_client_mode == "passive-fill-clients"
    assert args.compact_state is True
    assert args.empirical_depth_kernel == tmp_path / "kernel.parquet"

    cli_kernel = tmp_path / "cli_kernel.parquet"
    cli_args = module.parse_args(
        [
            "--input",
            str(tmp_path / "input.parquet"),
            "--quote-panel",
            str(tmp_path / "quotes.parquet"),
            "--output-dir",
            str(tmp_path / "out"),
            "--empirical-depth-kernel",
            str(cli_kernel),
        ]
    )
    assert cli_args.empirical_depth_kernel == cli_kernel

    override_args = module.parse_args(
        [
            "--config",
            str(config_path),
            "--input",
            str(tmp_path / "input.parquet"),
            "--quote-panel",
            str(tmp_path / "quotes.parquet"),
            "--output-dir",
            str(tmp_path / "out"),
            "--no-compact-state",
        ]
    )
    assert override_args.compact_state is False
