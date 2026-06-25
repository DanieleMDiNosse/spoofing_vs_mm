from __future__ import annotations

import importlib.util
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
