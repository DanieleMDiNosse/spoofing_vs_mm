from __future__ import annotations

import importlib.util
from pathlib import Path

import polars as pl

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "compute_client_session_spoofing_features.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("compute_client_session_spoofing_features", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_compute_and_write_features(tmp_path):
    module = _load_module()
    executions = pl.DataFrame({"client_id": ["A"], "MSCI": [0.8], "SCI": [0.7], "collapse_opposite_side": [0.6], "collapse_same_side": [0.1], "matched_deceptive_cancel_fraction_window": [0.9], "fill_qty": [100.0]})
    input_path = tmp_path / "execution_metrics.parquet"
    output_dir = tmp_path / "features"
    executions.write_parquet(input_path)
    output = module.compute_and_write(input_path=input_path, output_dir=output_dir, msci_threshold=0.5)
    assert output["parquet"].exists()
    assert output["csv"].exists()
    assert pl.read_parquet(output["parquet"]).height == 1
