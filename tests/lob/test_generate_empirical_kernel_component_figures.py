from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import polars as pl
import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "generate_empirical_kernel_component_figures.py"


def load_module():
    spec = importlib.util.spec_from_file_location("generate_empirical_kernel_component_figures", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_calibration_run(root: Path, *, top_n: int = 3) -> None:
    instruments = ["RISANAMENTO", "NEXI", "FERRARI"]
    (root / "run_manifest.json").write_text(
        json.dumps(
            {
                "created_at_utc": "2026-07-17T14:56:24Z",
                "horizon_seconds": 10,
                "top_n": top_n,
                "instruments": instruments,
            }
        )
    )
    scales = {"RISANAMENTO": 1e-7, "NEXI": 1e-6, "FERRARI": 1e-4}
    for instrument in instruments:
        output_dir = root / f"{instrument}_top{top_n}_h10"
        output_dir.mkdir(parents=True)
        rows = []
        for side, hit_values, visibility_multipliers in (
            ("ask", [0.30, 0.20, 0.10], [1.0, 3.0, 2.0]),
            ("bid", [0.25, 0.15, 0.05], [2.0, 1.0, 2.5]),
        ):
            for rank, (hit_probability, visibility_multiplier) in enumerate(
                zip(hit_values, visibility_multipliers, strict=True), start=1
            ):
                visibility = scales[instrument] * visibility_multiplier
                rows.append(
                    {
                        "instrument_id": instrument,
                        "side": side,
                        "rank": rank,
                        "hit_probability": hit_probability,
                        "visibility_covariance": visibility,
                        "protection_component": 1.0 - hit_probability,
                        "visibility_component": visibility,
                    }
                )
        pl.DataFrame(rows).write_csv(output_dir / "empirical_depth_kernel.csv")
        (output_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "instrument_id": instrument,
                    "top_n": top_n,
                    "horizon_seconds": 10.0,
                    "is_full_sample_calibration": True,
                    "protection_floor": 0.0,
                    "visibility_floor": 1e-12,
                }
            )
        )


def test_load_calibration_run_preserves_computed_component_values(tmp_path: Path):
    module = load_module()
    root = tmp_path / "calibration"
    root.mkdir()
    write_calibration_run(root)

    profile, provenance = module.load_calibration_run(root)

    assert profile.height == 18
    assert profile.select("instrument_id", "side", "rank").unique().height == 18
    assert provenance["instruments"] == ["RISANAMENTO", "NEXI", "FERRARI"]
    assert provenance["top_n"] == 3
    assert provenance["horizon_seconds"] == 10.0
    assert set(provenance["source_hashes"]) == {"RISANAMENTO", "NEXI", "FERRARI"}
    risanamento_ask = profile.filter(
        (pl.col("instrument_id") == "RISANAMENTO") & (pl.col("side") == "ask")
    ).sort("rank")
    assert risanamento_ask["hit_probability"].to_list() == pytest.approx([0.30, 0.20, 0.10])
    assert risanamento_ask["visibility_component"].to_list() == pytest.approx([1e-7, 3e-7, 2e-7])


def test_load_calibration_run_rejects_inconsistent_protection_component(tmp_path: Path):
    module = load_module()
    root = tmp_path / "calibration"
    root.mkdir()
    write_calibration_run(root)
    path = root / "RISANAMENTO_top3_h10" / "empirical_depth_kernel.csv"
    profile = pl.read_csv(path).with_columns(
        pl.when((pl.col("side") == "ask") & (pl.col("rank") == 1))
        .then(pl.lit(0.5))
        .otherwise(pl.col("protection_component"))
        .alias("protection_component")
    )
    profile.write_csv(path)

    with pytest.raises(ValueError, match="protection_component"):
        module.load_calibration_run(root)


def test_load_calibration_run_rejects_inconsistent_visibility_component(tmp_path: Path):
    module = load_module()
    root = tmp_path / "calibration"
    root.mkdir()
    write_calibration_run(root)
    path = root / "NEXI_top3_h10" / "empirical_depth_kernel.csv"
    profile = pl.read_csv(path).with_columns(
        pl.when((pl.col("side") == "bid") & (pl.col("rank") == 2))
        .then(pl.lit(9e-6))
        .otherwise(pl.col("visibility_component"))
        .alias("visibility_component")
    )
    profile.write_csv(path)

    with pytest.raises(ValueError, match="visibility_component"):
        module.load_calibration_run(root)


def test_load_calibration_run_rejects_null_component_values(tmp_path: Path):
    module = load_module()
    root = tmp_path / "calibration"
    root.mkdir()
    write_calibration_run(root)
    path = root / "FERRARI_top3_h10" / "empirical_depth_kernel.csv"
    profile = pl.read_csv(path).with_columns(
        pl.when((pl.col("side") == "ask") & (pl.col("rank") == 1))
        .then(pl.lit(None))
        .otherwise(pl.col("hit_probability"))
        .alias("hit_probability")
    )
    profile.write_csv(path)

    with pytest.raises(ValueError, match="null component values"):
        module.load_calibration_run(root)


def test_render_component_figures_handles_zero_visibility(tmp_path: Path):
    module = load_module()
    root = tmp_path / "calibration"
    root.mkdir()
    write_calibration_run(root)
    for path in root.glob("*_top3_h10/empirical_depth_kernel.csv"):
        profile = pl.read_csv(path).with_columns(
            pl.lit(0.0).alias("visibility_covariance"),
            pl.lit(0.0).alias("visibility_component"),
        )
        profile.write_csv(path)
        metadata_path = path.with_name("metadata.json")
        metadata = json.loads(metadata_path.read_text())
        metadata["visibility_floor"] = 0.0
        metadata_path.write_text(json.dumps(metadata))

    profile, provenance = module.load_calibration_run(root)

    tex = module.render_component_figures(profile, provenance=provenance)

    assert "(1,0.0) (2,0.0) (3,0.0)" in tex


def test_coordinates_preserve_float_precision():
    module = load_module()
    value = 0.00017228173668312345
    rows = pl.DataFrame({"rank": [1], "hit_probability": [value]})

    coordinates = module._coordinates(rows, "hit_probability")
    rendered_value = float(coordinates.removeprefix("(1,").removesuffix(")"))

    assert rendered_value == value


def test_render_component_figures_uses_rank_ticks_and_side_specific_values(tmp_path: Path):
    module = load_module()
    root = tmp_path / "calibration"
    root.mkdir()
    write_calibration_run(root)
    profile, provenance = module.load_calibration_run(root)

    tex = module.render_component_figures(profile, provenance=provenance)

    assert "\\label{fig:empirical_topn_visibility_profiles}" in tex
    assert "\\label{fig:empirical_topn_execution_risk_profiles}" in tex
    assert "book rank $k$" in tex
    assert "Ask" in tex and "Bid" in tex
    assert "(1,1.0) (2,3.0) (3,2.0)" in tex
    assert "(1,0.3) (2,0.2) (3,0.1)" in tex
    assert "\\times 10^{-7}" in tex
    assert "visibility_component" in tex
    assert "hit_probability" in tex
    assert tex.count("\\begin{figure}[!ht]") == 2
    assert "at (-1.6," in tex
    assert "at (-0.65," not in tex
    for digest in provenance["source_hashes"].values():
        assert digest in tex


def test_main_writes_generated_tex(tmp_path: Path):
    module = load_module()
    root = tmp_path / "calibration"
    root.mkdir()
    write_calibration_run(root)
    output = tmp_path / "empirical_kernel_component_figures.tex"

    module.main(["--calibration-root", str(root), "--output", str(output)])

    assert output.exists()
    text = output.read_text()
    assert "Generated by scripts/generate_empirical_kernel_component_figures.py" in text
    assert "\\begin{figure}[!ht]" in text
