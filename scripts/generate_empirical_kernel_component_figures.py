#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import polars as pl

REQUIRED_COLUMNS = {
    "instrument_id",
    "side",
    "rank",
    "hit_probability",
    "visibility_covariance",
    "protection_component",
    "visibility_component",
}
SIDES = ("ask", "bid")
SIDE_STYLES = {
    "ask": "blue!70!black, thick, mark=*, mark size=1.5pt",
    "bid": "orange!85!black, thick, dashed, mark=square*, mark size=1.4pt",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _horizon_token(horizon_seconds: float) -> str:
    return str(int(horizon_seconds)) if horizon_seconds.is_integer() else f"{horizon_seconds:g}"


def _validate_metadata(
    metadata: dict[str, Any],
    *,
    instrument: str,
    top_n: int,
    horizon_seconds: float,
    path: Path,
) -> None:
    if metadata.get("instrument_id") != instrument:
        raise ValueError(f"{path} has the wrong instrument_id")
    if int(metadata.get("top_n", -1)) != top_n:
        raise ValueError(f"{path} has the wrong top_n")
    if not math.isclose(
        float(metadata.get("horizon_seconds", math.nan)),
        horizon_seconds,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(f"{path} has the wrong horizon_seconds")
    if metadata.get("is_full_sample_calibration") is not True:
        raise ValueError(f"{path} is not a full-sample calibration")


def _validate_profile(
    profile: pl.DataFrame,
    *,
    metadata: dict[str, Any],
    instrument: str,
    top_n: int,
    path: Path,
) -> pl.DataFrame:
    missing = sorted(REQUIRED_COLUMNS - set(profile.columns))
    if missing:
        raise ValueError(f"{path} lacks required columns: {missing}")

    selected = profile.select(sorted(REQUIRED_COLUMNS)).with_columns(
        pl.col("instrument_id").cast(pl.Utf8),
        pl.col("side").cast(pl.Utf8).str.to_lowercase(),
        pl.col("rank").cast(pl.Int64),
        pl.col(
            "hit_probability",
            "visibility_covariance",
            "protection_component",
            "visibility_component",
        ).cast(pl.Float64),
    )
    if selected.height != 2 * top_n:
        raise ValueError(f"{path} must contain exactly {2 * top_n} side-rank rows")
    if selected.get_column("instrument_id").unique().to_list() != [instrument]:
        raise ValueError(f"{path} contains an unexpected instrument_id")
    if selected.select("side", "rank").unique().height != selected.height:
        raise ValueError(f"{path} contains duplicate side-rank rows")

    expected_ranks = list(range(1, top_n + 1))
    for side in SIDES:
        side_ranks = (
            selected.filter(pl.col("side") == side).sort("rank").get_column("rank").to_list()
        )
        if side_ranks != expected_ranks:
            raise ValueError(f"{path} does not contain ranks 1 through {top_n} for {side}")

    numeric_columns = [
        "hit_probability",
        "visibility_covariance",
        "protection_component",
        "visibility_component",
    ]
    if any(selected.get_column(column).null_count() for column in numeric_columns):
        raise ValueError(f"{path} contains null component values")
    if any(not selected.get_column(column).is_finite().all() for column in numeric_columns):
        raise ValueError(f"{path} contains non-finite component values")
    if selected.filter(~pl.col("hit_probability").is_between(0.0, 1.0)).height:
        raise ValueError(f"{path} has hit_probability outside [0, 1]")

    protection_floor = float(metadata.get("protection_floor", 0.0))
    visibility_floor = float(metadata.get("visibility_floor", 0.0))
    for row in selected.iter_rows(named=True):
        expected_protection = max(1.0 - row["hit_probability"], protection_floor)
        if not math.isclose(
            row["protection_component"],
            expected_protection,
            rel_tol=1e-10,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{path} has an inconsistent protection_component")
        expected_visibility = max(abs(row["visibility_covariance"]), visibility_floor)
        if not math.isclose(
            row["visibility_component"],
            expected_visibility,
            rel_tol=1e-10,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{path} has an inconsistent visibility_component")
    return selected


def load_calibration_run(calibration_root: Path) -> tuple[pl.DataFrame, dict[str, Any]]:
    manifest_path = calibration_root / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    instruments = [str(value) for value in manifest["instruments"]]
    top_n = int(manifest["top_n"])
    horizon_seconds = float(manifest["horizon_seconds"])
    if not instruments or top_n <= 0 or horizon_seconds <= 0:
        raise ValueError(f"invalid calibration manifest: {manifest_path}")

    horizon_token = _horizon_token(horizon_seconds)
    profiles: list[pl.DataFrame] = []
    source_hashes: dict[str, str] = {}
    source_paths: dict[str, str] = {}
    for instrument_index, instrument in enumerate(instruments):
        output_dir = calibration_root / f"{instrument}_top{top_n}_h{horizon_token}"
        metadata_path = output_dir / "metadata.json"
        kernel_path = output_dir / "empirical_depth_kernel.csv"
        if not metadata_path.exists():
            raise FileNotFoundError(metadata_path)
        if not kernel_path.exists():
            raise FileNotFoundError(kernel_path)
        metadata = json.loads(metadata_path.read_text())
        _validate_metadata(
            metadata,
            instrument=instrument,
            top_n=top_n,
            horizon_seconds=horizon_seconds,
            path=metadata_path,
        )
        profile = _validate_profile(
            pl.read_csv(kernel_path),
            metadata=metadata,
            instrument=instrument,
            top_n=top_n,
            path=kernel_path,
        ).with_columns(pl.lit(instrument_index).alias("instrument_order"))
        profiles.append(profile)
        source_hashes[instrument] = _sha256(kernel_path)
        source_paths[instrument] = str(kernel_path)

    combined = pl.concat(profiles).sort(["instrument_order", "side", "rank"])
    provenance: dict[str, Any] = {
        "calibration_root": str(calibration_root),
        "created_at_utc": manifest.get("created_at_utc"),
        "horizon_seconds": horizon_seconds,
        "instruments": instruments,
        "source_hashes": source_hashes,
        "source_paths": source_paths,
        "top_n": top_n,
    }
    return combined, provenance


def _nice_axis_upper(maximum: float, *, target_ticks: int = 5) -> tuple[float, float]:
    if not math.isfinite(maximum) or maximum <= 0:
        return 1.0, 0.2
    rough_step = maximum / target_ticks
    magnitude = 10.0 ** math.floor(math.log10(rough_step))
    normalized = rough_step / magnitude
    nice_fraction = next(value for value in (1.0, 2.0, 5.0, 10.0) if normalized <= value)
    step = nice_fraction * magnitude
    upper = math.ceil(maximum / step) * step
    return upper, step


def _format_tick(value: float) -> str:
    if math.isclose(value, round(value), abs_tol=1e-10):
        return str(int(round(value)))
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _coordinates(rows: pl.DataFrame, column: str, *, scale: float = 1.0) -> str:
    return " ".join(
        f"({int(rank)},{float(value) / scale!r})"
        for rank, value in rows.sort("rank").select("rank", column).iter_rows()
    )


def _raw_series_comment(
    rows: pl.DataFrame,
    *,
    instrument: str,
    side: str,
    column: str,
) -> str:
    values = ", ".join(
        f"{int(rank)}:{float(value):.17g}"
        for rank, value in rows.sort("rank").select("rank", column).iter_rows()
    )
    return f"% {instrument} {side} {column} raw: {values}"


def _axis_lines(
    *,
    top_n: int,
    y_upper: float,
    y_step: float,
    y_unit_cm: float,
    multiplier_exponent: int | None,
    show_y_title: bool,
    y_title: str,
) -> list[str]:
    lines = [
        f"\\begin{{scope}}[x=0.43cm,y={y_unit_cm:.8f}cm]",
        f"  \\draw[->] (0.7,0) -- ({top_n + 0.45},0);",
        f"  \\draw[->] (0.7,0) -- (0.7,{y_upper * 1.08:.8g});",
    ]
    tick_count = int(round(y_upper / y_step))
    for tick_index in range(tick_count + 1):
        value = tick_index * y_step
        lines.extend(
            [
                f"  \\draw[gray!15] (0.7,{value:.8g}) -- ({top_n + 0.25},{value:.8g});",
                f"  \\draw (0.63,{value:.8g}) -- (0.7,{value:.8g}) "
                f"node[left,font=\\scriptsize] {{{_format_tick(value)}}};",
            ]
        )
    for rank in range(1, top_n + 1):
        lines.append(
            f"  \\draw ({rank},0) -- ({rank},-0.025 * {y_upper:.8g}) "
            f"node[below,font=\\scriptsize] {{{rank}}};"
        )
    if multiplier_exponent is not None:
        lines.append(
            f"  \\node[anchor=south west,font=\\scriptsize] at (0.75,{y_upper:.8g}) "
            f"{{$\\times 10^{{{multiplier_exponent}}}$}};"
        )
    if show_y_title:
        lines.append(
            f"  \\node[rotate=90,font=\\footnotesize] at "
            f"(-1.6,{0.5 * y_upper:.8g}) {{{y_title}}};"
        )
    return lines


def _render_visibility_figure(profile: pl.DataFrame, provenance: dict[str, Any]) -> list[str]:
    instruments = provenance["instruments"]
    top_n = int(provenance["top_n"])
    horizon = _format_tick(float(provenance["horizon_seconds"]))
    lines = [
        "% data column: visibility_component",
        "\\begin{figure}[!ht]",
        "\\centering",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tikzpicture}",
    ]
    for panel_index, instrument in enumerate(instruments):
        panel = profile.filter(pl.col("instrument_id") == instrument)
        maximum = float(panel.get_column("visibility_component").max())
        exponent = math.floor(math.log10(maximum)) if maximum > 0.0 else 0
        display_scale = 10.0**exponent
        display_maximum = maximum / display_scale
        y_upper, y_step = _nice_axis_upper(display_maximum * 1.03)
        y_unit_cm = 3.0 / y_upper
        lines.append(f"\\begin{{scope}}[xshift={panel_index * 5.45:.2f}cm]")
        lines.extend(
            _axis_lines(
                top_n=top_n,
                y_upper=y_upper,
                y_step=y_step,
                y_unit_cm=y_unit_cm,
                multiplier_exponent=exponent,
                show_y_title=panel_index == 0,
                y_title="visibility proxy $\\nu_k^{i,s}$",
            )
        )
        lines.append(
            f"  \\node[font=\\footnotesize\\bfseries] at "
            f"({0.5 * (top_n + 1):.4g},{y_upper * 1.18:.8g}) "
            f"{{{instrument.title()}}};"
        )
        for side in SIDES:
            side_rows = panel.filter(pl.col("side") == side)
            lines.append(
                f"  \\draw[{SIDE_STYLES[side]}] plot coordinates "
                f"{{{_coordinates(side_rows, 'visibility_component', scale=display_scale)}}};"
            )
            lines.append(
                "  "
                + _raw_series_comment(
                    side_rows,
                    instrument=instrument,
                    side=side,
                    column="visibility_component",
                )
            )
        lines.extend(["\\end{scope}", "\\end{scope}"])
    lines.extend(
        [
            f"\\node[font=\\footnotesize] at ({0.5 * ((len(instruments) - 1) * 5.45 + top_n * 0.43):.4g},-0.65) "
            "{book rank $k$};",
            "\\draw[blue!70!black,thick,mark=*,mark size=1.5pt] "
            "(6.0,3.85) -- (6.7,3.85) node[right,font=\\footnotesize] {Ask};",
            "\\draw[orange!85!black,thick,dashed,mark=square*,mark size=1.4pt] "
            "(8.1,3.85) -- (8.8,3.85) node[right,font=\\footnotesize] {Bid};",
            "\\end{tikzpicture}%",
            "}",
            "\\caption{Side-specific empirical visibility proxy by book rank. Each panel shows the stored "
            "\\texttt{visibility\\_component} values from the full-sample top-$"
            f"{top_n}$, $h={horizon}$-second calibration. Separate axis multipliers preserve the "
            "instrument-specific price units; vertical magnitudes should not be compared across instruments.}",
            "\\label{fig:empirical_topn_visibility_profiles}",
            "\\end{figure}",
            "",
        ]
    )
    return lines


def _render_execution_risk_figure(profile: pl.DataFrame, provenance: dict[str, Any]) -> list[str]:
    instruments = provenance["instruments"]
    top_n = int(provenance["top_n"])
    horizon = _format_tick(float(provenance["horizon_seconds"]))
    maximum = float(profile.get_column("hit_probability").max())
    y_upper, y_step = _nice_axis_upper(maximum * 1.05, target_ticks=7)
    y_unit_cm = 3.0 / y_upper
    lines = [
        "% data column: hit_probability; protection_component = 1 - hit_probability",
        "\\begin{figure}[!ht]",
        "\\centering",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tikzpicture}",
    ]
    for panel_index, instrument in enumerate(instruments):
        panel = profile.filter(pl.col("instrument_id") == instrument)
        lines.append(f"\\begin{{scope}}[xshift={panel_index * 5.45:.2f}cm]")
        lines.extend(
            _axis_lines(
                top_n=top_n,
                y_upper=y_upper,
                y_step=y_step,
                y_unit_cm=y_unit_cm,
                multiplier_exponent=None,
                show_y_title=panel_index == 0,
                y_title="execution-risk proxy $p_k^{i,s}$",
            )
        )
        lines.append(
            f"  \\node[font=\\footnotesize\\bfseries] at "
            f"({0.5 * (top_n + 1):.4g},{y_upper * 1.18:.8g}) "
            f"{{{instrument.title()}}};"
        )
        for side in SIDES:
            side_rows = panel.filter(pl.col("side") == side)
            lines.append(
                f"  \\draw[{SIDE_STYLES[side]}] plot coordinates "
                f"{{{_coordinates(side_rows, 'hit_probability')}}};"
            )
            lines.append(
                "  "
                + _raw_series_comment(
                    side_rows,
                    instrument=instrument,
                    side=side,
                    column="hit_probability",
                )
            )
        lines.extend(["\\end{scope}", "\\end{scope}"])
    lines.extend(
        [
            f"\\node[font=\\footnotesize] at ({0.5 * ((len(instruments) - 1) * 5.45 + top_n * 0.43):.4g},-0.65) "
            "{book rank $k$};",
            "\\draw[blue!70!black,thick,mark=*,mark size=1.5pt] "
            "(6.0,3.85) -- (6.7,3.85) node[right,font=\\footnotesize] {Ask};",
            "\\draw[orange!85!black,thick,dashed,mark=square*,mark size=1.4pt] "
            "(8.1,3.85) -- (8.8,3.85) node[right,font=\\footnotesize] {Bid};",
            "\\end{tikzpicture}%",
            "}",
            "\\caption{Side-specific empirical execution-risk proxy by book rank. The plotted quantity is the "
            "stored $h$-second reach probability $p_k^{i,s}$ (\\texttt{hit\\_probability}) from the "
            f"full-sample top-${top_n}$, $h={horizon}$-second calibration. The empirical kernel uses its "
            "protection complement $\\rho_k^{i,s}=1-p_k^{i,s}$.}",
            "\\label{fig:empirical_topn_execution_risk_profiles}",
            "\\end{figure}",
            "",
        ]
    )
    return lines


def render_component_figures(profile: pl.DataFrame, *, provenance: dict[str, Any]) -> str:
    lines = [
        "% Generated by scripts/generate_empirical_kernel_component_figures.py; do not edit manually.",
        f"% calibration root: {provenance['calibration_root']}",
    ]
    for instrument in provenance["instruments"]:
        lines.append(
            f"% source {instrument}: {provenance['source_paths'][instrument]} "
            f"sha256={provenance['source_hashes'][instrument]}"
        )
    lines.append("")
    lines.extend(_render_visibility_figure(profile, provenance))
    lines.extend(_render_execution_risk_figure(profile, provenance))
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate TikZ profiles for the empirical depth-kernel components."
    )
    parser.add_argument("--calibration-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    profile, provenance = load_calibration_run(args.calibration_root)
    output = render_component_figures(profile, provenance=provenance)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output)


if __name__ == "__main__":
    main()
