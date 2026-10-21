#!/usr/bin/env python3
"""Render a compact paper package from the verified one-second sequential output.

The renderer consumes only public aggregate derivatives of the verified output
CSV files.  It neither replays raw data nor redefines the sequential screen.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any, Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INSTRUMENTS = ("FERRARI", "NEXI", "RISANAMENTO")
SCHEMA = "one_second_sequential_paper_package_v1"
SOURCE_SCHEMA = "h1_age30_withdrawal1_wmsci_price_pipeline_v2_complete_finite_rev"
SOURCE_REQUIRED = {
    "sequential_counts.csv",
    "refreshed_execution_diagnostics.csv",
    "selected_execution_diagnostics.csv",
    "step_memberships.csv",
    "metadata.json",
}
TABLES = ("funnel", "population", "components", "tail_overlap", "price_associations", "selected_composition")
FIGURES = ("wmsci_ecdf", "score_price_diagnostics")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tex_escape(value: object) -> str:
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(replacements.get(char, char) for char in str(value))


def _display(value: object) -> str:
    if value is None or pd.isna(value):
        return "NA"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.3f}"
    return str(value)


def _table(frame: pd.DataFrame, caption: str, label: str, *, headers: Mapping[str, str] | None = None) -> str:
    if frame.empty:
        raise ValueError(f"empty publication table: {label}")
    headers = dict(headers or {})
    columns = list(frame.columns)
    align = r">{\raggedright\arraybackslash}X" + r">{\raggedleft\arraybackslash}X" * (len(columns) - 1)
    title = " & ".join(tex_escape(headers.get(column, column.replace("_", " ").title())) for column in columns)
    body = [" & ".join(tex_escape(_display(value)) for value in row) + r" \\" for row in frame.itertuples(index=False, name=None)]
    return "\n".join([
        r"\begin{table}[htbp]", r"\centering", f"\\caption{{{tex_escape(caption)}}}", f"\\label{{{label}}}",
        r"\small", r"\setlength{\tabcolsep}{4pt}", r"\renewcommand{\arraystretch}{1.12}",
        f"\\begin{{tabularx}}{{\\textwidth}}{{{align}}}", r"\toprule", title + r" \\", r"\midrule",
        *body, r"\bottomrule", r"\end{tabularx}", r"\end{table}", "",
    ])


def _require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(f"{label} missing columns: {sorted(missing)[0]}")


def _finite(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.notna() & np.isfinite(numeric)


def _exact_top(frame: pd.DataFrame, score: str = "WMSCI_event") -> pd.DataFrame:
    _require_columns(frame, {"instrument", "execution_cluster_id", score}, "diagnostics")
    rows = []
    for instrument in INSTRUMENTS:
        group = frame.loc[frame["instrument"] == instrument].copy()
        values = pd.to_numeric(group[score], errors="coerce")
        positive = group.loc[_finite(values) & (values > 0)].copy()
        positive["_score"] = pd.to_numeric(positive[score], errors="coerce")
        if positive["execution_cluster_id"].isna().any() or positive["execution_cluster_id"].duplicated().any():
            raise ValueError("top-score ties require unique execution_cluster_id")
        k = math.ceil(0.05 * len(positive))
        rows.append(positive.sort_values(["_score", "execution_cluster_id"], ascending=[False, True], kind="stable").head(k).drop(columns="_score"))
    return pd.concat(rows, ignore_index=True)


def _spearman(left: pd.Series, right: pd.Series) -> tuple[int, float | None]:
    paired = pd.DataFrame({"left": pd.to_numeric(left, errors="coerce"), "right": pd.to_numeric(right, errors="coerce")}).dropna()
    paired = paired.loc[np.isfinite(paired["left"]) & np.isfinite(paired["right"])]
    if len(paired) < 2 or paired["left"].nunique() < 2 or paired["right"].nunique() < 2:
        return len(paired), None
    return len(paired), float(paired["left"].corr(paired["right"], method="spearman"))


def validate_source(source: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    source = Path(source)
    if not source.is_dir() or not SOURCE_REQUIRED.issubset({path.name for path in source.iterdir() if path.is_file()}):
        raise ValueError("one-second source package is incomplete")
    metadata = json.loads((source / "metadata.json").read_text())
    if metadata.get("schema") != SOURCE_SCHEMA:
        raise ValueError("unexpected source schema")
    contract = metadata.get("contract", {})
    expected = {
        "profile_window_seconds": 1.0, "candidate_age_seconds": 30.0,
        "withdrawal_window_seconds": 1.0, "reversion_horizon_seconds": 2.0,
        "top_positive_w_fraction": 0.05,
        "selection_order": ["positive_W_exact_top_5pct", "U_over_Q_gt_1", "FPM_gt_0", "REV_gt_0"],
        "episode_role": "recomputed_shared_order_reporting_only_not_a_selection_gate",
        "reversion_coverage": "all_assigned_positive_quantity_cancellations_must_have_finite_REV",
    }
    if {key: contract.get(key) for key in expected} != expected:
        raise ValueError("source sequential contract mismatch")
    outputs = metadata.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("source output hashes are missing")
    for name in SOURCE_REQUIRED - {"metadata.json"}:
        if outputs.get(name) != sha256_file(source / name):
            raise ValueError(f"source output hash mismatch: {name}")
    counts = pd.read_csv(source / "sequential_counts.csv")
    diagnostics = pd.read_csv(source / "refreshed_execution_diagnostics.csv")
    selected = pd.read_csv(source / "selected_execution_diagnostics.csv")
    _require_columns(counts, {"instrument", "all_executions", "finite_W_executions", "positive_W_executions", "top_W_exact_k", "top_W_after_U_over_Q_gt_1", "top_W_after_U_over_Q_gt_1_FPM_gt_0", "top_W_after_FPM_gt_0_REV_observed", "final_WMSCI_to_FPM_to_REV_executions"}, "counts")
    _require_columns(diagnostics, {"instrument", "execution_cluster_id", "WMSCI_event", "withdrawal_to_execution_ratio", "matched_deceptive_cancel_fraction_window", "execution_quantity", "favorable_mid_move_pre_fill", "post_cancel_mid_reversion", "post_cancel_mid_reversion_complete", "identity_level", "execution_anchor_mode"}, "diagnostics")
    _require_columns(selected, {"instrument", "execution_cluster_id", "identity_level", "execution_anchor_mode"}, "selected diagnostics")
    if tuple(counts["instrument"]) != INSTRUMENTS or tuple(sorted(diagnostics["instrument"].unique())) != INSTRUMENTS:
        raise ValueError("instrument inventory mismatch")
    if diagnostics.duplicated(["instrument", "execution_cluster_id"]).any():
        raise ValueError("execution IDs are not unique within instrument")
    top = _exact_top(diagnostics)
    final_ids = set(selected[["instrument", "execution_cluster_id"]].itertuples(index=False, name=None))
    for row in counts.itertuples(index=False):
        group = diagnostics.loc[diagnostics.instrument == row.instrument]
        if len(group) != row.all_executions or int((_finite(group.WMSCI_event)).sum()) != row.finite_W_executions or int((_finite(group.WMSCI_event) & (pd.to_numeric(group.WMSCI_event, errors="coerce") > 0)).sum()) != row.positive_W_executions:
            raise ValueError(f"diagnostic census mismatch: {row.instrument}")
        selected_n = int((selected.instrument == row.instrument).sum())
        if selected_n != row.final_WMSCI_to_FPM_to_REV_executions:
            raise ValueError(f"selected census mismatch: {row.instrument}")
    if len(top) != int(counts.top_W_exact_k.sum()):
        raise ValueError("exact top-five-percent census mismatch")
    if not final_ids.issubset(set(top[["instrument", "execution_cluster_id"]].itertuples(index=False, name=None))):
        raise ValueError("final selection is not contained in deterministic top-W tail")
    return counts, diagnostics, selected, metadata


def derive_products(counts: pd.DataFrame, diagnostics: pd.DataFrame, selected: pd.DataFrame) -> dict[str, pd.DataFrame]:
    top = _exact_top(diagnostics)
    funnel = pd.DataFrame({
        "Stage": ["All executions", "Positive W", "Exact top 5% W", "U/Q > 1", "FPM > 0", "Complete finite REV", "REV > 0 (final)"],
        "Ferrari": [counts.loc[counts.instrument == "FERRARI", field].item() for field in ("all_executions", "positive_W_executions", "top_W_exact_k", "top_W_after_U_over_Q_gt_1", "top_W_after_U_over_Q_gt_1_FPM_gt_0", "top_W_after_FPM_gt_0_REV_observed", "final_WMSCI_to_FPM_to_REV_executions")],
        "Nexi": [counts.loc[counts.instrument == "NEXI", field].item() for field in ("all_executions", "positive_W_executions", "top_W_exact_k", "top_W_after_U_over_Q_gt_1", "top_W_after_U_over_Q_gt_1_FPM_gt_0", "top_W_after_FPM_gt_0_REV_observed", "final_WMSCI_to_FPM_to_REV_executions")],
        "Risanamento": [counts.loc[counts.instrument == "RISANAMENTO", field].item() for field in ("all_executions", "positive_W_executions", "top_W_exact_k", "top_W_after_U_over_Q_gt_1", "top_W_after_U_over_Q_gt_1_FPM_gt_0", "top_W_after_FPM_gt_0_REV_observed", "final_WMSCI_to_FPM_to_REV_executions")],
    })
    population = counts.loc[:, ["instrument", "all_executions", "positive_W_executions", "top_W_exact_k", "final_WMSCI_to_FPM_to_REV_executions"]].copy()
    population["instrument"] = population["instrument"].str.title()
    population.columns = ["Instrument", "All executions", "Positive W", "Top 5% W", "Final"]
    components_rows, overlap_rows, price_rows = [], [], []
    for instrument in INSTRUMENTS:
        group = diagnostics.loc[diagnostics.instrument == instrument]
        positive = group.loc[_finite(group.WMSCI_event) & (pd.to_numeric(group.WMSCI_event, errors="coerce") > 0)]
        tail = top.loc[top.instrument == instrument]
        score_ids = set(tail.execution_cluster_id)
        for label, field in (("R = U/Q", "withdrawal_to_execution_ratio"), ("A = U/P", "matched_deceptive_cancel_fraction_window"), ("Q", "execution_quantity")):
            n, rho = _spearman(positive.WMSCI_event, positive[field])
            components_rows.append({"Instrument": instrument.title(), "Component": label, "Pairs": n, "Spearman with W": rho})
            alternative = positive.assign(_score=pd.to_numeric(positive[field], errors="coerce")).sort_values(["_score", "execution_cluster_id"], ascending=[False, True], kind="stable").head(len(tail))
            overlap_rows.append({"Instrument": instrument.title(), "Component": label, "Top-W k": len(tail), "Overlap": len(score_ids & set(alternative.execution_cluster_id))})
        for label, field in (("FPM", "favorable_mid_move_pre_fill"), ("REV", "post_cancel_mid_reversion"), ("ADV", "execution_price_advantage_vs_posture_mid")):
            n, rho = _spearman(positive.WMSCI_event, positive[field])
            price_rows.append({"Instrument": instrument.title(), "Diagnostic": label, "Observed pairs": n, "Spearman with W": rho})
    selected_view = selected.copy()
    if selected_view[["identity_level", "execution_anchor_mode"]].isna().any().any():
        raise ValueError("selected execution missing reporting fields")
    composition = (selected_view.assign(Instrument=selected_view.instrument.str.title(), Attribution=selected_view.identity_level.map({"client_original": "Original client", "firm": "Firm"}), Role=selected_view.execution_anchor_mode.str.title())
                   .groupby(["Instrument", "Attribution", "Role"], as_index=False).size().rename(columns={"size": "Final executions"}))
    return {
        "funnel": funnel, "population": population, "components": pd.DataFrame(components_rows),
        "tail_overlap": pd.DataFrame(overlap_rows), "price_associations": pd.DataFrame(price_rows),
        "selected_composition": composition,
    }


def _save(fig: plt.Figure, output: Path, stem: str) -> None:
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def render_figures(diagnostics: pd.DataFrame, output: Path) -> None:
    fig, axis = plt.subplots(figsize=(7.0, 4.0), layout="constrained")
    for instrument in INSTRUMENTS:
        values = np.sort(pd.to_numeric(diagnostics.loc[(diagnostics.instrument == instrument) & (pd.to_numeric(diagnostics.WMSCI_event, errors="coerce") > 0), "WMSCI_event"], errors="coerce").to_numpy(dtype=float))
        axis.step(np.log1p(values), np.arange(1, len(values) + 1) / len(values), where="post", label=f"{instrument.title()} (n={len(values):,})")
    axis.set(xlabel="log(1 + W)", ylabel="Cumulative fraction of positive W", ylim=(0, 1.02))
    axis.legend(frameon=False)
    _save(fig, output, "wmsci_ecdf")
    fig, axes = plt.subplots(2, 3, figsize=(10.0, 6.4), layout="constrained")
    for col, instrument in enumerate(INSTRUMENTS):
        positive = diagnostics.loc[(diagnostics.instrument == instrument) & (pd.to_numeric(diagnostics.WMSCI_event, errors="coerce") > 0)].copy()
        x = np.log1p(pd.to_numeric(positive.WMSCI_event, errors="coerce"))
        for row, (label, field) in enumerate((("FPM", "favorable_mid_move_pre_fill"), ("REV", "post_cancel_mid_reversion"))):
            y = pd.to_numeric(positive[field], errors="coerce")
            observed = np.isfinite(x) & np.isfinite(y)
            axes[row, col].scatter(x[observed], y[observed], s=7, alpha=0.32, rasterized=True)
            axes[row, col].axhline(0, color="0.45", linewidth=0.7, linestyle=":")
            axes[row, col].set(title=instrument.title() if row == 0 else "", ylabel=f"{label} (price units)")
            if row == 1:
                axes[row, col].set_xlabel("log(1 + W)")
    _save(fig, output, "score_price_diagnostics")


def _expected_inventory() -> set[str]:
    return {f"{name}.{extension}" for name in TABLES for extension in ("csv", "tex")} | {f"{name}.{extension}" for name in FIGURES for extension in ("pdf", "png")} | {"metadata.json"}


def write_package(output: Path, source: Path) -> dict[str, Any]:
    output, source = Path(output), Path(source)
    if output.exists():
        raise FileExistsError(output)
    counts, diagnostics, selected, source_metadata = validate_source(source)
    products = derive_products(counts, diagnostics, selected)
    captions = {
        "funnel": "Sequential execution-level screen. Exact top-five-percent W selection is calculated within the finite positive-W population of each instrument; subsequent rows are nested. Complete finite REV requires a finite reversion for every assigned positive-quantity cancellation.",
        "population": "Execution-level population after the verified one-second withdrawal reconstruction. Final denotes executions passing the full sequential screen, not independently selected episodes.",
        "components": "Descriptive Spearman associations in the positive-W execution population. R is withdrawal divided by execution quantity; A is the share of candidate quantity removed. Repeated shared postures mean these are not independent replications.",
        "tail_overlap": "Overlap of each component's deterministic exact-k upper tail with the exact top-five-percent W tail. Ties are resolved by stable execution identifier after descending score.",
        "price_associations": "Descriptive Spearman associations between W and observed price diagnostics in the positive-W population. FPM is favorable pre-execution midprice movement; REV is cancellation-anchored reversion; ADV is execution advantage versus the completed-display benchmark.",
        "selected_composition": "Composition of the final sequentially selected executions by resolved attribution and execution role. Episode membership is retained only for reporting and is not a screening condition.",
    }
    output.mkdir(parents=True)
    try:
        for name, frame in products.items():
            frame.to_csv(output / f"{name}.csv", index=False)
            (output / f"{name}.tex").write_text(_table(frame, captions[name], f"tab:one-second-{name.replace('_', '-') }"))
        render_figures(diagnostics, output)
        outputs = {path.name: sha256_file(path) for path in sorted(output.iterdir()) if path.is_file()}
        if set(outputs) | {"metadata.json"} != _expected_inventory():
            raise ValueError("package inventory mismatch")
        metadata = {
            "schema": SCHEMA,
            "generator_sha256": sha256_file(Path(__file__)),
            "source": {"metadata_sha256": sha256_file(source / "metadata.json"), "output_hashes": {name: source_metadata["outputs"][name] for name in sorted(SOURCE_REQUIRED - {"metadata.json"})}},
            "contract": {"profile_window_seconds": 1.0, "withdrawal_window_seconds": 1.0, "candidate_age_seconds": 30.0, "reversion_horizon_seconds": 2.0, "selection_order": ["positive_W_exact_top_5pct", "U_over_Q_gt_1", "FPM_gt_0", "REV_gt_0"], "episode_role": "reporting_only"},
            "outputs": outputs,
        }
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        return metadata
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise


def verify_package(output: Path, source: Path) -> dict[str, Any]:
    output = Path(output)
    if not output.is_dir() or any(path.is_dir() or path.is_symlink() for path in output.iterdir()) or {path.name for path in output.iterdir()} != _expected_inventory():
        raise ValueError("package inventory mismatch")
    metadata = json.loads((output / "metadata.json").read_text())
    if metadata.get("schema") != SCHEMA or metadata.get("generator_sha256") != sha256_file(Path(__file__)):
        raise ValueError("package metadata mismatch")
    counts, diagnostics, selected, source_metadata = validate_source(source)
    products = derive_products(counts, diagnostics, selected)
    for name, frame in products.items():
        if (output / f"{name}.csv").read_text() != frame.to_csv(index=False):
            raise ValueError(f"table mismatch: {name}.csv")
    actual = {path.name: sha256_file(path) for path in output.iterdir() if path.name != "metadata.json"}
    if metadata.get("outputs") != actual:
        raise ValueError("output hash mismatch")
    if metadata.get("source", {}).get("metadata_sha256") != sha256_file(Path(source) / "metadata.json"):
        raise ValueError("source metadata provenance mismatch")
    expected_source_hashes = {name: source_metadata["outputs"][name] for name in sorted(SOURCE_REQUIRED - {"metadata.json"})}
    if metadata.get("source", {}).get("output_hashes") != expected_source_hashes:
        raise ValueError("source output provenance mismatch")
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verify-package", action="store_true")
    args = parser.parse_args(argv)
    if args.verify_package:
        verify_package(args.output_dir, args.source_dir)
        print("ONE_SECOND_SEQUENTIAL_PAPER_PACKAGE_VERIFIED")
    else:
        write_package(args.output_dir, args.source_dir)
        verify_package(args.output_dir, args.source_dir)
        print("ONE_SECOND_SEQUENTIAL_PAPER_PACKAGE_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
