"""Six-object aggregate report; requires verified completion of every producer."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import polars as pl

from .temporal_calibration_report import (
    _verified_source, _stage_identity, _publish_stage, _atomic_json, _sha256,
    _split, _read_reference,
)

_REQUIRED_STAGES = ("estimate", "validate", "w_association", "endpoint_diagnostics",
                    "profile_diagnostics", "uncertainty_diagnostics")
# Only explicitly public numeric/semantic fields can reach aggregate reports.
_PUBLIC = frozenset((
    "instrument", "identity_level", "execution_anchor_mode", "role", "split", "parameter",
    "population", "weighting", "quantile", "value_ns", "unit", "status", "reason",
    "proposal_status", "proposed_window_seconds", "window_seconds", "horizon_seconds", "horizon_ns",
    "excess_count", "included_group_count", "excluded_group_count", "supported_risk_seconds",
    "n_obs", "n_actors", "n_days", "supported_seconds", "n_cases", "n_observed", "n_missing",
    "row_scope", "null_reason", "endpoint_kind", "n_negative_raw", "n_zero_raw", "n_positive_raw",
    "raw_delta_median_ticks", "raw_delta_p90_ticks", "raw_delta_p95_ticks", "recency_median_ns",
    "recency_p90_ns", "recency_p95_ns", "sci_median", "msci_median", "SCI_median", "MSCI_median",
    "reference_value_ns", "heldout_descriptive_p95_ns", "frozen_reference_coverage",
    "reference_delay_ns", "reference_quantile", "weight_at_reference", "tau_seconds",
    "reference_population", "identified_fraction", "ci_lower", "ci_upper", "replicates",
    "available_day_count", "supported_day_count", "calendar_day_count", "identified_replicates", "bootstrap_w_ci_lower_seconds",
    "bootstrap_w_ci_upper_seconds", "inference_note", "retained_day_count", "search_cap_binding",
    "chosen_window_seconds", "calibration_proposal_status", "heldout_excess_count",
    "heldout_included_group_count", "heldout_supported_risk_seconds", "interpolation",
    "direction", "rho", "reference_q90_delay_ns", "reference_pair_count",
    "physical_cancellation_count", "association_window_seconds", "model",
))


def _public_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [{key: value for key, value in row.items() if key in _PUBLIC}
            for row in pl.read_csv(path, infer_schema_length=None).to_dicts()]


def _figures(root: Path, manifest: Mapping[str, Any], instrument: str, destination: Path,
             tables: list[tuple[str, str, list[dict[str, Any]]]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    destination.mkdir()
    gaps = _read_reference(root, manifest, "gap_pairs.parquet", {"instrument", "identity_level", "event_date", "gap_ns"})
    ages = _read_reference(root, manifest, "age_exposure.parquet", {"instrument", "identity_level", "event_date", "age_start_ns", "age_end_ns"})
    for level in ("client_original", "firm"):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
        splits = ("calibration", "retrospective_validation") if instrument == "RISANAMENTO" else ("exploratory_single_day",)
        for split in splits:
            values = np.sort([int(row["gap_ns"]) for row in gaps if row["identity_level"] == level
                              and row["instrument"] == instrument and _split(instrument, row["event_date"]) == split])
            if values.size:
                zero_fraction = float(np.count_nonzero(values == 0) / values.size)
                axes[0].step(values / 1e9, np.arange(1, values.size + 1) / values.size, where="post",
                             label=f"{split}: n={values.size}; P(gap=0)={zero_fraction:.3f}")
            intervals = [(int(row["age_start_ns"]), int(row["age_end_ns"])) for row in ages
                         if row["identity_level"] == level and row["instrument"] == instrument
                         and _split(instrument, row["event_date"]) == split]
            total = sum(end - start for start, end in intervals)
            if total > 0:
                boundaries = sorted({value for interval in intervals for value in interval})
                cdf = [sum(max(0, min(end, value) - start) for start, end in intervals) / total for value in boundaries]
                axes[1].plot(np.asarray(boundaries) / 1e9, cdf, label=f"{split}: intervals={len(intervals)}")
        for axis, label in zip(axes, ("Adjacent execution-fill gap (s)", "First-seen visible-order age (s)")):
            axis.set_xlabel(label); axis.set_ylabel("Cumulative fraction")
            axis.set_ylim(0, 1.02); axis.grid(alpha=0.2)
            if axis.lines: axis.legend(fontsize=8)
            else: axis.text(0.5, 0.5, "No supported observations", ha="center", transform=axis.transAxes)
        fig.suptitle(f"{instrument} — {level}; descriptive pooled-role gap ECDF / order-time CDF")
        fig.savefig(destination / f"{level}_gap_age.pdf"); plt.close(fig)
        curves = next((rows for obj, kind, rows in tables if obj == "W" and kind == "curve"), [])
        fig, axis = plt.subplots(figsize=(7, 4.5), layout="constrained")
        for split in splits:
            for role in ("passive", "aggressive"):
                selected = sorted((r for r in curves if r.get("identity_level") == level and r.get("split") == split
                                   and r.get("execution_anchor_mode") == role and r.get("excess_count") is not None),
                                  key=lambda row: row["window_seconds"])
                if selected:
                    axis.plot([r["window_seconds"] for r in selected], [r["excess_count"] for r in selected],
                              marker="o", label=f"{split}/{role}")
        axis.set_xlabel("Association window (s)"); axis.set_ylabel("Summed excess physical cancellations (count)")
        axis.set_title(f"{instrument} — {level}; common-support shifted controls")
        axis.grid(alpha=0.2)
        if axis.lines: axis.legend(fontsize=8)
        else: axis.text(0.5, 0.5, "Association not identified / no support", ha="center", transform=axis.transAxes)
        fig.savefig(destination / f"{level}_association.pdf"); plt.close(fig)
        endpoints = next((rows for obj, kind, rows in tables if obj == "REV" and kind == "diagnostic"), [])
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
        for split in splits:
            selected = sorted((r for r in endpoints if r.get("identity_level") == level and r.get("split") == split
                               and r.get("row_scope") == "all_cases"), key=lambda row: row["horizon_ns"])
            observed = [r for r in selected if r.get("raw_delta_median_ticks") is not None]
            if observed:
                axes[0].plot([r["horizon_ns"] / 1e9 for r in observed], [r["raw_delta_median_ticks"] for r in observed],
                             marker="o", label=split)
            coverage = [r for r in selected if r.get("n_cases", 0)]
            if coverage:
                axes[1].plot([r["horizon_ns"] / 1e9 for r in coverage], [r["n_observed"] / r["n_cases"] for r in coverage],
                             marker="o", label=split)
        for axis in axes:
            axis.set_xlabel("Post-cancellation horizon (s)"); axis.grid(alpha=0.2)
            if axis.lines: axis.legend(fontsize=8)
            else: axis.text(0.5, 0.5, "No supported observations", ha="center", transform=axis.transAxes)
        axes[0].set_ylabel("Median raw midprice change (ticks)")
        axes[1].set_ylabel("Observed / all physical cancellation paths")
        axes[1].set_ylim(0, 1.02)
        fig.suptitle(f"{instrument} — {level}; market-only endpoints, not native episode REV")
        fig.savefig(destination / f"{level}_physical_endpoints.pdf"); plt.close(fig)
        profiles = next((rows for obj, kind, rows in tables if obj == "SCI/MSCI" and kind == "diagnostic"), [])
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout="constrained")
        for split in splits:
            for role in ("passive", "aggressive"):
                selected = sorted((r for r in profiles if r.get("identity_level") == level and r.get("split") == split
                                   and r.get("execution_anchor_mode") == role), key=lambda row: row["horizon_ns"])
                for axis, field in zip(axes[:2], ("sci_median", "msci_median")):
                    observed = [r for r in selected if r.get(field) is not None]
                    if observed:
                        axis.plot([r["horizon_ns"] / 1e9 for r in observed], [r[field] for r in observed],
                                  marker="o", label=f"{split}/{role}")
                covered = [r for r in selected if r.get("n_cases", 0)]
                if covered:
                    axes[2].plot([r["horizon_ns"] / 1e9 for r in covered], [r["n_observed"] / r["n_cases"] for r in covered],
                                 marker="o", label=f"{split}/{role}")
        for axis, label in zip(axes, ("Median native SCI", "Median native MSCI", "Observed / all execution profiles")):
            axis.set_xlabel("Post-execution horizon (s)"); axis.set_ylabel(label); axis.grid(alpha=0.2)
            if axis.lines: axis.legend(fontsize=7)
            else: axis.text(0.5, 0.5, "Not identified / no supported profiles", ha="center", transform=axis.transAxes)
        axes[2].set_ylim(0, 1.02)
        fig.suptitle(f"{instrument} — {level}; frozen train-only kernel; recency in aggregate table")
        fig.savefig(destination / f"{level}_native_profiles.pdf"); plt.close(fig)


def run_final_report(output_root: Path | str, design: Mapping[str, Any], instrument: str,
                     *, dates: list[str] | None = None) -> dict[str, Any]:
    root = Path(output_root).resolve()
    manifest, source_hash = _verified_source(root, design, instrument, dates=dates)
    for name in _REQUIRED_STAGES:
        if not (root / name / "stage_manifest.json").is_file():
            raise ValueError(f"complete verified {name} stage required before final report")
    # Calling completed producers verifies their current code/source contracts;
    # incompatible caches fail closed rather than being silently regenerated.
    from .temporal_calibration_report import run_estimate, run_validate
    from .temporal_calibration_w_stage import run_w_association
    from .temporal_calibration_endpoints import run_endpoint_diagnostics
    from .temporal_calibration_profiles import run_profile_diagnostics
    from .temporal_calibration_uncertainty import run_uncertainty_diagnostics
    for producer in (run_estimate, run_validate, run_w_association, run_endpoint_diagnostics,
                     run_profile_diagnostics, run_uncertainty_diagnostics):
        producer(root, design, instrument, dates=dates)
    predecessors = {name: _sha256(root / name / "stage_manifest.json") for name in _REQUIRED_STAGES}
    predecessors["final_report_module"] = _sha256(Path(__file__))
    identity = _stage_identity("final_report", source_hash, design, instrument, predecessors)

    def build(stage: Path) -> None:
        tables = [
            ("gap", "estimate", _public_rows(root / "estimate" / "estimates.csv")),
            ("gap_age", "validation", _public_rows(root / "validate" / "validation.csv")),
            ("W", "curve", _public_rows(root / "w_association" / "curve.csv")),
            ("W", "proposal", _public_rows(root / "w_association" / "proposals.csv")),
            ("REV", "diagnostic", _public_rows(root / "endpoint_diagnostics" / "aggregate.csv")),
            ("SCI/MSCI", "diagnostic", _public_rows(root / "profile_diagnostics" / "aggregate.csv")),
            ("tau", "proposal", _public_rows(root / "uncertainty_diagnostics" / "decay_proposals.csv")),
            ("W", "bootstrap", _public_rows(root / "uncertainty_diagnostics" / "bootstrap_summary.csv")),
            ("W", "actor_sensitivity", _public_rows(root / "uncertainty_diagnostics" / "actor_sensitivity.csv")),
            ("W", "validation", _public_rows(root / "uncertainty_diagnostics" / "frozen_validation.csv")),
        ]
        lines = [f"# Calibrazione temporale — {instrument}", "",
                 "Solo proposte e diagnostiche: nessuna configurazione del detector è stata modificata.",
                 "Clock replay/removal: SEQUENCETIME; anchor fill: TRADETIME. Timebase UTC comune non certificata.",
                 "Le finestre non sono misure di latenza fisica HFT. Firm fallback può aggregare più clienti.", ""]
        counts = Counter()
        for level, filename, title in (("client_original", "client_original.csv", "Client originale"),
                                       ("firm", "firm_fallback.csv", "Firm fallback")):
            records = []
            for obj, kind, rows in tables:
                for row in rows:
                    if row.get("identity_level") != level:
                        continue
                    effective_obj = "age" if obj == "gap" and str(row.get("parameter", "")).startswith("age") else obj
                    records.append({"instrument": instrument, "object": effective_obj, "record_kind": kind,
                                    "identity_level": level, "split": row.get("split"),
                                    "details_json": json.dumps(row, sort_keys=True, allow_nan=False)})
            pl.DataFrame(records, schema={name: pl.String for name in
                ("instrument", "object", "record_kind", "identity_level", "split", "details_json")}).write_csv(stage / filename)
            lines.extend((f"## {title}", "", "Oggetto | Righe aggregate | Stato", "--- | ---: | ---"))
            for obj in ("gap", "age", "W", "REV", "SCI/MSCI", "tau"):
                selected = [r for r in records if r["object"] == obj]
                counts[f"{level}/{obj}"] = len(selected)
                statuses = sorted({str(json.loads(row["details_json"]).get("status") or
                                       json.loads(row["details_json"]).get("proposal_status") or "diagnostic") for row in selected})
                lines.append(f"{obj} | {len(selected)} | {', '.join(statuses) if statuses else 'no_support'}")
            lines.append("")
        lines.extend(("## Interpretazione e limiti", "",
                      "RISANAMENTO: calibrazione fino al 30 settembre 2024; holdout successivo retrospettivo, senza refit.",
                      "NEXI/FERRARI: analisi esplorativa single-day, non validazione predittiva.",
                      "Età first-seen osservata e censura non equivalgono alla vita completa dell'ordine presso la venue.",
                      "W misura associazione descrittiva rispetto ai timing shift, non intenzione manipolativa.",
                      "REV qui indica diagnostiche market-only, non la metrica economica nativa episode REV.",
                      "Null/nonidentified restano distinti da zero. Bootstrap e leave-one-actor-out misurano stabilità, non verità causale."))
        (stage / "summary.md").write_text("\n".join(lines) + "\n")
        _atomic_json(stage / "summary.json", {"instrument": instrument, "proposal_only": True,
                      "source_manifest_sha256": source_hash, "objects": ["gap", "age", "W", "REV", "SCI/MSCI", "tau"],
                      "aggregate_row_counts": dict(counts), "is_pilot": manifest["identity"]["is_pilot"]})
        _figures(root, manifest, instrument, stage / "figures", tables)
        if _verified_source(root, design, instrument, dates=dates)[1] != source_hash:
            raise ValueError("reference source changed during final report")
    resumed = _publish_stage(root, "final_report", identity, build)
    return {"status": "final_report_complete", "resume_verified": resumed}
