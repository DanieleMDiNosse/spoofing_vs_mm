"""Publication-only validation and table materialization for verified cached analyses."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections.abc import Mapping
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd


HASH_MAP_FIELDS = (
    "outputs",
    "inputs",
    "input_hashes",
    "input_hashes_complete",
    "code_dependencies_sha256",
    "physical_assignment_trace_outputs",
)
REQUIRED_INPUT_FILENAMES = (
    "phase2_summary.json",
    "phase3_summary.json",
    "phase3_sequence_review_corrected.json",
    "phase4_price_summary.json",
    "phase4_recurrence_summary.json",
    "phase4_timing_summary.json",
    "phase5_strict_summary.json",
    "phase6_behavior_price_summary.json",
    "phase7_reversion_complete_provenance.json",
    "phase9_sensitivity_complete_provenance.json",
)
INSTRUMENTS = ("FERRARI", "NEXI", "RISANAMENTO")
IDENTITY_LEVELS = ("client_original", "firm")
ROLES = ("ALL", "aggressive", "passive")
ACCOUNTING_TOLERANCE = 1e-10
_DIGEST_RE = re.compile(r"[0-9a-fA-F]{64}\Z")
ROOT = Path(__file__).resolve().parents[1]
PACKAGE_FIGURES = (
    "wmsci_ecdf", "msci_wmsci_joint", "strict_score_space",
    "price_diagnostics",
)
BASELINE_CENSUS = {
    "FERRARI": {"executions": 12997, "candidates": 6600, "positive_W": 2544, "missing_M": 333},
    "NEXI": {"executions": 10512, "candidates": 4807, "positive_W": 1729, "missing_M": 621},
    "RISANAMENTO": {"executions": 24972, "candidates": 2477, "positive_W": 216, "missing_M": 4610},
}
_DISPLAY_NAMES = {
    "instrument": "Instrument",
    "identity": "Identity",
    "identity_level": "Identity",
    "comparison_mode": "Comparison",
    "fraction": "Tail fraction",
    "other": "Component",
    "n_pair": "Pairs",
    "n_actors": "Actors",
    "n_events": "Events",
    "k": "k",
    "executions": "Executions",
    "candidates": "Candidates",
    "positive_W": "Positive W",
    "missing_M": "Missing M",
    "overlap_count_exact_k": "Overlap at k",
    "component_tie_inclusive_n": "Component ties",
    "n_joint_price_observed": "Price observed",
    "n_favorable_and_reversal": "Favourable + REV",
    "strict_episodes": "Strict episodes",
    "strict_member_events": "Strict members",
    "strict_zero_W_members": "Zero-W members",
    "strict_mixed_anchor_episodes": "Mixed anchors",
    "selected_events": "Selected events",
    "selected_episodes": "Selected episodes",
    "n_complete_decomposition": "Complete decompositions",
    "total_REV_ticks_mean": "Mean REV ticks",
    "immediate_jump_ticks_mean": "Immediate ticks",
    "later_path_ticks_mean": "Later-path ticks",
    "selected_distinct_episodes": "Distinct episodes",
    "actors_with_ge2_selected_episodes": "Actors with ≥2 episodes",
    "actors_with_ge2_selected_days": "Actors with ≥2 days",
    "group_count": "Groups",
    "weighted_intensity_difference": "Weighted difference",
    "spec": "Specification",
    "fixed_k": "Fixed k",
    "topk_overlap_count": "Top-k overlap",
    "union_positive_score_spearman": "Positive-score Spearman",
    "changed_strict_episodes": "Changed strict episodes",
    "official_gate_comparable": "Official gate comparable",
    "manuscript_assignment": "Economic-time allocation",
}


def publication_contract():
    """Return the fixed descriptive semantics of this publication release."""
    return {
        "schema": "withdrawal_characterization_publication_v1",
        "assignment": "latest_prior_cluster_sort_index",
        "assignment_tie_break": ["-last_source_index", "first_source_index", "execution_id"],
        "posting_benchmark": "latest_candidate_first_seen_source_index",
        "withdrawal_boundary": "[0,2) seconds",
        "candidate_age_seconds": [0, 90],
        "top_n": 10,
        "structural_horizon_seconds": 10,
        "reversion_horizon_seconds": 2,
        "decay_seconds": 10,
        "units": {"behavior": "execution_cluster", "strict": "candidate_posture_episode"},
        "withdrawal_quantity": "gross_capped_uniquely_attributed_cancellations",
        "strict_price_sign": "literal >0",
        "rev_summary_positive_tolerance_ticks": 1e-10,
        "role_interpretation": "CONSOB_confirmed_as_reported_by_author",
        "clock_origin": "common_UTC_origin_operational_assumption",
        "severity_calibrated": False,
        "causal_counterfactual": False,
        "own_book_exclusion": False,
    }


def sha256_file(path: Path) -> str:
    """Hash a file in chunks without loading its content into memory."""
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _safe_file(root: Path, relative_path: str) -> Path:
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError("unsafe path")
    candidate = Path(relative_path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"unsafe path: {relative_path}")
    resolved_root = root.resolve()
    resolved = (resolved_root / candidate).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"unsafe path: {relative_path}") from exc
    if not resolved.is_file():
        raise ValueError(f"unknown path: {relative_path}")
    return resolved


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _validated_hash_map(root: Path, field: str, value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a dict")
    validated: dict[str, str] = {}
    for path, digest in value.items():
        if not isinstance(path, str) or not isinstance(digest, str) or not _DIGEST_RE.fullmatch(digest):
            raise ValueError(f"{field} has malformed hash map")
        _safe_file(root, path)
        validated[path] = digest
    return validated


def _merge_hashes(destination: dict[str, str], incoming: Mapping[str, str]) -> None:
    for path, digest in incoming.items():
        previous = destination.get(path)
        if previous is not None and previous != digest:
            raise ValueError(f"contradictory digest: {path}")
        destination[path] = digest


def verify_manifest(root: Path, manifest_path: Path) -> dict[str, str]:
    """Verify every present hash map and optional complete original-manifest linkage."""
    root = Path(root)
    manifest = _safe_file(root, str(manifest_path))
    try:
        payload = json.loads(manifest.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("malformed manifest") from exc
    if not isinstance(payload, dict):
        raise ValueError("malformed manifest")

    expected: dict[str, str] = {}
    for field in HASH_MAP_FIELDS:
        if field in payload:
            _merge_hashes(expected, _validated_hash_map(root, field, payload[field]))

    has_original_path = "original_manifest" in payload
    has_original_digest = "original_manifest_sha256" in payload
    if has_original_path != has_original_digest:
        raise ValueError("original manifest linkage incomplete")
    if has_original_path:
        original_path = payload["original_manifest"]
        original_digest = payload["original_manifest_sha256"]
        if not isinstance(original_path, str):
            raise ValueError("unsafe path")
        if not isinstance(original_digest, str) or not _DIGEST_RE.fullmatch(original_digest):
            raise ValueError("malformed original manifest digest")
        _safe_file(root, original_path)
        _merge_hashes(expected, {original_path: original_digest})

    for path, digest in expected.items():
        actual = sha256_file(_safe_file(root, path))
        if actual != digest:
            raise ValueError(f"hash mismatch: {path}")
    expected[_relative(root, manifest)] = sha256_file(manifest)
    return expected


def verify_inputs(root: Path) -> dict[str, str]:
    """Verify and merge the fixed upstream producer notes directly from audit/notes."""
    root = Path(root)
    merged: dict[str, str] = {}
    notes = root / "astra_metric_audit" / "notes"
    for filename in REQUIRED_INPUT_FILENAMES:
        manifest = notes / filename
        if not manifest.is_file():
            raise ValueError(f"missing required input: {filename}")
        _merge_hashes(merged, verify_manifest(root, Path("astra_metric_audit") / "notes" / filename))
    return merged


def population_census(events):
    """Validate the Phase-2 event population and return descriptive counts."""
    required = ("instrument", "execution_cluster_id", "identity_level", "P", "Q", "U", "weighted_U", "WMSCI", "MSCI")
    missing = [column for column in required if column not in events.columns]
    if missing:
        raise ValueError(f"missing field: {missing[0]}")
    if events.empty:
        raise ValueError("empty events")
    frame = events.loc[:, required].copy(deep=True)
    for field in ("instrument", "execution_cluster_id", "identity_level"):
        if frame[field].isna().any() or (frame[field].astype(str).str.len() == 0).any():
            raise ValueError(f"missing field: {field}")
    if not frame["instrument"].isin(INSTRUMENTS).all():
        raise ValueError("unknown instrument")
    if not frame["identity_level"].isin(IDENTITY_LEVELS).all():
        raise ValueError("unknown identity_level")
    if frame.duplicated(["instrument", "execution_cluster_id"]).any():
        raise ValueError("duplicate execution")
    numeric = {}
    for field in ("P", "Q", "U", "weighted_U", "WMSCI"):
        values = pd.to_numeric(frame[field], errors="coerce")
        if not values.map(math.isfinite).all():
            if field == "Q":
                raise ValueError("Q must be positive and finite")
            raise ValueError(f"{field} must be finite")
        numeric[field] = values
    if not (numeric["Q"] > 0).all():
        raise ValueError("Q must be positive and finite")
    for field in ("P", "U", "weighted_U", "WMSCI"):
        if (numeric[field] < 0).any():
            raise ValueError(f"{field} must be nonnegative")
    if (numeric["U"] - numeric["P"] > ACCOUNTING_TOLERANCE).any():
        raise ValueError("U exceeds P")
    if (numeric["weighted_U"] - numeric["U"] > ACCOUNTING_TOLERANCE).any():
        raise ValueError("weighted_U exceeds U")
    msci = pd.to_numeric(frame["MSCI"], errors="coerce")
    if (frame["MSCI"].notna() & ~msci.map(math.isfinite)).any():
        raise ValueError("MSCI must be finite or missing")
    return {
        instrument: {
            "executions": int((frame["instrument"] == instrument).sum()),
            "candidates": int(((frame["instrument"] == instrument) & (numeric["P"] > 0)).sum()),
            "positive_W": int(((frame["instrument"] == instrument) & (numeric["WMSCI"] > 0)).sum()),
            "missing_M": int(((frame["instrument"] == instrument) & ~msci.map(math.isfinite)).sum()),
        }
        for instrument in INSTRUMENTS
    }


def tex_escape(value: object) -> str:
    """Escape arbitrary text for a literal LaTex table cell."""
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
        "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in str(value)).replace(
        "≥", r"\ensuremath{\geq}"
    )


def display_value(value: object) -> str:
    """Format a display-only cell while preserving raw values in companion CSV text."""
    if value is None or pd.isna(value):
        return "NA"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            return "NA"
        return format(float(value), ".4g")
    return str(value)


def _header(column: str) -> str:
    return _DISPLAY_NAMES.get(column, column.replace("_", " "))


_TEXT_TABLE_COLUMNS = {
    "instrument", "identity", "identity_level", "comparison_mode", "other", "population", "spec", "price",
}


def _tex_cell(column: str, value: object) -> str:
    """Format public scientific names in TeX only; companion CSV values remain raw."""
    names = {
        "FERRARI": "Ferrari",
        "NEXI": "Nexi",
        "RISANAMENTO": "Risanamento",
        "strict_component": "Strict subset",
        "FPM_ticks": "FPM (ticks)",
        "REV_ticks": "REV (ticks)",
        "ADV_ticks": "ADV (ticks)",
        "execution_edge_ticks": "Execution edge (ticks)",
        "client_original": "Original client",
        "firm": "Firm",
        "baseline": "Baseline",
        "decay20": "20-second decay",
        "decay5": "5-second decay",
        "exclude_Q_lt10": "Exclude Q < 10",
        "no_decay": "No decay",
        "window0_5": "0.5-second window",
        "window1": "1-second window",
    }
    return tex_escape(names.get(str(value), display_value(value)))


def render_table(frame: pd.DataFrame, columns: list[str], caption: str, label: str) -> str:
    """Render bounded-width tabularx tables without resizing or tiny fonts."""
    if not columns:
        raise ValueError("columns required")
    if len(columns) > 6:
        raise ValueError("at most six columns")
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing column: {missing[0]}")
    text_columns = {column for column in columns if column in _TEXT_TABLE_COLUMNS}
    # Wrap headers as well as text: fixed-width numeric headers must not consume
    # all available width and leave the instrument/population columns too narrow.
    weights = [1.6 if column == "instrument" else 2.4 if column == "spec" else 1.0 for column in columns]
    total_weight = sum(weights)
    alignment = "".join(
        rf">{{\hsize={len(columns) * weight / total_weight:.8g}\hsize\linewidth=\hsize"
        + (r"\raggedright\arraybackslash}X" if column in text_columns else r"\raggedleft\arraybackslash}X")
        for column, weight in zip(columns, weights)
    )
    groups = [frame.loc[:, columns]]
    if len(frame) > 12 and "instrument" in columns:
        groups = [group for _, group in frame.loc[:, columns].groupby("instrument", sort=True)]
    lines = []
    for number, group in enumerate(groups):
        group_label = label if number == 0 else f"{label}-{str(group.iloc[0]['instrument']).lower()}"
        group_caption = caption if number == 0 else f"{caption} ({group.iloc[0]['instrument']})."
        lines.extend([r"\begin{table}[htbp]", r"\centering", f"\\caption{{{tex_escape(group_caption)}}}", f"\\label{{{tex_escape(group_label)}}}", f"\\begin{{tabularx}}{{\\textwidth}}{{{alignment}}}", r"\hline"])
        lines.append(" & ".join(tex_escape(_header(column)) for column in columns) + r" \\")
        lines.append(r"\hline")
        for row in group.itertuples(index=False, name=None):
            lines.append(" & ".join(_tex_cell(column, value) for column, value in zip(columns, row)) + r" \\")
        lines.extend([r"\hline", r"\end{tabularx}", r"\end{table}"])
    return "\n".join(lines)


def select_cached(path: Path, filters: Mapping[str, object]) -> pd.DataFrame:
    """Load a cached table and apply exact, validated literal stratum filters."""
    if not isinstance(filters, Mapping) or not filters:
        raise ValueError("empty filters")
    path = Path(path)
    if path.suffix == ".parquet":
        frame = pd.read_parquet(path)
    elif path.suffix == ".csv":
        frame = pd.read_csv(path)
    else:
        raise ValueError(f"unsupported cached table: {path}")
    selected = frame.copy(deep=True)
    for column, value in filters.items():
        if column not in selected.columns:
            raise ValueError(f"unknown filter column: {column}")
        if column == "instrument" and value not in INSTRUMENTS:
            raise ValueError(f"unknown instrument: {value}")
        if column in {"role", "tail_role"} and value not in ROLES:
            raise ValueError(f"unknown role: {value}")
        if column in {"identity", "tail_identity", "identity_level"} and value not in {"ALL", *IDENTITY_LEVELS}:
            raise ValueError(f"unknown identity: {value}")
        selected = selected.loc[selected[column] == value].copy(deep=True)
    if selected.empty:
        raise ValueError("empty selection")
    return selected


def table_specs() -> tuple[dict[str, object], ...]:
    """Return the immutable 12-table descriptive registry and fixed cache strata."""
    table_dir = "astra_metric_audit/tables/"
    return (
        {"stem": "population", "source": table_dir + "phase4_price_events.parquet", "filters": {}, "columns": ["instrument", "executions", "candidates", "positive_W", "missing_M"], "caption": "Counts of executions by instrument. The Candidates column counts executions with recent opposite-side candidate liquidity; Positive W counts positive WMSCI; Missing M counts unavailable MSCI profile comparisons. W denotes WMSCI and M denotes MSCI, not a price proxy. These are execution counts, not order or episode counts.", "label": "tab:withdrawal-population", "expected_rows": 3, "transform": "population_census"},
        {"stem": "components", "source": table_dir + "component_correlations.csv", "filters": {"population": "positive_WMSCI", "role": "ALL", "identity": "ALL"}, "include": {"other": ("R", "A", "Q")}, "columns": ["instrument", "other", "n_pair", "spearman"], "caption": "Component associations in the positive weighted-score execution-cluster population; paired descriptive counts are not IID observations.", "label": "tab:withdrawal-components", "expected_rows": 9},
        {"stem": "tail_overlap", "source": table_dir + "positive_tail_overlaps.csv", "filters": {"population": "positive_WMSCI", "role": "ALL", "identity": "ALL", "other": "R"}, "include": {"fraction": (0.01, 0.05, 0.1)}, "columns": ["instrument", "fraction", "k", "overlap_count_exact_k", "component_tie_inclusive_n"], "caption": "Exact-k positive-score tail overlap with R by instrument.", "label": "tab:withdrawal-tail-overlap", "expected_rows": 9},
        {"stem": "price_associations", "source": table_dir + "phase4_price_associations.csv", "filters": {"population": "positive_WMSCI", "role": "ALL", "identity": "ALL", "score": "WMSCI"}, "columns": ["instrument", "price", "n_pair", "n_actors", "spearman"], "caption": "Spearman associations between positive WMSCI and observed price diagnostics, with execution roles and attribution levels pooled within each instrument. FPM is favorable midprice movement, REV is cancellation-anchored reversion, ADV is execution advantage relative to candidate-layer completion, and execution edge is ADV minus FPM. Prices are in ticks. The Pairs column counts executions with both observations; Actors counts their attribution identities. Correlations are descriptive, not causal estimates.", "label": "tab:withdrawal-price-associations", "expected_rows": 12},
        {"stem": "joint_price", "source": table_dir + "phase6_behavior_price_headlines.csv", "filters": {"role": "ALL", "identity": "ALL", "selection": "pooled_W_top5", "price_definition": "native_REV"}, "columns": ["instrument", "n_events", "n_joint_price_observed", "n_favorable_and_reversal"], "caption": "Pooled top-5% WMSCI execution-cluster selection with native REV availability.", "label": "tab:withdrawal-joint-price", "expected_rows": 3},
        {"stem": "strict", "source": table_dir + "phase5_strict_episode_populations.csv", "filters": {}, "columns": ["instrument", "identity", "strict_episodes", "strict_member_events", "strict_zero_W_members", "strict_mixed_anchor_episodes"], "caption": "Episodes selected jointly for assigned opposite-side cancellation within 2 seconds (excluding the endpoint), unique withdrawal exceeding total executed quantity, positive favorable midprice movement, and positive cancellation-anchored reversion. The Strict members column counts executions in those episodes, not independently selected sequences. Zero-W members have zero WMSCI; Mixed anchors counts episodes containing both passive and aggressive executions. ALL pools the separately retained client and firm attribution levels.", "label": "tab:withdrawal-strict", "expected_rows": 9},
        {"stem": "strict_tail", "source": table_dir + "phase5_strict_tail_membership.csv", "filters": {"tail_role": "ALL", "tail_identity": "ALL", "score": "WMSCI", "selection": "exact_k", "fraction": 0.05}, "columns": ["instrument", "selected_events", "strict_member_events", "selected_episodes", "strict_episodes"], "caption": "Exact-k 5% WMSCI tail membership in strict candidate-posture episodes.", "label": "tab:withdrawal-strict-tail", "expected_rows": 3},
        {"stem": "rev_decomposition", "source": table_dir + "phase7_reversion_components_summary.csv", "filters": {"identity": "ALL", "role": "ALL", "grain": "episode"}, "include": {"population": ("ALL", "strict_component")}, "columns": ["instrument", "population", "n_complete_decomposition", "total_REV_ticks_mean", "immediate_jump_ticks_mean", "later_path_ticks_mean"], "caption": "Episode-grain native REV decomposition for all and strict-component populations.", "label": "tab:withdrawal-rev-decomposition", "expected_rows": 6},
        {"stem": "recurrence", "source": table_dir + "phase4_recurrence_tail_summary.csv", "filters": {"role": "ALL", "identity": "ALL", "score": "WMSCI", "selection": "exact_k", "fraction": 0.05}, "columns": ["instrument", "selected_events", "selected_distinct_episodes", "selected_actors", "actors_with_ge2_selected_episodes", "actors_with_ge2_selected_days"], "caption": "Exact-k 5% WMSCI tail recurrence across execution-cluster episodes and actor-days.", "label": "tab:withdrawal-recurrence", "expected_rows": 3},
        {"stem": "timing", "source": table_dir + "phase4_timing_reconciled_summary.csv", "filters": {"analysis_kind": "timing_shift_common_support", "offset_seconds": 0}, "columns": ["instrument", "identity_level", "comparison_mode", "represented_actors", "group_count", "weighted_intensity_difference"], "caption": "Cancellation-intensity contrast at observed execution timing: passive-only or aggressive-only own-execution windows minus the reference with no attributed own-execution window. Intensities and their weighted difference are in physical cancellations per posture-second of dynamically eligible liquidity. The Groups column counts actor-side-time-band groups supported at every schedule offset (-60, -30, 0, +30, +60 seconds); weights are fixed at zero-offset exposure. Actors counts distinct attribution identities. This comparison is separate from WMSCI ranking.", "label": "tab:withdrawal-timing", "expected_rows": 12},
        {"stem": "score_sensitivity", "source": table_dir + "phase9_W_score_stability.csv", "filters": {"role": "ALL", "identity": "ALL", "tail_fraction_baseline": 0.05}, "value_map": {"spec": {"manuscript_assignment": "economic-time allocation alternative"}}, "columns": ["instrument", "spec", "fixed_k", "topk_overlap_count", "union_positive_score_spearman"], "caption": "WMSCI 5% tail sensitivity across fixed-k specifications.", "label": "tab:withdrawal-score-sensitivity", "expected_rows": 24},
        {"stem": "gate_sensitivity", "source": table_dir + "phase9_episode_gate_stability.csv", "filters": {"identity": "ALL"}, "value_map": {"spec": {"manuscript_assignment": "economic-time allocation alternative"}}, "columns": ["instrument", "spec", "strict_episode_count", "changed_strict_episodes", "official_gate_comparable"], "caption": "Strict episode-gate sensitivity; the literal >0 native gate and Q restriction remain unchanged.", "label": "tab:withdrawal-gate-sensitivity", "expected_rows": 24},
    )


def build_tables(root: Path) -> dict[str, dict[str, object]]:
    """Materialize deterministic, narrow table products without writing publication files."""
    root = Path(root)
    products: dict[str, dict[str, object]] = {}
    specs = table_specs()
    stems = [str(spec["stem"]) for spec in specs]
    labels = [str(spec["label"]) for spec in specs]
    if len(stems) != len(set(stems)):
        raise ValueError("duplicate table stem")
    if len(labels) != len(set(labels)):
        raise ValueError("duplicate table label")
    for spec in specs:
        source = _safe_file(root, str(spec["source"]))
        if spec.get("transform") == "population_census":
            census = population_census(pd.read_parquet(source))
            frame = pd.DataFrame.from_dict(census, orient="index").reset_index(names="instrument")
        elif spec["filters"]:
            source_frame = pd.read_parquet(source) if source.suffix == ".parquet" else pd.read_csv(source)
            frame = source_frame.copy(deep=True)
            for field in ("role", "tail_role"):
                if field in frame and not frame[field].isin(ROLES).all():
                    raise ValueError("unknown role")
            for field in ("identity", "tail_identity", "identity_level"):
                if field in frame and not frame[field].isin(("ALL", *IDENTITY_LEVELS)).all():
                    raise ValueError("unknown identity")
            for column, value in spec["filters"].items():
                if column not in frame.columns:
                    raise ValueError(f"unknown filter column: {column}")
                frame = frame.loc[frame[column] == value].copy(deep=True)
        else:
            frame = pd.read_parquet(source) if source.suffix == ".parquet" else pd.read_csv(source)
            frame = frame.copy(deep=True)
        for column, values in spec.get("include", {}).items():
            if column not in frame.columns:
                raise ValueError(f"unknown filter column: {column}")
            frame = frame.loc[frame[column].isin(values)].copy(deep=True)
        for column, mapping in spec.get("value_map", {}).items():
            if column not in frame.columns:
                raise ValueError(f"missing column: {column}")
            frame[column] = frame[column].replace(mapping)
        if frame.empty:
            raise ValueError(f"empty selection for {spec['stem']}")
        columns = list(spec["columns"])
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise ValueError(f"missing column: {missing[0]}")
        sort_columns = [column for column in columns if column in frame.columns]
        frame = frame.sort_values(sort_columns, kind="stable").loc[:, columns].reset_index(drop=True)
        if len(frame) != spec["expected_rows"]:
            raise ValueError(f"unexpected row count for {spec['stem']}: {len(frame)}")
        products[str(spec["stem"])] = {
            "frame": frame,
            "tex": render_table(frame, columns, str(spec["caption"]), str(spec["label"])),
            "csv": frame.to_csv(index=False),
            "source": str(spec["source"]),
            "source_sha256": sha256_file(source),
            "filters": dict(spec["filters"]),
        }
    return products


def _figure_events(events: pd.DataFrame) -> pd.DataFrame:
    required = ("instrument", "WMSCI", "MSCI", "spoofing_compatible_episode")
    missing = [column for column in required if column not in events.columns]
    if missing:
        raise ValueError(f"missing field: {missing[0]}")
    frame = events.loc[:, required].copy(deep=True)
    if not frame["instrument"].isin(INSTRUMENTS).all():
        raise ValueError("unknown instrument")
    wmsci = pd.to_numeric(frame["WMSCI"], errors="coerce")
    if not wmsci.map(math.isfinite).all() or (wmsci < 0).any():
        raise ValueError("WMSCI must be finite and nonnegative")
    frame["WMSCI"] = wmsci
    frame["MSCI"] = pd.to_numeric(frame["MSCI"], errors="coerce")
    if not frame["spoofing_compatible_episode"].isin((True, False)).all() and not frame["spoofing_compatible_episode"].isna().all() and not frame["spoofing_compatible_episode"].dropna().isin((True, False)).all():
        raise ValueError("spoofing_compatible_episode must be boolean")
    frame["spoofing_compatible_episode"] = frame["spoofing_compatible_episode"].map(lambda value: bool(value) if pd.notna(value) else False)
    return frame


def _save_figure(figure, out: Path, name: str) -> None:
    paths = [out / f"{name}.{extension}" for extension in ("pdf", "png")]
    if any(path.exists() for path in paths):
        raise FileExistsError(next(path for path in paths if path.exists()))
    try:
        for path in paths:
            figure.savefig(path, dpi=220, bbox_inches="tight")
    finally:
        plt.close(figure)


def _plot_ecdf(events: pd.DataFrame, out: Path) -> None:
    figure, axis = plt.subplots(figsize=(6.3, 3.5), layout="constrained")
    for instrument in INSTRUMENTS:
        group = events.loc[events["instrument"] == instrument]
        positive = np.sort(group.loc[group["WMSCI"] > 0, "WMSCI"].to_numpy())
        if len(positive):
            axis.step(
                np.log1p(positive), np.arange(1, len(positive) + 1) / len(positive),
                where="post", label=f"{instrument}: positive={len(positive)}, zero-W={int((group['WMSCI'] == 0).sum())}",
            )
    axis.set(
        xlabel="log(1 + WMSCI), dimensionless",
        ylabel="Empirical CDF conditional on WMSCI > 0",
        ylim=(0, 1.02),
    )
    if axis.lines:
        axis.legend(frameon=False, fontsize=10, loc="lower right", title="Series counts", title_fontsize=10)
    _save_figure(figure, out, "wmsci_ecdf")


def _plot_score_space(events: pd.DataFrame, out: Path, *, strict: bool) -> None:
    finite = events.loc[np.isfinite(events["MSCI"])].copy(deep=True)
    ymax = max(1.0, float(np.log1p(events["WMSCI"].max())))
    figure, axes = plt.subplots(3, 1, figsize=(6.3, 7.4), layout="constrained", sharey=True)
    collections = []
    for axis, instrument in zip(axes, INSTRUMENTS):
        group = finite.loc[finite["instrument"] == instrument]
        zero_count = int(((events["instrument"] == instrument) & (events["WMSCI"] == 0)).sum())
        missing_count = int(((events["instrument"] == instrument) & ~np.isfinite(events["MSCI"])).sum())
        if strict:
            axis.scatter(
                group["MSCI"], np.log1p(group["WMSCI"]), s=10, color="0.35",
                rasterized=True, label="All executions with finite MSCI",
            )
            selected = group.loc[group["spoofing_compatible_episode"]]
            axis.scatter(
                selected["MSCI"], np.log1p(selected["WMSCI"]), s=26,
                facecolors="none", edgecolors="#D55E00", linewidths=1.1, rasterized=True,
                label="Strict episode members",
            )
            plotted = len(group)
        else:
            selected = group.loc[group["WMSCI"] > 0]
            plotted = len(selected)
            if plotted:
                collections.append(axis.hexbin(
                    selected["MSCI"], np.log1p(selected["WMSCI"]), gridsize=15,
                    extent=(-1, 2, 0, ymax), mincnt=1, cmap="viridis",
                ))
        strict_count = int(group["spoofing_compatible_episode"].sum()) if strict else None
        subtitle = f"finite MSCI={len(group)}; strict={strict_count}" if strict else f"finite MSCI={len(group)}; positive={plotted}"
        # Display padding prevents marker clipping; the common hex-bin extent
        # above remains unchanged, so this is not a new statistical binning.
        axis.set(title=f"{instrument}\n{subtitle}", xlabel="MSCI", xlim=(-1.08, 2.08), ylim=(-0.04 * ymax, 1.04 * ymax))
        axis.tick_params(labelsize=11)
    axes[0].set_ylabel("log(1 + WMSCI)", fontsize=12)
    if strict:
        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="outside upper center", ncol=1, frameon=False, fontsize=10)
    elif collections:
        maximum = max(float(collection.get_array().max()) for collection in collections)
        norm = LogNorm(vmin=1, vmax=max(2.0, maximum))
        for collection in collections:
            collection.set_norm(norm)
        figure.colorbar(collections[0], ax=list(axes), label="Executions per common bin")

    _save_figure(figure, out, "strict_score_space" if strict else "msci_wmsci_joint")


def _price_violin_groups(events: pd.DataFrame, bins: pd.DataFrame) -> dict[tuple[str, str, int], np.ndarray]:
    """Recover frozen candidate-present bins and reconcile all observed samples."""
    prices = ("FPM_ticks", "REV_ticks", "ADV_ticks")
    required = ("instrument", "execution_cluster_id", "P", "WMSCI", *prices)
    missing = [column for column in required if column not in events.columns]
    if missing:
        raise ValueError(f"missing field: {missing[0]}")
    frame = events.loc[:, required].copy(deep=True)
    if not frame["instrument"].isin(INSTRUMENTS).all():
        raise ValueError("unknown instrument")
    if frame["execution_cluster_id"].isna().any():
        raise ValueError("missing price execution identity")
    if frame.duplicated(["instrument", "execution_cluster_id"]).any():
        raise ValueError("duplicate price execution")
    for field in ("P", "WMSCI", *prices):
        frame[field] = pd.to_numeric(frame[field], errors="raise")
    if not np.isfinite(frame[["P", "WMSCI"]].to_numpy()).all() or (frame[["P", "WMSCI"]] < 0).any(axis=None):
        raise ValueError("P and WMSCI must be finite and nonnegative")
    if np.isinf(frame.loc[:, prices].to_numpy()).any():
        raise ValueError("price values must be finite or missing")
    summaries = _validate_price_bins(bins)
    pooled = summaries.loc[
        (summaries["role"] == "ALL") & (summaries["identity"] == "ALL")
        & (summaries["score"] == "WMSCI") & summaries["price"].isin(prices)
    ].set_index(["instrument", "price", "bin"])
    groups = {}
    for instrument in INSTRUMENTS:
        candidate = frame.loc[(frame["instrument"] == instrument) & (frame["P"] > 0)].copy()
        positive = candidate.loc[candidate["WMSCI"] > 0, "WMSCI"].to_numpy()
        # Match the frozen Phase-4 convention: instrument-level cut points,
        # right-closed bins, and tied endpoints removed before price filtering.
        interior = np.unique(np.quantile(positive, [0.2, 0.4, 0.6, 0.8])) if len(positive) else np.array([])
        if len(positive):
            interior = interior[(interior > positive.min()) & (interior < positive.max())]
        labels = pd.cut(candidate["WMSCI"], np.r_[-np.inf, interior, np.inf], labels=False, include_lowest=True) + 1
        candidate["bin"] = labels.where(candidate["WMSCI"] > 0, 0)
        for price in prices:
            for bin_number in range(6):
                member = candidate.loc[candidate["bin"] == bin_number, price].to_numpy(dtype=float)
                values = member[np.isfinite(member)]
                summary = pooled.loc[(instrument, price, bin_number)]
                if len(values) != int(summary["n_observed"]):
                    raise ValueError(f"price sample summary mismatch: {instrument} {price} {bin_number} count")
                if len(values):
                    observed = np.quantile(values, [0.25, 0.5, 0.75])
                    expected = summary[["q25", "median", "q75"]].to_numpy(dtype=float)
                    if not np.allclose(observed, expected, rtol=1e-10, atol=1e-10):
                        raise ValueError(f"price sample summary mismatch: {instrument} {price} {bin_number} quartiles")
                groups[(instrument, price, bin_number)] = values
    return groups


def _plot_price_violins(events: pd.DataFrame, bins: pd.DataFrame, out: Path) -> None:
    """Plot observed distributions, not densities inferred from summary quartiles."""
    groups = _price_violin_groups(events, bins)
    prices = ("FPM_ticks", "REV_ticks", "ADV_ticks")
    colors = ("#0072B2", "#D55E00", "#009E73")
    figure, axes = plt.subplots(3, 3, figsize=(9.2, 8.6), layout="constrained")
    for row, (price, color) in enumerate(zip(prices, colors)):
        for col, instrument in enumerate(INSTRUMENTS):
            axis = axes[row, col]
            medians, positions = [], []
            for bin_number in range(6):
                values = groups[(instrument, price, bin_number)]
                if not len(values):
                    continue  # Missing REV at W=0 is not a zero response.
                medians.append(float(np.median(values)))
                positions.append(bin_number)
                if len(values) >= 5 and values.min() < values.max():
                    parts = axis.violinplot(
                        [values], positions=[bin_number], widths=0.78,
                        showmeans=False, showmedians=False, showextrema=False,
                        bw_method="scott", points=200,
                    )
                    for body in parts["bodies"]:
                        body.set_facecolor(color)
                        body.set_edgecolor(color)
                        body.set_alpha(0.65)
                        body.set_linewidth(0.7)
                        body.set_gid("observed-density")
                else:
                    constant = len(values) >= 5 and values.min() == values.max()
                    ticks = axis.hlines(
                        values, bin_number - 0.13, bin_number + 0.13,
                        color=color, linewidth=1.4, zorder=2,
                    )
                    ticks.set_gid("constant-observations" if constant else "sparse-observations")
            if positions:
                lines = axis.hlines(medians, np.asarray(positions) - 0.25, np.asarray(positions) + 0.25,
                                    color="black", linewidth=2.0, zorder=3)
                lines.set_gid("observed-medians")
            axis.axhline(0, color="0.5", linestyle=":", linewidth=0.8, zorder=0)
            n_observed = sum(len(groups[(instrument, price, b)]) for b in range(6))
            axis.set(
                title=f"{instrument.title()}\nObserved n={n_observed:,}",
                ylabel=f"{price.removesuffix('_ticks')} (ticks)",
                xticks=range(6), xticklabels=["0", "1", "2", "3", "4", "5"],
                xlim=(-0.6, 5.6),
            )
            axis.title.set_fontsize(16)
            axis.yaxis.label.set_fontsize(15)
            axis.xaxis.label.set_fontsize(14)
            axis.tick_params(labelsize=14)
            axis.margins(y=0.08)
            if row == 2:
                axis.set_xlabel("WMSCI group", fontsize=14)
    figure.supxlabel("0: zero W with candidates; 1–5: positive-score quintiles", fontsize=15)
    _save_figure(figure, out, "price_diagnostics")


def make_figures(events: pd.DataFrame, bins: pd.DataFrame, out: Path) -> list[str]:
    """Write four non-overwriting descriptive figure pairs to an existing directory."""
    out = Path(out)
    if not out.is_dir():
        raise ValueError("figure output directory must exist")
    names: list[str] = list(PACKAGE_FIGURES)
    for name in names:
        for extension in ("pdf", "png"):
            path = out / f"{name}.{extension}"
            if path.exists():
                raise FileExistsError(path)
    frame = _figure_events(events)
    _price_violin_groups(events, bins)  # Validate samples before creating any files.
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 12, "axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11})
    _plot_ecdf(frame, out)
    _plot_score_space(frame, out, strict=False)
    _plot_score_space(frame, out, strict=True)
    _plot_price_violins(events, bins, out)
    return names


def _canonical_digest(value: Mapping[str, str]) -> str:
    """Return the stable public identity for a complete validated hash map."""
    return hashlib.sha256(json.dumps(dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _source_hash(root: Path, hashes: Mapping[str, str], relative: str) -> None:
    if relative not in hashes:
        raise ValueError(f"consumed source missing from input hashes: {relative}")
    if sha256_file(_safe_file(root, relative)) != hashes[relative]:
        raise ValueError(f"consumed source hash mismatch: {relative}")


def _validate_price_bins(bins: pd.DataFrame) -> pd.DataFrame:
    required = ("instrument", "role", "identity", "score", "price", "bin", "n_observed", "median", "q25", "q75")
    missing = [column for column in required if column not in bins.columns]
    if missing:
        raise ValueError(f"missing field: {missing[0]}")
    frame = bins.copy(deep=True)
    if not frame["instrument"].isin(INSTRUMENTS).all():
        raise ValueError("unknown instrument")
    if not frame["role"].isin(ROLES).all():
        raise ValueError("unknown role")
    if not frame["identity"].isin(("ALL", *IDENTITY_LEVELS)).all():
        raise ValueError("unknown identity")
    pooled = frame.loc[(frame["role"] == "ALL") & (frame["identity"] == "ALL") & (frame["score"] == "WMSCI") & frame["price"].isin(("FPM_ticks", "REV_ticks", "ADV_ticks"))].copy()
    raw_bins = pooled["bin"]
    numeric_bins = pd.to_numeric(raw_bins, errors="coerce")
    if (
        raw_bins.map(lambda value: isinstance(value, (bool, np.bool_))).any()
        or not numeric_bins.map(math.isfinite).all()
        or not (numeric_bins == np.floor(numeric_bins)).all()
        or not numeric_bins.isin(range(6)).all()
    ):
        raise ValueError("bin must be finite integers from 0 to 5")
    pooled["bin"] = numeric_bins.astype(int)
    expected = {(instrument, price, bin_number) for instrument in INSTRUMENTS for price in ("FPM_ticks", "REV_ticks", "ADV_ticks") for bin_number in range(6)}
    actual = {(row.instrument, row.price, int(row.bin)) for row in pooled.itertuples(index=False)}
    if actual != expected or len(pooled) != len(expected):
        raise ValueError("price-bin pooled strata are incomplete or duplicated")
    observed = pd.to_numeric(pooled["n_observed"], errors="coerce")
    if not observed.map(math.isfinite).all() or not (observed >= 0).all() or not (observed == np.floor(observed)).all():
        raise ValueError("n_observed must be nonnegative integers")
    values = pooled.loc[:, ["median", "q25", "q75"]].apply(pd.to_numeric, errors="coerce")
    has_observed = observed > 0
    if not np.isfinite(values.loc[has_observed].to_numpy(dtype=float)).all():
        raise ValueError("observed price quartiles must be finite")
    if not values.loc[~has_observed].isna().all(axis=None):
        raise ValueError("unobserved price quartiles must be missing")
    if not ((values.loc[has_observed, "q25"] <= values.loc[has_observed, "median"]) & (values.loc[has_observed, "median"] <= values.loc[has_observed, "q75"])).all():
        raise ValueError("price quartiles are not ordered")
    rev_zero = pooled.loc[(pooled["price"] == "REV_ticks") & (pooled["bin"] == 0)]
    if len(rev_zero) != len(INSTRUMENTS) or not (pd.to_numeric(rev_zero["n_observed"]) == 0).all() or not rev_zero[["median", "q25", "q75"]].isna().all(axis=None):
        raise ValueError("REV W=0 must remain legitimately unobserved")
    return frame


def validate_caches(root: Path) -> dict[str, object]:
    """Read and validate the immutable publication cache without creating output."""
    root = Path(root)
    hashes = verify_inputs(root)
    events_relative = "astra_metric_audit/tables/phase4_price_events.parquet"
    bins_relative = "astra_metric_audit/tables/phase4_price_bins.csv"
    _source_hash(root, hashes, events_relative)
    _source_hash(root, hashes, bins_relative)
    try:
        events = pd.read_parquet(root / events_relative)
    except Exception as exc:
        raise ValueError("unable to read phase4 price events") from exc
    try:
        bins = pd.read_csv(root / bins_relative)
    except Exception as exc:
        raise ValueError("unable to read phase4 price bins") from exc
    if population_census(events) != BASELINE_CENSUS:
        raise ValueError("baseline population census mismatch")
    _figure_events(events)
    bins = _validate_price_bins(bins)
    _price_violin_groups(events, bins)
    tables = build_tables(root)
    for product in tables.values():
        _source_hash(root, hashes, str(product["source"]))
        if product["source_sha256"] != hashes[str(product["source"])]:
            raise ValueError("table source hash mismatch")
    return {"hashes": hashes, "events": events, "bins": bins, "tables": tables}


def _expected_inventory() -> set[str]:
    tables = {f"{spec['stem']}.{extension}" for spec in table_specs() for extension in ("tex", "csv")}
    figures = {f"{name}.{extension}" for name in PACKAGE_FIGURES for extension in ("pdf", "png")}
    return tables | figures


def _output_hashes(output: Path) -> dict[str, str]:
    output = Path(output)
    return {path.name: sha256_file(path) for path in sorted(output.iterdir()) if path.is_file() and path.name != "metadata.json"}


def public_metadata(outputs: Path | Mapping[str, str], input_hashes: Mapping[str, str]) -> dict[str, object]:
    """Produce the strict, path-free metadata contract for public package verification."""
    output_hashes = _output_hashes(Path(outputs)) if isinstance(outputs, Path) else dict(outputs)
    if set(output_hashes) != _expected_inventory() or any(not _DIGEST_RE.fullmatch(digest) for digest in output_hashes.values()):
        raise ValueError("package inventory is incomplete or malformed")
    if not isinstance(input_hashes, Mapping) or not input_hashes or any(not isinstance(path, str) or not path or not isinstance(digest, str) or not _DIGEST_RE.fullmatch(digest) for path, digest in input_hashes.items()):
        raise ValueError("input hashes are malformed")
    return {
        "schema": "withdrawal_characterization_public_package_v1",
        "publication_contract": publication_contract(),
        "input_identity_digest": _canonical_digest(input_hashes),
        "generator_sha256": sha256_file(Path(__file__)),
        "price_rendering": {
            "rows": ["FPM_ticks", "REV_ticks", "ADV_ticks"],
            "columns": list(INSTRUMENTS),
            "cohort": "candidate_present_pooled_roles_and_identity_levels",
            "score_bins": "instrument_frozen_positive_WMSCI_quintiles_with_zero_separate",
            "kde_bandwidth": "scott",
            "kde_points": 200,
            "normalization": "equal_maximum_width_per_violin",
            "median": "observed_sample_horizontal_line",
            "independent_y_axes": True,
            "full_observed_range": True,
            "minimum_kde_samples": 5,
            "degenerate_samples": "observed_horizontal_marks_without_KDE",
            "missing_prices": "omitted_without_imputation",
            "zero_REV_imputed": False,
        },
        "outputs": dict(sorted(output_hashes.items())),
    }


def _validate_products(tables: Mapping[str, Mapping[str, object]]) -> None:
    specs = {str(spec["stem"]): spec for spec in table_specs()}
    if set(tables) != set(specs):
        raise ValueError("table products do not match registry")
    for stem, spec in specs.items():
        product = tables[stem]
        frame = product.get("frame")
        if not isinstance(frame, pd.DataFrame) or list(frame.columns) != list(spec["columns"]):
            raise ValueError(f"invalid table product: {stem}")
        if product.get("tex") != render_table(frame, list(spec["columns"]), str(spec["caption"]), str(spec["label"])):
            raise ValueError(f"invalid table TeX: {stem}")
        if product.get("csv") != frame.to_csv(index=False):
            raise ValueError(f"invalid table CSV: {stem}")


def write_package(out: Path, tables: Mapping[str, Mapping[str, object]], events: pd.DataFrame, bins: pd.DataFrame, input_hashes: Mapping[str, str]) -> dict[str, object]:
    """Create one complete package, refusing every existing target before any input work."""
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    if not isinstance(input_hashes, Mapping) or not input_hashes or any(not isinstance(path, str) or not path or not isinstance(digest, str) or not _DIGEST_RE.fullmatch(digest) for path, digest in input_hashes.items()):
        raise ValueError("input hashes are malformed")
    _validate_products(tables)
    _figure_events(events)
    _validate_price_bins(bins)
    _price_violin_groups(events, bins)

    serialized = {stem: (str(product["tex"]), str(product["csv"])) for stem, product in tables.items()}
    out.mkdir(parents=True)
    for stem, (tex, csv) in serialized.items():
        (out / f"{stem}.tex").write_text(tex)
        (out / f"{stem}.csv").write_text(csv)
    make_figures(events, bins, out)
    metadata = public_metadata(out, input_hashes)
    (out / "metadata.json").write_text(json.dumps(metadata, sort_keys=True, indent=2, allow_nan=False) + "\n")
    return metadata


def verify_package(out: Path, input_hashes: Mapping[str, str]) -> dict[str, object]:
    """Verify the fixed inventory, hashes, generator identity, and metadata contract."""
    out = Path(out)
    if not out.is_dir():
        raise ValueError("package directory is missing")
    contents = list(out.rglob("*"))
    if any(path.is_symlink() for path in contents):
        raise ValueError("package contains symbolic link")
    if any(path.is_dir() for path in contents):
        raise ValueError("package inventory mismatch")
    actual = {path.name for path in contents if path.is_file()}
    if actual != _expected_inventory() | {"metadata.json"}:
        raise ValueError("package inventory mismatch")
    try:
        metadata = json.loads((out / "metadata.json").read_text(), parse_constant=lambda _: (_ for _ in ()).throw(ValueError("non-finite metadata")))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("malformed metadata") from exc
    expected = public_metadata(out, input_hashes)
    if metadata != expected:
        if metadata.get("generator_sha256") != expected["generator_sha256"]:
            raise ValueError("generator identity mismatch")
        raise ValueError("metadata contract mismatch")
    for name, digest in metadata["outputs"].items():
        if sha256_file(out / name) != digest:
            raise ValueError(f"output hash mismatch: {name}")
    return metadata


def main(argv=None, *, root: Path = ROOT) -> int:
    """Validate, generate, or independently verify a cache-only public package."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--verify-package", action="store_true")
    args = parser.parse_args(argv)
    if not args.verify_package and args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    checked = validate_caches(Path(root))
    if args.validate_only:
        print("PUBLICATION_INPUTS_VALIDATED")
        return 0
    if args.verify_package:
        verify_package(args.output_dir, checked["hashes"])
        print("PUBLICATION_PACKAGE_VERIFIED")
        return 0
    write_package(args.output_dir, checked["tables"], checked["events"], checked["bins"], checked["hashes"])
    verify_package(args.output_dir, checked["hashes"])
    print("PUBLICATION_PACKAGE_COMPLETE")
    return 0


if __name__ == "__main__":
    main()
