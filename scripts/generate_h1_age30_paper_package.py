#!/usr/bin/env python3
"""Generate a fresh descriptive paper package from validated H1 age-30 bundles.

The generator is intentionally cache-free: it reads only the three persisted v3
bundles, validates each one with the native bundle validator, enforces the H1
age-30 metadata contract, and derives every publication aggregate directly
from the detected execution and episode artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

from scripts.validate_spoofing_detection_bundle import (
    V3_CLOCK_CONTRACT,
    V3_CLOCK_POLICY_ID,
    V3_OUTPUT_SCHEMA_VERSION,
    validate_bundle as native_validate_bundle,
)

ROOT = Path(__file__).resolve().parents[1]
INSTRUMENTS = ("FERRARI", "NEXI", "RISANAMENTO")
DEFAULT_BUNDLE_ROOT = ROOT / "outputs" / "spoofing_detection_h1_checkpointed_20261007_231228" / "runs"
PACKAGE_SCHEMA = "h1_age30_paper_package_v1"
FRACTION = 0.05
QUANTITY_TOLERANCE = 1e-9
EXPECTED = {
    "FERRARI": {"executions": 12997, "candidate_episodes": 3309, "strict_episodes": 108},
    "NEXI": {"executions": 10512, "candidate_episodes": 2196, "strict_episodes": 83},
    "RISANAMENTO": {"executions": 24972, "candidate_episodes": 1075, "strict_episodes": 2},
}
REQUIRED_EVENT_COLUMNS = (
    "execution_cluster_id", "episode_id", "identity_level", "actor_key", "event_ts",
    "candidate_deceptive_visible_qty_pre", "execution_quantity",
    "matched_deceptive_cancel_visible_qty_window",
    "matched_deceptive_cancel_fraction_window", "withdrawal_to_execution_ratio",
    "WMSCI_event", "MSCI", "favorable_mid_move_pre_fill",
    "post_cancel_mid_reversion", "execution_price_advantage_vs_posture_mid",
    "spoofing_compatible_episode", "episode_strict_detection", "episode_mixed_anchor",
)
REQUIRED_EPISODE_COLUMNS = (
    "episode_id", "identity_level", "mixed_anchor", "spoofing_compatible_episode",
)
TABLE_CAPTIONS = {
    "population": "Execution-cluster population and candidate-posture episode census. Candidate members are execution clusters with positive pre-execution candidate deceptive visible quantity; candidate episodes are separately counted posture episodes. Client and firm columns report the mutually exclusive resolved attribution level. W is WMSCI_event.",
    "components": "Spearman correlations among positive-score executions. R is withdrawal divided by executed quantity; A is the share of candidate quantity removed; Q is executed quantity. W denotes WMSCI. Correlations are descriptive; executions sharing orders are not independent replications.",
    "tail_overlap": "Exact top-5 percent overlap on the positive-W execution-cluster population. Each score tail has k=ceil(0.05 n_positive_W) members and ties are resolved by execution_cluster_id in ascending lexical order after descending score.",
    "price_associations": "Spearman correlations of positive W with observed price diagnostics in ticks. FPM is favourable midprice movement before the fill; REV is post-cancellation midprice reversion; ADV is execution-price advantage versus posture midprice. Pair counts are observed finite pairs.",
    "joint_price": "Among the deterministic exact top-5 percent positive-W execution clusters, the observed denominator has finite FPM and REV. The numerator has both FPM and REV strictly positive.",
    "strict": "Strict candidate-posture episodes and their execution-cluster members. Episode counts use strict episode representatives; member counts include all execution clusters in these episodes. Zero-W is evaluated among strict members and mixed episodes are counted once per episode.",
    "recurrence": "Recurrence among deterministic exact top-5 percent positive-W execution clusters. Actor and date counts are aggregate-only; no source actor identifiers are exported.",
    "strict_tail": "Membership of the deterministic exact top-5 percent positive-W execution-cluster tail in strict candidate-posture episodes. Strict episode counts are distinct selected episodes, while strict member executions are selected execution clusters.",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_path(bundle: Path, metadata: Mapping[str, Any], name: str) -> Path:
    value = metadata.get("paths", {}).get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"missing artifact path: {name}")
    candidate = Path(value)
    candidate = candidate if candidate.is_absolute() else bundle / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(bundle.resolve())
    except ValueError as exc:
        raise ValueError(f"artifact path escapes bundle: {name}") from exc
    if not resolved.is_file():
        raise ValueError(f"missing artifact file: {name}")
    return resolved


def _require_exact(metadata: Mapping[str, Any], key: str, value: object) -> None:
    if metadata.get(key) != value:
        raise ValueError(f"metadata contract mismatch: {key}")


def validate_bundle_contract(bundle: Path) -> dict[str, Any]:
    """Natively validate one bundle then require the fixed H1 age-30 contract."""
    bundle = Path(bundle).resolve()
    native_report = native_validate_bundle(bundle)
    try:
        metadata = json.loads((bundle / "metadata.json").read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("bundle metadata is malformed") from exc
    if not isinstance(metadata, dict):
        raise ValueError("bundle metadata is malformed")
    for key, value in (
        ("output_schema_version", V3_OUTPUT_SCHEMA_VERSION),
        ("clock_policy_id", V3_CLOCK_POLICY_ID),
        ("clock_contract", V3_CLOCK_CONTRACT),
        ("window_seconds", 1.0),
        ("max_deceptive_order_age_seconds", 30.0),
        ("top_n", 10),
        ("withdrawal_window_seconds", 2.0),
        ("reversion_horizon_seconds", 2.0),
        ("execution_cluster_max_gap_ms", 100),
    ):
        _require_exact(metadata, key, value)
    tick_size = metadata.get("tick_size")
    if not isinstance(tick_size, (int, float)) or not math.isfinite(float(tick_size)) or tick_size <= 0:
        raise ValueError("metadata contract mismatch: tick_size")
    for artifact in ("execution_metrics", "candidate_episodes"):
        _artifact_path(bundle, metadata, artifact)
    return {"native": native_report, "metadata": metadata, "metadata_sha256": sha256_file(bundle / "metadata.json")}


def _validate_events(frame: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_EVENT_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"execution metrics missing column: {missing[0]}")
    if frame.empty or frame["execution_cluster_id"].isna().any() or frame["execution_cluster_id"].duplicated().any():
        raise ValueError("execution_cluster_id must be present and unique")
    if not frame["identity_level"].isin(("client_original", "firm")).all():
        raise ValueError("unknown identity_level")
    for field in ("candidate_deceptive_visible_qty_pre", "execution_quantity", "matched_deceptive_cancel_visible_qty_window", "WMSCI_event"):
        values = pd.to_numeric(frame[field], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"invalid nonnegative numeric field: {field}")
    if not (pd.to_numeric(frame["execution_quantity"], errors="coerce") > 0).all():
        raise ValueError("execution_quantity must be positive")
    p = frame['candidate_deceptive_visible_qty_pre'].to_numpy(dtype=float)
    u = frame['matched_deceptive_cancel_visible_qty_window'].to_numpy(dtype=float)
    q = frame['execution_quantity'].to_numpy(dtype=float)
    a = frame['matched_deceptive_cancel_fraction_window'].to_numpy(dtype=float)
    r = frame['withdrawal_to_execution_ratio'].to_numpy(dtype=float)
    if not np.isnan(a[p == 0]).all() or not np.allclose(a[p > 0], u[p > 0] / p[p > 0], rtol=1e-12, atol=1e-12):
        raise ValueError('removal share must equal U/P when P>0 and be missing when P=0')
    if not np.isfinite(r).all() or not np.allclose(r, u / q, rtol=1e-12, atol=1e-12):
        raise ValueError('withdrawal ratio must equal U/Q')
    msci = pd.to_numeric(frame["MSCI"], errors="coerce")
    if (frame["MSCI"].notna() & ~np.isfinite(msci)).any():
        raise ValueError("MSCI must be finite or missing")
    for field in ("spoofing_compatible_episode", "episode_strict_detection", "episode_mixed_anchor"):
        if (frame[field].isna() & frame['episode_id'].notna()).any() or not frame[field].dropna().isin((True, False)).all():
            raise ValueError(f"{field} must be boolean on episode members")
        # Unmapped executions have no episode, hence cannot belong to a strict one.
        frame[field] = frame[field].eq(True)


def validate_quantity_only_w(frame: pd.DataFrame) -> float:
    """Check WMSCI_event against the H1 quantity-only formula, row by row."""
    required = (
        "candidate_deceptive_visible_qty_pre", "execution_quantity",
        "matched_deceptive_cancel_visible_qty_window", "WMSCI_event",
    )
    if any(column not in frame.columns for column in required):
        raise ValueError("quantity-only W formula columns are missing")
    p = pd.to_numeric(frame[required[0]], errors="coerce").to_numpy(dtype=float)
    q = pd.to_numeric(frame[required[1]], errors="coerce").to_numpy(dtype=float)
    u = pd.to_numeric(frame[required[2]], errors="coerce").to_numpy(dtype=float)
    observed = pd.to_numeric(frame[required[3]], errors="coerce").to_numpy(dtype=float)
    if not (np.isfinite(p).all() and np.isfinite(q).all() and np.isfinite(u).all() and np.isfinite(observed).all()):
        raise ValueError("quantity-only W formula requires finite values")
    expected = np.zeros_like(observed)
    active = (p > 0) & (u > 0)
    expected[active] = np.log1p(p[active] / q[active]) * np.log1p(u[active] / q[active]) * (u[active] / p[active])
    error = float(np.max(np.abs(observed - expected), initial=0.0))
    if error > QUANTITY_TOLERANCE:
        raise ValueError(f"quantity-only W formula mismatch: max_abs_error={error:.3g}")
    return error


def exact_positive_top_k(frame: pd.DataFrame, score: str, *, fraction: float = FRACTION) -> tuple[pd.DataFrame, int]:
    """Return deterministic exact-k positive-score tail with an ID tie break."""
    if not (0 < fraction <= 1) or not math.isfinite(fraction):
        raise ValueError("tail fraction must be in (0, 1]")
    if score not in frame or "execution_cluster_id" not in frame:
        raise ValueError("tail score or execution_cluster_id missing")
    positive = frame.loc[pd.to_numeric(frame[score], errors="coerce") > 0].copy()
    if positive.empty:
        return positive, 0
    if positive["execution_cluster_id"].isna().any() or positive["execution_cluster_id"].duplicated().any():
        raise ValueError("tail requires unique execution_cluster_id")
    values = pd.to_numeric(positive[score], errors="coerce")
    if not np.isfinite(values).all():
        raise ValueError("tail score must be finite")
    k = int(math.ceil(fraction * len(positive)))
    selected = positive.assign(_score=values).sort_values(
        ["_score", "execution_cluster_id"], ascending=[False, True], kind="stable"
    ).head(k).drop(columns="_score").reset_index(drop=True)
    return selected, k


def _spearman(left: pd.Series, right: pd.Series) -> tuple[int, float | None]:
    paired = pd.DataFrame({"left": pd.to_numeric(left, errors="coerce"), "right": pd.to_numeric(right, errors="coerce")}).dropna()
    n = len(paired)
    if n < 2 or paired["left"].nunique() < 2 or paired["right"].nunique() < 2:
        return n, None
    return n, float(paired["left"].corr(paired["right"], method="spearman"))


def _attribution_name(value: str) -> str:
    return {"client_original": "Client", "firm": "Firm"}.get(value, value)


def strict_summary(instrument: str, events: pd.DataFrame, candidate_episodes: pd.DataFrame) -> pd.DataFrame:
    """Reconcile strict representative episodes, all members, and attribution strata."""
    needed_events = {"episode_id", "identity_level", "WMSCI_event", "spoofing_compatible_episode", "episode_strict_detection", "episode_mixed_anchor"}
    needed_episodes = {"episode_id", "identity_level", "mixed_anchor", "spoofing_compatible_episode"}
    if needed_events - set(events) or needed_episodes - set(candidate_episodes):
        raise ValueError("strict reconciliation columns are missing")
    certified = candidate_episodes.loc[candidate_episodes["spoofing_compatible_episode"], list(needed_episodes)].copy()
    if certified["episode_id"].isna().any() or certified["episode_id"].duplicated().any():
        raise ValueError("strict candidate episodes must have unique IDs")
    members = events.loc[events["spoofing_compatible_episode"]].copy()
    representatives = events.loc[events["episode_strict_detection"]].copy()
    member_ids = set(members["episode_id"].dropna())
    certified_ids = set(certified["episode_id"])
    representative_ids = set(representatives["episode_id"].dropna())
    if member_ids != certified_ids or representative_ids != certified_ids or len(representatives) != len(certified):
        raise ValueError("strict episode membership does not reconcile across artifacts")
    episode_identity = certified.set_index("episode_id")["identity_level"].to_dict()
    if any(episode_identity.get(episode) != identity for episode, identity in zip(members["episode_id"], members["identity_level"])):
        raise ValueError("strict member attribution does not reconcile")
    if any(episode_identity.get(episode) != identity for episode, identity in zip(representatives["episode_id"], representatives["identity_level"])):
        raise ValueError("strict representative attribution does not reconcile")

    rows: list[dict[str, object]] = []
    for identity in ("ALL", "client_original", "firm"):
        episodes = certified if identity == "ALL" else certified.loc[certified["identity_level"] == identity]
        ids = set(episodes["episode_id"])
        selected_members = members.loc[members["episode_id"].isin(ids)]
        rows.append({
            "instrument": instrument,
            "attribution": identity if identity == "ALL" else _attribution_name(identity),
            "strict_episodes": int(len(episodes)),
            "strict_member_executions": int(len(selected_members)),
            "strict_zero_W_members": int((pd.to_numeric(selected_members["WMSCI_event"], errors="coerce") == 0).sum()),
            "strict_mixed_episodes": int(episodes["mixed_anchor"].sum()),
        })
    return pd.DataFrame(rows)


def _read_bundle(instrument: str, bundle: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    checked = validate_bundle_contract(bundle)
    metadata = checked["metadata"]
    events_path = _artifact_path(bundle, metadata, "execution_metrics")
    episodes_path = _artifact_path(bundle, metadata, "candidate_episodes")
    events = pl.read_parquet(events_path, columns=list(REQUIRED_EVENT_COLUMNS)).to_pandas()
    episodes = pl.read_parquet(episodes_path, columns=list(REQUIRED_EPISODE_COLUMNS)).to_pandas()
    _validate_events(events)
    max_abs_error = validate_quantity_only_w(events)
    expected = EXPECTED[instrument]
    strict_count = int(episodes["spoofing_compatible_episode"].sum())
    if len(events) != expected["executions"] or len(episodes) != expected["candidate_episodes"] or strict_count != expected["strict_episodes"]:
        raise ValueError(f"expected H1 census mismatch for {instrument}")
    checked["execution_metrics_sha256"] = sha256_file(events_path)
    checked["candidate_episodes_sha256"] = sha256_file(episodes_path)
    checked["max_quantity_only_w_abs_error"] = max_abs_error
    return events, episodes, checked


def _population(instrument: str, events: pd.DataFrame, episodes: pd.DataFrame) -> dict[str, object]:
    return {
        "instrument": instrument,
        "executions": int(len(events)),
        "candidate_member_executions": int((events["candidate_deceptive_visible_qty_pre"] > 0).sum()),
        "candidate_episodes": int(len(episodes)),
        "positive_W_executions": int((events["WMSCI_event"] > 0).sum()),
        "missing_MSCI_executions": int(events["MSCI"].isna().sum()),
        "client_executions": int((events["identity_level"] == "client_original").sum()),
        "firm_executions": int((events["identity_level"] == "firm").sum()),
    }


def _components(instrument: str, events: pd.DataFrame) -> pd.DataFrame:
    positive = events.loc[events["WMSCI_event"] > 0]
    components = {
        "A": "matched_deceptive_cancel_fraction_window",
        "R": "withdrawal_to_execution_ratio",
        "Q": "execution_quantity",
    }
    rows = []
    for label, field in components.items():
        n, rho = _spearman(positive["WMSCI_event"], positive[field])
        rows.append({"instrument": instrument, "component": label, "component_definition": {"A": "matched-cancel fraction", "R": "withdrawal/executed quantity", "Q": "executed quantity"}[label], "n_pairs": n, "spearman_W_component": rho})
    return pd.DataFrame(rows)


def _tail_overlap(instrument: str, events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    top_w, k = exact_positive_top_k(events, "WMSCI_event")
    w_ids = set(top_w["execution_cluster_id"])
    positive = events.loc[events["WMSCI_event"] > 0]
    rows = []
    for label, field in {"A": "matched_deceptive_cancel_fraction_window", "R": "withdrawal_to_execution_ratio", "Q": "execution_quantity"}.items():
        top_component, component_k = exact_positive_top_k(positive, field)
        if component_k != k:
            raise ValueError("tail k differs across positive-W components")
        rows.append({"instrument": instrument, "component": label, "positive_W_executions": int(len(positive)), "k_exact_ceil_5pct": k, "overlap_count_exact_k": int(len(w_ids & set(top_component["execution_cluster_id"])),), "component_tail_tie_break": "execution_cluster_id ascending"})
    return pd.DataFrame(rows), top_w, k


def _price_associations(instrument: str, events: pd.DataFrame, tick_size: float) -> pd.DataFrame:
    positive = events.loc[events["WMSCI_event"] > 0]
    rows = []
    for label, field in {
        "FPM_ticks": "favorable_mid_move_pre_fill",
        "REV_ticks": "post_cancel_mid_reversion",
        "ADV_ticks": "execution_price_advantage_vs_posture_mid",
    }.items():
        values = pd.to_numeric(positive[field], errors="coerce") / tick_size
        n, rho = _spearman(positive["WMSCI_event"], values)
        rows.append({"instrument": instrument, "price_diagnostic": label, "n_observed_pairs": n, "spearman_W_price_ticks": rho})
    return pd.DataFrame(rows)


def _joint_price(instrument: str, tail: pd.DataFrame, tick_size: float) -> dict[str, object]:
    fpm = pd.to_numeric(tail["favorable_mid_move_pre_fill"], errors="coerce") / tick_size
    rev = pd.to_numeric(tail["post_cancel_mid_reversion"], errors="coerce") / tick_size
    observed = np.isfinite(fpm) & np.isfinite(rev)
    return {
        "instrument": instrument,
        "top5_selected_executions": int(len(tail)),
        "joint_FPM_REV_observed": int(observed.sum()),
        "joint_positive_FPM_and_REV": int(((fpm > 0) & (rev > 0) & observed).sum()),
    }


def _recurrence(instrument: str, tail: pd.DataFrame) -> dict[str, object]:
    date = pd.to_datetime(tail["event_ts"], errors="coerce").dt.date
    # Actor identifiers remain internal and are reduced to aggregate recurrence counts.
    enriched = tail.assign(_date=date)
    per_actor_episode = enriched.dropna(subset=["actor_key", "episode_id"]).groupby("actor_key")["episode_id"].nunique()
    per_actor_day = enriched.dropna(subset=["actor_key", "_date"]).groupby("actor_key")["_date"].nunique()
    per_day = enriched.groupby("_date", dropna=True).size()
    return {
        "instrument": instrument,
        "top5_selected_executions": int(len(tail)),
        "selected_distinct_episodes": int(tail["episode_id"].nunique(dropna=True)),
        "selected_days": int(date.nunique(dropna=True)),
        "days_with_ge2_selected_executions": int((per_day >= 2).sum()),
        "actors_with_ge2_selected_episodes": int((per_actor_episode >= 2).sum()),
        "actors_with_ge2_selected_days": int((per_actor_day >= 2).sum()),
    }


def _strict_tail(instrument: str, tail: pd.DataFrame) -> dict[str, object]:
    strict_members = tail.loc[tail["spoofing_compatible_episode"]]
    return {
        "instrument": instrument,
        "top5_selected_executions": int(len(tail)),
        "top5_strict_member_executions": int(len(strict_members)),
        "top5_strict_distinct_episodes": int(strict_members["episode_id"].nunique(dropna=True)),
    }


def tex_escape(value: object) -> str:
    substitutions = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(substitutions.get(char, char) for char in str(value)).replace("≥", r"\ensuremath{\geq}")


def _display(value: object) -> str:
    if value is None or pd.isna(value):
        return "NA"
    if isinstance(value, (float, np.floating)):
        return format(float(value), ".4g")
    if isinstance(value, bool):
        return "Yes" if value else "No"
    return str(value)


def render_table(frame: pd.DataFrame, caption: str, label: str) -> str:
    if frame.empty:
        raise ValueError("cannot render empty table")
    aliases = {
        'instrument': 'Instrument', 'executions': 'All executions',
        'candidate_member_executions': 'Candidate-present executions',
        'candidate_episodes': 'Candidate episodes', 'positive_W_executions': 'Positive W',
        'missing_MSCI_executions': 'Missing MSCI', 'client_executions': 'Client executions',
        'firm_executions': 'Firm executions', 'component': 'Component', 'n_pairs': 'Pairs',
        'spearman_W_component': 'Spearman', 'k_exact_ceil_5pct': 'Selected',
        'overlap_count_exact_k': 'Overlap', 'price_diagnostic': 'Price',
        'n_observed_pairs': 'Observed pairs', 'spearman_W_price_ticks': 'Spearman',
        'top5_selected_executions': 'Selected executions', 'joint_FPM_REV_observed': 'Both observed',
        'joint_positive_FPM_and_REV': 'Both positive', 'attribution': 'Attribution',
        'strict_episodes': 'Episodes', 'strict_member_executions': 'Members',
        'strict_zero_W_members': 'Zero-W members', 'strict_mixed_episodes': 'Mixed roles',
        'selected_distinct_episodes': 'Distinct episodes', 'selected_days': 'Days',
        'days_with_ge2_selected_executions': 'Days with repeat executions',
        'actors_with_ge2_selected_episodes': 'Actors with repeat episodes',
        'actors_with_ge2_selected_days': 'Actors with repeat days',
        'top5_strict_member_executions': 'Strict members',
        'top5_strict_distinct_episodes': 'Strict episodes',
    }
    shown = frame.drop(columns=['component_definition', 'component_tail_tie_break'], errors='ignore').copy()
    if 'executions' in shown.columns:
        shown = shown.set_index('instrument').T.reset_index().rename(columns={'index': 'Measure'})
        shown['Measure'] = shown['Measure'].map(aliases)
    shown = shown.rename(columns=lambda c: aliases.get(c, c.title()))
    shown = shown.replace({'FERRARI': 'Ferrari', 'NEXI': 'Nexi', 'RISANAMENTO': 'Risanamento',
                           'FPM_ticks': 'FPM', 'REV_ticks': 'REV', 'ADV_ticks': 'ADV'})
    columns = list(shown.columns)
    align = r'>{\raggedright\arraybackslash}X' + r'>{\raggedleft\arraybackslash}X' * (len(columns) - 1)
    header = ' & '.join(tex_escape(c) for c in columns)
    body = [' & '.join(tex_escape(_display(value)) for value in row) + r' \\' for row in shown.itertuples(index=False, name=None)]
    return '\n'.join([
        r'\begin{table}[htbp]', r'\centering', f'\\caption{{{tex_escape(caption)}}}', f'\\label{{{label}}}',
        r'\small', r'\setlength{\tabcolsep}{4pt}', r'\renewcommand{\arraystretch}{1.15}',
        f'\\begin{{tabularx}}{{\\textwidth}}{{{align}}}', r'\toprule', header + r' \\', r'\midrule',
        *body, r'\bottomrule', r'\end{tabularx}', r'\end{table}', '',
    ])


def _save_figure(figure: plt.Figure, output: Path, stem: str) -> None:
    figure.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    figure.savefig(output / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def _plot_ecdf(all_events: Mapping[str, pd.DataFrame], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2), layout="constrained")
    for instrument, events in all_events.items():
        values = np.sort(events.loc[events["WMSCI_event"] > 0, "WMSCI_event"].to_numpy(dtype=float))
        ax.step(np.log1p(values), np.arange(1, len(values) + 1) / len(values), where="post", label=f"{instrument} (n={len(values):,})")
    ax.set(xlabel="log(1 + WMSCI)", ylabel="Cumulative fraction of positive scores", ylim=(0, 1.02))
    ax.legend(frameon=False)
    _save_figure(fig, out, "wmsci_ecdf")


def _plot_msci_joint(all_events: Mapping[str, pd.DataFrame], out: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.4), sharex=True, layout="constrained")
    for ax, (instrument, events) in zip(axes, all_events.items()):
        finite = events.loc[events["MSCI"].notna()]
        ax.scatter(finite["MSCI"], np.log1p(finite["WMSCI_event"]), s=7, alpha=0.35, rasterized=True)
        ax.set_title(f"{instrument}: finite MSCI={len(finite):,}", loc="left")
        ax.set_ylabel("log(1 + W)")
    axes[-1].set_xlabel("MSCI")
    _save_figure(fig, out, "msci_wmsci_joint")


def _plot_price_diagnostics(all_events: Mapping[str, pd.DataFrame], tick_sizes: Mapping[str, float], out: Path) -> None:
    diagnostics = (("FPM", "favorable_mid_move_pre_fill"), ("REV", "post_cancel_mid_reversion"), ("ADV", "execution_price_advantage_vs_posture_mid"))
    fig, axes = plt.subplots(3, 3, figsize=(10.2, 8.6), layout="constrained")
    for col, (instrument, events) in enumerate(all_events.items()):
        positive = events.loc[events["WMSCI_event"] > 0]
        for row, (label, field) in enumerate(diagnostics):
            ax = axes[row, col]
            y = pd.to_numeric(positive[field], errors="coerce") / tick_sizes[instrument]
            good = np.isfinite(y)
            ax.scatter(np.log1p(positive.loc[good, "WMSCI_event"]), y.loc[good], s=7, alpha=0.35, rasterized=True)
            ax.axhline(0, color="0.5", linewidth=0.7, linestyle=":")
            ax.set_title(instrument if row == 0 else "")
            ax.set_ylabel(f"{label} (ticks)")
            if row == 2:
                ax.set_xlabel("log(1 + W)")
    _save_figure(fig, out, "price_diagnostics")


def _plot_strict_score_space(all_events: Mapping[str, pd.DataFrame], out: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.4), sharex=True, sharey=True, layout="constrained")
    for ax, (instrument, events) in zip(axes, all_events.items()):
        finite = events.loc[events["MSCI"].notna()]
        strict = finite.loc[finite["spoofing_compatible_episode"]]
        ax.scatter(finite["MSCI"], np.log1p(finite["WMSCI_event"]), s=6, color="0.55", alpha=0.35, rasterized=True, label="All finite-MSCI executions")
        ax.scatter(strict["MSCI"], np.log1p(strict["WMSCI_event"]), s=25, facecolors="none", edgecolors="#D55E00", linewidths=1.0, label="Strict episode members")
        ax.set_title(instrument, loc="left")
        ax.set_ylabel("log(1 + W)")
    axes[-1].set_xlabel("MSCI")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, loc="upper right", frameon=False, fontsize=9)
    _save_figure(fig, out, "strict_score_space")


def _expected_inventory() -> set[str]:
    table_stems = set(TABLE_CAPTIONS)
    figures = {f"{stem}.{extension}" for stem in ("wmsci_ecdf", "msci_wmsci_joint", "price_diagnostics", "strict_score_space") for extension in ("pdf", "png")}
    tables = {f"{stem}.{extension}" for stem in table_stems for extension in ("csv", "tex")}
    return tables | figures | {"summary.json"}


def _output_hashes(output: Path) -> dict[str, str]:
    return {path.name: sha256_file(path) for path in sorted(output.iterdir()) if path.is_file() and path.name != "metadata.json"}


def build_package_data(bundle_root: Path) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, Any], dict[str, float]]:
    all_events: dict[str, pd.DataFrame] = {}
    all_episodes: dict[str, pd.DataFrame] = {}
    validations: dict[str, Any] = {}
    tick_sizes: dict[str, float] = {}
    for instrument in INSTRUMENTS:
        events, episodes, validated = _read_bundle(instrument, Path(bundle_root) / instrument)
        all_events[instrument] = events
        all_episodes[instrument] = episodes
        validations[instrument] = {
            "bundle": str((Path(bundle_root) / instrument).resolve()),
            "metadata_sha256": validated["metadata_sha256"],
            "execution_metrics_sha256": validated["execution_metrics_sha256"],
            "candidate_episodes_sha256": validated["candidate_episodes_sha256"],
            "native_artifact_count": validated["native"]["artifact_count"],
            "max_quantity_only_w_abs_error": validated["max_quantity_only_w_abs_error"],
        }
        tick_sizes[instrument] = float(validated["metadata"]["tick_size"])
    return all_events, all_episodes, validations, tick_sizes


def derive_tables(all_events: Mapping[str, pd.DataFrame], all_episodes: Mapping[str, pd.DataFrame], tick_sizes: Mapping[str, float]) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    products: dict[str, list[Any]] = {stem: [] for stem in TABLE_CAPTIONS}
    summary: dict[str, Any] = {"instruments": {}}
    for instrument in INSTRUMENTS:
        events = all_events[instrument]
        episodes = all_episodes[instrument]
        population = _population(instrument, events, episodes)
        components = _components(instrument, events)
        overlaps, tail, k = _tail_overlap(instrument, events)
        price = _price_associations(instrument, events, tick_sizes[instrument])
        joint = _joint_price(instrument, tail, tick_sizes[instrument])
        strict = strict_summary(instrument, events, episodes)
        recurrence = _recurrence(instrument, tail)
        strict_tail = _strict_tail(instrument, tail)
        products["population"].append(population)
        products["components"].append(components)
        products["tail_overlap"].append(overlaps)
        products["price_associations"].append(price)
        products["joint_price"].append(joint)
        products["strict"].append(strict)
        products["recurrence"].append(recurrence)
        products["strict_tail"].append(strict_tail)
        summary["instruments"][instrument] = {
            "executions": population["executions"], "candidate_episodes": population["candidate_episodes"],
            "positive_W_executions": population["positive_W_executions"], "tail_k_exact_ceil_5pct": k,
            "strict_episodes": int(strict.loc[strict["attribution"] == "ALL", "strict_episodes"].item()),
            "strict_member_executions": int(strict.loc[strict["attribution"] == "ALL", "strict_member_executions"].item()),
            "joint_positive_FPM_and_REV": joint["joint_positive_FPM_and_REV"],
            "joint_FPM_REV_observed": joint["joint_FPM_REV_observed"],
        }
    tables: dict[str, pd.DataFrame] = {}
    for stem, pieces in products.items():
        if pieces and isinstance(pieces[0], pd.DataFrame):
            tables[stem] = pd.concat(pieces, ignore_index=True)
        else:
            tables[stem] = pd.DataFrame(pieces)
        tables[stem] = tables[stem].sort_values(list(tables[stem].columns), kind="stable").reset_index(drop=True)
    return tables, summary


def write_package(output: Path, bundle_root: Path) -> dict[str, Any]:
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    all_events, all_episodes, validations, tick_sizes = build_package_data(bundle_root)
    tables, summary = derive_tables(all_events, all_episodes, tick_sizes)
    output.mkdir(parents=True)
    try:
        for stem, frame in tables.items():
            frame.to_csv(output / f"{stem}.csv", index=False)
            (output / f"{stem}.tex").write_text(render_table(frame, TABLE_CAPTIONS[stem], f"tab:h1-age30-{stem.replace('_', '-')}") )
        summary.update({
            "schema": PACKAGE_SCHEMA,
            "tail_rule": "positive W only; k=ceil(0.05*n_positive_W); descending score then ascending execution_cluster_id",
            "quantity_only_W": "log1p(P/Q)*log1p(U/Q)*(U/P), where P=candidate_deceptive_visible_qty_pre, U=matched_deceptive_cancel_visible_qty_window, Q=execution_quantity",
            "price_units": "FPM, REV and ADV divided by per-instrument metadata tick_size",
        })
        (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
        _plot_ecdf(all_events, output)
        _plot_msci_joint(all_events, output)
        _plot_price_diagnostics(all_events, tick_sizes, output)
        _plot_strict_score_space(all_events, output)
        outputs = _output_hashes(output)
        if set(outputs) != _expected_inventory():
            raise ValueError("generated package inventory mismatch")
        metadata = {
            "schema": PACKAGE_SCHEMA,
            "generator_sha256": sha256_file(Path(__file__)),
            "contract": {
                "window_seconds": 1.0, "max_deceptive_order_age_seconds": 30.0, "top_n": 10,
                "withdrawal_window_seconds": 2.0, "reversion_horizon_seconds": 2.0,
                "execution_cluster_max_gap_ms": 100, "v3_output_schema_version": V3_OUTPUT_SCHEMA_VERSION,
                "v3_clock_policy_id": V3_CLOCK_POLICY_ID,
            },
            "native_bundle_validation": validations,
            "tick_sizes": tick_sizes,
            "outputs": outputs,
        }
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n")
        return metadata
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise


def verify_package(output: Path, bundle_root: Path) -> dict[str, Any]:
    output = Path(output)
    if not output.is_dir() or any(path.is_dir() or path.is_symlink() for path in output.iterdir()):
        raise ValueError("package directory inventory is invalid")
    if {path.name for path in output.iterdir()} != _expected_inventory() | {"metadata.json"}:
        raise ValueError("package inventory mismatch")
    try:
        metadata = json.loads((output / "metadata.json").read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("package metadata malformed") from exc
    if metadata.get("schema") != PACKAGE_SCHEMA or metadata.get("generator_sha256") != sha256_file(Path(__file__)):
        raise ValueError("package metadata contract mismatch")
    expected_contract = {"window_seconds": 1.0, "max_deceptive_order_age_seconds": 30.0, "top_n": 10, "withdrawal_window_seconds": 2.0, "reversion_horizon_seconds": 2.0, "execution_cluster_max_gap_ms": 100, "v3_output_schema_version": V3_OUTPUT_SCHEMA_VERSION, "v3_clock_policy_id": V3_CLOCK_POLICY_ID}
    if metadata.get("contract") != expected_contract:
        raise ValueError("package metadata contract mismatch")
    all_events, all_episodes, validations, tick_sizes = build_package_data(bundle_root)
    tables, summary = derive_tables(all_events, all_episodes, tick_sizes)
    for stem, frame in tables.items():
        if (output / f"{stem}.csv").read_text() != frame.to_csv(index=False):
            raise ValueError(f"table mismatch: {stem}.csv")
        expected_tex = render_table(frame, TABLE_CAPTIONS[stem], f"tab:h1-age30-{stem.replace('_', '-')}")
        if (output / f"{stem}.tex").read_text() != expected_tex:
            raise ValueError(f"table mismatch: {stem}.tex")
    summary.update({"schema": PACKAGE_SCHEMA, "tail_rule": "positive W only; k=ceil(0.05*n_positive_W); descending score then ascending execution_cluster_id", "quantity_only_W": "log1p(P/Q)*log1p(U/Q)*(U/P), where P=candidate_deceptive_visible_qty_pre, U=matched_deceptive_cancel_visible_qty_window, Q=execution_quantity", "price_units": "FPM, REV and ADV divided by per-instrument metadata tick_size"})
    if json.loads((output / "summary.json").read_text()) != summary:
        raise ValueError("summary mismatch")
    if metadata.get("native_bundle_validation") != validations or metadata.get("tick_sizes") != tick_sizes:
        raise ValueError("input validation metadata mismatch")
    actual_hashes = _output_hashes(output)
    if metadata.get("outputs") != actual_hashes:
        raise ValueError("output hash mismatch")
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, default=DEFAULT_BUNDLE_ROOT)
    parser.add_argument("--verify-package", action="store_true")
    args = parser.parse_args(argv)
    if args.verify_package:
        verify_package(args.output_dir, args.bundle_root)
        print("H1_AGE30_PAPER_PACKAGE_VERIFIED")
    else:
        write_package(args.output_dir, args.bundle_root)
        verify_package(args.output_dir, args.bundle_root)
        print("H1_AGE30_PAPER_PACKAGE_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
