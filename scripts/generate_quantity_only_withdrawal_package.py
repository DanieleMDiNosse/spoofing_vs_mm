"""Derive a quantity-only paper package from frozen cached sufficient statistics.

This never modifies the historical audit cache or its original package.  It replaces
only the elapsed-time discount in WMSCI and cancellation-anchored REV; all eligibility,
assignment, horizons, candidate age, depth, clustering and selection conventions are
read from the frozen cache unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from scripts import generate_withdrawal_characterization_paper as render

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "astra_metric_audit" / "tables"
OLD = ROOT / "paper" / "generated" / "withdrawal_characterization_v1"
INSTRUMENTS = ("FERRARI", "NEXI", "RISANAMENTO")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score(frame: pd.DataFrame) -> pd.Series:
    valid = (frame.P > 0) & (frame.Q > 0) & (frame.U > 0)
    value = pd.Series(0.0, index=frame.index)
    value.loc[valid] = (
        np.log1p(frame.loc[valid, "P"] / frame.loc[valid, "Q"])
        * np.log1p(frame.loc[valid, "U"] / frame.loc[valid, "Q"])
        * (frame.loc[valid, "U"] / frame.loc[valid, "P"])
    )
    return value


def qmean(frame: pd.DataFrame, value: str) -> float:
    observed = frame.loc[frame.has_cancel_reversion_state & np.isfinite(frame[value])].copy()
    if len(observed) != len(frame) or observed.empty:
        return np.nan
    return np.average(observed[value], weights=observed.attributed_cancel_visible_qty)


def rank(frame: pd.DataFrame, fraction: float = .05) -> pd.DataFrame:
    positive = frame.loc[frame.WMSCI > 0]
    return positive.sort_values(["WMSCI", "event_key"], ascending=[False, True]).head(max(1, int(np.ceil(fraction * len(positive)))))


def corr(x: pd.Series, y: pd.Series) -> float:
    paired = pd.DataFrame({"x": x, "y": y}).dropna()
    return float(spearmanr(paired.x, paired.y).statistic) if len(paired) > 1 and paired.x.nunique() > 1 and paired.y.nunique() > 1 else np.nan


def table(out: Path, stem: str, frame: pd.DataFrame, columns: list[str], caption: str, label: str) -> None:
    selected = frame.loc[:, columns].sort_values(columns, kind="stable").reset_index(drop=True)
    (out / f"{stem}.csv").write_text(selected.to_csv(index=False))
    (out / f"{stem}.tex").write_text(render.render_table(selected, columns, caption, label))


def quantity_events() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    events = pd.read_parquet(AUDIT / "phase4_price_events.parquet").copy()
    variants = pd.read_parquet(AUDIT / "phase9_variant_W_scores.parquet")
    current = variants.loc[variants.spec == "no_decay", ["event_key", "W_recomputed", "U_recomputed", "WU_recomputed"]]
    if len(current) != len(events) or current.event_key.duplicated().any():
        raise ValueError("quantity-only score cache is incomplete")
    events = events.merge(current, on="event_key", validate="one_to_one")
    events["U"] = events.U_recomputed
    events["weighted_U"] = events.U_recomputed
    events["weighted_R"] = events.U_recomputed / events.Q
    events["delay_factor"] = np.where(events.U_recomputed > 0, 1.0, np.nan)
    events["WMSCI"] = events.W_recomputed
    if not np.allclose(events.WMSCI, score(events), rtol=1e-12, atol=1e-12):
        raise ValueError("quantity-only WMSCI identity failed")
    cancels = pd.read_parquet(AUDIT / "phase7_cancel_reversion_components.parquet").copy()
    grouped = cancels.groupby("execution_cluster_id", sort=False).apply(lambda x: qmean(x, "post_cancel_mid_reversion"), include_groups=False).rename("REV").reset_index()
    events = events.drop(columns=["REV"]).merge(grouped, on="execution_cluster_id", how="left", validate="one_to_one")
    events["REV_ticks"] = events.REV / events.tick_size
    episodes = pd.read_parquet(AUDIT / "phase5_episode_gate_map.parquet").copy()
    episode_rev = cancels.groupby("episode_id", sort=False).apply(lambda x: qmean(x, "post_cancel_mid_reversion"), include_groups=False).rename("post_cancel_mid_reversion").reset_index()
    episodes = episodes.drop(columns=["post_cancel_mid_reversion"]).merge(episode_rev, on="episode_id", how="left", validate="one_to_one")
    episodes["quantity_only_strict"] = (
        (episodes.withdrawn_quantity > episodes.total_execution_quantity)
        & np.isfinite(episodes.favorable_mid_move) & np.isfinite(episodes.post_cancel_mid_reversion)
        & (episodes.favorable_mid_move > 0) & (episodes.post_cancel_mid_reversion > 0)
    )
    expected = pd.read_csv(AUDIT / "phase9_episode_gate_stability.csv")
    expected = expected.loc[(expected.spec == "no_decay") & (expected.identity == "ALL"), ["instrument", "strict_episode_count"]]
    observed = episodes.groupby("instrument").quantity_only_strict.sum().reindex(INSTRUMENTS).astype(int)
    if observed.to_dict() != dict(zip(expected.instrument, expected.strict_episode_count)):
        raise ValueError("quantity-only strict gate does not match cached recomputation")
    flags = episodes.set_index("episode_id").quantity_only_strict
    events["spoofing_compatible_episode"] = events.episode_id.map(flags).fillna(False).astype(bool)
    return events, cancels, episodes


def build(out: Path) -> None:
    if out.exists():
        raise FileExistsError(out)
    events, cancels, episodes = quantity_events()
    out.mkdir(parents=True)
    # Timing is mathematically invariant to the removed score/reversion discount.
    for stem in ("population", "timing"):
        for suffix in ("csv", "tex"):
            shutil.copy2(OLD / f"{stem}.{suffix}", out / f"{stem}.{suffix}")
    # Score-dependent descriptive tables.
    components = []
    tails = []
    associations = []
    strict_rows = []
    strict_tail = []
    recurrence = []
    joint = []
    rev_rows = []
    for instrument in INSTRUMENTS:
        f = events.loc[events.instrument == instrument].copy()
        positive = f.loc[f.WMSCI > 0]
        for other in ("R", "A", "Q"):
            source = {"R": positive.R, "A": positive.A, "Q": positive.P / positive.Q}[other]
            components.append(dict(instrument=instrument, other=other, n_pair=len(positive), spearman=corr(positive.WMSCI, source)))
        for fraction in (.01, .05, .1):
            k = max(1, int(np.ceil(fraction * len(positive))))
            selected = positive.sort_values(["WMSCI", "event_key"], ascending=[False, True]).head(k)
            cutoff = selected.WMSCI.min()
            ratio = positive.sort_values(["R", "event_key"], ascending=[False, True]).head(k)
            tails.append(dict(instrument=instrument, fraction=fraction, k=k, overlap_count_exact_k=len(set(selected.event_key) & set(ratio.event_key)), component_tie_inclusive_n=int((positive.R >= ratio.R.min()).sum())))
        for price in ("FPM_ticks", "REV_ticks", "ADV_ticks", "execution_edge_ticks"):
            paired = positive.loc[:, ["WMSCI", price, "actor_key"]].dropna()
            associations.append(dict(instrument=instrument, price=price, n_pair=len(paired), n_actors=paired.actor_key.nunique(), spearman=corr(paired.WMSCI, paired[price])))
        selected = rank(f)
        observed = selected.FPM_ticks.notna() & selected.REV_ticks.notna()
        joint.append(dict(instrument=instrument, n_events=len(selected), n_joint_price_observed=int(observed.sum()), n_favorable_and_reversal=int(((selected.FPM_ticks > 0) & (selected.REV_ticks > 0)).sum())))
        member = selected.spoofing_compatible_episode
        strict_tail.append(dict(instrument=instrument, selected_events=len(selected), strict_member_events=int(member.sum()), selected_episodes=selected.episode_id.dropna().nunique(), strict_episodes=selected.loc[member, "episode_id"].dropna().nunique()))
        recurrence.append(dict(instrument=instrument, selected_events=len(selected), selected_distinct_episodes=selected.episode_id.dropna().nunique(), selected_actors=selected.actor_key.nunique(), actors_with_ge2_selected_episodes=int((selected.groupby("actor_key").episode_id.nunique() >= 2).sum()), actors_with_ge2_selected_days=int((selected.assign(day=pd.to_datetime(selected.cluster_end_ts).dt.date).groupby("actor_key").day.nunique() >= 2).sum())))
        for population, subset in (("ALL", episodes.loc[episodes.instrument == instrument]), ("strict_component", episodes.loc[(episodes.instrument == instrument) & episodes.quantity_only_strict])):
            valid = subset.dropna(subset=["post_cancel_mid_reversion"])
            values = cancels.loc[(cancels.instrument == instrument) & cancels.episode_id.isin(valid.episode_id)]
            if len(valid):
                rev_rows.append(dict(instrument=instrument, population=population, n_complete_decomposition=len(valid), total_REV_ticks_mean=(valid.post_cancel_mid_reversion / events.loc[events.instrument == instrument, "tick_size"].iloc[0]).mean(), immediate_jump_ticks_mean=np.average(values.immediate_jump_ticks.dropna(), weights=values.loc[values.immediate_jump_ticks.notna(), "attributed_cancel_visible_qty"]) if values.immediate_jump_ticks.notna().any() else np.nan, later_path_ticks_mean=np.average(values.later_path_ticks.dropna(), weights=values.loc[values.later_path_ticks.notna(), "attributed_cancel_visible_qty"]) if values.later_path_ticks.notna().any() else np.nan))
    for _, row in episodes.groupby(["instrument", "identity_level"], sort=True):
        first = row.iloc[0]
        strict_rows.append(dict(instrument=first.instrument, identity=first.identity_level, strict_episodes=int(row.quantity_only_strict.sum()), strict_member_events=int(events.loc[(events.instrument == first.instrument) & events.spoofing_compatible_episode & (events.identity_level == first.identity_level), "execution_cluster_id"].nunique()), strict_zero_W_members=int(events.loc[(events.instrument == first.instrument) & events.spoofing_compatible_episode & (events.identity_level == first.identity_level) & (events.WMSCI == 0), "execution_cluster_id"].nunique()), strict_mixed_anchor_episodes=int(row.loc[row.quantity_only_strict, "mixed_anchor"].sum())))
    all_strict = episodes.groupby("instrument").quantity_only_strict.sum()
    for instrument in INSTRUMENTS:
        f = events.loc[(events.instrument == instrument) & events.spoofing_compatible_episode]
        strict_rows.append(dict(instrument=instrument, identity="ALL", strict_episodes=int(all_strict[instrument]), strict_member_events=f.execution_cluster_id.nunique(), strict_zero_W_members=int((f.WMSCI == 0).sum()), strict_mixed_anchor_episodes=int(episodes.loc[(episodes.instrument == instrument) & episodes.quantity_only_strict, "mixed_anchor"].sum())))
    table(out, "components", pd.DataFrame(components), ["instrument", "other", "n_pair", "spearman"], "Quantity-only WMSCI component associations in the positive-score execution population; paired counts are not IID observations.", "tab:withdrawal-components")
    table(out, "tail_overlap", pd.DataFrame(tails), ["instrument", "fraction", "k", "overlap_count_exact_k", "component_tie_inclusive_n"], "Exact-k positive quantity-only WMSCI tail overlap with R by instrument.", "tab:withdrawal-tail-overlap")
    table(out, "price_associations", pd.DataFrame(associations), ["instrument", "price", "n_pair", "n_actors", "spearman"], "Spearman associations between quantity-only WMSCI and observed price diagnostics. Correlations are descriptive, not causal estimates.", "tab:withdrawal-price-associations")
    table(out, "joint_price", pd.DataFrame(joint), ["instrument", "n_events", "n_joint_price_observed", "n_favorable_and_reversal"], "Top-5% quantity-only WMSCI selection with observed native REV.", "tab:withdrawal-joint-price")
    table(out, "strict", pd.DataFrame(strict_rows), ["instrument", "identity", "strict_episodes", "strict_member_events", "strict_zero_W_members", "strict_mixed_anchor_episodes"], "Quantity-only reversion episode screen. Members count executions in selected episodes, not independently selected sequences.", "tab:withdrawal-strict")
    table(out, "strict_tail", pd.DataFrame(strict_tail), ["instrument", "selected_events", "strict_member_events", "selected_episodes", "strict_episodes"], "Exact-k 5% quantity-only WMSCI tail membership in selected episodes.", "tab:withdrawal-strict-tail")
    table(out, "rev_decomposition", pd.DataFrame(rev_rows), ["instrument", "population", "n_complete_decomposition", "total_REV_ticks_mean", "immediate_jump_ticks_mean", "later_path_ticks_mean"], "Episode-grain quantity-weighted native REV decomposition.", "tab:withdrawal-rev-decomposition")
    table(out, "recurrence", pd.DataFrame(recurrence), ["instrument", "selected_events", "selected_distinct_episodes", "selected_actors", "actors_with_ge2_selected_episodes", "actors_with_ge2_selected_days"], "Exact-k 5% quantity-only WMSCI tail recurrence.", "tab:withdrawal-recurrence")
    sensitivity = []
    gates = []
    for instrument in INSTRUMENTS:
        f = events.loc[events.instrument == instrument]
        k = len(rank(f))
        sensitivity.append(dict(instrument=instrument, spec="quantity_only", fixed_k=k, topk_overlap_count=k, union_positive_score_spearman=1.0))
        for identity, sub in episodes.loc[episodes.instrument == instrument].groupby("identity_level"):
            gates.append(dict(instrument=instrument, identity=identity, spec="quantity_only", strict_episode_count=int(sub.quantity_only_strict.sum()), changed_strict_episodes=0, official_gate_comparable=True))
        gates.append(dict(instrument=instrument, identity="ALL", spec="quantity_only", strict_episode_count=int(episodes.loc[episodes.instrument == instrument, "quantity_only_strict"].sum()), changed_strict_episodes=0, official_gate_comparable=True))
    table(out, "score_sensitivity", pd.DataFrame(sensitivity), ["instrument", "spec", "fixed_k", "topk_overlap_count", "union_positive_score_spearman"], "Current quantity-only WMSCI 5% exact-k reference.", "tab:withdrawal-score-sensitivity")
    table(out, "gate_sensitivity", pd.DataFrame(gates), ["instrument", "identity", "spec", "strict_episode_count", "changed_strict_episodes", "official_gate_comparable"], "Current quantity-only strict episode screen by attribution level.", "tab:withdrawal-gate-sensitivity")
    bins = []
    for instrument in INSTRUMENTS:
        f = events.loc[(events.instrument == instrument) & (events.P > 0)].copy()
        positive = f.loc[f.WMSCI > 0, "WMSCI"]
        cuts = np.unique(np.quantile(positive, [.2, .4, .6, .8])) if len(positive) else np.array([])
        cuts = cuts[(cuts > positive.min()) & (cuts < positive.max())] if len(positive) else cuts
        labels = pd.cut(f.WMSCI, np.r_[-np.inf, cuts, np.inf], labels=False, include_lowest=True) + 1
        f["bin"] = labels.where(f.WMSCI > 0, 0).astype(int)
        for price in ("FPM_ticks", "REV_ticks", "ADV_ticks"):
            for b in range(6):
                values = f.loc[f.bin == b, price].dropna()
                bins.append(dict(instrument=instrument, role="ALL", identity="ALL", score="WMSCI", price=price, bin=b, n_observed=len(values), median=values.median() if len(values) else np.nan, q25=values.quantile(.25) if len(values) else np.nan, q75=values.quantile(.75) if len(values) else np.nan))
    render.make_figures(events, pd.DataFrame(bins), out)
    inputs = {str(p.relative_to(ROOT)): digest(p) for p in (AUDIT / "phase4_price_events.parquet", AUDIT / "phase5_episode_gate_map.parquet", AUDIT / "phase7_cancel_reversion_components.parquet", AUDIT / "phase9_variant_W_scores.parquet", AUDIT / "phase9_episode_gate_stability.csv")}
    metadata = {"schema": "withdrawal_characterization_quantity_only_v1", "publication_contract": {"withdrawal_time_discount": "none", "reversion_weight": "attributed_cancellation_quantity", "withdrawal_boundary": "[0,2) seconds", "candidate_age_seconds": [0, 90], "top_n": 10, "structural_horizon_seconds": 10, "reversion_horizon_seconds": 2, "assignment": "latest_prior_cluster_sort_index"}, "inputs": inputs, "generator_sha256": digest(Path(__file__)), "outputs": {p.name: digest(p) for p in sorted(out.iterdir()) if p.is_file()}}
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    build(args.output_dir)
    print("QUANTITY_ONLY_PACKAGE_COMPLETE", args.output_dir)
