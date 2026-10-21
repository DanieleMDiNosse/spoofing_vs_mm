import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import generate_h1_age30_paper_package as package


def _metadata(**overrides):
    metadata = {
        "output_schema_version": package.V3_OUTPUT_SCHEMA_VERSION,
        "clock_policy_id": package.V3_CLOCK_POLICY_ID,
        "clock_contract": package.V3_CLOCK_CONTRACT,
        "window_seconds": 1.0,
        "max_deceptive_order_age_seconds": 30.0,
        "top_n": 10,
        "withdrawal_window_seconds": 2.0,
        "reversion_horizon_seconds": 2.0,
        "execution_cluster_max_gap_ms": 100,
        "tick_size": 0.01,
        "paths": {"execution_metrics": "execution_metrics.parquet", "candidate_episodes": "candidate_episodes.parquet"},
    }
    metadata.update(overrides)
    return metadata


def test_validate_bundle_contract_is_fail_closed_on_h1_age30_policy(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "metadata.json").write_text(json.dumps(_metadata(max_deceptive_order_age_seconds=90.0)))
    monkeypatch.setattr(package, "native_validate_bundle", lambda _bundle: {"artifact_count": 14})

    with pytest.raises(ValueError, match="max_deceptive_order_age_seconds"):
        package.validate_bundle_contract(bundle)


def test_exact_positive_top_k_uses_ceil_and_execution_cluster_id_ties():
    frame = pd.DataFrame(
        {
            "execution_cluster_id": ["z", "b", "a", "c", "d"],
            "WMSCI_event": [9.0, 8.0, 8.0, 8.0, 1.0],
        }
    )

    selected, k = package.exact_positive_top_k(frame, "WMSCI_event", fraction=0.40)

    assert k == 2
    assert selected["execution_cluster_id"].tolist() == ["z", "a"]


def test_quantity_only_score_validation_rejects_nonquantity_w_formula():
    frame = pd.DataFrame(
        {
            "execution_cluster_id": ["x", "y"],
            "candidate_deceptive_visible_qty_pre": [10.0, 0.0],
            "execution_quantity": [5.0, 3.0],
            "matched_deceptive_cancel_visible_qty_window": [4.0, 0.0],
            "WMSCI_event": [np.log1p(2.0) * np.log1p(0.8) * 0.4, 0.0],
        }
    )
    assert package.validate_quantity_only_w(frame) < 1e-12
    frame.loc[0, "WMSCI_event"] += 0.01

    with pytest.raises(ValueError, match="quantity-only W formula"):
        package.validate_quantity_only_w(frame)


def test_strict_summary_reconciles_episode_representatives_and_members():
    events = pd.DataFrame(
        {
            "execution_cluster_id": ["a", "b", "c", "d"],
            "episode_id": ["ep1", "ep1", "ep2", None],
            "identity_level": ["firm", "firm", "client_original", "firm"],
            "WMSCI_event": [0.0, 1.0, 2.0, 0.0],
            "spoofing_compatible_episode": [True, True, True, False],
            "episode_strict_detection": [True, False, True, False],
            "episode_mixed_anchor": [True, True, False, False],
        }
    )
    candidate_episodes = pd.DataFrame(
        {
            "episode_id": ["ep1", "ep2"],
            "identity_level": ["firm", "client_original"],
            "mixed_anchor": [True, False],
            "spoofing_compatible_episode": [True, True],
        }
    )

    result = package.strict_summary("TEST", events, candidate_episodes)

    all_row = result.loc[result["attribution"] == "ALL"].iloc[0]
    assert all_row[["strict_episodes", "strict_member_executions", "strict_zero_W_members", "strict_mixed_episodes"]].tolist() == [2, 3, 1, 1]
    assert result.loc[result["attribution"] == "Firm", "strict_episodes"].item() == 1


def test_strict_summary_rejects_member_episode_not_certified_by_candidate_table():
    events = pd.DataFrame(
        {
            "execution_cluster_id": ["a"], "episode_id": ["missing"], "identity_level": ["firm"],
            "WMSCI_event": [1.0], "spoofing_compatible_episode": [True],
            "episode_strict_detection": [True], "episode_mixed_anchor": [False],
        }
    )
    candidate_episodes = pd.DataFrame(
        {
            "episode_id": ["other"], "identity_level": ["firm"], "mixed_anchor": [False],
            "spoofing_compatible_episode": [True],
        }
    )

    with pytest.raises(ValueError, match="strict episode membership"):
        package.strict_summary("TEST", events, candidate_episodes)


def test_component_labels_preserve_manuscript_denominators():
    frame = pd.DataFrame({
        'WMSCI_event': [1., 2., 3.],
        'matched_deceptive_cancel_fraction_window': [.9, .5, .1],
        'withdrawal_to_execution_ratio': [2., 4., 8.],
        'execution_quantity': [6., 4., 2.],
    })
    result = package._components('TEST', frame).set_index('component')
    assert result.loc['R', 'spearman_W_component'] == pytest.approx(1.)
    assert result.loc['A', 'spearman_W_component'] == pytest.approx(-1.)
    assert result.loc['Q', 'spearman_W_component'] == pytest.approx(-1.)


def test_undefined_removal_share_and_unmapped_episode_are_preserved():
    frame = pd.DataFrame({c: [None, None] for c in package.REQUIRED_EVENT_COLUMNS})
    frame['execution_cluster_id'] = ['a', 'b']
    frame['identity_level'] = ['firm', 'client_original']
    frame['candidate_deceptive_visible_qty_pre'] = [0., 10.]
    frame['execution_quantity'] = [2., 2.]
    frame['matched_deceptive_cancel_visible_qty_window'] = [0., 5.]
    frame['WMSCI_event'] = [0., np.log1p(5) * np.log1p(2.5) * .5]
    frame['matched_deceptive_cancel_fraction_window'] = [np.nan, .5]
    frame['withdrawal_to_execution_ratio'] = [0., 2.5]
    package._validate_events(frame)
    assert np.isnan(frame.loc[0, 'matched_deceptive_cancel_fraction_window'])
    assert not frame['spoofing_compatible_episode'].any()
    wrong = frame.copy()
    wrong.loc[0, 'matched_deceptive_cancel_fraction_window'] = 0.
    with pytest.raises(ValueError, match='removal share'):
        package._validate_events(wrong)


def test_rendered_tables_use_wrapped_headers_without_rescaling():
    text = package.render_table(pd.DataFrame({'instrument': ['FERRARI'], 'strict_episodes': [108]}), 'Counts.', 'tab:counts')
    assert r'\begin{tabularx}{\textwidth}' in text
    assert 'Episodes' in text
    assert 'resizebox' not in text
    assert '108' in text

