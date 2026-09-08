"""Shared episode artifact contract for standard and depth-grid producers."""
from __future__ import annotations

import hashlib
import platform
from pathlib import Path

import polars as pl

from spoofing_detection.lob.candidate_episodes import EPISODE_VERSION, EpisodeResult

EPISODE_ARTIFACTS = {
    'candidate_episodes': 'episodes',
    'episode_cluster_members': 'members',
    'episode_withdrawals': 'withdrawals',
    'episode_anchor_summary': 'anchor_summary',
    'actor_day_episode_summary': 'actor_day_summary',
}
ANALYSIS_SEMANTICS_VERSION = 'posture_episodes_placement_day_cluster_smallness_v1'


def episode_metadata() -> dict:
    return {
        'analysis_semantics_version': ANALYSIS_SEMANTICS_VERSION,
        'episode_semantics': {
            'primary_unit': 'candidate_posture_episode',
            'definition': EPISODE_VERSION,
            'grouping': ['partition_id', 'actor_key', 'event_date', 'execution_side', 'shared_candidate_lifecycle_component'],
            'cluster_strict_flags': 'diagnostics_not_independent_episode_detections',
            'fpm_baseline': 'exact_post_latest_candidate_placement_before_first_fill_complete_placement_required',
            'fpm_aggregation': 'execution_quantity_weighted_all_members_complete_coverage',
            'reversion_aggregation': 'quantity_delay_weighted_unique_withdrawals_complete_coverage',
            'withdrawal_accounting': 'unique_physical_cancellation_once_per_episode',
            'smallness': 'sum_unique_withdrawal_quantity_gt_sum_all_member_execution_quantity',
            'branch_summary': 'participation_counts_nonadditive_for_mixed_anchor_episodes; quantities_branch_specific',
            'inference': 'descriptive_only; episodes_and_actors_within_day_not_assumed_independent',
            'day_boundary': 'trading_partition_and_normalized_recorded_event_date; no_cross_day_horizons',
        },
        'execution_role_provenance': {
            'verification_status': 'unverified_upstream_mapping',
            'role_fields': ['PASSIVEORDER', 'AGGRESSIVEORDER'],
            'classification': 'exclusive_Y_flags',
            'meaning': 'role_of_the_order_and_actor_represented_by_the_row',
            'required_evidence': 'source_field_mapping_export_transform_and_sample_trade_reconciliation',
        },
        'software_versions': {'python': platform.python_version(), 'polars': pl.__version__},
    }


def scientific_source_hashes() -> dict[str, str]:
    directory = Path(__file__).parent
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(directory.glob('*.py'))}


def episode_frames(result: EpisodeResult) -> dict[str, pl.DataFrame]:
    return {name: getattr(result, attribute) for name, attribute in EPISODE_ARTIFACTS.items()}


def write_episode_artifacts(result: EpisodeResult, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, frame in episode_frames(result).items():
        paths[name] = output_dir / f'{name}.parquet'
        frame.write_parquet(paths[name])
    return paths
