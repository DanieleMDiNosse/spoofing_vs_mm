"""Shared episode artifact contract for standard and depth-grid producers."""
from __future__ import annotations

import hashlib
import platform
from pathlib import Path

import polars as pl

from spoofing_detection.lob.candidate_episodes import (
    ANCHOR_SCHEMA,
    DAY_SCHEMA,
    EPISODE_SCHEMA,
    EPISODE_VERSION,
    LIFECYCLE_SCHEMA,
    MEMBER_SCHEMA,
    WITHDRAWAL_SCHEMA,
    EpisodeResult,
)

EPISODE_ARTIFACTS = {
    'candidate_episodes': 'episodes',
    'episode_cluster_members': 'members',
    'episode_withdrawals': 'withdrawals',
    'candidate_lifecycles': 'candidate_lifecycles',
    'episode_anchor_summary': 'anchor_summary',
    'actor_day_episode_summary': 'actor_day_summary',
}
ANALYSIS_SEMANTICS_VERSION = 'posture_episodes_supported_lifecycle_cancellation_v2'


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
            'withdrawal_accounting': 'unique_physical_cancellation_once_per_episode',
            'primary_exposure_quantity': 'sum_lifecycle_maximum_admissible_pre_cancel_snapshot_quantity_not_simultaneous_depth',
            'primary_execution_quantity': 'sum_all_member_execution_quantity; nonpositive_is_undefined',
            'primary_cancellation_quantity': 'supported_unique_physical_cancellation_quantity_capped_once_per_lifecycle_by_primary_exposure; raw_quantity_audit_only',
            'primary_withdrawal_to_exposure_ratio': 'A=U/P; defined_only_for_positive_P; bounded_in_[0,1]',
            'primary_score': 'log1p(P/Q)*log1p(U/Q)*(U/P); undefined_for_nonpositive_Q',
            'reversion_aggregation': 'supported_cancellation_quantity_weighted_unique_withdrawals_complete_coverage; not_winner_attribution_weighted',
            'smallness': 'sum_supported_withdrawal_quantity_gt_sum_all_member_execution_quantity',
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


def validate_episode_result(result: EpisodeResult) -> None:
    """Fail closed on the serialized episode score/audit contract."""
    expected = {
        'episodes': EPISODE_SCHEMA,
        'members': MEMBER_SCHEMA,
        'withdrawals': WITHDRAWAL_SCHEMA,
        'candidate_lifecycles': LIFECYCLE_SCHEMA,
        'anchor_summary': ANCHOR_SCHEMA,
        'actor_day_summary': DAY_SCHEMA,
    }
    for attribute, schema in expected.items():
        if getattr(result, attribute).schema != schema:
            raise ValueError(f'episode artifact schema mismatch: {attribute}')

    episodes = result.episodes
    lifecycles = result.candidate_lifecycles
    invalid_ratio = episodes.filter(
        pl.col('primary_withdrawal_to_lifecycle_exposure_ratio').is_not_null()
        & (
            (pl.col('primary_withdrawal_to_lifecycle_exposure_ratio') < 0)
            | (pl.col('primary_withdrawal_to_lifecycle_exposure_ratio') > 1)
        )
    )
    if not invalid_ratio.is_empty():
        raise ValueError('episode primary A=U/P must be in [0, 1]')
    invalid_lifecycle = lifecycles.filter(
        (pl.col('supported_unique_cancellation_quantity') < 0)
        | (pl.col('supported_unique_cancellation_quantity') > pl.col('lifecycle_union_exposure_quantity'))
        | (pl.col('supported_unique_cancellation_quantity') > pl.col('raw_unique_cancellation_quantity'))
    )
    if not invalid_lifecycle.is_empty():
        raise ValueError('lifecycle supported cancellation exceeds admissible exposure or raw audit quantity')
    if episodes.is_empty():
        return
    lifecycle_totals = lifecycles.group_by('episode_id').agg(
        pl.col('lifecycle_union_exposure_quantity').sum().alias('_exposure'),
        pl.col('supported_unique_cancellation_quantity').sum().alias('_supported'),
    )
    reconciliation = episodes.join(lifecycle_totals, on='episode_id', how='left').filter(
        (pl.col('primary_lifecycle_union_exposure_quantity') != pl.col('_exposure'))
        | (pl.col('primary_supported_unique_cancellation_quantity') != pl.col('_supported'))
    )
    if not reconciliation.is_empty():
        raise ValueError('episode primary quantity does not reconcile to candidate lifecycles')


def write_episode_artifacts(result: EpisodeResult, output_dir: Path) -> dict[str, Path]:
    validate_episode_result(result)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, frame in episode_frames(result).items():
        paths[name] = output_dir / f'{name}.parquet'
        frame.write_parquet(paths[name])
    return paths
