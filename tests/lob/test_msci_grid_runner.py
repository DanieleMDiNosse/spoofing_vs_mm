from __future__ import annotations

import copy
import random

import polars as pl
import pytest

from spoofing_detection.lob.msci_grid import sample_execution_grid
from spoofing_detection.lob.msci_grid_runner import (
    compute_partition_grids,
    summarize_grid,
    verify_partition_grids,
)
from spoofing_detection.lob.msci_grid_store import SUMMARY_SCHEMA, TRAJECTORY_SCHEMA


CONFIG = {
    "schema": "msci_grid_study_v1",
    "source_clock_policy": "spoofing_detection_sequence_replay_trade_fill_sequence_cancel_v3_utc_provisional",
    "grid_policy": "fixed_pre_exact_post_execution_closed_grid_v1",
    "grid_step_ns": 10,
    "lag_ns": 10,
    "horizon_ns": 30,
    "complete_grid_required": True,
    "cohort": "exact_top_positive_W_5pct_then_U_over_Q_gt_1_before_prices",
    "grid_sensitivity_ns": [1, 2, 5],
    "lag_sensitivity_ns": [10, 20, 30],
    "lag_sensitivity_grid_ns": 1,
    "selection_threshold": None,
}


def _coverage(*, gap: tuple[int, int] | None = None, end_ns: int = 30) -> dict[str, object]:
    value: dict[str, object] = {
        "segments": [
            {
                "partition_id": "P",
                "start_ns": -10,
                "end_ns": end_ns,
                "attested": True,
                "source": "frozen-producer-witness",
            }
        ]
    }
    if gap is not None:
        value["invalid_intervals"] = [
            {"partition_id": "P", "start_ns": gap[0], "end_ns": gap[1]}
        ]
    return value


def _execution(**changes: object) -> dict[str, object]:
    execution: dict[str, object] = {
        "instrument": "XYZ",
        "partition_id": "P",
        "actor_key": "A",
        "execution_cluster_id": "E1",
        "deceptive_side": "bid",
        "cluster_first_sort_index": 10,
        "cluster_last_sort_index": 10,
        "cluster_end_replay_time_ns": 0,
    }
    execution.update(changes)
    return execution


def _states(*, post_dwi: tuple[float, float, float, float] = (0.0, 0.2, 0.4, 0.6)) -> pl.DataFrame:
    rows = [
        ("P", "A", 9, -1, 0.0, 0.5, 0.5),
        ("P", "A", 10, 0, post_dwi[0], 0.5, 0.5),
        ("P", "A", 11, 10, post_dwi[1], 0.5, 0.5),
        ("P", "A", 12, 20, post_dwi[2], 0.5, 0.5),
        ("P", "A", 13, 30, post_dwi[3], 0.5, 0.5),
    ]
    return pl.DataFrame(
        rows,
        schema=[
            "partition_id", "actor_key", "sort_index", "replay_time_ns", "DWI",
            "L_bid_topN", "L_ask_topN",
        ],
        orient="row",
    )


def _summary_row(summary: pl.DataFrame, cluster: str = "E1") -> dict[str, object]:
    return summary.filter(pl.col("execution_cluster_id") == cluster).row(0, named=True)


def test_compute_partition_grids_emits_store_schemas_closed_nodes_and_independent_ramp_j():
    executions = pl.DataFrame([_execution()])

    trajectory, summary = compute_partition_grids(executions, _states(), _coverage(), CONFIG)

    assert trajectory.schema == TRAJECTORY_SCHEMA
    assert summary.schema == SUMMARY_SCHEMA
    assert trajectory["grid_offset_ns"].to_list() == [0, 10, 20, 30]
    assert trajectory["target_replay_ns"].to_list() == [0, 10, 20, 30]
    assert trajectory["selected_replay_ns"].to_list() == [0, 10, 20, 30]
    # With constant liquidity, MSCI is the core's signed SCI contribution.
    assert trajectory["MSCI"].to_list() == pytest.approx([0.0, 0.1, 0.2, 0.3])
    assert _summary_row(summary) == {
        "instrument": "XYZ", "partition_id": "P", "actor_key": "A", "execution_cluster_id": "E1",
        "J": pytest.approx(0.1), "J_observed_partial": pytest.approx(0.1),
        "argmax_end_offset_ns": 10, "argmax_start_offset_ns": 0,
        "complete_grid": True, "valid_pair_count": 3,
    }
    assert verify_partition_grids(executions, trajectory, summary, CONFIG, states=_states(), coverage=_coverage()) is None


def test_compute_partition_grids_uses_canonical_cluster_end_not_fill_anchor_and_exact_t0_on_same_clock():
    execution = _execution(fill_replay_time_ns=99)
    states = _states().with_columns(
        pl.when(pl.col("sort_index") == 11).then(pl.lit(0)).otherwise(pl.col("replay_time_ns")).alias("replay_time_ns")
    )

    trajectory, summary = compute_partition_grids(pl.DataFrame([execution]), states, _coverage(), CONFIG)

    assert trajectory["target_replay_ns"].to_list() == [0, 10, 20, 30]
    assert trajectory.row(0, named=True)["selected_sort_index"] == 10
    # Positive nodes use canonical as-of selection, so the later same-clock
    # state is eligible after the exact-t0 node.
    assert trajectory.filter(pl.col("grid_offset_ns") == 10).row(0, named=True)["selected_sort_index"] == 11
    assert _summary_row(summary)["J"] == pytest.approx(0.1)


def test_compute_partition_grids_requires_source_canonical_cluster_end_instead_of_fill_fallback():
    execution = _execution()
    execution.pop("cluster_end_replay_time_ns")
    execution["fill_replay_time_ns"] = 0

    with pytest.raises(ValueError, match="cluster_end_replay_time_ns"):
        compute_partition_grids(pl.DataFrame([execution]), _states(), _coverage(), CONFIG)


def test_compute_partition_grids_retains_invalid_missing_baseline_and_null_profile_as_closed_grids():
    no_baseline = _execution(execution_cluster_id="NO_BASELINE")
    null_profile = _execution(execution_cluster_id="NULL_PROFILE")
    states = _states().with_columns(
        pl.when(pl.col("sort_index") == 9).then(pl.lit(0.0)).otherwise(pl.lit(None)).alias("DWI")
    )
    no_pre_states = states.filter(pl.col("sort_index") >= 10)
    combined = pl.concat([states, no_pre_states.with_columns(pl.lit("B").alias("actor_key"))])
    no_baseline["actor_key"] = "B"

    trajectory, summary = compute_partition_grids(
        pl.DataFrame([no_baseline, null_profile]), combined, _coverage(), CONFIG
    )

    assert trajectory.group_by("execution_cluster_id").len().sort("execution_cluster_id")["len"].to_list() == [4, 4]
    assert trajectory.filter(pl.col("execution_cluster_id") == "NO_BASELINE")["reason"].unique().to_list() == ["missing_pre_execution_state"]
    assert trajectory.filter(pl.col("execution_cluster_id") == "NULL_PROFILE")["reason"].unique().to_list() == ["invalid_post_execution_profile"]
    assert summary["J"].null_count() == 2
    assert summary["complete_grid"].to_list() == [False, False]


def test_compute_partition_grids_preserves_long_attested_pause_but_marks_known_gap_and_partial_separately():
    long_pause = _states().filter(pl.col("sort_index") != 11)

    complete_trajectory, complete_summary = compute_partition_grids(
        pl.DataFrame([_execution()]), long_pause, _coverage(), CONFIG
    )
    gapped_trajectory, gapped_summary = compute_partition_grids(
        pl.DataFrame([_execution()]), long_pause, _coverage(gap=(5, 6)), CONFIG
    )

    assert complete_trajectory["valid"].all()
    assert complete_trajectory["selected_sort_index"].to_list() == [10, 10, 12, 13]
    assert _summary_row(complete_summary)["J"] == pytest.approx(0.2)
    assert gapped_trajectory.filter(pl.col("grid_offset_ns") >= 10)["reason"].unique().to_list() == ["known_coverage_gap"]
    assert _summary_row(gapped_summary)["J"] is None
    assert _summary_row(gapped_summary)["J_observed_partial"] is None


def test_compute_partition_grids_keeps_actors_isolated_and_inputs_unchanged():
    executions = pl.DataFrame([_execution(), _execution(actor_key="B", execution_cluster_id="E2")])
    states = pl.concat([
        _states(),
        _states(post_dwi=(0.0, 0.9, 0.9, 0.9)).with_columns(pl.lit("B").alias("actor_key")),
    ])
    coverage = _coverage()
    executions_before, states_before, coverage_before = executions.clone(), states.clone(), copy.deepcopy(coverage)

    trajectory, summary = compute_partition_grids(executions, states, coverage, CONFIG)

    assert trajectory.filter(pl.col("actor_key") == "A")["selected_sort_index"].to_list() == [10, 11, 12, 13]
    assert _summary_row(summary, "E1")["J"] == pytest.approx(0.1)
    assert _summary_row(summary, "E2")["J"] == pytest.approx(0.45)
    assert executions.equals(executions_before)
    assert states.equals(states_before)
    assert coverage == coverage_before


@pytest.mark.parametrize(
    ("post_dwi", "expected_msci", "expected_j", "expected_argmax"),
    [
        ((0.0, 0.0, 0.0, 0.0), [0.0, 0.0, 0.0, 0.0], 0.0, None),
        ((0.0, 0.2, 0.4, 0.6), [0.0, 0.1, 0.2, 0.3], 0.1, 10),
        ((0.0, 0.6, 0.0, 0.0), [0.0, 0.3, 0.0, 0.0], 0.3, 10),
    ],
)
def test_compute_partition_grids_matches_independent_constant_ramp_and_jump_cases(
    post_dwi: tuple[float, float, float, float],
    expected_msci: list[float],
    expected_j: float,
    expected_argmax: int | None,
):
    trajectory, summary = compute_partition_grids(
        pl.DataFrame([_execution()]), _states(post_dwi=post_dwi), _coverage(), CONFIG
    )

    assert trajectory["MSCI"].to_list() == pytest.approx(expected_msci)
    row = _summary_row(summary)
    assert row["J"] == pytest.approx(expected_j)
    assert row["argmax_end_offset_ns"] == expected_argmax


def test_summarize_grid_retains_endpoint_and_maximum_scientific_components_without_changing_store_schema():
    trajectory, _ = compute_partition_grids(pl.DataFrame([_execution()]), _states(), _coverage(), CONFIG)

    rich = summarize_grid(trajectory, lag_ns=10)

    assert rich["J"] == pytest.approx(0.1)
    assert rich["MSCI_t0"] == pytest.approx(0.0)
    assert rich["MSCI_horizon"] == pytest.approx(0.3)
    assert rich["MSCI_max"] == pytest.approx(0.3)
    assert rich["MSCI_max_offset_ns"] == 30


def test_verify_partition_grids_reconstructs_summary_and_rejects_wrong_grid_or_components():
    executions = pl.DataFrame([_execution()])
    trajectory, summary = compute_partition_grids(executions, _states(), _coverage(), CONFIG)
    wrong_summary = summary.with_columns(pl.lit(9.0).alias("J"))
    missing_node = trajectory.filter(pl.col("grid_offset_ns") != 30)
    wrong_component = trajectory.with_columns(
        pl.when(pl.col("grid_offset_ns") == 10).then(pl.lit(0.7)).otherwise(pl.col("MSCI")).alias("MSCI")
    )
    wrong_component_summary = summary.with_columns([
        pl.lit(0.7).alias("J"), pl.lit(0.7).alias("J_observed_partial"),
        pl.lit(10).alias("argmax_end_offset_ns"), pl.lit(0).alias("argmax_start_offset_ns"),
    ])
    wrong_component_summary = pl.DataFrame(
        wrong_component_summary.to_dicts(), schema=SUMMARY_SCHEMA, strict=False
    )

    with pytest.raises(ValueError, match="summary J"):
        verify_partition_grids(executions, trajectory, wrong_summary, CONFIG, states=_states(), coverage=_coverage())
    with pytest.raises(ValueError, match="closed grid"):
        verify_partition_grids(executions, missing_node, summary, CONFIG, states=_states(), coverage=_coverage())
    with pytest.raises(ValueError, match="MSCI"):
        verify_partition_grids(executions, wrong_component, wrong_component_summary, CONFIG, states=_states(), coverage=_coverage())


def test_verify_partition_grids_checks_recorded_baseline_against_supplied_local_states():
    executions = pl.DataFrame([_execution()])
    trajectory, summary = compute_partition_grids(executions, _states(), _coverage(), CONFIG)
    wrong_baseline = trajectory.with_columns(pl.lit(8).cast(pl.Int64).alias("baseline_sort_index"))

    with pytest.raises(ValueError, match="baseline"):
        verify_partition_grids(executions, wrong_baseline, summary, CONFIG, states=_states(), coverage=_coverage())


@pytest.mark.parametrize(
    ("states", "reason"),
    [
        (_states().with_columns(pl.when(pl.col("sort_index") == 9).then(pl.lit(None)).otherwise(pl.col("DWI")).alias("DWI")), "invalid_baseline_profile"),
        (_states().filter(pl.col("sort_index") != 10), "missing_post_execution_state"),
        (_states().with_columns(pl.when(pl.col("sort_index") == 11).then(pl.lit(None)).otherwise(pl.col("replay_time_ns")).alias("replay_time_ns")), "invalid_state_clock"),
        (_states().with_columns(pl.when(pl.col("sort_index") == 11).then(pl.lit(10)).otherwise(pl.col("sort_index")).alias("sort_index")), "invalid_canonical_sort_index"),
        (_states().with_columns(pl.when(pl.col("sort_index") == 12).then(pl.lit(5)).otherwise(pl.col("replay_time_ns")).alias("replay_time_ns")), "replay_time_regression"),
    ],
)
def test_verify_partition_grids_accepts_exact_closed_terminal_invalid_grid(
    states: pl.DataFrame, reason: str,
):
    executions = pl.DataFrame([_execution()])
    trajectory, summary = compute_partition_grids(executions, states, _coverage(), CONFIG)

    assert trajectory["reason"].unique().to_list() == [reason]
    assert trajectory["valid"].unique().to_list() == [False]
    assert trajectory.select(["selected_sort_index", "selected_replay_ns", "baseline_sort_index", "DWI", "L_bid_topN", "L_ask_topN", "SCI", "C_opposite", "C_same", "MSCI"]).null_count().row(0) == (4,) * 10
    assert verify_partition_grids(executions, trajectory, summary, CONFIG, states=states, coverage=_coverage()) is None


def test_verify_partition_grids_rejects_forged_valid_grid_when_local_baseline_is_missing():
    executions = pl.DataFrame([_execution()])
    states = _states().filter(pl.col("sort_index") >= 10)
    trajectory, _ = compute_partition_grids(executions, states, _coverage(), CONFIG)
    forged = trajectory.with_columns([
        pl.lit(999).cast(pl.Int64).alias("selected_sort_index"),
        pl.lit(0).cast(pl.Int64).alias("selected_replay_ns"),
        pl.lit(8).cast(pl.Int64).alias("baseline_sort_index"),
        pl.lit(0.0).alias("DWI"), pl.lit(0.5).alias("L_bid_topN"), pl.lit(0.5).alias("L_ask_topN"),
        pl.lit(0.0).alias("SCI"), pl.lit(0.0).alias("C_opposite"), pl.lit(0.0).alias("C_same"), pl.lit(0.0).alias("MSCI"),
        pl.lit(True).alias("valid"), pl.lit(None).cast(pl.String).alias("reason"),
    ])
    forged_summary = pl.DataFrame([{
        "instrument": "XYZ", "partition_id": "P", "actor_key": "A", "execution_cluster_id": "E1",
        "J": 0.0, "J_observed_partial": 0.0, "argmax_end_offset_ns": None,
        "argmax_start_offset_ns": None, "complete_grid": True, "valid_pair_count": 3,
    }], schema=SUMMARY_SCHEMA, strict=False)

    with pytest.raises(ValueError, match="missing_pre_execution_state"):
        verify_partition_grids(executions, forged, forged_summary, CONFIG, states=states, coverage=_coverage())


def test_verify_partition_grids_requires_attested_coverage_and_checks_partial_invalid_nodes():
    executions = pl.DataFrame([_execution()])
    states = _states()
    valid_trajectory, valid_summary = compute_partition_grids(executions, states, _coverage(), CONFIG)
    partial_coverage = _coverage(gap=(15, 15))
    partial_trajectory, partial_summary = compute_partition_grids(executions, states, partial_coverage, CONFIG)
    forged_reason = partial_trajectory.with_columns(
        pl.when(pl.col("grid_offset_ns") == 20).then(pl.lit("outside_coverage")).otherwise(pl.col("reason")).alias("reason")
    )

    with pytest.raises(ValueError, match="coverage_unverified"):
        verify_partition_grids(executions, valid_trajectory, valid_summary, CONFIG, states=states)
    assert verify_partition_grids(executions, valid_trajectory, valid_summary, CONFIG, states=states, coverage=_coverage()) is None
    assert partial_trajectory.filter(pl.col("grid_offset_ns") >= 20)["reason"].unique().to_list() == ["known_coverage_gap"]
    with pytest.raises(ValueError, match="reason"):
        verify_partition_grids(executions, forged_reason, partial_summary, CONFIG, states=states, coverage=partial_coverage)


def test_verify_partition_grids_ignores_irrelevant_later_clock_regression_beyond_boundary():
    executions = pl.DataFrame([_execution()])
    states = pl.concat([
        _states(),
        pl.DataFrame([("P", "A", 14, 40, 0.7, 0.5, 0.5), ("P", "A", 15, 20, 0.8, 0.5, 0.5)], schema=_states().schema, orient="row"),
    ])
    trajectory, summary = compute_partition_grids(executions, states, _coverage(), CONFIG)

    assert trajectory["selected_sort_index"].to_list() == [10, 11, 12, 13]
    assert verify_partition_grids(executions, trajectory, summary, CONFIG, states=states, coverage=_coverage()) is None


@pytest.mark.parametrize(
    ("coverage", "offset", "reason"),
    [
        (None, 0, "coverage_unverified"),
        (_coverage(end_ns=10), 20, "outside_coverage"),
        (_coverage(gap=(15, 15)), 20, "known_coverage_gap"),
    ],
)
def test_verify_partition_grids_accepts_exact_coverage_invalid_reasons(
    coverage: dict[str, object] | None, offset: int, reason: str,
):
    executions = pl.DataFrame([_execution()])
    trajectory, summary = compute_partition_grids(executions, _states(), coverage, CONFIG)

    assert trajectory.filter(pl.col("grid_offset_ns") == offset).row(0, named=True)["reason"] == reason
    assert verify_partition_grids(executions, trajectory, summary, CONFIG, states=_states(), coverage=coverage) is None


def test_compute_partition_grids_matches_approved_sampler_on_40_deterministic_random_fixtures():
    generator = random.Random(7)
    executions = pl.DataFrame([_execution()])
    coverage = _coverage()
    for _ in range(40):
        clocks = [-1, 0]
        for _ in range(3):
            clocks.append(clocks[-1] + generator.randint(1, 20))
        states = pl.DataFrame(
            [
                ("P", "A", index + 9, clock, generator.uniform(-1, 1), generator.random(), generator.random())
                for index, clock in enumerate(clocks)
            ],
            schema=_states().schema,
            orient="row",
        )

        trajectory, _ = compute_partition_grids(executions, states, coverage, CONFIG)
        approved = sample_execution_grid(_execution(), states, coverage, CONFIG)

        assert trajectory.to_dicts() == approved.to_dicts()
