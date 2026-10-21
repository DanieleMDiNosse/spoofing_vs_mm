from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from spoofing_detection.lob.msci_grid import (
    profile_components,
    sample_execution_grid,
    summarize_increment,
    validate_grid_config,
)


CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "msci_grid_study_v1.json"
EXPECTED_CONFIG = {
    "schema": "msci_grid_study_v1",
    "source_clock_policy": "spoofing_detection_sequence_replay_trade_fill_sequence_cancel_v3_utc_provisional",
    "grid_policy": "fixed_pre_exact_post_execution_closed_grid_v1",
    "grid_step_ns": 10_000_000,
    "lag_ns": 50_000_000,
    "horizon_ns": 1_000_000_000,
    "complete_grid_required": True,
    "cohort": "exact_top_positive_W_5pct_then_U_over_Q_gt_1_before_prices",
    "grid_sensitivity_ns": [5_000_000, 10_000_000, 25_000_000],
    "lag_sensitivity_ns": [25_000_000, 50_000_000, 100_000_000],
    "lag_sensitivity_grid_ns": 5_000_000,
    "selection_threshold": None,
}


def test_config_uses_integer_ns_and_exact_scientific_contract():
    payload = json.loads(CONFIG_PATH.read_text())

    assert payload == EXPECTED_CONFIG
    assert validate_grid_config(payload) is None


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("grid_step_ns", True),
        ("grid_step_ns", 0),
        ("lag_ns", False),
        ("lag_ns", -1),
        ("horizon_ns", 1.0),
        ("horizon_ns", 0),
        ("lag_sensitivity_grid_ns", True),
    ],
)
def test_config_rejects_nonpositive_or_noninteger_time_values(key, value):
    payload = copy.deepcopy(EXPECTED_CONFIG)
    payload[key] = value

    with pytest.raises(ValueError, match="positive integer"):
        validate_grid_config(payload)


def test_config_rejects_nondivisible_primary_grid():
    payload = copy.deepcopy(EXPECTED_CONFIG)
    payload["lag_ns"] = 55_000_000

    with pytest.raises(ValueError, match="grid_step_ns"):
        validate_grid_config(payload)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("schema", "msci_grid_study_v0"),
        ("source_clock_policy", "other_clock"),
        ("grid_policy", "interpolated_grid"),
        ("complete_grid_required", False),
        ("cohort", "price_selected"),
        ("selection_threshold", 0.1),
    ],
)
def test_config_rejects_incompatible_primary_semantics(key, value):
    payload = copy.deepcopy(EXPECTED_CONFIG)
    payload[key] = value

    with pytest.raises(ValueError):
        validate_grid_config(payload)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("grid_sensitivity_ns", [5_000_000, 7_000_000]),
        ("grid_sensitivity_ns", [5_000_000, True]),
        ("lag_sensitivity_ns", [25_000_000, 26_000_000]),
        ("lag_sensitivity_ns", [25_000_000, 1_500_000_000]),
    ],
)
def test_config_validates_sensitivity_cases_independently(key, value):
    payload = copy.deepcopy(EXPECTED_CONFIG)
    payload[key] = value

    with pytest.raises(ValueError):
        validate_grid_config(payload)


def test_config_rejects_missing_or_unknown_fields():
    missing = copy.deepcopy(EXPECTED_CONFIG)
    missing.pop("cohort")
    unknown = {**EXPECTED_CONFIG, "unrecognized": "value"}

    with pytest.raises(ValueError, match="missing"):
        validate_grid_config(missing)
    with pytest.raises(ValueError, match="unknown"):
        validate_grid_config(unknown)


def test_profile_components_maps_bid_opposite_collapse_independently():
    pre = {"DWI": 0.5, "L_bid_topN": 0.8, "L_ask_topN": 0.2}
    post = {"DWI": -1.0, "L_bid_topN": 0.0, "L_ask_topN": 0.2}

    out = profile_components(pre, post, "bid")

    assert out["SCI"] == pytest.approx(1.5)
    assert out["C_opposite"] == pytest.approx(1.0)
    assert out["C_same"] == pytest.approx(0.0)
    assert out["MSCI"] == pytest.approx(1.75)


def test_profile_components_maps_ask_opposite_collapse_symmetrically():
    pre = {"DWI": 0.5, "L_bid_topN": 0.2, "L_ask_topN": 0.8}
    post = {"DWI": -1.0, "L_bid_topN": 0.2, "L_ask_topN": 0.0}

    out = profile_components(pre, post, "ask")

    assert out == {
        "SCI": pytest.approx(1.5),
        "C_opposite": pytest.approx(1.0),
        "C_same": pytest.approx(0.0),
        "MSCI": pytest.approx(1.75),
    }


def test_profile_components_uses_exact_zero_denominator_policy_without_epsilon():
    pre = {"DWI": 0.0, "L_bid_topN": 0.0, "L_ask_topN": 0.0}
    post = {"DWI": 0.0, "L_bid_topN": 0.0, "L_ask_topN": 1.0}

    out = profile_components(pre, post, "bid")

    assert out == {"SCI": 0.0, "C_opposite": 0.0, "C_same": 0.0, "MSCI": 0.0}


def test_profile_components_accepts_closed_profile_domain_boundaries():
    pre = {"DWI": -1.0, "L_bid_topN": 1.0, "L_ask_topN": 0.0}
    post = {"DWI": 1.0, "L_bid_topN": 0.0, "L_ask_topN": 0.0}

    out = profile_components(pre, post, "bid")

    assert out == {
        "SCI": 2.0,
        "C_opposite": 1.0,
        "C_same": 0.0,
        "MSCI": 2.0,
    }


@pytest.mark.parametrize(
    ("profile_name", "field", "value"),
    [
        ("pre", "DWI", -1.000_001),
        ("post", "DWI", 1.000_001),
        ("pre", "L_bid_topN", -0.000_001),
        ("post", "L_bid_topN", 1.000_001),
        ("pre", "L_ask_topN", -0.000_001),
        ("post", "L_ask_topN", 1.000_001),
        ("pre", "DWI", True),
        ("post", "L_bid_topN", np.bool_(False)),
    ],
)
def test_profile_components_invalid_domain_or_boolean_component_yields_null_msci(
    profile_name, field, value
):
    pre = {"DWI": 0.0, "L_bid_topN": 0.5, "L_ask_topN": 0.5}
    post = {"DWI": 0.0, "L_bid_topN": 0.5, "L_ask_topN": 0.5}
    {"pre": pre, "post": post}[profile_name][field] = value

    out = profile_components(pre, post, "bid")

    assert out["MSCI"] is None


@pytest.mark.parametrize(
    "pre, post",
    [
        ({"DWI": None, "L_bid_topN": 0.5, "L_ask_topN": 0.5}, {"DWI": 0.0, "L_bid_topN": 0.0, "L_ask_topN": 0.0}),
        ({"DWI": math.nan, "L_bid_topN": 0.5, "L_ask_topN": 0.5}, {"DWI": 0.0, "L_bid_topN": 0.0, "L_ask_topN": 0.0}),
        ({"DWI": 0.0, "L_bid_topN": math.inf, "L_ask_topN": 0.5}, {"DWI": 0.0, "L_bid_topN": 0.0, "L_ask_topN": 0.0}),
        ({"DWI": 0.0, "L_bid_topN": -0.5, "L_ask_topN": 0.5}, {"DWI": 0.0, "L_bid_topN": 0.0, "L_ask_topN": 0.0}),
    ],
)
def test_profile_components_preserves_undefined_profiles_as_none(pre, post):
    out = profile_components(pre, post, "bid")

    assert out["MSCI"] is None


def test_profile_components_rejects_invalid_side_and_does_not_mutate_inputs():
    pre = {"DWI": 0.5, "L_bid_topN": 0.8, "L_ask_topN": 0.2}
    post = {"DWI": -1.0, "L_bid_topN": 0.0, "L_ask_topN": 0.2}
    pre_before, post_before = copy.deepcopy(pre), copy.deepcopy(post)

    with pytest.raises(ValueError, match="deceptive_side"):
        profile_components(pre, post, "neither")
    profile_components(pre, post, "bid")

    assert pre == pre_before
    assert post == post_before


def _grid(values: list[float], *, step_ns: int = 10_000_000):
    return np.arange(len(values), dtype=np.int64) * step_ns, np.asarray(values, dtype=float), np.ones(len(values), dtype=bool)


def test_summarize_increment_constant_series_is_zero_without_argmax():
    offset_ns, msci, valid = _grid([0.5] * 101)

    result = summarize_increment(offset_ns, msci, valid, lag_ns=50_000_000)

    assert result == {
        "J": 0.0,
        "J_observed_partial": 0.0,
        "argmax_end_offset_ns": None,
        "argmax_start_offset_ns": None,
        "complete_grid": True,
        "valid_pair_count": 96,
    }


def test_summarize_increment_ramp_has_expected_lagged_gain():
    offset_ns, msci, valid = _grid([index / 100 for index in range(101)])

    result = summarize_increment(offset_ns, msci, valid, lag_ns=50_000_000)

    assert result["J"] == pytest.approx(0.05)
    assert result["J_observed_partial"] == pytest.approx(0.05)


def test_summarize_increment_allows_valid_three_point_increment_from_msci_boundaries():
    offset_ns, msci, valid = _grid([-1.0, 2.0], step_ns=10)

    result = summarize_increment(offset_ns, msci, valid, lag_ns=10)

    assert result == {
        "J": 3.0,
        "J_observed_partial": 3.0,
        "argmax_end_offset_ns": 10,
        "argmax_start_offset_ns": 0,
        "complete_grid": True,
        "valid_pair_count": 1,
    }


@pytest.mark.parametrize(
    "values",
    [[0.0, 2.000_001], [-1.000_001, 2.0], [0.0, 3.0], [-2.0, 2.0]],
)
def test_summarize_increment_invalid_msci_endpoint_cannot_produce_complete_primary(values):
    offset_ns, _, valid = _grid([0.0, 0.0], step_ns=10)

    result = summarize_increment(offset_ns, values, valid, lag_ns=10)

    assert result == {
        "J": None,
        "J_observed_partial": None,
        "argmax_end_offset_ns": None,
        "argmax_start_offset_ns": None,
        "complete_grid": False,
        "valid_pair_count": 0,
    }


def test_summarize_increment_mixed_invalid_endpoints_leave_only_valid_descriptive_pairs():
    offset_ns = np.array([0, 10, 20, 30, 40], dtype=np.int64)
    msci = np.array([0.0, 1.0, None, 2.000_001, np.bool_(True)], dtype=object)
    valid = np.ones(5, dtype=bool)

    result = summarize_increment(offset_ns, msci, valid, lag_ns=10)

    assert result == {
        "J": None,
        "J_observed_partial": 1.0,
        "argmax_end_offset_ns": None,
        "argmax_start_offset_ns": None,
        "complete_grid": False,
        "valid_pair_count": 1,
    }


@pytest.mark.parametrize("invalid_value", [True, np.bool_(False), None, math.nan])
def test_summarize_increment_invalid_endpoint_preserves_partial_descriptive_value(invalid_value):
    offset_ns = np.array([0, 10, 20, 30], dtype=np.int64)
    msci = np.array([0.0, 1.0, invalid_value, 2.0], dtype=object)
    valid = np.ones(4, dtype=bool)

    result = summarize_increment(offset_ns, msci, valid, lag_ns=10)

    assert result == {
        "J": None,
        "J_observed_partial": 1.0,
        "argmax_end_offset_ns": None,
        "argmax_start_offset_ns": None,
        "complete_grid": False,
        "valid_pair_count": 1,
    }


@pytest.mark.parametrize(
    "values",
    [
        [0.0] * 5 + [1.0] * 96,
        [0.0] * 5 + [1.0] + [0.0] * 95,
    ],
)
def test_summarize_increment_captures_persistent_and_transient_grid_jumps(values):
    offset_ns, msci, valid = _grid(values)

    result = summarize_increment(offset_ns, msci, valid, lag_ns=50_000_000)

    assert result["J"] == pytest.approx(1.0)
    assert result["argmax_start_offset_ns"] == 0
    assert result["argmax_end_offset_ns"] == 50_000_000


def test_summarize_increment_negative_only_gain_is_zero_without_argmax():
    offset_ns, msci, valid = _grid([2.0, 1.0, 0.0, -1.0])

    result = summarize_increment(offset_ns, msci, valid, lag_ns=10_000_000)

    assert result["J"] == 0.0
    assert result["argmax_end_offset_ns"] is None
    assert result["argmax_start_offset_ns"] is None


def test_summarize_increment_keeps_partial_result_distinct_from_primary():
    offset_ns, msci, valid = _grid([0.0, 1.0, math.nan, 2.0])

    result = summarize_increment(offset_ns, msci, valid, lag_ns=10_000_000)

    assert result["complete_grid"] is False
    assert result["valid_pair_count"] == 1
    assert result["J"] is None
    assert result["J_observed_partial"] == pytest.approx(1.0)
    assert result["argmax_end_offset_ns"] is None
    assert result["argmax_start_offset_ns"] is None


def test_summarize_increment_equal_positive_maxima_use_earliest_pair():
    offset_ns, msci, valid = _grid([0.0, 1.0, 0.0, 1.0])

    result = summarize_increment(offset_ns, msci, valid, lag_ns=10_000_000)

    assert result["J"] == pytest.approx(1.0)
    assert result["argmax_start_offset_ns"] == 0
    assert result["argmax_end_offset_ns"] == 10_000_000


def test_summarize_increment_returns_null_when_no_valid_pairs_exist():
    offset_ns, msci, valid = _grid([math.nan, math.nan])

    result = summarize_increment(offset_ns, msci, valid, lag_ns=10_000_000)

    assert result["J"] is None
    assert result["J_observed_partial"] is None
    assert result["valid_pair_count"] == 0


def test_summarize_increment_short_transient_between_grid_nodes_is_not_observed():
    offset_ns, msci, valid = _grid([0.0, 0.0, 0.0, 0.0])

    result = summarize_increment(offset_ns, msci, valid, lag_ns=10_000_000)

    assert result["J"] == 0.0
    assert result["argmax_end_offset_ns"] is None


@pytest.mark.parametrize(
    "offset_ns, msci, valid, lag_ns",
    [
        (np.array([0, 10], dtype=np.int64), np.array([0.0]), np.array([True]), 10),
        (np.array([1, 11], dtype=np.int64), np.array([0.0, 1.0]), np.array([True, True]), 10),
        (np.array([0, 10, 30], dtype=np.int64), np.array([0.0, 1.0, 2.0]), np.array([True, True, True]), 10),
        (np.array([0.0, 10.0]), np.array([0.0, 1.0]), np.array([True, True]), 10),
        (np.array([0, 10], dtype=np.int64), np.array([0.0, 1.0]), np.array([True, True]), True),
        (np.array([0, 10], dtype=np.int64), np.array([0.0, 1.0]), np.array([True, True]), 20),
    ],
)
def test_summarize_increment_rejects_invalid_arrays_grid_or_lag(offset_ns, msci, valid, lag_ns):
    with pytest.raises(ValueError):
        summarize_increment(offset_ns, msci, valid, lag_ns=lag_ns)


def test_summarize_increment_does_not_mutate_inputs():
    offset_ns, msci, valid = _grid([0.0, 1.0, math.nan])
    offset_before, msci_before, valid_before = offset_ns.copy(), msci.copy(), valid.copy()

    summarize_increment(offset_ns, msci, valid, lag_ns=10_000_000)

    np.testing.assert_array_equal(offset_ns, offset_before)
    np.testing.assert_equal(msci, msci_before)
    np.testing.assert_array_equal(valid, valid_before)


def test_sample_execution_grid_uses_exact_t0_and_canonical_asof_post_state():
    execution = {
        "partition_id": "P",
        "actor_key": "A",
        "execution_cluster_id": "E",
        "deceptive_side": "bid",
        "cluster_first_sort_index": 10,
        "cluster_last_sort_index": 10,
        "cluster_end_replay_time_ns": 0,
    }
    states = pl.DataFrame(
        {
            "partition_id": ["P", "P", "P", "P"],
            "actor_key": ["A", "A", "A", "A"],
            "sort_index": [12, 9, 11, 10],
            "replay_time_ns": [10_000_000, -1_000_000, 0, 0],
            "DWI": [-0.5, 0.5, -0.8, -1.0],
            "L_bid_topN": [0.1, 0.8, 0.3, 0.0],
            "L_ask_topN": [0.2, 0.2, 0.2, 0.2],
        }
    )
    coverage = {
        "segments": [
            {
                "partition_id": "P",
                "start_ns": -1_000_000,
                "end_ns": 1_000_000_000,
                "attested": True,
                "source": "frozen-producer-v1",
            }
        ]
    }

    out = sample_execution_grid(execution, states, coverage, EXPECTED_CONFIG)

    assert out.height == 101
    assert out.row(0, named=True)["selected_sort_index"] == 10
    assert out.row(0, named=True)["selected_replay_ns"] == 0
    assert out.row(0, named=True)["baseline_sort_index"] == 9
    assert out.row(0, named=True)["MSCI"] == pytest.approx(1.75)
    assert out.filter(pl.col("grid_offset_ns") == 10_000_000).row(0, named=True)[
        "selected_sort_index"
    ] == 12


def _sampler_fixture(*, include_update: bool = True):
    execution = {
        "instrument": "XYZ",
        "partition_id": "P",
        "actor_key": "A",
        "execution_cluster_id": "E",
        "deceptive_side": "bid",
        "cluster_first_sort_index": 10,
        "cluster_last_sort_index": 10,
        "cluster_end_replay_time_ns": 0,
    }
    states = [
        ("P", "A", 9, -1_000_000, 0.5, 0.8, 0.2),
        ("P", "A", 10, 0, -1.0, 0.0, 0.2),
        ("P", "A", 11, 0, -0.8, 0.3, 0.2),
    ]
    if include_update:
        states.append(("P", "A", 12, 10_000_000, -0.5, 0.1, 0.2))
    return (
        execution,
        pl.DataFrame(
            states,
            schema=[
                "partition_id",
                "actor_key",
                "sort_index",
                "replay_time_ns",
                "DWI",
                "L_bid_topN",
                "L_ask_topN",
            ],
            orient="row",
        ),
        {
            "segments": [
                {
                    "partition_id": "P",
                    "start_ns": -1_000_000,
                    "end_ns": 1_000_000_000,
                    "attested": True,
                    "source": "frozen-producer-v1",
                }
            ]
        },
    )


def test_sample_execution_grid_fixed_baseline_never_becomes_incremental_and_includes_endpoint():
    execution, states, coverage = _sampler_fixture()

    out = sample_execution_grid(execution, states, coverage, EXPECTED_CONFIG)
    ten_ms = out.filter(pl.col("grid_offset_ns") == 10_000_000).row(0, named=True)
    end = out.filter(pl.col("grid_offset_ns") == 1_000_000_000).row(0, named=True)

    assert out.columns == [
        "instrument",
        "partition_id",
        "actor_key",
        "execution_cluster_id",
        "grid_offset_ns",
        "target_replay_ns",
        "selected_sort_index",
        "selected_replay_ns",
        "baseline_sort_index",
        "DWI",
        "L_bid_topN",
        "L_ask_topN",
        "SCI",
        "C_opposite",
        "C_same",
        "MSCI",
        "valid",
        "reason",
    ]
    assert ten_ms["baseline_sort_index"] == 9
    assert ten_ms["SCI"] == pytest.approx(1.0)
    assert end["selected_sort_index"] == 12
    assert end["valid"] is True


def test_sample_execution_grid_never_uses_future_or_other_actor_partition_state():
    execution, states, coverage = _sampler_fixture()
    states = pl.concat(
        [
            states,
            pl.DataFrame(
                [
                    ("P", "A", 13, 20_000_000, 1.0, 1.0, 1.0),
                    ("P", "OTHER", 10, 0, 1.0, 1.0, 1.0),
                    ("OTHER", "A", 10, 0, 1.0, 1.0, 1.0),
                ],
                schema=states.schema,
                orient="row",
            ),
        ]
    )

    out = sample_execution_grid(execution, states, coverage, EXPECTED_CONFIG)

    assert out.filter(pl.col("grid_offset_ns") == 10_000_000).row(0, named=True)[
        "selected_sort_index"
    ] == 12
    assert out.filter(pl.col("grid_offset_ns") == 0).row(0, named=True)["selected_sort_index"] == 10


def test_sample_execution_grid_uses_greatest_canonical_index_for_positive_same_clock_ties():
    execution, states, coverage = _sampler_fixture()
    same_clock_later = pl.concat(
        [
            states,
            pl.DataFrame(
                [("P", "A", 13, 10_000_000, 0.0, 0.2, 0.2)],
                schema=states.schema,
                orient="row",
            ),
        ]
    )

    out = sample_execution_grid(execution, same_clock_later, coverage, EXPECTED_CONFIG)

    assert out.filter(pl.col("grid_offset_ns") == 10_000_000).row(0, named=True)[
        "selected_sort_index"
    ] == 13


def test_sample_execution_grid_missing_exact_t0_and_missing_pre_have_explicit_null_rows():
    execution, states, coverage = _sampler_fixture()
    wrong_t0 = states.filter(pl.col("sort_index") != 11).with_columns(
        pl.when(pl.col("sort_index") == 10)
        .then(pl.lit(1))
        .otherwise(pl.col("replay_time_ns"))
        .alias("replay_time_ns")
    )
    no_pre = states.filter(pl.col("sort_index") >= 10)

    wrong_t0_out = sample_execution_grid(execution, wrong_t0, coverage, EXPECTED_CONFIG)
    no_pre_out = sample_execution_grid(execution, no_pre, coverage, EXPECTED_CONFIG)

    assert wrong_t0_out.select("reason").unique().to_series().to_list() == [
        "missing_post_execution_state"
    ]
    assert no_pre_out.select("reason").unique().to_series().to_list() == [
        "missing_pre_execution_state"
    ]
    assert wrong_t0_out["MSCI"].null_count() == wrong_t0_out.height
    assert no_pre_out["MSCI"].null_count() == no_pre_out.height


def test_sample_execution_grid_rejects_relevant_clock_regression_without_reordering():
    execution, states, coverage = _sampler_fixture()
    regressed = states.with_columns(
        pl.when(pl.col("sort_index") == 12)
        .then(pl.lit(-2_000_000))
        .otherwise(pl.col("replay_time_ns"))
        .alias("replay_time_ns")
    )

    out = sample_execution_grid(execution, regressed, coverage, EXPECTED_CONFIG)

    assert out.select("reason").unique().to_series().to_list() == ["replay_time_regression"]
    assert out["MSCI"].null_count() == out.height


def test_sample_execution_grid_rejects_float_replay_clock_without_parsing_or_clamping():
    execution, states, coverage = _sampler_fixture()
    float_clock = states.with_columns(pl.col("replay_time_ns").cast(pl.Float64))

    out = sample_execution_grid(execution, float_clock, coverage, EXPECTED_CONFIG)

    assert out.select("reason").unique().to_series().to_list() == ["invalid_state_clock"]
    assert out["MSCI"].null_count() == out.height


def test_sample_execution_grid_only_forward_fills_inside_attested_coverage_and_known_gaps():
    execution, states, coverage = _sampler_fixture(include_update=False)
    unverified = sample_execution_grid(execution, states, None, EXPECTED_CONFIG)
    continuous = sample_execution_grid(execution, states, coverage, EXPECTED_CONFIG)
    ending_early = {"segments": [{**coverage["segments"][0], "end_ns": 20_000_000}]}
    bounded = sample_execution_grid(execution, states, ending_early, EXPECTED_CONFIG)
    gapped = {
        **coverage,
        "invalid_intervals": [{"partition_id": "P", "start_ns": 20_000_000, "end_ns": 30_000_000}],
    }
    gap_out = sample_execution_grid(execution, states, gapped, EXPECTED_CONFIG)

    assert unverified.row(0, named=True)["reason"] == "coverage_unverified"
    assert continuous.filter(pl.col("grid_offset_ns") == 1_000_000_000).row(0, named=True)["valid"] is True
    assert bounded.filter(pl.col("grid_offset_ns") == 10_000_000).row(0, named=True)["valid"] is True
    assert bounded.filter(pl.col("grid_offset_ns") == 30_000_000).row(0, named=True)["reason"] == "outside_coverage"
    assert gap_out.filter(pl.col("grid_offset_ns") == 20_000_000).row(0, named=True)["reason"] == "known_coverage_gap"
    assert gap_out.filter(pl.col("grid_offset_ns") == 10_000_000).row(0, named=True)["valid"] is True


def test_sample_execution_grid_cannot_take_baseline_or_t0_from_other_actor_partition():
    execution, states, coverage = _sampler_fixture()
    local_post_only = states.filter(pl.col("sort_index") >= 10)
    other_identity = states.with_columns(
        [pl.lit("OTHER").alias("partition_id"), pl.lit("OTHER").alias("actor_key")]
    )

    out = sample_execution_grid(
        execution, pl.concat([local_post_only, other_identity]), coverage, EXPECTED_CONFIG
    )

    assert out.select("reason").unique().to_series().to_list() == ["missing_pre_execution_state"]


def test_sample_execution_grid_rejects_invalid_fixed_baseline_profile_without_substitution():
    execution, states, coverage = _sampler_fixture()
    invalid_pre = states.with_columns(
        pl.when(pl.col("sort_index") == 9).then(pl.lit(None)).otherwise(pl.col("DWI")).alias("DWI")
    )

    out = sample_execution_grid(execution, invalid_pre, coverage, EXPECTED_CONFIG)

    assert out.select("reason").unique().to_series().to_list() == ["invalid_baseline_profile"]
    assert out["MSCI"].null_count() == out.height


def test_sample_execution_grid_keeps_explicit_zero_valid_and_null_disappearance_invalid_without_mutation():
    execution, states, coverage = _sampler_fixture()
    zero = states.with_columns(
        [
            pl.when(pl.col("sort_index") == 10).then(pl.lit(0.0)).otherwise(pl.col("DWI")).alias("DWI"),
            pl.when(pl.col("sort_index") == 10)
            .then(pl.lit(0.0))
            .otherwise(pl.col("L_bid_topN"))
            .alias("L_bid_topN"),
        ]
    )
    disappeared = states.with_columns(
        pl.when(pl.col("sort_index") == 12).then(pl.lit(None)).otherwise(pl.col("DWI")).alias("DWI")
    )
    states_before = states.clone()
    coverage_before = copy.deepcopy(coverage)
    execution_before = copy.deepcopy(execution)

    zero_out = sample_execution_grid(execution, zero, coverage, EXPECTED_CONFIG)
    disappeared_out = sample_execution_grid(execution, disappeared, coverage, EXPECTED_CONFIG)

    assert zero_out.row(0, named=True)["valid"] is True
    assert zero_out.row(0, named=True)["MSCI"] == pytest.approx(1.25)
    assert disappeared_out.filter(pl.col("grid_offset_ns") == 10_000_000).row(0, named=True)["reason"] == (
        "invalid_post_execution_profile"
    )
    assert disappeared_out.filter(pl.col("grid_offset_ns") == 10_000_000).row(0, named=True)["MSCI"] is None
    assert states.equals(states_before)
    assert coverage == coverage_before
    assert execution == execution_before


def test_sample_execution_grid_known_gap_between_nodes_poison_later_fresh_state():
    execution, states, coverage = _sampler_fixture(include_update=False)
    fresh_after_gap = pl.concat(
        [
            states,
            pl.DataFrame(
                [("P", "A", 12, 5_000_000, -0.5, 0.1, 0.2)],
                schema=states.schema,
                orient="row",
            ),
        ]
    )
    gapped = {
        **coverage,
        "invalid_intervals": [{"partition_id": "P", "start_ns": 1_000_000, "end_ns": 2_000_000}],
    }

    out = sample_execution_grid(execution, fresh_after_gap, gapped, EXPECTED_CONFIG)

    assert out.row(0, named=True)["valid"] is True
    for offset_ns in (10_000_000, 20_000_000):
        row = out.filter(pl.col("grid_offset_ns") == offset_ns).row(0, named=True)
        assert row["selected_sort_index"] == 12
        assert row["valid"] is False
        assert row["reason"] == "known_coverage_gap"


def test_sample_execution_grid_gap_between_baseline_and_t0_invalidates_closed_grid():
    execution, states, coverage = _sampler_fixture()
    gapped = {
        **coverage,
        "invalid_intervals": [{"partition_id": "P", "start_ns": -500_000, "end_ns": -100_000}],
    }

    out = sample_execution_grid(execution, states, gapped, EXPECTED_CONFIG)

    assert out.row(0, named=True)["reason"] == "known_coverage_gap"
    assert out["valid"].sum() == 0


def test_sample_execution_grid_missing_attested_segment_between_nodes_invalidates_later_nodes():
    execution, states, coverage = _sampler_fixture(include_update=False)
    fresh_after_gap = pl.concat(
        [
            states,
            pl.DataFrame(
                [("P", "A", 12, 5_000_000, -0.5, 0.1, 0.2)],
                schema=states.schema,
                orient="row",
            ),
        ]
    )
    split_coverage = {
        "segments": [
            {**coverage["segments"][0], "end_ns": 1_000_000},
            {**coverage["segments"][0], "start_ns": 2_000_000},
        ]
    }

    out = sample_execution_grid(execution, fresh_after_gap, split_coverage, EXPECTED_CONFIG)

    assert out.row(0, named=True)["valid"] is True
    assert out.filter(pl.col("grid_offset_ns") == 10_000_000).row(0, named=True)["reason"] == (
        "outside_coverage"
    )


def test_sample_execution_grid_ignores_declared_gaps_wholly_outside_required_window():
    execution, states, coverage = _sampler_fixture()
    coverage = {
        **coverage,
        "invalid_intervals": [
            {"partition_id": "P", "start_ns": -3_000_000, "end_ns": -2_000_000},
            {"partition_id": "P", "start_ns": 1_000_000_001, "end_ns": 1_001_000_000},
        ],
    }

    out = sample_execution_grid(execution, states, coverage, EXPECTED_CONFIG)

    assert out["valid"].all()


def test_sample_execution_grid_ignores_clock_regression_after_first_strict_horizon_crossing():
    execution, states, coverage = _sampler_fixture()
    future_regression = pl.concat(
        [
            states,
            pl.DataFrame(
                [
                    ("P", "A", 13, 2_000_000_000, 0.0, 0.2, 0.2),
                    ("P", "A", 14, 1_999_000_000, 0.0, 0.2, 0.2),
                ],
                schema=states.schema,
                orient="row",
            ),
        ]
    )

    out = sample_execution_grid(execution, future_regression, coverage, EXPECTED_CONFIG)

    assert out["valid"].all()


def test_sample_execution_grid_rejects_clock_regression_before_horizon_crossing():
    execution, states, coverage = _sampler_fixture()
    internal_regression = pl.concat(
        [
            states,
            pl.DataFrame(
                [
                    ("P", "A", 13, 500_000_000, 0.0, 0.2, 0.2),
                    ("P", "A", 14, 400_000_000, 0.0, 0.2, 0.2),
                    ("P", "A", 15, 2_000_000_000, 0.0, 0.2, 0.2),
                ],
                schema=states.schema,
                orient="row",
            ),
        ]
    )

    out = sample_execution_grid(execution, internal_regression, coverage, EXPECTED_CONFIG)

    assert out.select("reason").unique().to_series().to_list() == ["replay_time_regression"]


def test_sample_execution_grid_ignores_late_malformed_clock_after_horizon_boundary():
    execution, states, coverage = _sampler_fixture()
    with_late_null = pl.concat(
        [
            states,
            pl.DataFrame(
                [
                    ("P", "A", 13, 2_000_000_000, 0.0, 0.2, 0.2),
                    ("P", "A", 14, None, 0.0, 0.2, 0.2),
                ],
                schema=states.schema,
                orient="row",
            ),
        ]
    )

    out = sample_execution_grid(execution, with_late_null, coverage, EXPECTED_CONFIG)

    assert out["valid"].all()


def test_sample_execution_grid_uses_one_explicit_schema_for_valid_and_invalid_rows():
    execution, states, coverage = _sampler_fixture()
    valid = sample_execution_grid(execution, states, coverage, EXPECTED_CONFIG)
    invalid = sample_execution_grid(
        execution, states.filter(pl.col("sort_index") >= 10), coverage, EXPECTED_CONFIG
    )

    assert valid.schema == invalid.schema
    combined = pl.concat([valid, invalid])
    assert combined.schema == valid.schema
    assert combined.height == valid.height + invalid.height


def test_sample_execution_grid_accepts_abutting_attested_segments_as_continuous_support():
    execution, states, coverage = _sampler_fixture()
    abutting = {
        "segments": [
            {**coverage["segments"][0], "end_ns": 500_000_000},
            {**coverage["segments"][0], "start_ns": 500_000_000},
        ]
    }

    out = sample_execution_grid(execution, states, abutting, EXPECTED_CONFIG)

    assert out["valid"].all()


def test_sample_execution_grid_validates_equal_horizon_clock_batch_before_boundary():
    execution, states, coverage = _sampler_fixture()
    horizon_batch_regression = pl.concat(
        [
            states,
            pl.DataFrame(
                [
                    ("P", "A", 13, 1_000_000_000, 0.0, 0.2, 0.2),
                    ("P", "A", 14, 999_000_000, 0.0, 0.2, 0.2),
                    ("P", "A", 15, 2_000_000_000, 0.0, 0.2, 0.2),
                ],
                schema=states.schema,
                orient="row",
            ),
        ]
    )

    out = sample_execution_grid(execution, horizon_batch_regression, coverage, EXPECTED_CONFIG)

    assert out.select("reason").unique().to_series().to_list() == ["replay_time_regression"]
