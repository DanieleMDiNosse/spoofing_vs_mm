from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from spoofing_detection.lob.depth_kernel_calibration import (
    build_calibration_snapshot_tables,
    build_empirical_depth_kernel,
    calibrate_empirical_depth_kernel,
    estimate_hit_probability_profile,
    estimate_visibility_covariance_profile,
    fit_log_decay_slope,
    load_empirical_kernel_weights,
    summarise_instrument_kernel,
)


def raw_event(
    seq,
    event_type,
    order_id,
    side,
    price,
    qty,
    displayed,
    client,
    *,
    trade_time=None,
    bookout=None,
    last_shares=None,
    aggressive="N",
):
    timestamp = bookout or f"2024-01-02 09:30:{seq:02d}"
    return {
        "TRADEDATE": "2024-01-02",
        "MIC": "XMIL",
        "MARKETCODE": "MTA",
        "SYMBOLINDEX": 123,
        "EMM (*)": 1,
        "SEQUENCETIME": timestamp,
        "BOOKIN": timestamp,
        "BOOKOUTTIME": timestamp,
        "TRADETIME": trade_time,
        "HDR_APPLKEYSEQUENCENUMBER": seq,
        "HDR_HWMSEQUENCENUMBER": seq,
        "HDR_OFFSETID": seq,
        "ROW_NUMBER": seq,
        "ORDEREVENTTYPE (*)": event_type,
        "ORDERID": order_id,
        "ORDERPRIORITY": str(seq),
        "ORDERSIDE (*)": side,
        "ORDERPX": price,
        "ORDERQTY": qty,
        "DISPLAYEDQTY": displayed,
        "LEAVESQTY": qty,
        "LASTSHARES": last_shares,
        "LASTTRADEDPX": price,
        "ORDERTYPE (*)": 2,
        "TIMEINFORCE (*)": 0,
        "FIRMID": "F1",
        "NMSC_ORIGINALCLIENTIDSHORTCODE": client,
        "ORDER_TRADINGCAPACITY (*)": 3 if client is not None else 1,
        "PASSIVEORDER": "Y" if event_type == 3 and aggressive != "Y" else None,
        "AGGRESSIVEORDER": aggressive if event_type == 3 else None,
    }


def test_empirical_kernel_combines_protection_and_covariance_by_side():
    profile = pl.DataFrame(
        {
            "instrument_id": ["ABC", "ABC", "ABC", "ABC"],
            "side": ["bid", "bid", "ask", "ask"],
            "rank": [1, 2, 1, 2],
            "depth_distance_ticks": [1.0, 2.0, 1.0, 2.0],
            "exposure_count": [100, 100, 100, 100],
            "hit_count": [80, 20, 50, 25],
            "hit_probability": [0.8, 0.2, 0.5, 0.25],
            "visibility_covariance": [2.0, 1.0, 3.0, 1.0],
        }
    )

    out = build_empirical_depth_kernel(profile, protection_floor=0.0, visibility_floor=0.0)
    bid = out.filter(pl.col("side") == "bid").sort("rank")
    ask = out.filter(pl.col("side") == "ask").sort("rank")

    assert bid["raw_weight"].to_list() == pytest.approx([0.4, 0.8])
    assert bid["kernel_weight"].to_list() == pytest.approx([1 / 3, 2 / 3])
    assert ask["raw_weight"].to_list() == pytest.approx([1.5, 0.75])
    assert ask["kernel_weight"].to_list() == pytest.approx([2 / 3, 1 / 3])


def test_fit_log_decay_slope_estimates_positive_decay():
    distances = [1.0, 2.0, 3.0, 4.0]
    values = [2.718281828459045 ** (2.0 - 0.7 * d) for d in distances]

    slope = fit_log_decay_slope(distances, values, weights=[10, 10, 10, 10])

    assert slope == pytest.approx(0.7)


def test_estimate_hit_probability_profile_uses_at_or_through_prices():
    snapshots = pl.DataFrame(
        {
            "instrument_id": ["ABC", "ABC", "ABC", "ABC"],
            "partition_id": ["P", "P", "P", "P"],
            "snapshot_id": [1, 1, 2, 2],
            "sort_index": [1, 1, 2, 2],
            "event_ts": [0.0, 0.0, 10.0, 10.0],
            "side": ["ask", "ask", "ask", "ask"],
            "rank": [1, 2, 1, 2],
            "depth_distance_ticks": [1.0, 2.0, 1.0, 2.0],
            "price": [101.0, 102.0, 101.0, 102.0],
            "visible_qty": [10.0, 20.0, 10.0, 20.0],
            "mid": [100.5, 100.5, 100.5, 100.5],
        }
    )
    fills = pl.DataFrame(
        {
            "instrument_id": ["ABC", "ABC"],
            "partition_id": ["P", "P"],
            "event_ts": [1.0, 11.0],
            "side": ["ask", "ask"],
            "price": [102.0, 101.0],
        }
    )

    out = estimate_hit_probability_profile(snapshots, fills, horizon_seconds=2.0).sort("rank")

    assert out["exposure_count"].to_list() == [2, 2]
    assert out["hit_count"].to_list() == [2, 1]
    assert out["hit_probability"].to_list() == pytest.approx([1.0, 0.5])


def test_estimate_visibility_covariance_profile_uses_future_mid_change():
    snapshots = pl.DataFrame(
        {
            "instrument_id": ["ABC", "ABC", "ABC", "ABC"],
            "partition_id": ["P", "P", "P", "P"],
            "snapshot_id": [1, 1, 2, 2],
            "sort_index": [1, 1, 2, 2],
            "event_ts": [0.0, 0.0, 10.0, 10.0],
            "side": ["bid", "ask", "bid", "ask"],
            "rank": [1, 1, 1, 1],
            "depth_distance_ticks": [1.0, 1.0, 1.0, 1.0],
            "price": [100.0, 101.0, 100.0, 101.0],
            "visible_qty": [10.0, 30.0, 30.0, 10.0],
            "mid": [100.5, 100.5, 100.5, 100.5],
            "future_mid": [101.0, 101.0, 100.0, 100.0],
        }
    )

    out = estimate_visibility_covariance_profile(snapshots).sort(["side", "rank"])

    assert set(out.columns) >= {"side", "rank", "visibility_covariance", "visibility_observation_count"}
    assert out.filter(pl.col("side") == "ask").item(0, "visibility_covariance") > 0
    assert out.filter(pl.col("side") == "bid").item(0, "visibility_covariance") > 0


def test_calibrate_empirical_depth_kernel_replays_full_sample():
    df = pl.DataFrame(
        [
            raw_event(1, 1, "B1", 1, 100.0, 10, 10, "C1"),
            raw_event(2, 1, "B2", 1, 99.9, 20, 20, "C2"),
            raw_event(3, 1, "A1", 2, 100.1, 10, 10, "C3"),
            raw_event(4, 1, "A2", 2, 100.2, 20, 20, "C4"),
            raw_event(5, 3, "A2", 2, 100.2, 0, 0, "C4", last_shares=5),
        ]
    )

    snapshots, fills = build_calibration_snapshot_tables(df, instrument_id="ABC", top_n=2, tick_size=0.1)
    assert not snapshots.is_empty()
    assert fills.height == 1
    assert snapshots.select(pl.col("future_mid").is_not_null().sum()).item() > 0

    out = calibrate_empirical_depth_kernel(
        df,
        instrument_id="ABC",
        top_n=2,
        tick_size=0.1,
        horizon_seconds=10.0,
        visibility_floor=1e-12,
    )

    assert set(out["side"].to_list()) == {"ask", "bid"}
    sums = out.group_by("side").agg(pl.col("kernel_weight").sum().alias("sum")).sort("side")
    assert sums["sum"].to_list() == pytest.approx([1.0, 1.0])
    assert "kappa_hat_side" in out.columns
    assert "lambda_hat_side" in out.columns
    assert "kappa_hat_instrument" in out.columns
    assert "lambda_hat_instrument" in out.columns


def test_load_empirical_kernel_weights_filters_instrument_and_normalizes(tmp_path: Path):
    path = tmp_path / "kernel.parquet"
    pl.DataFrame(
        {
            "instrument_id": ["ABC", "ABC", "XYZ"],
            "side": ["bid", "bid", "bid"],
            "rank": [1, 2, 1],
            "kernel_weight": [2.0, 6.0, 1.0],
        }
    ).write_parquet(path)

    weights = load_empirical_kernel_weights(path, instrument_id="ABC")

    assert weights == {"bid": {1: pytest.approx(0.25), 2: pytest.approx(0.75)}}


def test_summarise_instrument_kernel_collapses_side_slopes_to_one_instrument_value():
    kernel = pl.DataFrame(
        {
            "instrument_id": ["ABC", "ABC", "ABC", "ABC"],
            "side": ["ask", "ask", "bid", "bid"],
            "rank": [1, 2, 1, 2],
            "exposure_count": [10, 10, 30, 30],
            "visibility_observation_count": [5, 5, 15, 15],
            "kappa_hat_side": [0.2, 0.2, 0.6, 0.6],
            "lambda_hat_side": [0.1, 0.1, 0.5, 0.5],
        }
    )

    summary = summarise_instrument_kernel(kernel).to_dicts()[0]

    assert summary["instrument_id"] == "ABC"
    assert summary["kappa_hat_instrument"] == pytest.approx(0.5)
    assert summary["lambda_hat_instrument"] == pytest.approx(0.4)
