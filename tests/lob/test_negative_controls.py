from __future__ import annotations

import polars as pl

from spoofing_detection.lob.negative_controls import add_time_shift_placebo, add_wrong_side_placebo


def test_add_time_shift_placebo_offsets_sort_index():
    events = pl.DataFrame({"review_event_id": ["S1"], "sort_index": [100], "client_id": ["A"]})
    shifted = add_time_shift_placebo(events, shift_events=50)
    row = shifted.row(0, named=True)
    assert row["placebo_type"] == "time_shift"
    assert row["placebo_sort_index"] == 150


def test_add_wrong_side_placebo_flips_side():
    events = pl.DataFrame({"review_event_id": ["S1"], "execution_side": ["ask"], "deceptive_side": ["bid"]})
    wrong = add_wrong_side_placebo(events)
    row = wrong.row(0, named=True)
    assert row["placebo_type"] == "wrong_side"
    assert row["placebo_deceptive_side"] == "ask"
