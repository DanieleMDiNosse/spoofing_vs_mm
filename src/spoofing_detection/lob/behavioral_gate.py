from __future__ import annotations

import math
from typing import Any

import polars as pl


def _positive_finite(value: Any) -> bool:
    if value is None:
        return False
    numeric = float(value)
    return math.isfinite(numeric) and numeric > 0


def attach_spoofing_compatible_sequence_gate(
    executions: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Attach a transparent same-episode gate and return the passing events.

    The gate is not a label of intent. It requires a rapid attributed
    cancellation, a fill smaller than the withdrawn quantity, a favorable
    pre-fill mid move, and positive cancel-anchored mid reversion.
    """

    gate_columns = (
        "gate_rapid_matched_withdrawal",
        "gate_small_fill_relative_to_withdrawal",
        "gate_favorable_pre_fill_move",
        "gate_cancel_anchored_reversion",
        "spoofing_compatible_sequence",
    )
    if executions.is_empty():
        enriched = executions.clone()
        for column in gate_columns:
            enriched = enriched.with_columns(pl.Series(column, [], dtype=pl.Boolean))
        return enriched, enriched.clone()

    rows: list[dict[str, Any]] = []
    for execution in executions.iter_rows(named=True):
        rapid = bool(execution.get("has_matched_deceptive_cancel_window"))
        fill_qty = execution.get("fill_qty")
        withdrawn_qty = execution.get("matched_deceptive_cancel_visible_qty_window")
        small_fill = (
            _positive_finite(fill_qty)
            and _positive_finite(withdrawn_qty)
            and float(fill_qty) < float(withdrawn_qty)
        )
        favorable_pre_move = _positive_finite(execution.get("favorable_mid_move_pre_fill"))
        cancel_reversion = _positive_finite(execution.get("post_cancel_mid_reversion"))
        sequence = rapid and small_fill and favorable_pre_move and cancel_reversion
        row = {
            **execution,
            "gate_rapid_matched_withdrawal": rapid,
            "gate_small_fill_relative_to_withdrawal": small_fill,
            "gate_favorable_pre_fill_move": favorable_pre_move,
            "gate_cancel_anchored_reversion": cancel_reversion,
            "spoofing_compatible_sequence": sequence,
        }
        rows.append(row)

    enriched = pl.DataFrame(rows, infer_schema_length=None)
    return enriched, enriched.filter(pl.col("spoofing_compatible_sequence"))
