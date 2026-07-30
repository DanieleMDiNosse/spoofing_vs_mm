from __future__ import annotations

import polars as pl
import pytest

from spoofing_detection.lob.behavioral_gate import attach_spoofing_compatible_sequence_gate


def _execution(cluster_id: str, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "partition_id": "P",
        "actor_key": "client_original:C1",
        "execution_anchor_mode": "passive",
        "execution_cluster_id": cluster_id,
        "fill_qty": 10.0,
        "has_matched_deceptive_cancel_window": True,
        "matched_deceptive_cancel_visible_qty_window": 50.0,
        "favorable_mid_move_pre_fill": 0.10,
        "post_cancel_mid_reversion": 0.05,
    }
    row.update(overrides)
    return row


def test_gate_requires_all_interpretable_same_episode_components():
    executions = pl.DataFrame(
        [
            _execution("PASS"),
            _execution("NO_RAPID", has_matched_deceptive_cancel_window=False),
            _execution("NOT_SMALL", fill_qty=50.0),
            _execution("NO_PRE_MOVE", favorable_mid_move_pre_fill=0.0),
            _execution("NO_REVERSION", post_cancel_mid_reversion=None),
        ],
        infer_schema_length=None,
    )

    enriched, gated = attach_spoofing_compatible_sequence_gate(executions)
    by_id = {row["execution_cluster_id"]: row for row in enriched.to_dicts()}

    assert by_id["PASS"]["gate_rapid_matched_withdrawal"] is True
    assert by_id["PASS"]["gate_small_fill_relative_to_withdrawal"] is True
    assert by_id["PASS"]["gate_favorable_pre_fill_move"] is True
    assert by_id["PASS"]["gate_cancel_anchored_reversion"] is True
    assert by_id["PASS"]["spoofing_compatible_sequence"] is True

    assert by_id["NO_RAPID"]["spoofing_compatible_sequence"] is False
    assert by_id["NOT_SMALL"]["spoofing_compatible_sequence"] is False
    assert by_id["NO_PRE_MOVE"]["spoofing_compatible_sequence"] is False
    assert by_id["NO_REVERSION"]["spoofing_compatible_sequence"] is False
    assert gated["execution_cluster_id"].to_list() == ["PASS"]
    assert "composite_score" not in enriched.columns
    assert "gate_conditioned_withdrawal_excess" not in enriched.columns
    assert "mcnemar_exact_one_sided_pvalue" not in enriched.columns


def test_gate_preserves_schema_for_empty_execution_frame():
    enriched, gated = attach_spoofing_compatible_sequence_gate(pl.DataFrame())

    assert enriched.is_empty()
    assert gated.is_empty()
    assert gated.schema == enriched.schema


@pytest.mark.parametrize("execution_anchor_mode", ["passive", "aggressive"])
def test_gate_semantics_and_anchor_provenance_are_branch_invariant(
    execution_anchor_mode: str,
):
    executions = pl.DataFrame(
        [_execution("PASS", execution_anchor_mode=execution_anchor_mode)],
        infer_schema_length=None,
    )

    enriched, gated = attach_spoofing_compatible_sequence_gate(executions)

    assert enriched.item(0, "spoofing_compatible_sequence") is True
    assert gated.item(0, "execution_anchor_mode") == execution_anchor_mode
