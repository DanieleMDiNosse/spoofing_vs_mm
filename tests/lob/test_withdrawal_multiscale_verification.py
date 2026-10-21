"""Independent reconstruction checks, not merely a second call to production."""
import importlib

import polars as pl
import pytest

from tests.lob.test_withdrawal_multiscale import frames, NS
from spoofing_detection.lob.withdrawal_multiscale import evaluate_cells, add_priority_flags

POLICIES = [{"id": "all", "max_age_ns": None, "half_life_ns": None},
            {"id": "exp30", "max_age_ns": None, "half_life_ns": 30 * NS}]


def sample():
    e, c, l, x = frames()
    p, a = evaluate_cells(e, c, l, x, [NS, 2 * NS], POLICIES)
    return e, c, x, p, a


def verifier():
    return importlib.import_module("spoofing_detection.lob.withdrawal_multiscale_verification")


def test_independent_accounting_reconstructs_all_cells():
    e, c, x, p, a = sample()
    receipt = verifier().verify_accounting(e, c, x, p, a, [NS, 2 * NS], POLICIES)
    assert receipt["status"] == "verified"
    assert receipt["execution_cells_checked"] == 4


@pytest.mark.parametrize("field", ["withdrawal_qty_raw", "withdrawal_qty_age_weighted", "WMSCI_market",
                                    "opposite_market_visible_qty_pre_topN"])
def test_independent_accounting_rejects_changed_quantities(field):
    e, c, x, p, a = sample()
    p = p.with_columns((pl.col(field) + 1).alias(field))
    with pytest.raises(ValueError):
        verifier().verify_accounting(e, c, x, p, a, [NS, 2 * NS], POLICIES)


def test_independent_accounting_rejects_null_winner_flag():
    e, c, x, p, a = sample()
    a = a.with_columns(pl.lit(None, dtype=pl.Boolean).alias("assigned_flag"))
    with pytest.raises(ValueError):
        verifier().verify_accounting(e, c, x, p, a, [NS, 2 * NS], POLICIES)


@pytest.mark.parametrize("field,replacement", [
    ("cluster_first_sort_index", 999), ("cluster_last_sort_index", 999),
    ("age_ns", 123), ("deceptive_order_first_seen_sort_index", 999),
    ("cancel_order_first_seen_sort_index", 999), ("cancel_visible_qty", 999.0),
    ("candidate_visible_qty_pre", 999.0), ("assignment_rule", "forged"),
])
def test_independent_accounting_checks_source_audit_fields(field, replacement):
    e, c, x, p, a = sample()
    a = a.with_columns(pl.lit(replacement).cast(a.schema[field]).alias(field))
    with pytest.raises(ValueError):
        verifier().verify_accounting(e, c, x, p, a, [NS, 2 * NS], POLICIES)


def test_independent_accounting_avoids_finite_market_score_overflow():
    e, c, l, x = frames()
    e = e.with_columns(pl.lit(1.0).alias("execution_quantity"),
        pl.lit(1e308).alias("opposite_market_visible_qty_pre_topN"))
    c = c.with_columns(pl.lit(1e308).alias("deceptive_order_visible_qty_pre"))
    l = l.with_columns(pl.lit(1e308).alias("cancel_visible_qty"))
    x = x.with_columns(pl.lit(1e308).alias("visible_qty_pre_cancel"))
    policies = [POLICIES[0]]
    p, a = evaluate_cells(e, c, l, x, [2 * NS], policies)
    assert verifier().verify_accounting(e, c, x, p, a, [2 * NS], policies)["status"] == "verified"


def test_independent_priorities_rejects_corruption_and_preserves_zero_j():
    _, _, _, p, _ = sample()
    p = p.with_columns(pl.lit(0.0).alias("J"), pl.lit(True).alias("complete_grid"))
    p = add_priority_flags(p, 0.1)
    assert verifier().verify_priorities(p, 0.1)["status"] == "verified"
    altered = p.with_columns((~pl.col("priority_top_w_self")).alias("priority_top_w_self"))
    with pytest.raises(ValueError):
        verifier().verify_priorities(altered, 0.1)
