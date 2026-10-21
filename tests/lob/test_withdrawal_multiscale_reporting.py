"""Report populations must retain candidates and distinguish common support."""
import importlib

import polars as pl

from tests.lob.test_withdrawal_multiscale import frames, NS
from spoofing_detection.lob.withdrawal_multiscale import evaluate_cells, add_priority_flags


def population():
    e,c,l,x = frames()
    p,_ = evaluate_cells(e,c,l,x,[NS,2*NS],[{"id":"all","max_age_ns":None,"half_life_ns":None}])
    p = p.with_columns(pl.lit(0.0).alias("J"), pl.lit(True).alias("complete_grid"),
        pl.lit("complete").alias("profile_status"))
    return add_priority_flags(p,0.1)


def api():
    return importlib.import_module("spoofing_detection.lob.withdrawal_multiscale_reporting")


def test_report_counts_keep_unselected_and_common_support_uses_union():
    p=population()
    tables=api().build_tables(p)
    assert set(tables)=={"population_counts","priority_counts","common_support_counts","score_components","review"}
    assert tables["population_counts"]["total_execution_rows"].to_list()==[1,1]
    assert tables["population_counts"]["raw_candidates"].to_list()==[0,1]
    assert tables["review"].height==1
    assert tables["common_support_counts"]["union_candidates"].to_list()==[1,1]
    assert tables["common_support_counts"]["common_complete_support"].to_list()==[1,1]


def test_report_support_missing_is_not_an_absent_candidate():
    p=population().with_columns(pl.when(pl.col("horizon_ns")==2*NS).then(False).otherwise(True).alias("complete_grid"))
    tables=api().build_tables(p)
    assert tables["review"].height==1
    assert tables["population_counts"]["incomplete_J_candidates"].to_list()==[0,1]
    assert tables["common_support_counts"]["common_complete_support"].to_list()==[0,0]
