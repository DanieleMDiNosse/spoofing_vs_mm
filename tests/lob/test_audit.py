from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from spoofing_detection.lob.audit import audit_dataframe, audit_paths, write_audit_report


def minimal_event(**overrides):
    row = {
        "TRADEDATE": "2024-01-02",
        "MIC": "XMIL",
        "MARKETCODE": "MTA",
        "SYMBOLINDEX": 123,
        "EMM (*)": 1,
        "SEQUENCETIME": 1,
        "HDR_APPLKEYSEQUENCENUMBER": 1,
        "HDR_HWMSEQUENCENUMBER": 1,
        "HDR_OFFSETID": 1,
        "ROW_NUMBER": 1,
        "ORDEREVENTTYPE (*)": 1,
        "ORDERID": "O1",
        "ORDERPRIORITY": "1",
        "ORDERSIDE (*)": 1,
        "ORDERPX": 100.0,
        "ORDERQTY": 10,
        "DISPLAYEDQTY": 10,
        "LEAVESQTY": 10,
        "ORDERTYPE (*)": 2,
        "TIMEINFORCE (*)": 0,
        "FIRMID": "F1",
        "NMSC_ORIGINALCLIENTIDSHORTCODE": "C1",
        "PASSIVEORDER": True,
        "AGGRESSIVEORDER": False,
    }
    row.update(overrides)
    return row


def test_audit_dataframe_reports_missing_required_columns_and_unknown_enums():
    df = pl.DataFrame([
        minimal_event(**{"ORDEREVENTTYPE (*)": 999}),
    ]).drop("ORDERID")

    audit = audit_dataframe(df, source_name="synthetic.parquet")

    assert audit["source_name"] == "synthetic.parquet"
    assert audit["row_count"] == 1
    assert "ORDERID" in audit["missing_required_columns"]
    event_audit = audit["enum_fields"]["ORDEREVENTTYPE (*)"]
    assert event_audit["observed_codes"] == [999]
    assert event_audit["unknown_codes"] == [999]
    assert audit["identity_fields"]["FIRMID"]["missing_count"] == 0
    assert audit["passive_aggressive_fields"]["PASSIVEORDER"]["observed_values"] == ["True"]


def test_audit_paths_and_report_write_json_and_markdown(tmp_path: Path):
    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"
    pl.DataFrame([minimal_event()]).write_parquet(first)
    pl.DataFrame([minimal_event(**{"NMSC_ORIGINALCLIENTIDSHORTCODE": None})]).write_parquet(second)

    audit = audit_paths([first, second])
    paths = write_audit_report(audit, tmp_path / "audit")

    assert paths["json"].exists()
    assert paths["markdown"].exists()

    saved = json.loads(paths["json"].read_text())
    markdown = paths["markdown"].read_text()

    assert saved["files_audited"] == 2
    assert saved["aggregate"]["row_count"] == 2
    assert saved["aggregate"]["identity_missing_counts"]["NMSC_ORIGINALCLIENTIDSHORTCODE"] == 1
    assert "# LOB Schema and Enum Audit" in markdown
    assert "Unknown enum codes" in markdown
    assert "PASSIVEORDER" in markdown
    assert "AGGRESSIVEORDER" in markdown
