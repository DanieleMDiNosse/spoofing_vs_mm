from __future__ import annotations

import polars as pl

from spoofing_detection.lob.client_identity_audit import audit_missing_client_trading_capacity
from spoofing_detection.lob.config import LOBConfig
from spoofing_detection.lob.normalize import normalize_event


def minimal_raw_event(**overrides):
    row = {
        "TRADEDATE": "2024-01-02",
        "MIC": "XMIL",
        "MARKETCODE": "MTA",
        "SYMBOLINDEX": 123,
        "EMM (*)": 1,
        "ORDEREVENTTYPE (*)": 1,
        "ORDERID": "O1",
        "ORDERPRIORITY": "1",
        "ORDERSIDE (*)": 1,
        "ORDERPX": 100.0,
        "ORDERQTY": 10.0,
        "DISPLAYEDQTY": 10.0,
        "LEAVESQTY": 10.0,
        "ORDERTYPE (*)": 2,
        "TIMEINFORCE (*)": 0,
        "FIRMID": "F1",
        "NMSC_ORIGINALCLIENTIDSHORTCODE": None,
        "ORDER_TRADINGCAPACITY (*)": 1,
        "ORDER_TRADINGCAPACITY (*) (Tooltip)": "1 : Dealing_on_own_account",
    }
    row.update(overrides)
    return row


def test_normalize_event_preserves_order_trading_capacity_code_and_tooltip():
    event = normalize_event(minimal_raw_event(), sort_index=1, config=LOBConfig())

    assert event["order_trading_capacity_code"] == 1
    assert event["order_trading_capacity_label"] == "1 : Dealing_on_own_account"
    assert event["ORDER_TRADINGCAPACITY (*)"] == 1
    assert event["ORDER_TRADINGCAPACITY (*) (Tooltip)"] == "1 : Dealing_on_own_account"


def test_audit_missing_client_trading_capacity_accepts_own_account_rows():
    df = pl.DataFrame(
        {
            "NMSC_ORIGINALCLIENTIDSHORTCODE": [None, "C1"],
            "ORDER_TRADINGCAPACITY (*)": [1, 3],
            "ORDER_TRADINGCAPACITY (*) (Tooltip)": [
                "1 : Dealing_on_own_account",
                "3 : Any_other_capacity",
            ],
        }
    )

    audit = audit_missing_client_trading_capacity(df)

    assert audit["missing_client_rows"] == 1
    assert audit["missing_client_bad_capacity_rows"] == 0
    assert audit["missing_client_bad_tooltip_rows"] == 0
    assert audit["tooltip_available"] is True
    assert audit["claim_holds"] is True


def test_audit_missing_client_trading_capacity_reports_bad_capacity():
    df = pl.DataFrame(
        {
            "NMSC_ORIGINALCLIENTIDSHORTCODE": [None],
            "ORDER_TRADINGCAPACITY (*)": [3],
            "ORDER_TRADINGCAPACITY (*) (Tooltip)": ["3 : Any_other_capacity"],
        }
    )

    audit = audit_missing_client_trading_capacity(df)

    assert audit["missing_client_rows"] == 1
    assert audit["missing_client_bad_capacity_rows"] == 1
    assert audit["missing_client_bad_tooltip_rows"] == 1
    assert audit["claim_holds"] is False


def test_audit_treats_zero_client_sentinel_as_missing_and_reports_it_separately():
    df = pl.DataFrame(
        {
            "NMSC_ORIGINALCLIENTIDSHORTCODE": [0.0, None, 123.0],
            "ORDER_TRADINGCAPACITY (*)": [3, 1, 3],
            "ORDER_TRADINGCAPACITY (*) (Tooltip)": [
                "3 : Any_other_capacity",
                "1 : Dealing_on_own_account",
                "3 : Any_other_capacity",
            ],
        }
    )

    audit = audit_missing_client_trading_capacity(df)

    assert audit["zero_client_sentinel_rows"] == 1
    assert audit["missing_client_rows"] == 2
    assert audit["missing_client_bad_capacity_rows"] == 1
    assert audit["missing_client_bad_tooltip_rows"] == 1
    assert audit["claim_holds"] is False


def test_normalize_event_recognizes_scaled_and_scientific_zero_client_sentinels():
    for sentinel in ("0.00", "0e0", "-0.00"):
        event = normalize_event(
            minimal_raw_event(NMSC_ORIGINALCLIENTIDSHORTCODE=sentinel),
            sort_index=1,
            config=LOBConfig(),
        )

        assert event["client_original_id"] is None
        assert event["client_original_id_missing_flag"] is True
        assert "zero_client_original_id_sentinel" in event["normalization_issue_flags"]


def test_audit_counts_scaled_and_scientific_zero_client_sentinels():
    df = pl.DataFrame(
        {
            "NMSC_ORIGINALCLIENTIDSHORTCODE": ["0.00", "0e0", "-0.00", "C1"],
            "ORDER_TRADINGCAPACITY (*)": [1, 1, 1, 3],
        }
    )

    audit = audit_missing_client_trading_capacity(df)

    assert audit["zero_client_sentinel_rows"] == 3
    assert audit["missing_client_rows"] == 3


def test_audit_missing_client_trading_capacity_uses_code_when_tooltip_absent():
    df = pl.DataFrame(
        {
            "NMSC_ORIGINALCLIENTIDSHORTCODE": [None, "C1"],
            "ORDER_TRADINGCAPACITY (*)": [1, 3],
        }
    )

    audit = audit_missing_client_trading_capacity(df)

    assert audit["missing_client_rows"] == 1
    assert audit["missing_client_bad_capacity_rows"] == 0
    assert audit["missing_client_bad_tooltip_rows"] == 0
    assert audit["tooltip_available"] is False
    assert audit["claim_holds"] is True
