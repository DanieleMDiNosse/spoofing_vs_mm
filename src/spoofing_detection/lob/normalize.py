from __future__ import annotations

import math
from typing import Any

from .config import LOBConfig
from .enums import (
    EVENT_CLASS_BY_CODE,
    EVENT_LABEL_BY_CODE,
    KILL_REASON_BY_CODE,
    ORDER_SIDE_BY_CODE,
    ORDER_TYPE_BY_CODE,
    TIME_IN_FORCE_BY_CODE,
    normalize_enum_code,
    require_known,
)


NON_VISIBLE_UNPRICED_ORDER_TYPES = {
    "market",
    "stop_market_or_stop_market_on_quote",
    "stop_limit_or_stop_limit_on_quote",
    "mid_point_peg",
}


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null", "nan"}:
        return True
    return False


def get_first(row: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in row and not _is_missing(row[name]):
            return row[name]
    return None


def to_float(value: Any) -> float | None:
    if _is_missing(value):
        return None
    return float(value)


def to_str_or_none(value: Any) -> str | None:
    if _is_missing(value):
        return None
    # Avoid converting integral floats to strings with a trailing .0 for IDs.
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def join_flags(flags: list[str]) -> str | None:
    if not flags:
        return None
    return ";".join(dict.fromkeys(flags))


def normalize_event(row: dict[str, Any], *, sort_index: int, config: LOBConfig) -> dict[str, Any]:
    event_type_code = normalize_enum_code(
        get_first(row, "ORDEREVENTTYPE (*)", "ORDEREVENTTYPE", "ORDEREVENTTYPE (*) (Tooltip)")
    )
    event_class = require_known(
        event_type_code, EVENT_CLASS_BY_CODE, field_name="ORDEREVENTTYPE", strict=config.strict_enums
    )
    event_type_label = EVENT_LABEL_BY_CODE.get(event_type_code) if event_type_code is not None else None

    side_code = normalize_enum_code(get_first(row, "ORDERSIDE (*)", "ORDERSIDE", "ORDERSIDE (*) (Tooltip)"))
    side_label = require_known(
        side_code, ORDER_SIDE_BY_CODE, field_name="ORDERSIDE", strict=config.strict_enums
    ) if side_code is not None else None

    order_type_code = normalize_enum_code(get_first(row, "ORDERTYPE (*)", "ORDERTYPE", "ORDERTYPE (*) (Tooltip)"))
    order_type_label = require_known(
        order_type_code, ORDER_TYPE_BY_CODE, field_name="ORDERTYPE", strict=config.strict_enums
    ) if order_type_code is not None else None

    tif_code = normalize_enum_code(get_first(row, "TIMEINFORCE (*)", "TIMEINFORCE", "TIMEINFORCE (*) (Tooltip)"))
    tif_label = require_known(
        tif_code, TIME_IN_FORCE_BY_CODE, field_name="TIMEINFORCE", strict=config.strict_enums
    ) if tif_code is not None else None

    kill_code = normalize_enum_code(get_first(row, "KILLREASON (*)", "KILLREASON", "KILLREASON (*) (Tooltip)"))
    kill_label = require_known(
        kill_code, KILL_REASON_BY_CODE, field_name="KILLREASON", strict=config.strict_enums
    ) if kill_code is not None else None

    price = to_float(get_first(row, "ORDERPX"))
    order_qty = to_float(get_first(row, "ORDERQTY"))
    displayed_qty = to_float(get_first(row, "DISPLAYEDQTY"))
    leaves_qty = to_float(get_first(row, "LEAVESQTY"))
    last_shares = to_float(get_first(row, "LASTSHARES"))
    last_px = to_float(get_first(row, "LASTTRADEDPX"))
    trading_capacity_code = normalize_enum_code(
        get_first(row, "ORDER_TRADINGCAPACITY (*)", "ORDER_TRADINGCAPACITY")
    )
    trading_capacity_label = get_first(row, "ORDER_TRADINGCAPACITY (*) (Tooltip)")

    firm_id = to_str_or_none(get_first(row, "FIRMID"))
    client_original_id = to_str_or_none(get_first(row, "NMSC_ORIGINALCLIENTIDSHORTCODE"))

    flags: list[str] = []
    if client_original_id is None:
        flags.append("missing_client_original_id")
    if firm_id is None:
        flags.append("missing_firm_id")
    if price is None and order_type_label in NON_VISIBLE_UNPRICED_ORDER_TYPES:
        flags.append("non_resting_unpriced_event")
    elif price is None and event_class in {"new_order", "session_reload", "modify_order"}:
        flags.append("missing_price_for_potential_resting_event")

    return {
        "sort_index": sort_index,
        "TRADEDATE": get_first(row, "TRADEDATE"),
        "MIC": get_first(row, "MIC"),
        "MARKETCODE": get_first(row, "MARKETCODE"),
        "SYMBOLINDEX": get_first(row, "SYMBOLINDEX"),
        "EMM (*)": get_first(row, "EMM (*)", "EMM"),
        "ISIN": get_first(row, "ISIN"),
        "SEQUENCETIME": get_first(row, "SEQUENCETIME"),
        "BOOKIN": get_first(row, "BOOKIN"),
        "BOOKOUTTIME": get_first(row, "BOOKOUTTIME"),
        "TRADETIME": get_first(row, "TRADETIME"),
        "HDR_APPLKEYSEQUENCENUMBER": get_first(row, "HDR_APPLKEYSEQUENCENUMBER"),
        "HDR_HWMSEQUENCENUMBER": get_first(row, "HDR_HWMSEQUENCENUMBER"),
        "HDR_OFFSETID": get_first(row, "HDR_OFFSETID"),
        "ROW_NUMBER": get_first(row, "ROW_NUMBER"),
        "EVENTID": get_first(row, "EVENTID"),
        "ORDERID": to_str_or_none(get_first(row, "ORDERID")),
        "ORDERPRIORITY": to_str_or_none(get_first(row, "ORDERPRIORITY")),
        "CLIENTORDERID": to_str_or_none(get_first(row, "CLIENTORDERID")),
        "ORIGCLIENTORDERID": to_str_or_none(get_first(row, "ORIGCLIENTORDERID")),
        "EXECUTIONID": get_first(row, "EXECUTIONID"),
        "TRADEUNIQUEIDENTIFIER": get_first(row, "TRADEUNIQUEIDENTIFIER"),
        "event_type_code": event_type_code,
        "event_type_label_observed": get_first(row, "ORDEREVENTTYPE (*) (Tooltip)") or event_type_label,
        "event_class": event_class,
        "side_code": side_code,
        "side_label": side_label,
        "event_order_type_code": order_type_code,
        "event_order_type_label": order_type_label,
        "time_in_force_code": tif_code,
        "time_in_force_label": tif_label,
        "kill_reason_code": kill_code,
        "kill_reason_label": kill_label,
        "ORDERPX": price,
        "ORDERQTY": order_qty,
        "DISPLAYEDQTY": displayed_qty,
        "LEAVESQTY": leaves_qty,
        "LASTSHARES": last_shares,
        "LASTTRADEDPX": last_px,
        "ORDERSTATUS": get_first(row, "ORDERSTATUS"),
        "PASSIVEORDER": get_first(row, "PASSIVEORDER"),
        "AGGRESSIVEORDER": get_first(row, "AGGRESSIVEORDER"),
        "UNCROSSINGTRADE": get_first(row, "UNCROSSINGTRADE"),
        "DARKINDICATOR": get_first(row, "DARKINDICATOR"),
        "SWEEPORDERINDICATOR": get_first(row, "SWEEPORDERINDICATOR"),
        "QUOTEINDICATOR": get_first(row, "QUOTEINDICATOR"),
        "firm_id": firm_id,
        "client_original_id": client_original_id,
        "client_original_id_missing_flag": client_original_id is None,
        "FIRMID": firm_id,
        "NMSC_ORIGINALCLIENTIDSHORTCODE": client_original_id,
        "MSC_EVENTCLIENTIDSHORTCODE": to_str_or_none(get_first(row, "MSC_EVENTCLIENTIDSHORTCODE")),
        "NMSC_ORIGINALEXECWFIRMSHORTCODE": to_str_or_none(get_first(row, "NMSC_ORIGINALEXECWFIRMSHORTCODE")),
        "MSC_EVENTEXECWFIRMSHORTCODE": to_str_or_none(get_first(row, "MSC_EVENTEXECWFIRMSHORTCODE")),
        "NMSC_ORIGINALINVESTDECISWFIRMSHORTCODE": to_str_or_none(get_first(row, "NMSC_ORIGINALINVESTDECISWFIRMSHORTCODE")),
        "NMSC_ORIGINALNONEXECBROKERSHORTCODE": to_str_or_none(get_first(row, "NMSC_ORIGINALNONEXECBROKERSHORTCODE")),
        "ACCOUNTTYPEINTERNAL (*)": get_first(row, "ACCOUNTTYPEINTERNAL (*)"),
        "LPROLE (*)": get_first(row, "LPROLE (*)"),
        "order_trading_capacity_code": trading_capacity_code,
        "order_trading_capacity_label": trading_capacity_label,
        "ORDER_TRADINGCAPACITY (*)": trading_capacity_code,
        "ORDER_TRADINGCAPACITY (*) (Tooltip)": trading_capacity_label,
        "DEAINDICATOR": get_first(row, "DEAINDICATOR"),
        "INVESTMENTALGOINDICATOR": get_first(row, "INVESTMENTALGOINDICATOR"),
        "EXECUTIONALGOINDICATOR": get_first(row, "EXECUTIONALGOINDICATOR"),
        "FRMARAMPLP": get_first(row, "FRMARAMPLP"),
        "unknown_enum_flag": False,
        "normalization_issue_flags": join_flags(flags),
    }


def is_visible_resting_event(event: dict[str, Any]) -> bool:
    """Whether an event row can create/update visible resting liquidity."""
    if event["ORDERID"] is None or event["side_label"] not in {"bid", "ask"}:
        return False
    if event["ORDERPX"] is None:
        return False
    if (event["LEAVESQTY"] or 0) <= 0 or (event["DISPLAYEDQTY"] or 0) <= 0:
        return False
    if event["event_order_type_label"] == "market":
        return False
    if event["event_order_type_label"] in {
        "stop_market_or_stop_market_on_quote",
        "stop_limit_or_stop_limit_on_quote",
        "mid_point_peg",
    }:
        return False
    return True
