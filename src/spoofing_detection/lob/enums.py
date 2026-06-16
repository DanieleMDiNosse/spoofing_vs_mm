from __future__ import annotations

import math
import re
from typing import Mapping, TypeVar

T = TypeVar("T")


class UnknownEnumError(ValueError):
    """Raised when an accepted provisional enum mapping is missing a code."""


EVENT_LABEL_BY_CODE: dict[int, str] = {
    1: "New",
    2: "Modify",
    3: "Fill",
    4: "Cancel",
    6: "Trigger",
    7: "Refill",
    9: "VFA_VFC",
    11: "GTC_GTD_Reload",
    23: "Move_Dark_to_COB",
}

EVENT_CLASS_BY_CODE: dict[int, str] = {
    1: "new_order",
    2: "modify_order",
    3: "fill",
    4: "cancel",
    6: "trigger",
    7: "iceberg_refill",
    9: "special_validity_event",
    11: "session_reload",
    23: "move_dark_to_cob",
}

ORDER_SIDE_BY_CODE: dict[int, str] = {
    1: "bid",
    2: "ask",
}

ORDER_TYPE_BY_CODE: dict[int, str] = {
    1: "market",
    2: "limit",
    3: "stop_market_or_stop_market_on_quote",
    4: "stop_limit_or_stop_limit_on_quote",
    8: "mid_point_peg",
    10: "iceberg",
}

TIME_IN_FORCE_BY_CODE: dict[int, str] = {
    0: "day",
    1: "good_till_cancel",
    2: "valid_for_uncrossing",
    3: "immediate_or_cancel",
    4: "fill_or_kill",
    6: "good_till_date",
    7: "valid_for_closing_uncrossing",
}

KILL_REASON_BY_CODE: dict[int, str] = {
    1: "order_cancelled_by_client",
    2: "order_expired",
    3: "order_cancelled_by_market_operations",
    5: "done_for_day",
    7: "cancelled_by_stp",
    8: "remaining_quantity_killed_ioc",
    11: "cancelled_due_to_cancel_on_disconnect",
    41: "cancelled_due_to_order_price_control_collar_breach",
}

ACCOUNT_TYPE_INTERNAL_BY_CODE: dict[int, str] = {
    1: "client",
    2: "house",
    6: "liquidity_provider",
}

LP_ROLE_BY_CODE: dict[int, str] = {
    1: "liquidity_provider_or_market_maker",
}

TRADING_CAPACITY_BY_CODE: dict[int, str] = {
    1: "dealing_on_own_account",
    2: "matched_principal",
    3: "any_other_capacity",
}

TRADE_TYPE_BY_CODE: dict[int, str] = {
    1: "conventional_trade",
    33: "dark_trade",
}

EXECUTION_PHASE_BY_CODE: dict[int, str] = {
    1: "continuous_trading_phase",
    2: "uncrossing_phase",
    3: "observed_unknown_phase_3",
}

ACK_TYPE_BY_CODE: dict[int, str] = {
    0: "new_order_ack",
    1: "replace_ack",
    3: "stop_triggered_ack",
    5: "refilled_iceberg_ack",
    14: "iceberg_transformed_to_limit",
    16: "vfu_vfc_triggered_ack",
    18: "reload_ack",
    23: "move_dark_to_cob_as_limit",
}

ACK_PHASE_BY_CODE: dict[int, str] = {
    1: "new",
    2: "modify",
    5: "reject",
    6: "trigger",
}


def normalize_enum_code(value) -> int | None:
    """Normalize numeric enum values and tooltip strings such as '1 : New'."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        raise ValueError(f"non-integer enum float: {value!r}")
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "nan"}:
        return None
    match = re.match(r"^([+-]?\d+)(?:\.0+)?(?:\s*:.*)?$", text)
    if match:
        return int(match.group(1))
    raise ValueError(f"cannot parse enum code from {value!r}")


def require_known(value, mapping: Mapping[int, T], *, field_name: str, strict: bool = True) -> T | None:
    code = normalize_enum_code(value)
    if code is None:
        return None
    if code in mapping:
        return mapping[code]
    message = f"Unknown {field_name} enum code: {code!r}"
    if strict:
        raise UnknownEnumError(message)
    return None
