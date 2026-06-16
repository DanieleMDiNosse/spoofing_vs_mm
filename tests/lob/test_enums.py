import pytest

from spoofing_detection.lob.enums import (
    EVENT_CLASS_BY_CODE,
    ORDER_SIDE_BY_CODE,
    ORDER_TYPE_BY_CODE,
    TIME_IN_FORCE_BY_CODE,
    UnknownEnumError,
    normalize_enum_code,
    require_known,
)


def test_accepted_provisional_enum_mappings_are_explicit():
    assert EVENT_CLASS_BY_CODE[1] == "new_order"
    assert EVENT_CLASS_BY_CODE[2] == "modify_order"
    assert EVENT_CLASS_BY_CODE[3] == "fill"
    assert EVENT_CLASS_BY_CODE[4] == "cancel"
    assert EVENT_CLASS_BY_CODE[7] == "iceberg_refill"
    assert EVENT_CLASS_BY_CODE[11] == "session_reload"
    assert EVENT_CLASS_BY_CODE[23] == "move_dark_to_cob"

    assert ORDER_SIDE_BY_CODE[1] == "bid"
    assert ORDER_SIDE_BY_CODE[2] == "ask"

    assert ORDER_TYPE_BY_CODE[1] == "market"
    assert ORDER_TYPE_BY_CODE[2] == "limit"
    assert ORDER_TYPE_BY_CODE[10] == "iceberg"

    assert TIME_IN_FORCE_BY_CODE[0] == "day"
    assert TIME_IN_FORCE_BY_CODE[3] == "immediate_or_cancel"


def test_enum_code_normalization_accepts_numeric_and_tooltip_like_values():
    assert normalize_enum_code(1) == 1
    assert normalize_enum_code(1.0) == 1
    assert normalize_enum_code("1") == 1
    assert normalize_enum_code("1 : New") == 1
    assert normalize_enum_code(None) is None


def test_unknown_enum_codes_fail_loudly_in_strict_mode():
    with pytest.raises(UnknownEnumError, match="ORDEREVENTTYPE"):
        require_known(999, EVENT_CLASS_BY_CODE, field_name="ORDEREVENTTYPE")
