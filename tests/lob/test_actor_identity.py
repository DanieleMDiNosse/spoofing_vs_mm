from __future__ import annotations

import math
from decimal import Decimal

import pytest

from spoofing_detection.lob.actor_identity import (
    actor_identity_from_event,
    actor_identity_from_order,
    normalize_identity_value,
    resolve_actor_identity,
    same_actor,
)
from spoofing_detection.lob.models import ActiveOrder


def active_order(*, client_original_id, firm_id) -> ActiveOrder:
    return ActiveOrder(
        order_id="O1",
        side="bid",
        price=100.0,
        leaves_qty=10.0,
        displayed_qty=10.0,
        order_qty=10.0,
        order_priority="1",
        order_type_code=2,
        order_type_label="limit",
        time_in_force_code=0,
        firm_id=firm_id,
        client_original_id=client_original_id,
        first_seen_sort_index=1,
        last_update_sort_index=1,
        last_event_class="new_order",
    )


def test_client_identity_has_priority_over_firm():
    actor = resolve_actor_identity(client_original_id="31425", firm_id="130358")

    assert actor is not None
    assert actor.actor_key == "client_original:31425"
    assert actor.actor_id == "31425"
    assert actor.identity_level == "client_original"
    assert actor.identity_source == "NMSC_ORIGINALCLIENTIDSHORTCODE"
    assert actor.identity_fallback_flag is False


def test_missing_client_falls_back_to_firm_without_calling_it_client():
    actor = resolve_actor_identity(client_original_id=None, firm_id="157922_3")

    assert actor is not None
    assert actor.actor_key == "firm:157922_3"
    assert actor.actor_id == "157922_3"
    assert actor.identity_level == "firm"
    assert actor.identity_source == "FIRMID"
    assert actor.identity_fallback_flag is True


@pytest.mark.parametrize(
    "client_sentinel",
    [0, 0.0, -0.0, "0", " 0 ", "0.0", "0.00", "0e0", "-0.00", Decimal("0.00")],
)
def test_zero_client_sentinel_falls_back_to_firm(client_sentinel):
    actor = resolve_actor_identity(client_original_id=client_sentinel, firm_id="F1")

    assert actor is not None
    assert actor.actor_key == "firm:F1"
    assert actor.actor_id == "F1"
    assert actor.identity_level == "firm"
    assert actor.identity_source == "FIRMID"
    assert actor.identity_fallback_flag is True


def test_zero_client_sentinel_without_firm_is_unattributable():
    assert resolve_actor_identity(client_original_id=0.0, firm_id=None) is None


def test_zero_client_sentinel_does_not_merge_distinct_firms():
    first = resolve_actor_identity(client_original_id=0.0, firm_id="F1")
    second = resolve_actor_identity(client_original_id=0.0, firm_id="F2")

    assert same_actor(first, second) is False


def test_zero_firm_identifier_is_preserved_when_client_is_missing():
    actor = resolve_actor_identity(client_original_id=None, firm_id=0.0)

    assert actor is not None
    assert actor.actor_key == "firm:0"
    assert actor.identity_level == "firm"


def test_namespaces_prevent_raw_identifier_collision():
    client = resolve_actor_identity(client_original_id="123", firm_id="F")
    firm = resolve_actor_identity(client_original_id=None, firm_id="123")

    assert client is not None
    assert firm is not None
    assert client.actor_key != firm.actor_key


def test_missing_both_identities_returns_none():
    assert resolve_actor_identity(client_original_id=" ", firm_id=None) is None


@pytest.mark.parametrize("value", [None, "", " \t ", "null", " NONE ", "NaN", math.nan])
def test_normalize_identity_value_treats_missing_and_sentinel_forms_as_none(value):
    assert normalize_identity_value(value) is None


def test_normalize_identity_value_trims_text_and_removes_spurious_integral_float_suffix():
    assert normalize_identity_value("  C-31425  ") == "C-31425"
    assert normalize_identity_value(31425) == "31425"
    assert normalize_identity_value(31425.0) == "31425"
    assert normalize_identity_value(31425.5) == "31425.5"
    assert normalize_identity_value("31425.0") == "31425.0"


def test_event_extraction_uses_normalized_event_identity_fields():
    actor = actor_identity_from_event(
        {
            "client_original_id": " C1 ",
            "firm_id": "F1",
            "MSC_EVENTCLIENTIDSHORTCODE": "different-client",
        }
    )

    assert actor is not None
    assert actor.actor_key == "client_original:C1"
    assert actor.identity_source == "NMSC_ORIGINALCLIENTIDSHORTCODE"


def test_event_extraction_does_not_fall_back_to_event_client_shortcode():
    assert actor_identity_from_event(
        {
            "client_original_id": None,
            "firm_id": None,
            "MSC_EVENTCLIENTIDSHORTCODE": "different-client",
        }
    ) is None


def test_order_extraction_resolves_active_order_identity():
    actor = actor_identity_from_order(active_order(client_original_id=None, firm_id="F1"))

    assert actor is not None
    assert actor.actor_key == "firm:F1"
    assert actor.identity_level == "firm"


def test_same_actor_requires_matching_namespaced_identity_keys():
    client = resolve_actor_identity(client_original_id="C1", firm_id="F1")
    same_client = resolve_actor_identity(client_original_id="C1", firm_id="other-firm")
    firm_with_same_raw_id = resolve_actor_identity(client_original_id=None, firm_id="C1")

    assert same_actor(client, same_client) is True
    assert same_actor(client, firm_with_same_raw_id) is False
    assert same_actor(client, None) is False
