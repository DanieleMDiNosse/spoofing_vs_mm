from __future__ import annotations

from typing import Any

import polars as pl

from .actor_identity import (
    is_zero_client_original_identity_sentinel,
    normalize_client_original_identity_value,
)

CLIENT_ID_COL = "NMSC_ORIGINALCLIENTIDSHORTCODE"
CAPACITY_CODE_COL = "ORDER_TRADINGCAPACITY (*)"
CAPACITY_TOOLTIP_COL = "ORDER_TRADINGCAPACITY (*) (Tooltip)"
OWN_ACCOUNT_CODE = 1
OWN_ACCOUNT_TOKEN = "dealing_on_own_account"


def _missing_expr(column: str) -> pl.Expr:
    return pl.col(column).map_elements(
        normalize_client_original_identity_value,
        return_dtype=pl.String,
        skip_nulls=False,
    ).is_null()


def _zero_sentinel_expr(column: str) -> pl.Expr:
    return pl.col(column).map_elements(
        is_zero_client_original_identity_sentinel,
        return_dtype=pl.Boolean,
        skip_nulls=False,
    )


def audit_missing_client_trading_capacity(df: pl.DataFrame) -> dict[str, Any]:
    """Check whether missing client ids coincide with own-account capacity.

    The tooltip column is present in some extracts but absent in others. The
    numeric trading-capacity code is therefore the hard requirement; tooltip
    consistency is checked only when the column exists.
    """

    if CLIENT_ID_COL not in df.columns:
        raise ValueError(f"missing required column: {CLIENT_ID_COL}")
    if CAPACITY_CODE_COL not in df.columns:
        raise ValueError(f"missing required column: {CAPACITY_CODE_COL}")

    has_tooltip = CAPACITY_TOOLTIP_COL in df.columns
    missing_client = _missing_expr(CLIENT_ID_COL)
    zero_client_sentinel = _zero_sentinel_expr(CLIENT_ID_COL)
    capacity_code = pl.col(CAPACITY_CODE_COL).cast(pl.Int64, strict=False)
    bad_capacity = missing_client & (capacity_code.fill_null(-1) != OWN_ACCOUNT_CODE)

    if has_tooltip:
        tooltip_contains_own_account = (
            pl.col(CAPACITY_TOOLTIP_COL)
            .cast(pl.Utf8, strict=False)
            .str.to_lowercase()
            .str.contains(OWN_ACCOUNT_TOKEN, literal=True)
            .fill_null(False)
        )
        bad_tooltip = missing_client & ~tooltip_contains_own_account
    else:
        bad_tooltip = pl.lit(False)

    summary = df.select(
        pl.len().alias("rows"),
        zero_client_sentinel.sum().alias("zero_client_sentinel_rows"),
        missing_client.sum().alias("missing_client_rows"),
        bad_capacity.sum().alias("missing_client_bad_capacity_rows"),
        bad_tooltip.sum().alias("missing_client_bad_tooltip_rows"),
    ).to_dicts()[0]
    out = {key: int(value) for key, value in summary.items()}
    out["tooltip_available"] = has_tooltip
    out["claim_holds"] = (
        out["missing_client_bad_capacity_rows"] == 0
        and out["missing_client_bad_tooltip_rows"] == 0
    )
    return out
