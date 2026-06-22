from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots

REQUIRED_QUOTE_COLUMNS = ("sort_index", "post_best_bid", "post_best_ask")
TOP_OF_BOOK_QTY_COLUMNS = (
    "post_bid_level_1_visible_qty",
    "post_ask_level_1_visible_qty",
)
HOVER_COLUMNS = (
    "TRADEDATE",
    "event_class",
    "event_order_type_label",
    "event_side",
    "post_best_bid",
    "post_best_ask",
    "spread",
    "relative_spread_bps",
    "lob_issue_flags",
    "normalization_issue_flags",
)
BID_LEVEL_COLORS = ("#1f77b4", "#6baed6", "#9ecae1", "#3182bd", "#08519c")
ASK_LEVEL_COLORS = ("#d62728", "#fb6a4a", "#fcae91", "#cb181d", "#99000d")
LEVEL_DASHES = ("solid", "dash", "dot", "dashdot", "longdash")


def validate_quote_panel_columns(panel: pl.DataFrame) -> None:
    missing = [column for column in REQUIRED_QUOTE_COLUMNS if column not in panel.columns]
    if missing:
        raise ValueError(f"panel is missing required quote columns: {', '.join(missing)}")


def read_panel(panel_path: str | Path) -> pl.DataFrame:
    panel_path = Path(panel_path)
    if panel_path.suffix.lower() != ".parquet":
        raise ValueError(f"expected a parquet panel, got {panel_path}")
    panel = pl.read_parquet(panel_path)
    validate_quote_panel_columns(panel)
    return panel


def compute_quote_diagnostics(panel: pl.DataFrame) -> pl.DataFrame:
    validate_quote_panel_columns(panel)

    out = panel.with_columns(
        [
            (
                pl.col("post_best_bid").is_not_null()
                & pl.col("post_best_ask").is_not_null()
                & (pl.col("post_best_bid") < pl.col("post_best_ask"))
            ).alias("valid_touch"),
            ((pl.col("post_best_bid") + pl.col("post_best_ask")) / 2.0).alias("mid_price"),
            (pl.col("post_best_ask") - pl.col("post_best_bid")).alias("spread"),
        ]
    ).with_columns(
        [
            pl.when(pl.col("mid_price").is_not_null() & (pl.col("mid_price") > 0))
            .then(pl.col("spread") / pl.col("mid_price") * 10_000.0)
            .otherwise(None)
            .alias("relative_spread_bps")
        ]
    )

    if all(column in out.columns for column in TOP_OF_BOOK_QTY_COLUMNS):
        bid_qty, ask_qty = TOP_OF_BOOK_QTY_COLUMNS
        out = out.with_columns(
            [(pl.col(bid_qty) + pl.col(ask_qty)).alias("top_of_book_visible_qty")]
        ).with_columns(
            [
                pl.when(pl.col("top_of_book_visible_qty") > 0)
                .then(pl.col(bid_qty) / pl.col("top_of_book_visible_qty"))
                .otherwise(None)
                .alias("top_of_book_imbalance")
            ]
        )
    else:
        out = out.with_columns(
            [
                pl.lit(None, dtype=pl.Float64).alias("top_of_book_visible_qty"),
                pl.lit(None, dtype=pl.Float64).alias("top_of_book_imbalance"),
            ]
        )

    return out


def collapse_session_reload_rows(panel: pl.DataFrame) -> pl.DataFrame:
    """Keep one completed session-reload snapshot per day plus all live rows.

    Session reload rows replay the opening visible book snapshot. Plotting each
    reload row as a separate live quote state exposes partial book construction:
    one side may be absent and spreads can be mechanically huge until the full
    snapshot has been replayed. For quote-path visualization, the meaningful
    reload state is the completed snapshot, i.e. the last reload row per day.
    """

    if "event_class" not in panel.columns or "TRADEDATE" not in panel.columns:
        return panel

    is_reload = pl.col("event_class") == "session_reload"
    with_row_id = panel.with_row_index("__quote_diag_row_id")
    last_reload = (
        with_row_id.filter(is_reload)
        .group_by("TRADEDATE")
        .agg(pl.col("__quote_diag_row_id").max().alias("__last_reload_row_id"))
    )
    if last_reload.is_empty():
        return panel

    return (
        with_row_id.join(last_reload, on="TRADEDATE", how="left")
        .filter((~is_reload) | (pl.col("__quote_diag_row_id") == pl.col("__last_reload_row_id")))
        .drop(["__quote_diag_row_id", "__last_reload_row_id"])
    )


def write_metadata(
    *,
    output_path: str | Path,
    panel_path: str | Path,
    diagnostics: pl.DataFrame,
    html_path: str | Path,
    command: list[str] | None = None,
    session_reload_mode: str = "all",
    input_row_count: int | None = None,
    best_levels: int = 1,
) -> None:
    output_path = Path(output_path)
    observed_input_rows = diagnostics.height if input_row_count is None else int(input_row_count)
    all_raw_panel_rows = session_reload_mode == "all" and diagnostics.height == observed_input_rows
    payload: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "panel_path": str(panel_path),
        "html_path": str(html_path),
        "input_row_count": observed_input_rows,
        "row_count": diagnostics.height,
        "x_axis": "sort_index",
        "event_time": True,
        "all_events": all_raw_panel_rows,
        "all_raw_panel_rows": all_raw_panel_rows,
        "all_live_events": True,
        "session_reload_mode": session_reload_mode,
        "best_levels": best_levels,
        "partitioned_by_day": False,
        "columns": diagnostics.columns,
        "valid_touch_rows": int(diagnostics.select(pl.col("valid_touch").fill_null(False).sum()).item()),
        "command": command,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _available_columns(df: pl.DataFrame, columns: tuple[str, ...]) -> list[str]:
    return [column for column in columns if column in df.columns]


def _level_price_column(side: str, level: int) -> str:
    if level == 1:
        return "post_best_bid" if side == "bid" else "post_best_ask"
    return f"post_{side}_level_{level}_price"


def _level_qty_column(side: str, level: int) -> str:
    return f"post_{side}_level_{level}_visible_qty"


def _validate_best_levels(df: pl.DataFrame, best_levels: int) -> None:
    if best_levels < 1:
        raise ValueError("best_levels must be at least 1")
    missing: list[str] = []
    for level in range(1, best_levels + 1):
        for side in ("bid", "ask"):
            column = _level_price_column(side, level)
            if column not in df.columns:
                missing.append(column)
    if missing:
        raise ValueError(
            f"requested best_levels={best_levels}, but missing price columns: "
            + ", ".join(missing)
        )


def _customdata(df: pl.DataFrame) -> tuple[list[str], list[list[object]] | None]:
    hover_columns = _available_columns(df, HOVER_COLUMNS)
    if not hover_columns:
        return [], None
    rows = [list(row) for row in df.select(hover_columns).iter_rows()]
    return hover_columns, rows


def _hovertemplate(y_label: str, hover_columns: list[str]) -> str:
    lines = ["sort_index=%{x}", f"{y_label}=%{{y}}"]
    for idx, column in enumerate(hover_columns):
        lines.append(f"{column}=%{{customdata[{idx}]}}")
    return "<br>".join(lines) + "<extra></extra>"


def _add_trace(
    fig: go.Figure,
    *,
    df: pl.DataFrame,
    y_column: str,
    name: str,
    row: int,
    color: str,
    hover_columns: list[str],
    customdata: list[list[object]] | None,
    dash: str = "solid",
) -> None:
    if y_column not in df.columns:
        return

    fig.add_trace(
        go.Scatter(
            x=df.get_column("sort_index").to_list(),
            y=df.get_column(y_column).to_list(),
            name=name,
            mode="lines",
            line={"color": color, "shape": "hv", "dash": dash},
            customdata=customdata,
            hovertemplate=_hovertemplate(name, hover_columns),
        ),
        row=row,
        col=1,
    )


def write_interactive_quote_html(
    diagnostics: pl.DataFrame,
    output_html: str | Path,
    *,
    title: str,
    source_label: str,
    best_levels: int = 1,
) -> None:
    validate_quote_panel_columns(diagnostics)
    _validate_best_levels(diagnostics, best_levels)
    output_html = Path(output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(
            "Best bid / best ask / mid price",
            "Spread diagnostics",
            "Top-of-book depth diagnostics",
        ),
    )
    hover_columns, customdata = _customdata(diagnostics)

    # Put the rich per-event context on the first trace only. With unified hover
    # it is still visible at each x-position, while avoiding duplicating the
    # same customdata payload across every diagnostic trace in large all-event
    # HTML outputs.
    for level in range(1, best_levels + 1):
        bid_name = "Best bid" if level == 1 else f"Bid L{level}"
        ask_name = "Best ask" if level == 1 else f"Ask L{level}"
        bid_hover = hover_columns if level == 1 else []
        bid_customdata = customdata if level == 1 else None
        dash = LEVEL_DASHES[(level - 1) % len(LEVEL_DASHES)]
        _add_trace(
            fig,
            df=diagnostics,
            y_column=_level_price_column("bid", level),
            name=bid_name,
            row=1,
            color=BID_LEVEL_COLORS[(level - 1) % len(BID_LEVEL_COLORS)],
            hover_columns=bid_hover,
            customdata=bid_customdata,
            dash=dash,
        )
        _add_trace(
            fig,
            df=diagnostics,
            y_column=_level_price_column("ask", level),
            name=ask_name,
            row=1,
            color=ASK_LEVEL_COLORS[(level - 1) % len(ASK_LEVEL_COLORS)],
            hover_columns=[],
            customdata=None,
            dash=dash,
        )

    _add_trace(fig, df=diagnostics, y_column="mid_price", name="Mid price", row=1, color="#2ca02c", hover_columns=[], customdata=None)
    _add_trace(fig, df=diagnostics, y_column="spread", name="Spread", row=2, color="#9467bd", hover_columns=[], customdata=None)
    _add_trace(fig, df=diagnostics, y_column="relative_spread_bps", name="Relative spread (bps)", row=2, color="#8c564b", hover_columns=[], customdata=None)
    for level in range(1, best_levels + 1):
        dash = LEVEL_DASHES[(level - 1) % len(LEVEL_DASHES)]
        _add_trace(
            fig,
            df=diagnostics,
            y_column=_level_qty_column("bid", level),
            name=f"Bid L{level} visible qty",
            row=3,
            color=BID_LEVEL_COLORS[(level - 1) % len(BID_LEVEL_COLORS)],
            hover_columns=[],
            customdata=None,
            dash=dash,
        )
        _add_trace(
            fig,
            df=diagnostics,
            y_column=_level_qty_column("ask", level),
            name=f"Ask L{level} visible qty",
            row=3,
            color=ASK_LEVEL_COLORS[(level - 1) % len(ASK_LEVEL_COLORS)],
            hover_columns=[],
            customdata=None,
            dash=dash,
        )
    _add_trace(fig, df=diagnostics, y_column="top_of_book_imbalance", name="Top-of-book imbalance", row=3, color="#7f7f7f", hover_columns=[], customdata=None)

    fig.update_layout(
        title=(
            f"{title}<br>"
            f"<sup>Source: {source_label}; x-axis is event time (`sort_index`); "
            f"Best levels plotted: {best_levels}; all events included; no day partitioning.</sup>"
        ),
        hovermode="x unified",
        template="plotly_white",
        height=950,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(title_text="Event time (`sort_index`)", row=3, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Spread", row=2, col=1)
    fig.update_yaxes(title_text="Qty / imbalance", row=3, col=1)

    fig.write_html(output_html, include_plotlyjs="cdn", full_html=True)
    # Plotly JSON-escapes forward slashes in text fields. Keep the generated
    # HTML human-searchable for report titles and labels without changing the
    # plotted data.
    output_html.write_text(output_html.read_text().replace("\\u002f", "/"))
