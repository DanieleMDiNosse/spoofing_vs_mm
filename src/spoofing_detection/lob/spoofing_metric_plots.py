from __future__ import annotations

from pathlib import Path

import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _top_mcps_table(mcps_scores: pl.DataFrame | None, execution_metrics: pl.DataFrame) -> pl.DataFrame:
    if mcps_scores is not None and not mcps_scores.is_empty():
        columns = [
            column
            for column in (
                "client_id",
                "top_n",
                "gamma",
                "executions",
                "finite_msci_executions",
                "MCPS",
                "max_MSCI",
                "mean_MSCI",
                "mean_favorable_mid_move_pre_fill",
                "mean_post_cancel_mid_reversion",
                "mean_execution_price_advantage_vs_posture_mid",
                "matched_deceptive_cancel_share",
            )
            if column in mcps_scores.columns
        ]
        return mcps_scores.sort(["MCPS", "max_MSCI", "executions"], descending=[True, True, True]).head(20).select(columns)

    if execution_metrics.is_empty() or "client_id" not in execution_metrics.columns:
        return pl.DataFrame({"client_id": [], "executions": [], "max_MSCI": [], "mean_MSCI": []})
    return (
        execution_metrics.group_by("client_id")
        .agg(
            [
                pl.len().alias("executions"),
                pl.col("MSCI").max().alias("max_MSCI"),
                pl.col("MSCI").mean().alias("mean_MSCI"),
            ]
        )
        .sort(["max_MSCI", "executions"], descending=[True, True])
        .head(20)
    )


def write_spoofing_metric_dashboard(
    *,
    execution_metrics: pl.DataFrame,
    state_time_series: pl.DataFrame | None,
    output_html: str | Path,
    title: str,
    client_id: str | None = None,
    mcps_scores: pl.DataFrame | None = None,
) -> None:
    output_html = Path(output_html)
    fig = make_subplots(
        rows=7,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.055,
        specs=[[{}], [{}], [{}], [{}], [{}], [{"type": "table"}], [{}]],
        subplot_titles=(
            "Event-level MSCI over time",
            "MSCI distribution",
            "Opposite-side collapse versus same-side collapse",
            "Candidate deceptive profile size versus small execution size",
            "Price-response diagnostics for spoofing-like executions",
            "Top clients by MCPS",
            "Selected-client DWI time series",
        ),
    )

    if not execution_metrics.is_empty():
        if "has_matched_deceptive_cancel_window" in execution_metrics.columns:
            plotted_executions = execution_metrics.filter(pl.col("has_matched_deceptive_cancel_window"))
        else:
            plotted_executions = execution_metrics
        x_col = "event_ts" if "event_ts" in execution_metrics.columns else "sort_index"
        colors = "#d62728"
        hover_cols = [
            column
            for column in (
                "client_id",
                "execution_side",
                "deceptive_side",
                "fill_qty",
                "SCI",
                "collapse_opposite_side",
                "collapse_same_side",
                "candidate_deceptive_visible_qty_pre",
                "matched_deceptive_cancel_visible_qty_window",
                "matched_deceptive_cancel_fraction_window",
                "favorable_mid_move_pre_fill",
                "post_cancel_mid_reversion",
                "execution_price_advantage_vs_posture_mid",
            )
            if column in execution_metrics.columns
        ]
        customdata = [list(row) for row in plotted_executions.select(hover_cols).iter_rows()] if hover_cols else None
        hover_lines = ["%{x}", "MSCI=%{y}"]
        for idx, column in enumerate(hover_cols):
            hover_lines.append(f"{column}=%{{customdata[{idx}]}}")
        if plotted_executions.is_empty():
            fig.add_annotation(text="No spoofing-like executions with matched deceptive cancels", row=1, col=1, showarrow=False)
        else:
            fig.add_trace(
                go.Scattergl(
                    x=plotted_executions.get_column(x_col).to_list(),
                    y=plotted_executions.get_column("MSCI").to_list(),
                    mode="markers",
                    marker={"color": colors, "size": 7, "opacity": 0.75},
                    customdata=customdata,
                    hovertemplate="<br>".join(hover_lines) + "<extra></extra>",
                    name="spoofing-like executions: matched deceptive cancel",
                ),
                row=1,
                col=1,
            )
        fig.add_trace(
            go.Histogram(
                x=plotted_executions.get_column("MSCI").drop_nulls().to_list(),
                nbinsx=40,
                marker_color="#d62728",
                name="MSCI distribution for spoofing-like executions",
            ),
            row=2,
            col=1,
        )
        if {"collapse_same_side", "collapse_opposite_side"}.issubset(execution_metrics.columns):
            if not plotted_executions.is_empty():
                fig.add_trace(
                    go.Scattergl(
                        x=plotted_executions.get_column("collapse_same_side").to_list(),
                        y=plotted_executions.get_column("collapse_opposite_side").to_list(),
                        mode="markers",
                        marker={"color": colors, "size": 7, "opacity": 0.65},
                        name="side collapse for spoofing-like executions",
                    ),
                    row=3,
                    col=1,
                )
            fig.add_trace(
                go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line={"dash": "dash", "color": "#777"}, name="equal collapse"),
                row=3,
                col=1,
            )
        if {"fill_qty", "candidate_deceptive_visible_qty_pre"}.issubset(execution_metrics.columns):
            if not plotted_executions.is_empty():
                fig.add_trace(
                    go.Scattergl(
                        x=plotted_executions.get_column("fill_qty").to_list(),
                        y=plotted_executions.get_column("candidate_deceptive_visible_qty_pre").to_list(),
                        mode="markers",
                        marker={"color": colors, "size": 7, "opacity": 0.65},
                        text=plotted_executions.get_column("client_id").to_list() if "client_id" in plotted_executions.columns else None,
                        hovertemplate=(
                            "small execution qty=%{x}<br>candidate deceptive profile volume before execution=%{y}"
                            "<br>client=%{text}<extra></extra>"
                        ),
                        name="candidate deceptive profile before execution",
                    ),
                    row=4,
                    col=1,
                )
        if {"favorable_mid_move_pre_fill", "post_cancel_mid_reversion"}.issubset(execution_metrics.columns):
            if not plotted_executions.is_empty():
                fig.add_trace(
                    go.Scattergl(
                        x=plotted_executions.get_column("favorable_mid_move_pre_fill").to_list(),
                        y=plotted_executions.get_column("post_cancel_mid_reversion").to_list(),
                        mode="markers",
                        marker={"color": colors, "size": 7, "opacity": 0.65},
                        text=plotted_executions.get_column("client_id").to_list()
                        if "client_id" in plotted_executions.columns
                        else None,
                        customdata=(
                            [
                                list(row)
                                for row in plotted_executions.select(
                                    [
                                        col
                                        for col in ("execution_price_advantage_vs_posture_mid", "MSCI")
                                        if col in plotted_executions.columns
                                    ]
                                ).iter_rows()
                            ]
                            if any(
                                col in plotted_executions.columns
                                for col in ("execution_price_advantage_vs_posture_mid", "MSCI")
                            )
                            else None
                        ),
                        hovertemplate=(
                            "favorable pre-fill mid move=%{x}<br>post-cancel mid reversion=%{y}"
                            "<br>client=%{text}<br>extra metrics=%{customdata}<extra></extra>"
                        ),
                        name="price response for spoofing-like executions",
                    ),
                    row=5,
                    col=1,
                )
            fig.add_trace(
                go.Scatter(x=[0, 0], y=[-1, 1], mode="lines", line={"dash": "dot", "color": "#999"}, name="zero FPM"),
                row=5,
                col=1,
            )
            fig.add_trace(
                go.Scatter(x=[-1, 1], y=[0, 0], mode="lines", line={"dash": "dot", "color": "#bbb"}, name="zero REV"),
                row=5,
                col=1,
            )
    else:
        fig.add_annotation(text="No eligible executions", row=1, col=1, showarrow=False)

    top_clients = _top_mcps_table(mcps_scores, execution_metrics)
    fig.add_trace(
        go.Table(
            header={"values": top_clients.columns, "fill_color": "#e5ecf6", "align": "left"},
            cells={
                "values": [top_clients.get_column(column).to_list() for column in top_clients.columns],
                "align": "left",
            },
        ),
        row=6,
        col=1,
    )

    if state_time_series is not None and not state_time_series.is_empty() and "DWI" in state_time_series.columns:
        state_df = state_time_series
        selected_client = client_id
        if selected_client is None and "client_id" in state_df.columns and state_df.height:
            selected_client = str(state_df.item(0, "client_id"))
        if selected_client is not None:
            state_df = state_df.filter(pl.col("client_id") == selected_client)
        if not state_df.is_empty():
            x_col = "event_ts" if "event_ts" in state_df.columns else "sort_index"
            fig.add_trace(
                go.Scattergl(
                    x=state_df.get_column(x_col).to_list(),
                    y=state_df.get_column("DWI").to_list(),
                    mode="lines",
                    line={"color": "#2ca02c"},
                    name=f"DWI {selected_client}",
                ),
                row=7,
                col=1,
            )

    fig.update_layout(
        title=title,
        template="plotly_white",
        height=2050,
        hovermode="closest",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    fig.update_yaxes(title_text="MSCI", row=1, col=1)
    fig.update_xaxes(title_text="Execution time", row=1, col=1)
    fig.update_xaxes(title_text="MSCI", row=2, col=1)
    fig.update_yaxes(title_text="count", row=2, col=1)
    fig.update_xaxes(title_text="same-side collapse", row=3, col=1)
    fig.update_yaxes(title_text="opposite-side collapse", row=3, col=1)
    fig.update_xaxes(title_text="small execution quantity", row=4, col=1)
    fig.update_yaxes(title_text="candidate deceptive profile volume", row=4, col=1)
    fig.update_xaxes(title_text="favorable pre-fill mid-price movement", row=5, col=1)
    fig.update_yaxes(title_text="post-cancel mid-price reversion", row=5, col=1)
    fig.update_yaxes(title_text="DWI", row=7, col=1)
    fig.update_xaxes(title_text="Event time", row=7, col=1)

    note = (
        "<p><b>How to read this dashboard:</b> DWI is the client's ask-minus-bid weighted top-n depth profile. "
        "MSCI becomes large only when DWI changes quickly and the liquidity that disappears is mostly on the side "
        "opposite to the small execution. MCPS is the client-level repetition score: it asks how often MSCI is above "
        "a chosen threshold. Price-response diagnostics are signed so positive values indicate movement or execution "
        "price advantage in the direction favorable to the small execution; they are economic consistency checks, not "
        "causal proof. The event-level scatter plots show only spoofing-like executions: red points directly "
        "cancel one of the pre-existing opposite-side candidate deceptive orders after the execution. Other executions "
        "are omitted from these scatter plots. These are surveillance cues, not proof of intent.</p>"
    )
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(note + fig.to_html(include_plotlyjs="cdn", full_html=True))