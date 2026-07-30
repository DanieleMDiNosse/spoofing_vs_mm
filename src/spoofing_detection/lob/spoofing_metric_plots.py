from __future__ import annotations

from pathlib import Path

import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots


_ANCHOR_COLORS = {"passive": "#d62728", "aggressive": "#1f77b4"}


def _identity_column(frame: pl.DataFrame) -> str | None:
    return next((column for column in ("actor_key", "client_id") if column in frame.columns), None)


def _actor_marker_colors(frame: pl.DataFrame) -> str | list[str]:
    if "execution_anchor_mode" not in frame.columns:
        return "#d62728"
    return [
        _ANCHOR_COLORS.get(str(mode), "#777777")
        for mode in frame.get_column("execution_anchor_mode").to_list()
    ]


def _select_actor_state(
    state_time_series: pl.DataFrame,
    *,
    actor_key: str | None,
    client_id: str | None,
) -> tuple[pl.DataFrame, str | None]:
    if actor_key is not None and client_id is not None:
        raise ValueError("specify actor_key or the legacy client_id selector, not both")

    state_df = state_time_series
    identity_column = _identity_column(state_df)
    if identity_column is None:
        return state_df, None

    selected_actor = actor_key
    if client_id is not None:
        if identity_column == "actor_key" and {"actor_id", "identity_level"}.issubset(state_df.columns):
            matches = state_df.filter(
                (pl.col("actor_id").cast(pl.String) == client_id)
                & (pl.col("identity_level") == "client_original")
            )
            selected_actor = (
                str(matches.item(0, "actor_key"))
                if not matches.is_empty()
                else f"client_original:{client_id}"
            )
        else:
            selected_actor = client_id
    if selected_actor is None and state_df.height:
        value = state_df.item(0, identity_column)
        selected_actor = str(value) if value is not None else None
    if selected_actor is not None:
        state_df = state_df.filter(pl.col(identity_column).cast(pl.String) == selected_actor)
    return state_df, selected_actor


def _top_mcps_table(mcps_scores: pl.DataFrame | None, execution_metrics: pl.DataFrame) -> pl.DataFrame:
    if mcps_scores is not None and not mcps_scores.is_empty():
        mcps_column = "MCPS_resting_profile" if "MCPS_resting_profile" in mcps_scores.columns else "MCPS"
        max_msci_column = (
            "max_MSCI_resting_profile"
            if "max_MSCI_resting_profile" in mcps_scores.columns
            else "max_MSCI"
        )
        mean_msci_column = (
            "mean_MSCI_resting_profile"
            if "mean_MSCI_resting_profile" in mcps_scores.columns
            else "mean_MSCI"
        )
        columns = [
            column
            for column in (
                "actor_key",
                "actor_id",
                "identity_level",
                "identity_fallback_flag",
                "execution_anchor_mode",
                "client_id",
                "top_n",
                "gamma",
                "executions",
                "finite_msci_executions",
                mcps_column,
                max_msci_column,
                mean_msci_column,
                "mean_favorable_mid_move_pre_fill",
                "mean_post_cancel_mid_reversion",
                "mean_execution_price_advantage_vs_posture_mid",
                "matched_deceptive_cancel_share",
            )
            if column in mcps_scores.columns
        ]
        ranked = mcps_scores.sort(
            [mcps_column, max_msci_column, "executions"],
            descending=[True, True, True],
        )
        identity_column = _identity_column(ranked)
        if identity_column is not None:
            subset = [
                column
                for column in (identity_column, "identity_level", "execution_anchor_mode")
                if column in ranked.columns
            ]
            ranked = ranked.unique(subset=subset, keep="first", maintain_order=True)
        return ranked.head(20).select(columns)

    identity_column = _identity_column(execution_metrics)
    if execution_metrics.is_empty() or identity_column is None:
        return pl.DataFrame(
            {
                "actor_key": [],
                "executions": [],
                "max_MSCI_resting_profile": [],
                "mean_MSCI_resting_profile": [],
            }
        )
    msci_column = (
        "MSCI_resting_profile"
        if "MSCI_resting_profile" in execution_metrics.columns
        else "MSCI"
    )
    group_columns = [
        column
        for column in (
            identity_column,
            "actor_id",
            "identity_level",
            "identity_fallback_flag",
            "execution_anchor_mode",
        )
        if column in execution_metrics.columns
    ]
    return (
        execution_metrics.group_by(group_columns)
        .agg(
            [
                pl.len().alias("executions"),
                pl.col(msci_column).max().alias("max_MSCI_resting_profile"),
                pl.col(msci_column).mean().alias("mean_MSCI_resting_profile"),
            ]
        )
        .sort(["max_MSCI_resting_profile", "executions"], descending=[True, True])
        .head(20)
    )


def write_spoofing_metric_dashboard(
    *,
    execution_metrics: pl.DataFrame,
    state_time_series: pl.DataFrame | None,
    output_html: str | Path,
    title: str,
    actor_key: str | None = None,
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
            "Top actors by MCPS",
            "Selected-actor DWI time series",
        ),
    )

    if not execution_metrics.is_empty():
        msci_column = (
            "MSCI_resting_profile"
            if "MSCI_resting_profile" in execution_metrics.columns
            else "MSCI"
        )
        execution_quantity_column = (
            "execution_quantity" if "execution_quantity" in execution_metrics.columns else "fill_qty"
        )
        if "has_matched_deceptive_cancel_window" in execution_metrics.columns:
            plotted_executions = execution_metrics.filter(pl.col("has_matched_deceptive_cancel_window"))
        else:
            plotted_executions = execution_metrics
        x_col = "event_ts" if "event_ts" in execution_metrics.columns else "sort_index"
        colors = _actor_marker_colors(plotted_executions)
        hover_cols = [
            column
            for column in (
                "actor_key",
                "actor_id",
                "identity_level",
                "identity_fallback_flag",
                "execution_anchor_mode",
                "client_id",
                "execution_side",
                "deceptive_side",
                execution_quantity_column,
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
        identity_column = _identity_column(plotted_executions)
        hover_lines = ["%{x}", "MSCI_resting_profile=%{y}"]
        for idx, column in enumerate(hover_cols):
            hover_lines.append(f"{column}=%{{customdata[{idx}]}}")
        if plotted_executions.is_empty():
            fig.add_annotation(text="No spoofing-like executions with matched deceptive cancels", row=1, col=1, showarrow=False)
        else:
            fig.add_trace(
                go.Scattergl(
                    x=plotted_executions.get_column(x_col).to_list(),
                    y=plotted_executions.get_column(msci_column).to_list(),
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
                x=plotted_executions.get_column(msci_column).drop_nulls().to_list(),
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
        if {execution_quantity_column, "candidate_deceptive_visible_qty_pre"}.issubset(
            execution_metrics.columns
        ):
            if not plotted_executions.is_empty():
                fig.add_trace(
                    go.Scattergl(
                        x=plotted_executions.get_column(execution_quantity_column).to_list(),
                        y=plotted_executions.get_column("candidate_deceptive_visible_qty_pre").to_list(),
                        mode="markers",
                        marker={"color": colors, "size": 7, "opacity": 0.65},
                        text=(
                            plotted_executions.get_column(identity_column).to_list()
                            if identity_column is not None
                            else None
                        ),
                        hovertemplate=(
                            "small execution qty=%{x}<br>candidate deceptive profile volume before execution=%{y}"
                            "<br>actor=%{text}<extra></extra>"
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
                        text=(
                            plotted_executions.get_column(identity_column).to_list()
                            if identity_column is not None
                            else None
                        ),
                        customdata=(
                            [
                                list(row)
                                for row in plotted_executions.select(
                                    [
                                        col
                                        for col in ("execution_price_advantage_vs_posture_mid", msci_column)
                                        if col in plotted_executions.columns
                                    ]
                                ).iter_rows()
                            ]
                            if any(
                                col in plotted_executions.columns
                                for col in ("execution_price_advantage_vs_posture_mid", msci_column)
                            )
                            else None
                        ),
                        hovertemplate=(
                            "favorable pre-fill mid move=%{x}<br>post-cancel mid reversion=%{y}"
                            "<br>actor=%{text}<br>extra metrics=%{customdata}<extra></extra>"
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

    top_actors = _top_mcps_table(mcps_scores, execution_metrics)
    fig.add_trace(
        go.Table(
            header={"values": top_actors.columns, "fill_color": "#e5ecf6", "align": "left"},
            cells={
                "values": [top_actors.get_column(column).to_list() for column in top_actors.columns],
                "align": "left",
            },
        ),
        row=6,
        col=1,
    )

    if state_time_series is not None and not state_time_series.is_empty() and "DWI" in state_time_series.columns:
        state_df, selected_actor = _select_actor_state(
            state_time_series,
            actor_key=actor_key,
            client_id=client_id,
        )
        if not state_df.is_empty():
            x_col = "event_ts" if "event_ts" in state_df.columns else "sort_index"
            fig.add_trace(
                go.Scattergl(
                    x=state_df.get_column(x_col).to_list(),
                    y=state_df.get_column("DWI").to_list(),
                    mode="lines",
                    line={"color": "#2ca02c"},
                    name=f"DWI {selected_actor}",
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
    fig.update_yaxes(title_text="MSCI resting profile", row=1, col=1)
    fig.update_xaxes(title_text="Execution time", row=1, col=1)
    fig.update_xaxes(title_text="MSCI resting profile", row=2, col=1)
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
        "<p><b>How to read this dashboard:</b> DWI is the selected actor's ask-minus-bid weighted top-n depth profile. "
        "MSCI is the signed contrast of normalized SCI plus opposite-side collapse minus same-side collapse; "
        "negative values mean same-side collapse dominates. "
        "It is a secondary aggregate shape diagnostic, not matched-withdrawal evidence. MCPS is the actor-level "
        "repetition score: it asks how often MSCI is above "
        "a chosen threshold. Price-response diagnostics are signed so positive values indicate movement or execution "
        "price advantage in the direction favorable to the small execution; they are economic consistency checks, not "
        "causal proof. Passive-anchor points are red and aggressive-anchor points are blue when anchor metadata is "
        "available. Firm-fallback rows aggregate activity at the firm level and must not be interpreted as client "
        "identity. The event-level scatter plots show only spoofing-like executions that directly "
        "cancel one of the pre-existing opposite-side candidate deceptive orders after the execution. Other executions "
        "are omitted from these scatter plots. These are surveillance cues, not proof of intent.</p>"
    )
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(note + fig.to_html(include_plotlyjs="cdn", full_html=True))