# Run with:
# streamlit run ta_dashboard.py

from __future__ import annotations

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ta_pipe import PipelineConfig, run_pipeline, SignalResult


st.set_page_config(
    page_title="TA Candidate Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    [data-testid="stSidebar"], [data-testid="collapsedControl"] {
        display: none;
    }

    .st-key-candidate_table_sticky {
        position: fixed;
        top: 3.25rem;
        left: 0;
        right: 0;
        width: auto;
        max-width: none;
        z-index: 1000;
        background: var(--background-color);
        padding: 0.6rem 0 0.6rem;
        border-bottom: 1px solid var(--border-color);
        box-shadow: 0 8px 18px rgba(0, 0, 0, 0.18);
    }

    .st-key-candidate_table_sticky [data-testid="stHorizontalBlock"],
    .st-key-candidate_table_sticky [data-testid="column"],
    .st-key-candidate_table_sticky .stMarkdown {
        background: var(--background-color);
        color: var(--text-color);
        margin-bottom: 0;
    }

    .st-key-candidate_table_sticky .stMarkdown p,
    .st-key-candidate_table_sticky .stMarkdown strong {
        color: var(--text-color);
    }

    .st-key-candidate_table_sticky [data-testid="stButton"] button {
        background: var(--secondary-background-color) !important;
        border: 2px solid var(--border-color) !important;
        color: var(--text-color) !important;
        font-size: 1.35rem;
        font-weight: 700;
        min-height: 2.75rem;
        padding: 0;
    }

    .st-key-refresh_pipeline_fixed {
        position: fixed;
        right: 1rem;
        bottom: 1rem;
        z-index: 1001;
        background: var(--background-color);
        padding: 0.4rem;
        border: 1px solid var(--border-color);
        border-radius: 0.5rem;
        box-shadow: 0 8px 18px rgba(0, 0, 0, 0.18);
    }

    .st-key-refresh_pipeline_fixed [data-testid="stButton"] button {
        background: var(--secondary-background-color) !important;
        border: 1px solid var(--border-color) !important;
        color: var(--text-color) !important;
        font-weight: 700;
        min-height: 2.5rem;
    }

    @media (max-width: 900px) {
        .st-key-candidate_table_sticky {
            left: 0;
            right: 0;
            width: auto;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _format_signal_date(value) -> object:
    if value is None:
        return None
    if hasattr(value, "date"):
        return value.date()
    return value


def _latest_valid_value(series: pd.Series | None) -> float | None:
    if series is None:
        return None
    valid = series.replace([float("inf"), -float("inf")], pd.NA).dropna()
    if valid.empty:
        return None
    return float(valid.iloc[-1])


def result_summary_df(results: list[SignalResult], top_n: int) -> pd.DataFrame:
    rows = []
    for rank, r in enumerate(results[:top_n], start=1):
        latest_spread = _latest_valid_value(r.spread)
        rows.append({
            "rank": rank,
            "ticker": r.ticker,
            "pre_golden_score": round(r.pre_golden_score, 2) if r.pre_golden_score is not None else None,
            "spread_%": round(latest_spread * 100, 2) if latest_spread is not None else None,
            "est_days_to_cross": round(r.days_to_cross_estimate, 1) if r.days_to_cross_estimate is not None else None,
            "latest_signal": r.latest_signal,
            "signal_date": _format_signal_date(r.latest_signal_date),
            "total_crosses": len(r.cross_dates),
            "golden_crosses": r.cross_colors.count("green"),
            "death_crosses": r.cross_colors.count("red"),
        })
    return pd.DataFrame(rows)


def selected_result_details_df(r: SignalResult) -> pd.DataFrame:
    latest_spread = _latest_valid_value(r.spread)
    values = [
        ("Ticker", r.ticker),
        (
            "Pre-golden score",
            f"{r.pre_golden_score:.2f}" if r.pre_golden_score is not None else "None",
        ),
        (
            "Latest spread %",
            f"{latest_spread * 100:.2f}" if latest_spread is not None else "None",
        ),
        (
            "Estimated days to cross",
            f"{r.days_to_cross_estimate:.1f}"
            if r.days_to_cross_estimate is not None
            else "None",
        ),
        ("Latest signal", r.latest_signal or "None"),
        ("Signal date", str(_format_signal_date(r.latest_signal_date))),
        ("Total crosses", len(r.cross_dates)),
        ("Golden crosses", r.cross_colors.count("green")),
        ("Death crosses", r.cross_colors.count("red")),
    ]
    return pd.DataFrame(values, columns=["metric", "value"])


def selected_result_reasons_df(r: SignalResult) -> pd.DataFrame:
    if not r.pre_golden_reasons:
        return pd.DataFrame([{"reason": "None"}])
    return pd.DataFrame(
        {"reason": reason}
        for reason in r.pre_golden_reasons
    )


def _add_series_trace(
    fig: go.Figure,
    series: pd.Series | None,
    *,
    name: str,
    row: int,
    col: int,
    color: str,
    dash: str | None = None,
) -> None:
    if series is None or series.empty:
        return
    fig.add_trace(
        go.Scatter(
            x=series.index,
            y=series,
            mode="lines",
            name=name,
            line={"color": color, **({"dash": dash} if dash else {})},
        ),
        row=row,
        col=col,
    )


def make_signal_chart(r: SignalResult, cfg: PipelineConfig) -> go.Figure:
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.50, 0.18, 0.17, 0.15],
        subplot_titles=("Price", "SMA spread %", "MACD", "RSI"),
    )

    _add_series_trace(fig, r.close, name="Close", row=1, col=1, color="#4da3ff")
    _add_series_trace(
        fig,
        r.sma_50,
        name=f"SMA {cfg.short_sma_period}",
        row=1,
        col=1,
        color="#f59e0b",
        dash="dash",
    )
    _add_series_trace(
        fig,
        r.sma_200,
        name=f"SMA {cfg.long_sma_period}",
        row=1,
        col=1,
        color="#facc15",
    )
    _add_series_trace(
        fig,
        r.vwma_50,
        name=f"VWMA {cfg.vwma_period}",
        row=1,
        col=1,
        color="#fb7185",
        dash="dot",
    )

    golden = []
    death = []
    for date, price, color in zip(r.cross_dates, r.cross_prices, r.cross_colors):
        if color == "green":
            golden.append((date, price))
        elif color == "red":
            death.append((date, price))

    if golden:
        dates, prices = zip(*golden)
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=prices,
                mode="markers",
                name=f"Golden crosses ({len(golden)})",
                marker={
                    "color": "#22c55e",
                    "size": 11,
                    "symbol": "triangle-up",
                    "line": {"color": "white", "width": 1},
                },
            ),
            row=1,
            col=1,
        )

    if death:
        dates, prices = zip(*death)
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=prices,
                mode="markers",
                name=f"Death crosses ({len(death)})",
                marker={
                    "color": "#ef4444",
                    "size": 11,
                    "symbol": "triangle-down",
                    "line": {"color": "white", "width": 1},
                },
            ),
            row=1,
            col=1,
        )

    if r.spread is not None and not r.spread.empty:
        spread_pct = r.spread * 100
        fig.add_trace(
            go.Scatter(
                x=spread_pct.index,
                y=spread_pct,
                mode="lines",
                name="Spread %",
                line={"color": "#22d3ee"},
            ),
            row=2,
            col=1,
        )

    fig.add_hline(y=0, line_dash="dash", row=2, col=1)
    fig.add_hline(y=cfg.min_spread * 100, line_dash="dot", row=2, col=1)
    fig.add_hline(y=-cfg.min_spread * 100, line_dash="dot", row=2, col=1)

    if r.macd_hist is not None and not r.macd_hist.empty:
        hist = r.macd_hist.dropna()
        if not hist.empty:
            fig.add_trace(
                go.Bar(
                    x=hist.index,
                    y=hist,
                    name="MACD hist",
                    marker_color=[
                        "#22c55e" if value >= 0 else "#ef4444"
                        for value in hist
                    ],
                    opacity=0.55,
                ),
                row=3,
                col=1,
            )
    _add_series_trace(fig, r.macd, name="MACD", row=3, col=1, color="#60a5fa")
    _add_series_trace(fig, r.macd_signal, name="Signal", row=3, col=1, color="#f97316")
    fig.add_hline(y=0, line_dash="dash", row=3, col=1)

    _add_series_trace(fig, r.rsi, name="RSI", row=4, col=1, color="#a78bfa")
    fig.add_hline(y=45, line_dash="dot", row=4, col=1)
    fig.add_hline(y=70, line_dash="dot", row=4, col=1)

    fig.update_layout(
        title=r.ticker,
        height=980,
        hovermode="x unified",
        template="plotly_dark",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
        margin={"l": 40, "r": 30, "t": 80, "b": 40},
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Spread %", row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    fig.update_yaxes(title_text="RSI", row=4, col=1)
    fig.update_xaxes(title_text="Date", row=4, col=1)
    return fig


@st.cache_data(show_spinner=True, ttl=3600)
def load_results() -> tuple[list[SignalResult], PipelineConfig]:
    cfg = PipelineConfig(
        top_n=20,
        plot=False,
        save_plots=False,
        start_date="2022-01-01",
        news_days=7,
        forward_days=20,
    )
    ranked = run_pipeline(cfg)
    return ranked, cfg


def main() -> None:
    if "candidate_table_expanded" not in st.session_state:
        st.session_state.candidate_table_expanded = True
    if "selected_candidate_index" not in st.session_state:
        st.session_state.selected_candidate_index = 0

    with st.container(key="refresh_pipeline_fixed"):
        if st.button("Refresh pipeline", use_container_width=True):
            load_results.clear()
            st.session_state.selected_candidate_index = 0

    ranked, cfg = load_results()

    if not ranked:
        st.warning("No ranked candidates found.")
        st.stop()

    summary = result_summary_df(ranked, cfg.top_n)
    with st.container(key="candidate_table_sticky"):
        toggle_label = "▲" if st.session_state.candidate_table_expanded else "▼"
        toggle_col, title_col = st.columns(
            [0.06, 0.94],
            vertical_alignment="center",
        )
        if toggle_col.button(
            toggle_label,
            key="toggle_candidate_table",
            use_container_width=True,
            help=(
                "Collapse candidate table"
                if st.session_state.candidate_table_expanded
                else "Expand candidate table"
            ),
        ):
            st.session_state.candidate_table_expanded = not st.session_state.candidate_table_expanded
            st.rerun()
        title_col.markdown("**Top pre-golden-cross candidates**")

        if st.session_state.candidate_table_expanded:
            table_state = st.dataframe(
                summary,
                use_container_width=True,
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row",
                height=280,
            )
            selected_rows = table_state.selection.rows
            if selected_rows and selected_rows[0] < len(summary):
                st.session_state.selected_candidate_index = selected_rows[0]

    spacer_height = 410 if st.session_state.candidate_table_expanded else 140
    st.markdown(
        f'<div style="height: {spacer_height}px;"></div>',
        unsafe_allow_html=True,
    )
    selected_index = st.session_state.selected_candidate_index
    if selected_index >= len(summary):
        selected_index = 0
        st.session_state.selected_candidate_index = selected_index
    selected_ticker = summary.iloc[selected_index]["ticker"]
    selected_result = next(r for r in ranked if r.ticker == selected_ticker)

    fig = make_signal_chart(selected_result, cfg)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader(f"{selected_result.ticker} details")
    st.dataframe(
        selected_result_details_df(selected_result),
        use_container_width=True,
        hide_index=True,
    )
    st.dataframe(
        selected_result_reasons_df(selected_result),
        use_container_width=True,
        hide_index=True,
    )


if __name__ == "__main__":
    main()
