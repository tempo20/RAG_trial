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
)


def _format_signal_date(value) -> object:
    if value is None:
        return None
    if hasattr(value, "date"):
        return value.date()
    return value


def result_summary_df(results: list[SignalResult], top_n: int) -> pd.DataFrame:
    rows = []
    for rank, r in enumerate(results[:top_n], start=1):
        last_fwd = r.forward_returns[-1] if r.forward_returns else None
        rows.append({
            "rank": rank,
            "ticker": r.ticker,
            "latest_signal": r.latest_signal,
            "signal_date": _format_signal_date(r.latest_signal_date),
            "forward_return": f"{last_fwd * 100:+.1f}" if last_fwd is not None else "pend.",
            "total_crosses": len(r.cross_dates),
            "golden_crosses": r.cross_colors.count("green"),
            "death_crosses": r.cross_colors.count("red"),
        })
    return pd.DataFrame(rows)


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
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.72, 0.28],
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
        color="#f8fafc",
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

    fig.update_layout(
        title=f"{r.ticker} Technical Signal",
        height=820,
        hovermode="x unified",
        template="plotly_dark",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
        margin={"l": 40, "r": 30, "t": 80, "b": 40},
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Spread %", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
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
    st.title("Technical Analysis Candidate Dashboard")

    if st.button("Refresh pipeline"):
        st.cache_data.clear()

    ranked, cfg = load_results()

    if not ranked:
        st.warning("No ranked candidates found.")
        st.stop()

    summary = result_summary_df(ranked, cfg.top_n)
    st.subheader("Top candidates")
    st.dataframe(summary, use_container_width=True, hide_index=True)

    selected_ticker = st.sidebar.radio(
        "Candidates",
        summary["ticker"].tolist(),
        index=0,
    )
    selected_result = next(r for r in ranked if r.ticker == selected_ticker)

    st.sidebar.subheader(selected_result.ticker)
    st.sidebar.metric("Latest signal", selected_result.latest_signal or "None")
    st.sidebar.metric("Signal date", str(_format_signal_date(selected_result.latest_signal_date)))
    st.sidebar.metric("Total crosses", len(selected_result.cross_dates))
    st.sidebar.metric("Golden crosses", selected_result.cross_colors.count("green"))
    st.sidebar.metric("Death crosses", selected_result.cross_colors.count("red"))

    fig = make_signal_chart(selected_result, cfg)
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
