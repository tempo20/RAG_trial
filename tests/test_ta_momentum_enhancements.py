from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, TESTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ta_pipe import (  # noqa: E402
    PipelineConfig,
    SignalResult,
    _cheap_market_signal,
    compute_adx,
    compute_momentum_metrics,
    compute_obv,
)
from ta_dashboard import make_signal_chart, result_summary_df  # noqa: E402


def _series(values: list[float], start: str = "2025-01-01") -> pd.Series:
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq="D"))


def _result(ticker: str, rs_score: float | None) -> SignalResult:
    return SignalResult(
        ticker=ticker,
        cross_dates=[],
        cross_prices=[],
        cross_colors=[],
        forward_returns=[],
        latest_signal=None,
        latest_signal_date=None,
        relative_strength_score=rs_score,
    )


def test_compute_obv_tracks_up_flat_and_down_days() -> None:
    close = _series([10.0, 11.0, 11.0, 10.0, 12.0])
    volume = _series([100.0, 200.0, 300.0, 400.0, 500.0])

    obv = compute_obv(close, volume)

    assert obv.tolist() == [0.0, 200.0, 200.0, -200.0, 300.0]
    assert obv.index.equals(close.index)


def test_compute_adx_returns_aligned_finite_values_after_warmup() -> None:
    close = _series([float(value) for value in range(20, 100)])
    high = close + 1.0
    low = close - 1.0

    adx = compute_adx(high=high, low=low, close=close, period=14)

    assert adx.index.equals(close.index)
    assert not adx.dropna().empty
    assert np.isfinite(adx.dropna().iloc[-1])


def test_compute_momentum_metrics_scores_adx_and_obv_confirmation() -> None:
    close = _series([float(value) for value in range(20, 100)])
    high = close + 1.0
    low = close - 1.0
    volume = _series([1_000_000.0 + value * 1_000.0 for value in range(len(close))])

    score, reasons, adx, obv = compute_momentum_metrics(
        close=close,
        high=high,
        low=low,
        volume=volume,
        cfg=PipelineConfig(),
    )

    assert score >= 5.0
    assert any("ADX above 25" in reason for reason in reasons)
    assert any("ADX above 40" in reason for reason in reasons)
    assert any("OBV rising" in reason for reason in reasons)
    assert any("OBV at 20-day high" in reason for reason in reasons)
    assert not adx.dropna().empty
    assert not obv.dropna().empty


def test_compute_momentum_metrics_penalizes_obv_divergence() -> None:
    close = _series(
        [100.0] * 30
        + [100.0, 99.0, 98.0, 97.0, 96.0, 95.0, 94.0, 93.0, 92.0, 91.0, 101.0]
    )
    high = close + 1.0
    low = close - 1.0
    volume = _series([100.0] * 30 + [100.0] + [1_000.0] * 9 + [10.0])

    cfg = PipelineConfig(adx_trending_threshold=1_000.0, adx_strong_threshold=2_000.0)
    score, reasons, _, _ = compute_momentum_metrics(
        close=close,
        high=high,
        low=low,
        volume=volume,
        cfg=cfg,
    )

    assert score < 0
    assert any("OBV divergence" in reason for reason in reasons)


def test_cheap_market_signal_filters_far_from_52_week_high() -> None:
    cfg = PipelineConfig()
    index = pd.date_range("2025-01-01", periods=253, freq="D")
    retained_hist = pd.DataFrame(
        {
            "Close": [100.0] * 252 + [81.0],
            "Volume": [100_000.0] * 253,
        },
        index=index,
    )
    filtered_hist = pd.DataFrame(
        {
            "Close": [100.0] * 252 + [79.0],
            "Volume": [100_000.0] * 253,
        },
        index=index,
    )

    retained = _cheap_market_signal("NEAR", retained_hist, base_score=1.0, cfg=cfg)
    filtered = _cheap_market_signal("FAR", filtered_hist, base_score=1.0, cfg=cfg)

    assert retained is not None
    assert round(retained.distance_from_52w_high or 0.0, 2) == -0.19
    assert filtered is None


def test_result_summary_df_percentile_rank_uses_full_result_list() -> None:
    results = [
        _result("LOW", 10.0),
        _result("MID", 20.0),
        _result("HIGH", 30.0),
    ]

    df = result_summary_df(results, top_n=2)

    assert df["ticker"].tolist() == ["LOW", "MID"]
    assert df["rs_pct_rank"].tolist() == [33.3, 66.7]


def test_make_signal_chart_adds_obv_subplot_and_trace() -> None:
    dates = pd.date_range("2025-01-01", periods=5, freq="D")
    result = _result("OBV", 10.0)
    result.close = pd.Series([10.0, 11.0, 12.0, 11.0, 13.0], index=dates)
    result.obv = pd.Series([0.0, 100.0, 250.0, 150.0, 350.0], index=dates)

    fig = make_signal_chart(result, PipelineConfig())

    assert any(trace.name == "OBV" for trace in fig.data)
    assert hasattr(fig.layout, "yaxis5")
