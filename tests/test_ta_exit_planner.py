from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, TESTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ta_dashboard import (  # noqa: E402
    _make_exit_plan_chart,
    _position_slice_from_entry,
    _series_value_on_date,
    compute_exit_levels,
    compute_position_context,
    recommended_exits,
)
from ta_pipe import PipelineConfig, SignalResult  # noqa: E402


def _result(ticker: str = "TEST") -> SignalResult:
    return SignalResult(
        ticker=ticker,
        cross_dates=[],
        cross_prices=[],
        cross_colors=[],
        forward_returns=[],
        latest_signal=None,
        latest_signal_date=None,
    )


def test_series_value_on_date_uses_nearest_prior_valid_value() -> None:
    values = pd.Series(
        [10.0, None, 12.0],
        index=pd.to_datetime(["2026-06-01", "2026-06-02", "2026-06-04"]),
    )

    assert _series_value_on_date(values, date(2026, 6, 3)) == 10.0
    assert _series_value_on_date(values, date(2026, 5, 31)) is None


def test_position_slice_resolves_non_market_and_out_of_range_dates() -> None:
    close = pd.Series(
        [100.0, 105.0, 110.0],
        index=pd.to_datetime(["2026-06-05", "2026-06-08", "2026-06-09"]),
    )

    weekend = _position_slice_from_entry(close, date(2026, 6, 6))
    assert weekend["entry_bar_date"] == date(2026, 6, 8)
    assert weekend["entry_bar_close"] == 105.0
    assert weekend["from_entry"].tolist() == [105.0, 110.0]
    assert weekend["entry_data_warning"] is None

    before_history = _position_slice_from_entry(close, date(2026, 6, 1))
    assert before_history["entry_bar_date"] is None
    assert before_history["from_entry"].empty
    assert "before the available price history" in before_history["entry_data_warning"]

    after_history = _position_slice_from_entry(close, date(2026, 6, 12))
    assert after_history["entry_bar_date"] == date(2026, 6, 9)
    assert after_history["from_entry"].tolist() == [110.0]
    assert "after the latest loaded price bar" in after_history["entry_data_warning"]


def test_compute_position_context_tracks_entry_regime_window_and_return() -> None:
    dates = pd.bdate_range(end=pd.Timestamp(date.today()), periods=5)
    result = _result()
    result.close = pd.Series([90.0, 95.0, 100.0, 110.0, 105.0], index=dates)
    result.sma_50 = pd.Series([85.0, 90.0, 92.0, 94.0, 96.0], index=dates)
    result.sma_200 = pd.Series([80.0] * 5, index=dates)
    result.spread = (result.sma_50 - result.sma_200) / result.sma_200
    result.regime_label = "bearish_or_weak"

    context = compute_position_context(
        result,
        PipelineConfig(forward_days=2),
        entry_price=100.0,
        entry_date=dates[1].date(),
    )

    assert context["entry_bar_date"] == dates[1].date()
    assert context["regime_on_entry"] == "confirmed_bullish"
    assert context["regime_on_entry_is_approx"] is True
    assert context["regime_changed"] is True
    assert context["trading_days_held"] == 3
    assert context["window_expired"] is True
    assert context["days_remaining_in_window"] == 0
    assert context["highest_close_since_entry"] == 110.0
    assert context["highest_close_date"] == dates[3].date()
    assert context["return_pct"] == pytest.approx(5.0)


def test_compute_position_context_after_latest_bar_has_no_negative_trading_days() -> None:
    dates = pd.bdate_range(end=pd.Timestamp(date.today()) - pd.Timedelta(days=3), periods=3)
    result = _result()
    result.close = pd.Series([100.0, 101.0, 102.0], index=dates)

    context = compute_position_context(
        result,
        PipelineConfig(),
        entry_price=100.0,
        entry_date=date.today(),
    )

    assert context["entry_bar_date"] == dates[-1].date()
    assert context["trading_days_held"] == 0
    assert context["highest_close_since_entry"] == 102.0
    assert "after the latest loaded price bar" in context["entry_data_warning"]


def test_compute_exit_levels_calculates_current_levels_and_raw_rr() -> None:
    dates = pd.date_range("2026-06-01", periods=5)
    result = _result()
    result.close = pd.Series([100.0, 105.0, 110.0, 115.0, 120.0], index=dates)
    result.atr_14 = pd.Series([4.0, 4.0, 4.5, 4.5, 5.0], index=dates)
    result.sma_50 = pd.Series([97.0] * 5, index=dates)
    result.sma_200 = pd.Series([95.0] * 5, index=dates)
    result.ema_21 = pd.Series([102.0] * 5, index=dates)
    result.donchian_20_high = pd.Series([121.0] * 5, index=dates)
    result.donchian_55_high = pd.Series([130.0] * 5, index=dates)
    cfg = PipelineConfig(bollinger_period=3, bollinger_std_mult=2.0)

    levels = compute_exit_levels(result, cfg, 100.0, dates[0].date())
    expected_upper = (
        result.close.rolling(3).mean() + 2.0 * result.close.rolling(3).std()
    ).dropna().iloc[-1]

    assert levels["latest_atr"] == 5.0
    assert levels["atr_pct"] == 5.0
    assert levels["highest_since_entry"] == 120.0
    assert levels["stop_atr_1x"] == 95.0
    assert levels["stop_atr_2x"] == 90.0
    assert levels["stop_trailing_atr_2x"] == 110.0
    assert levels["stop_sma50"] == 97.0
    assert levels["stop_ema21"] == 102.0
    assert levels["target_atr_1x"] == 105.0
    assert levels["target_atr_2x"] == 110.0
    assert levels["target_atr_3x"] == 115.0
    assert levels["target_extension_1_5x"] == 107.5
    assert levels["target_extension_2_5x"] == 112.5
    assert levels["target_extension_3x"] == 115.0
    assert levels["target_donchian_20"] == 121.0
    assert levels["target_donchian_55"] == 130.0
    assert levels["target_bb_upper"] == pytest.approx(expected_upper)
    assert levels["target_sma200"] == 95.0
    assert levels["rr_atr_1x"] == 0.5
    assert levels["rr_atr_2x"] == 1.0
    assert levels["rr_atr_3x"] == 1.5
    assert levels["rr_extension_1_5x"] == 0.75
    assert levels["rr_extension_2_5x"] == 1.25
    assert levels["rr_extension_3x"] == 1.5
    assert levels["rr_donchian_20"] == 2.1
    assert levels["rr_donchian_55"] == 3.0
    assert levels["rr_bb_upper"] == pytest.approx((expected_upper - 100.0) / 10.0)
    assert levels["rr_sma200"] == -0.5


def test_compute_exit_levels_keeps_non_atr_levels_when_atr_is_missing() -> None:
    dates = pd.date_range("2026-06-01", periods=3)
    result = _result()
    result.close = pd.Series([100.0, 101.0, 102.0], index=dates)
    result.sma_50 = pd.Series([98.0] * 3, index=dates)
    result.sma_200 = pd.Series([96.0] * 3, index=dates)
    result.ema_21 = pd.Series([99.0] * 3, index=dates)
    result.donchian_55_high = pd.Series([104.0] * 3, index=dates)

    levels = compute_exit_levels(
        result,
        PipelineConfig(bollinger_period=2),
        100.0,
        dates[0].date(),
    )

    assert levels["stop_atr_2x"] is None
    assert levels["target_atr_2x"] is None
    assert levels["rr_donchian_55"] is None
    assert levels["stop_sma50"] == 98.0
    assert levels["stop_ema21"] == 99.0
    assert levels["target_donchian_55"] == 104.0
    assert levels["target_bb_upper"] is not None


def _recommendation_levels() -> dict[str, float | None]:
    return {
        "target_atr_1x": 110.0,
        "rr_atr_1x": 0.5,
        "target_atr_2x": 120.0,
        "rr_atr_2x": 1.0,
        "target_extension_1_5x": 115.0,
        "rr_extension_1_5x": 0.75,
        "target_extension_2_5x": 125.0,
        "rr_extension_2_5x": 1.25,
        "target_extension_3x": 130.0,
        "rr_extension_3x": 1.5,
        "target_donchian_20": 118.0,
        "rr_donchian_20": 0.9,
        "target_donchian_55": 128.0,
        "rr_donchian_55": 1.4,
        "target_bb_upper": 116.0,
        "rr_bb_upper": 0.8,
        "target_sma200": 112.0,
        "rr_sma200": 0.6,
        "stop_trailing_atr_2x": 98.0,
        "stop_atr_2x": 90.0,
        "stop_atr_1x": 95.0,
        "stop_sma50": 97.0,
        "stop_ema21": 101.0,
    }


@pytest.mark.parametrize(
    ("regime", "primary_label", "secondary_labels"),
    [
        (
            "confirmed_bullish",
            "2× ATR target",
            {"55-day high resistance", "2.5× ATR extension"},
        ),
        (
            "bullish_impulse",
            "1.5× ATR extension",
            {"2× ATR target", "20-day high resistance"},
        ),
        (
            "bullish_transition",
            "SMA200 reclaim target",
            {"1× ATR target", "Bollinger upper band"},
        ),
        (
            "pre_golden_setup",
            "SMA200 reclaim target",
            {"1× ATR target", "Bollinger upper band"},
        ),
        ("neutral", "1× ATR target", {"Bollinger upper band"}),
    ],
)
def test_recommended_exits_uses_regime_mapping_and_deduplicates(
    regime: str,
    primary_label: str,
    secondary_labels: set[str],
) -> None:
    context = {
        "window_expired": False,
        "regime_changed": False,
        "regime_now": regime,
        "regime_on_entry": regime,
        "trading_days_held": 5,
    }

    exits = recommended_exits(
        _recommendation_levels(),
        context,
        100.0,
        PipelineConfig(),
    )
    targets = [item for item in exits if item["type"] == "target"]

    assert any(
        item["label"] == primary_label and item["priority"] == 1
        for item in targets
    )
    assert secondary_labels <= {
        item["label"] for item in targets if item["priority"] == 2
    }
    assert sum(item["label"] == "55-day high resistance" for item in targets) == 1


def test_recommended_exits_adds_review_flags_and_impulse_stop_priority() -> None:
    context = {
        "window_expired": True,
        "regime_changed": True,
        "regime_now": "bearish_or_weak",
        "regime_on_entry": "confirmed_bullish",
        "trading_days_held": 25,
    }
    exits = recommended_exits(
        _recommendation_levels(),
        context,
        100.0,
        PipelineConfig(forward_days=20),
    )

    assert exits[0]["label"] == "Signal window expired — review exit"
    assert exits[1]["label"] == "Regime deteriorated — review position"
    assert next(item for item in exits if item["label"] == "EMA21 trailing stop")[
        "priority"
    ] == 3

    context.update({"window_expired": False, "regime_changed": False, "regime_now": "bullish_impulse"})
    impulse_exits = recommended_exits(
        _recommendation_levels(),
        context,
        100.0,
        PipelineConfig(),
    )
    assert next(
        item for item in impulse_exits if item["label"] == "EMA21 trailing stop"
    )["priority"] == 2


def test_exit_plan_chart_uses_timestamp_and_priority_styles() -> None:
    dates = pd.date_range("2026-06-01", periods=3)
    result = _result()
    result.close = pd.Series([100.0, 105.0, 110.0], index=dates)
    exits = [
        {
            "label": "Primary target",
            "type": "target",
            "price": 120.0,
            "rr": 1.0,
            "rationale": "Target",
            "priority": 1,
        },
        {
            "label": "Secondary stop",
            "type": "stop",
            "price": 95.0,
            "rr": None,
            "rationale": "Stop",
            "priority": 2,
        },
        {
            "label": "Review only",
            "type": "stop",
            "price": None,
            "rr": None,
            "rationale": "Review",
            "priority": 1,
        },
    ]

    fig = _make_exit_plan_chart(
        result,
        PipelineConfig(),
        100.0,
        dates[0].date(),
        exits,
        {"highest_close_since_entry": 110.0},
    )
    shapes = list(fig.layout.shapes)
    annotations = [annotation.text for annotation in fig.layout.annotations]

    assert any(
        shape.x0 == pd.Timestamp(dates[0].date())
        and shape.x1 == pd.Timestamp(dates[0].date())
        for shape in shapes
    )
    assert any(
        shape.line.color == "#22c55e" and shape.line.dash == "solid"
        for shape in shapes
    )
    assert any(
        shape.line.color == "#f97316" and shape.line.dash == "dash"
        for shape in shapes
    )
    assert any(
        shape.line.color == "#38bdf8" and shape.line.dash == "dot"
        for shape in shapes
    )
    assert "Entry $100.00" in annotations
    assert "Review only" not in annotations
