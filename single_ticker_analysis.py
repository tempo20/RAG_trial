from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import argparse
import csv
import json
import math
import os
import re
import sqlite3
import uuid
from difflib import SequenceMatcher

try:
    import numpy as np
except Exception:  # pragma: no cover - dependency-light environments
    np = None

try:
    import pandas as pd
except Exception:  # pragma: no cover - dependency-light environments
    pd = None


SQLITE_DB = os.getenv("SQLITE_DB", "my_database.db")
TICKER_MAP_PATH = Path(os.getenv("TICKER_MAP_PATH", "ticker_company_map.csv"))
DEFAULT_PARQUET = Path(os.getenv("HIST_DATA_PARQUET", "hist_data.parquet"))

FINANCIAL_METRIC_KEYS = [
    "market_cap",
    "revenue_ttm",
    "revenue_growth_yoy",
    "gross_margin",
    "operating_margin",
    "net_margin",
    "free_cash_flow",
    "fcf_margin",
    "total_debt",
    "cash_and_equivalents",
    "debt_to_equity",
    "current_ratio",
    "pe_ratio",
    "forward_pe",
    "price_to_sales",
    "price_to_fcf",
    "roe",
    "roa",
]

TECHNICAL_INDICATOR_KEYS = [
    "latest_close",
    "sma_20",
    "sma_50",
    "sma_200",
    "return_20d",
    "return_63d",
    "return_126d",
    "return_252d",
    "volatility_20d",
    "volatility_63d",
    "rsi_14",
    "max_drawdown",
    "volume_trend",
    "relative_strength_spy_63d",
    "relative_strength_spy_126d",
]

STRATEGY_NAMES = {"moving_average_trend", "momentum", "rsi_mean_reversion"}


@dataclass
class ResolvedTicker:
    query: str
    ticker: str | None
    company_name: str | None
    canonical_name: str | None
    exchange: str | None
    confidence: float
    resolution_mode: str
    candidates: list[dict[str, Any]]
    needs_disambiguation: bool
    warnings: list[str]


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _json_default(value: Any) -> Any:
    if pd is not None:
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass
    if np is not None:
        if isinstance(value, (np.integer, np.floating)):
            value = value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, (datetime,)):
        return value.isoformat()
    return str(value)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, default=_json_default)


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _score_ratio(value: float | None, points: list[tuple[float, float]]) -> float | None:
    if value is None:
        return None
    for threshold, score in points:
        if value <= threshold:
            return score
    return points[-1][1] if points else None


_LEGAL_SUFFIX_PATTERN = re.compile(
    r"""[\s,\.]*\b(
        incorporated|corporation|international|holdings|holding|group|
        technologies|technology|systems|solutions|pharmaceuticals|
        financial|services|limited|enterprises|partners|associates|
        class\s+a|class\s+b|inc|corp|ltd|llc|llp|plc|co|company|sa|ag|nv|bv|se
    )\b[\s\.]*$""",
    re.IGNORECASE | re.VERBOSE,
)


def _strip_legal_suffixes(name: str) -> str:
    current = str(name or "").strip().rstrip(".,")
    previous = None
    while current and current != previous:
        previous = current
        current = _LEGAL_SUFFIX_PATTERN.sub("", current).strip().rstrip(".,")
    return current


def _normalize_name(name: str) -> str:
    text = str(name or "").strip().lower()
    text = text.replace("&", " and ")
    text = re.sub(r"['`]", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _ticker_map_path() -> Path:
    return Path(os.getenv("TICKER_MAP_PATH", str(TICKER_MAP_PATH)))


def _load_ticker_rows(path: Path | None = None) -> list[dict[str, Any]]:
    csv_path = path or _ticker_map_path()
    if not csv_path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with open(csv_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            ticker = str(row.get("ticker") or "").strip().upper()
            company_name = str(row.get("company_name") or "").strip()
            if not ticker or not company_name:
                continue
            aliases = [
                alias.strip()
                for alias in str(row.get("aliases") or "").split(";")
                if alias.strip()
            ]
            rows.append(
                {
                    "ticker": ticker,
                    "company_name": company_name,
                    "canonical_name": company_name,
                    "exchange": (str(row.get("exchange") or "").strip() or None),
                    "aliases": aliases,
                }
            )
    return rows


def _candidate_score(query: str, row: dict[str, Any]) -> tuple[float, str, str | None]:
    q_raw = str(query or "").strip()
    q_norm = _normalize_name(q_raw)
    q_no_suffix = _normalize_name(_strip_legal_suffixes(q_raw))
    ticker = str(row.get("ticker") or "").upper()
    surfaces = [str(row.get("company_name") or ""), *list(row.get("aliases") or [])]

    if q_raw.upper() == ticker or q_norm == ticker.lower():
        return 0.99, "ticker_exact", ticker

    best_score = 0.0
    best_mode = "no_match"
    best_alias: str | None = None
    for surface in surfaces:
        surface_norm = _normalize_name(surface)
        stripped_norm = _normalize_name(_strip_legal_suffixes(surface))
        if q_norm and q_norm == surface_norm:
            return 0.98, "company_exact", surface
        if q_no_suffix and q_no_suffix == stripped_norm:
            return 0.97, "company_exact", surface
        if not q_norm or not surface_norm:
            continue

        if q_norm in {surface_norm, stripped_norm}:
            score = 0.96
            mode = "company_exact"
        elif re.search(r"\b" + re.escape(q_norm) + r"\b", surface_norm):
            score = min(0.88, 0.72 + 0.03 * len(q_norm.split()))
            mode = "company_partial"
        elif stripped_norm and re.search(r"\b" + re.escape(q_no_suffix) + r"\b", stripped_norm):
            score = min(0.87, 0.72 + 0.03 * len(q_no_suffix.split()))
            mode = "company_partial"
        else:
            ratio = SequenceMatcher(None, q_norm, surface_norm).ratio()
            token_overlap = 0.0
            q_tokens = set(q_norm.split())
            s_tokens = set(surface_norm.split())
            if q_tokens and s_tokens:
                token_overlap = len(q_tokens & s_tokens) / len(q_tokens | s_tokens)
            score = max(ratio * 0.86, token_overlap * 0.82)
            mode = "company_fuzzy"

        if score > best_score:
            best_score = score
            best_mode = mode
            best_alias = surface

    return best_score, best_mode, best_alias


def resolve_single_ticker(query: str) -> ResolvedTicker:
    rows = _load_ticker_rows()
    warnings: list[str] = []
    if not rows:
        return ResolvedTicker(
            query=query,
            ticker=None,
            company_name=None,
            canonical_name=None,
            exchange=None,
            confidence=0.0,
            resolution_mode="unresolved",
            candidates=[],
            needs_disambiguation=False,
            warnings=["ticker_company_map.csv unavailable or empty"],
        )

    candidates: list[dict[str, Any]] = []
    for row in rows:
        score, mode, matched_alias = _candidate_score(query, row)
        if score < 0.45:
            continue
        candidates.append(
            {
                "ticker": row["ticker"],
                "company_name": row["company_name"],
                "canonical_name": row["canonical_name"],
                "exchange": row.get("exchange"),
                "confidence": round(float(score), 4),
                "resolution_mode": mode,
                "matched_alias": matched_alias,
            }
        )

    candidates.sort(key=lambda item: item["confidence"], reverse=True)
    candidates = candidates[:8]
    if not candidates:
        return ResolvedTicker(
            query=query,
            ticker=None,
            company_name=None,
            canonical_name=None,
            exchange=None,
            confidence=0.0,
            resolution_mode="unresolved",
            candidates=[],
            needs_disambiguation=False,
            warnings=[f"no confident ticker match for query: {query}"],
        )

    best = candidates[0]
    second = candidates[1] if len(candidates) > 1 else None
    confidence = float(best["confidence"])
    margin = confidence - float(second["confidence"]) if second else confidence
    clearly_dominant = confidence >= 0.90 or (confidence >= 0.72 and margin >= 0.14)
    plausible_multiple = second is not None and float(second["confidence"]) >= 0.68 and margin < 0.14

    if plausible_multiple and best.get("resolution_mode") != "ticker_exact":
        warnings.append("multiple plausible ticker candidates; disambiguation required")
        return ResolvedTicker(
            query=query,
            ticker=None,
            company_name=None,
            canonical_name=None,
            exchange=None,
            confidence=confidence,
            resolution_mode="ambiguous",
            candidates=candidates,
            needs_disambiguation=True,
            warnings=warnings,
        )

    if not clearly_dominant:
        warnings.append("best ticker match is below confidence threshold")
        return ResolvedTicker(
            query=query,
            ticker=None,
            company_name=None,
            canonical_name=None,
            exchange=None,
            confidence=0.0,
            resolution_mode="unresolved",
            candidates=candidates,
            needs_disambiguation=False,
            warnings=warnings,
        )

    return ResolvedTicker(
        query=query,
        ticker=best["ticker"],
        company_name=best["company_name"],
        canonical_name=best["canonical_name"],
        exchange=best.get("exchange"),
        confidence=confidence,
        resolution_mode=best["resolution_mode"],
        candidates=candidates,
        needs_disambiguation=False,
        warnings=warnings,
    )


def _empty_financial_metrics(ticker: str, warnings: list[str] | None = None) -> dict[str, Any]:
    metrics = {key: None for key in FINANCIAL_METRIC_KEYS}
    missing = [key for key, value in metrics.items() if value is None]
    return {
        "ticker": ticker.upper(),
        "as_of_date": _today_utc(),
        "source_provider": None,
        "metrics": metrics,
        "scores": {
            "growth_score": 0.0,
            "profitability_score": 0.0,
            "liquidity_score": 0.0,
            "leverage_score": 0.0,
            "valuation_score": 0.0,
            "financial_quality_score": 0.0,
        },
        "data_quality": {
            "missing_metrics": missing,
            "available_metrics_count": 0,
            "warnings": list(warnings or []),
        },
    }


def _latest_statement_value(df: Any, ticker: str, names: list[str]) -> float | None:
    if pd is None or df is None:
        return None
    try:
        if df.empty:
            return None
    except Exception:
        return None
    sub = df
    try:
        if hasattr(sub.columns, "levels"):
            for level in range(sub.columns.nlevels):
                if ticker in set(map(str, sub.columns.get_level_values(level))):
                    sub = sub.xs(ticker, axis=1, level=level)
                    break
    except Exception:
        pass
    for name in names:
        try:
            if name in sub.index:
                series = sub.loc[name]
                for _, value in reversed(list(series.items())):
                    value = _finite_float(value)
                    if value is not None:
                        return value
        except Exception:
            continue
    return None


def _latest_ratio_value(df: Any, ticker: str, names: list[str] | None = None) -> float | None:
    if pd is None or df is None:
        return None
    try:
        if df.empty:
            return None
    except Exception:
        return None
    names = names or []
    candidates: list[Any] = []
    try:
        if ticker in df.index:
            candidates.append(df.loc[ticker])
    except Exception:
        pass
    for name in names:
        try:
            if name in df.index:
                candidates.append(df.loc[name])
        except Exception:
            pass
    try:
        if ticker in df.columns:
            candidates.append(df[ticker])
    except Exception:
        pass
    try:
        if len(df.index) == 1:
            candidates.append(df.iloc[0])
    except Exception:
        pass
    for candidate in candidates:
        try:
            if hasattr(candidate, "items"):
                for _, value in reversed(list(candidate.items())):
                    value = _finite_float(value)
                    if value is not None:
                        return value
            else:
                value = _finite_float(candidate)
                if value is not None:
                    return value
        except Exception:
            continue
    return None


def _fetch_financial_metrics_from_provider(ticker: str) -> tuple[dict[str, Any], str | None, list[str]]:
    warnings: list[str] = []
    metrics = {key: None for key in FINANCIAL_METRIC_KEYS}
    api_key = os.getenv("FMP_API_KEY", "").strip()
    if not api_key:
        return metrics, None, ["FMP_API_KEY not set; financial metrics unavailable"]
    try:
        from financetoolkit import Toolkit
    except Exception as exc:  # pragma: no cover - environment-dependent
        return metrics, None, [f"FinanceToolkit unavailable: {exc}"]

    try:
        toolkit = Toolkit([ticker], api_key=api_key, start_date="2020-01-01")
    except Exception as exc:  # pragma: no cover - provider-dependent
        return metrics, "FinanceToolkit/FMP", [f"FinanceToolkit init failed: {exc}"]

    profile = income = balance = cashflow = None
    ratios: dict[str, Any] = {}
    for name, getter in [
        ("profile", lambda: toolkit.get_profile()),
        ("income", lambda: toolkit.get_income_statement()),
        ("balance", lambda: toolkit.get_balance_sheet_statement()),
        ("cashflow", lambda: toolkit.get_cash_flow_statement()),
    ]:
        try:
            value = getter()
            if name == "profile":
                profile = value
            elif name == "income":
                income = value
            elif name == "balance":
                balance = value
            else:
                cashflow = value
        except Exception as exc:
            warnings.append(f"{name} unavailable: {exc}")

    ratio_getters = {
        "pe_ratio": ("get_price_earnings_ratio", ["Price Earnings Ratio", "Price-to-Earnings Ratio"]),
        "forward_pe": ("get_forward_price_earnings_ratio", ["Forward Price Earnings Ratio"]),
        "debt_to_equity": ("get_debt_to_equity_ratio", ["Debt-to-Equity Ratio", "Debt to Equity Ratio"]),
        "current_ratio": ("get_current_ratio", ["Current Ratio"]),
        "roe": ("get_return_on_equity", ["Return on Equity", "ROE"]),
        "roa": ("get_return_on_assets", ["Return on Assets", "ROA"]),
    }
    ratios_obj = getattr(toolkit, "ratios", None)
    if ratios_obj is not None:
        for key, (method, names) in ratio_getters.items():
            try:
                fn = getattr(ratios_obj, method)
            except Exception:
                continue
            try:
                ratios[key] = _latest_ratio_value(fn(rounding=6), ticker, names)
            except Exception:
                continue

    revenue = _latest_statement_value(income, ticker, ["Revenue"])
    prior_revenue = None
    if pd is not None and income is not None:
        try:
            row = income.xs(ticker, axis=1, level=1) if hasattr(income.columns, "levels") else income
            values = [
                _finite_float(value)
                for value in row.loc["Revenue"].tolist()
                if _finite_float(value) is not None
            ]
            if len(values) >= 2:
                prior_revenue = values[-2]
        except Exception:
            pass
    gross_profit = _latest_statement_value(income, ticker, ["Gross Profit"])
    operating_income = _latest_statement_value(income, ticker, ["Operating Income"])
    net_income = _latest_statement_value(income, ticker, ["Net Income"])
    operating_cash_flow = _latest_statement_value(cashflow, ticker, ["Operating Cash Flow"])
    capex = _latest_statement_value(cashflow, ticker, ["Capital Expenditure", "Capital Expenditures"])
    total_debt = _latest_statement_value(balance, ticker, ["Total Debt", "Long Term Debt"])
    cash = _latest_statement_value(balance, ticker, ["Cash and Cash Equivalents", "Cash and Short Term Investments"])
    total_equity = _latest_statement_value(balance, ticker, ["Total Stockholders Equity", "Total Equity"])
    current_assets = _latest_statement_value(balance, ticker, ["Total Current Assets"])
    current_liabilities = _latest_statement_value(balance, ticker, ["Total Current Liabilities"])
    total_assets = _latest_statement_value(balance, ticker, ["Total Assets"])

    market_cap = None
    try:
        if profile is not None and not profile.empty:
            col = ticker if ticker in profile.columns else profile.columns[0]
            for label in ["Market Capitalization", "Market Cap"]:
                if label in profile.index:
                    market_cap = _finite_float(profile.loc[label, col])
                    break
    except Exception:
        pass

    fcf = None
    if operating_cash_flow is not None and capex is not None:
        fcf = operating_cash_flow - abs(capex)

    metrics.update(
        {
            "market_cap": market_cap,
            "revenue_ttm": revenue,
            "revenue_growth_yoy": ((revenue / prior_revenue) - 1.0) if revenue is not None and prior_revenue else None,
            "gross_margin": (gross_profit / revenue) if gross_profit is not None and revenue else None,
            "operating_margin": (operating_income / revenue) if operating_income is not None and revenue else None,
            "net_margin": (net_income / revenue) if net_income is not None and revenue else None,
            "free_cash_flow": fcf,
            "fcf_margin": (fcf / revenue) if fcf is not None and revenue else None,
            "total_debt": total_debt,
            "cash_and_equivalents": cash,
            "debt_to_equity": ratios.get("debt_to_equity")
            if ratios.get("debt_to_equity") is not None
            else ((total_debt / total_equity) if total_debt is not None and total_equity else None),
            "current_ratio": ratios.get("current_ratio")
            if ratios.get("current_ratio") is not None
            else ((current_assets / current_liabilities) if current_assets is not None and current_liabilities else None),
            "pe_ratio": ratios.get("pe_ratio"),
            "forward_pe": ratios.get("forward_pe"),
            "price_to_sales": (market_cap / revenue) if market_cap is not None and revenue else None,
            "price_to_fcf": (market_cap / fcf) if market_cap is not None and fcf else None,
            "roe": ratios.get("roe")
            if ratios.get("roe") is not None
            else ((net_income / total_equity) if net_income is not None and total_equity else None),
            "roa": ratios.get("roa")
            if ratios.get("roa") is not None
            else ((net_income / total_assets) if net_income is not None and total_assets else None),
        }
    )
    return metrics, "FinanceToolkit/FMP", warnings


def _score_financial_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    growth_inputs = []
    growth = _finite_float(metrics.get("revenue_growth_yoy"))
    if growth is not None:
        growth_inputs.append(_clamp((growth + 0.05) / 0.35))

    profitability_inputs = []
    for key, scale in [("gross_margin", 0.65), ("operating_margin", 0.30), ("net_margin", 0.22), ("fcf_margin", 0.18), ("roe", 0.25), ("roa", 0.12)]:
        value = _finite_float(metrics.get(key))
        if value is not None:
            profitability_inputs.append(_clamp(value / scale))

    liquidity_inputs = []
    current_ratio = _finite_float(metrics.get("current_ratio"))
    if current_ratio is not None:
        liquidity_inputs.append(_clamp(current_ratio / 2.0))
    cash = _finite_float(metrics.get("cash_and_equivalents"))
    debt = _finite_float(metrics.get("total_debt"))
    if cash is not None and debt is not None:
        liquidity_inputs.append(1.0 if debt <= 0 else _clamp(cash / debt))

    leverage_inputs = []
    dte = _finite_float(metrics.get("debt_to_equity"))
    if dte is not None:
        leverage_inputs.append(_clamp(1.0 - (dte / 3.0)))
    if cash is not None and debt is not None:
        leverage_inputs.append(1.0 if cash >= debt else _clamp(cash / max(debt, 1.0)))

    valuation_inputs = []
    for key, ideal, high in [("pe_ratio", 18.0, 50.0), ("forward_pe", 16.0, 45.0), ("price_to_sales", 4.0, 20.0), ("price_to_fcf", 20.0, 60.0)]:
        value = _finite_float(metrics.get(key))
        if value is not None and value > 0:
            if value <= ideal:
                valuation_inputs.append(1.0)
            else:
                valuation_inputs.append(_clamp(1.0 - ((value - ideal) / (high - ideal))))

    def avg(values: list[float]) -> float:
        return round(sum(values) / len(values), 4) if values else 0.0

    scores = {
        "growth_score": avg(growth_inputs),
        "profitability_score": avg(profitability_inputs),
        "liquidity_score": avg(liquidity_inputs),
        "leverage_score": avg(leverage_inputs),
        "valuation_score": avg(valuation_inputs),
    }
    available = [
        scores["growth_score"],
        scores["profitability_score"],
        scores["liquidity_score"],
        scores["leverage_score"],
        scores["valuation_score"],
    ]
    weights = [0.20, 0.30, 0.15, 0.15, 0.20]
    weighted = [
        score * weight
        for score, weight in zip(available, weights)
        if score > 0.0
    ]
    used_weights = [
        weight
        for score, weight in zip(available, weights)
        if score > 0.0
    ]
    scores["financial_quality_score"] = round(sum(weighted) / sum(used_weights), 4) if used_weights else 0.0
    return scores


def _connect_analysis_db(db_path: str | None) -> sqlite3.Connection:
    from create_sql_db import create_database, connect_sqlite

    path = db_path or os.getenv("SQLITE_DB", SQLITE_DB)
    create_database(path)
    conn = connect_sqlite(path)
    return conn


def _persist_financial_metrics(payload: dict[str, Any], db_path: str | None) -> None:
    conn = _connect_analysis_db(db_path)
    try:
        source_provider = payload.get("source_provider") or "unavailable"
        conn.execute(
            """
            INSERT OR REPLACE INTO single_ticker_financial_metrics (
                id, ticker, as_of_date, source_provider, metrics_json, score_json,
                data_quality_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                payload["ticker"],
                payload["as_of_date"],
                source_provider,
                _json_dumps(payload["metrics"]),
                _json_dumps(payload["scores"]),
                _json_dumps(payload["data_quality"]),
                _now_utc(),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def collect_financial_metrics(
    ticker: str,
    *,
    db_path: str | None = None,
    persist: bool = True
) -> dict[str, Any]:
    ticker = str(ticker or "").strip().upper()
    if not ticker:
        return _empty_financial_metrics("", ["missing ticker"])
    metrics, source_provider, warnings = _fetch_financial_metrics_from_provider(ticker)
    metrics = {key: _finite_float(metrics.get(key)) for key in FINANCIAL_METRIC_KEYS}
    missing = [key for key, value in metrics.items() if value is None]
    available_count = len(FINANCIAL_METRIC_KEYS) - len(missing)
    critical = ["revenue_ttm", "net_margin", "free_cash_flow", "total_debt", "cash_and_equivalents"]
    missing_critical = [key for key in critical if metrics.get(key) is None]
    if missing_critical:
        warnings.append("missing critical financial fields: " + ", ".join(missing_critical))
    if available_count < 5:
        warnings.append("too few financial metrics available for high-confidence scoring")
    payload = {
        "ticker": ticker,
        "as_of_date": _today_utc(),
        "source_provider": source_provider,
        "metrics": metrics,
        "scores": _score_financial_metrics(metrics),
        "data_quality": {
            "missing_metrics": missing,
            "available_metrics_count": available_count,
            "warnings": warnings,
        },
    }
    if persist:
        try:
            _persist_financial_metrics(payload, db_path)
        except Exception as exc:
            payload["data_quality"]["warnings"].append(f"persistence failed: {exc}")
    return payload


def _load_price_history(ticker: str, lookback_days: int | None = None) -> tuple[Any | None, list[str]]:
    warnings: list[str] = []
    if pd is None:
        return None, ["pandas unavailable; price history cannot be loaded"]
    parquet_path = Path(os.getenv("HIST_DATA_PARQUET", str(DEFAULT_PARQUET)))
    if not parquet_path.is_file():
        return None, [f"price parquet not found: {parquet_path}"]
    try:
        raw = pd.read_parquet(parquet_path)
    except Exception as exc:
        return None, [f"price parquet load failed: {exc}"]

    ticker = str(ticker or "").upper()
    try:
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns.names = ["metric", "ticker"]
            if ticker not in set(map(str, raw.columns.get_level_values("ticker"))):
                return None, [f"ticker {ticker} not found in local price parquet"]
            sub = raw.xs(ticker, axis=1, level="ticker").copy()
            rename = {
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adj_close",
                "Volume": "volume",
            }
            sub = sub.rename(columns=rename)
            sub["bar_date"] = pd.to_datetime(sub.index).strftime("%Y-%m-%d")
        elif {"ticker", "bar_date"}.issubset(set(raw.columns)):
            sub = raw[raw["ticker"].astype(str).str.upper() == ticker].copy()
            if sub.empty:
                return None, [f"ticker {ticker} not found in local price parquet"]
        else:
            return None, ["unsupported price parquet shape"]
    except Exception as exc:
        return None, [f"price parquet normalization failed: {exc}"]

    if "close" not in sub.columns and "Close" in sub.columns:
        sub = sub.rename(columns={"Close": "close", "Adj Close": "adj_close", "Volume": "volume"})
    if "close" not in sub.columns:
        return None, ["close column missing from price data"]
    sub["bar_date"] = pd.to_datetime(sub["bar_date"]).dt.strftime("%Y-%m-%d")
    sub = sub.sort_values("bar_date").reset_index(drop=True)
    if lookback_days and lookback_days > 0:
        sub = sub.tail(max(int(lookback_days) + 260, int(lookback_days))).reset_index(drop=True)
    return sub, warnings


def _price_series(frame: Any) -> Any | None:
    if pd is None or frame is None:
        return None
    price_col = "adj_close" if "adj_close" in frame.columns and frame["adj_close"].notna().any() else "close"
    try:
        prices = pd.to_numeric(frame[price_col], errors="coerce").dropna()
    except Exception:
        return None
    return prices if not prices.empty else None


def _simple_moving_average(values: Any, window: int) -> float | None:
    if pd is None:
        return None
    series = pd.Series(values).dropna()
    if len(series) < window:
        return None
    return _finite_float(series.tail(window).mean())


def _period_return(values: Any, periods: int) -> float | None:
    if pd is None:
        return None
    series = pd.Series(values).dropna()
    if len(series) <= periods:
        return None
    start = _finite_float(series.iloc[-1 - periods])
    end = _finite_float(series.iloc[-1])
    if start is None or end is None or start == 0.0:
        return None
    return (end / start) - 1.0


def _volatility(values: Any, window: int) -> float | None:
    if pd is None:
        return None
    series = pd.Series(values).dropna()
    if len(series) <= window:
        return None
    returns = series.pct_change().dropna().tail(window)
    if returns.empty:
        return None
    return _finite_float(returns.std() * math.sqrt(252))


def _rsi(values: Any, period: int = 14) -> float | None:
    if pd is None:
        return None
    series = pd.Series(values, dtype="float64").dropna()
    if len(series) <= period:
        return None
    delta = series.diff().dropna()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    avg_gain = gains.tail(period).mean()
    avg_loss = losses.tail(period).mean()
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return _finite_float(100.0 - (100.0 / (1.0 + rs)))


def _max_drawdown(values: Any) -> float | None:
    if pd is None:
        return None
    series = pd.Series(values, dtype="float64").dropna()
    if len(series) < 2:
        return None
    running_max = series.cummax()
    drawdown = (series / running_max) - 1.0
    return _finite_float(drawdown.min())


def _average(values: list[float | None]) -> float:
    present = [float(value) for value in values if value is not None]
    return round(sum(present) / len(present), 4) if present else 0.0


def _score_technical_indicators(indicators: dict[str, Any]) -> dict[str, float]:
    latest = _finite_float(indicators.get("latest_close"))
    sma50 = _finite_float(indicators.get("sma_50"))
    sma200 = _finite_float(indicators.get("sma_200"))
    trend_inputs = []
    if latest is not None and sma50 is not None:
        trend_inputs.append(1.0 if latest > sma50 else 0.35)
    if sma50 is not None and sma200 is not None:
        trend_inputs.append(1.0 if sma50 > sma200 else 0.30)

    momentum_inputs = []
    for key in ("return_63d", "return_126d"):
        value = _finite_float(indicators.get(key))
        if value is not None:
            momentum_inputs.append(_clamp((value + 0.05) / 0.25))

    volatility_inputs = []
    vol63 = _finite_float(indicators.get("volatility_63d"))
    drawdown = _finite_float(indicators.get("max_drawdown"))
    if vol63 is not None:
        volatility_inputs.append(_clamp(1.0 - (vol63 / 0.70)))
    if drawdown is not None:
        volatility_inputs.append(_clamp(1.0 + drawdown / 0.45))

    rs_inputs = []
    for key in ("relative_strength_spy_63d", "relative_strength_spy_126d"):
        value = _finite_float(indicators.get(key))
        if value is not None:
            rs_inputs.append(_clamp((value + 0.05) / 0.20))

    scores = {
        "trend_score": _average(trend_inputs),
        "momentum_score": _average(momentum_inputs),
        "volatility_score": _average(volatility_inputs),
        "relative_strength_score": _average(rs_inputs),
    }
    available = [value for value in scores.values() if value > 0.0]
    scores["technical_score"] = round(sum(available) / len(available), 4) if available else 0.0
    return scores


def _persist_technical_indicators(payload: dict[str, Any], db_path: str | None) -> None:
    conn = _connect_analysis_db(db_path)
    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO single_ticker_technical_indicators (
                id, ticker, as_of_date, lookback_days, indicators_json, score_json,
                data_quality_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                payload["ticker"],
                payload["as_of_date"],
                payload.get("lookback_days"),
                _json_dumps(payload["indicators"]),
                _json_dumps(payload["scores"]),
                _json_dumps(payload["data_quality"]),
                _now_utc(),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def compute_technical_indicators(
    ticker: str,
    *,
    lookback_days: int = 252,
    db_path: str | None = None,
    persist: bool = True
) -> dict[str, Any]:
    ticker = str(ticker or "").strip().upper()
    warnings: list[str] = []
    frame, load_warnings = _load_price_history(ticker, lookback_days)
    warnings.extend(load_warnings)
    indicators = {key: None for key in TECHNICAL_INDICATOR_KEYS}
    price_rows = 0
    as_of_date = _today_utc()
    if frame is not None and pd is not None:
        prices = _price_series(frame)
        price_rows = int(len(prices)) if prices is not None else 0
        if price_rows:
            as_of_date = str(frame.iloc[-1].get("bar_date") or as_of_date)
            indicators["latest_close"] = _finite_float(prices.iloc[-1])
            for window, key in [(20, "sma_20"), (50, "sma_50"), (200, "sma_200")]:
                indicators[key] = _simple_moving_average(prices, window)
            for periods, key in [(20, "return_20d"), (63, "return_63d"), (126, "return_126d"), (252, "return_252d")]:
                indicators[key] = _period_return(prices, periods)
            indicators["volatility_20d"] = _volatility(prices, 20)
            indicators["volatility_63d"] = _volatility(prices, 63)
            indicators["rsi_14"] = _rsi(prices, 14)
            indicators["max_drawdown"] = _max_drawdown(prices.tail(max(lookback_days, 1)))
            if "volume" in frame.columns:
                volume = pd.to_numeric(frame["volume"], errors="coerce").dropna()
                if len(volume) >= 40:
                    recent = _finite_float(volume.tail(20).mean())
                    prior = _finite_float(volume.iloc[-40:-20].mean())
                    indicators["volume_trend"] = ((recent / prior) - 1.0) if recent is not None and prior else None
        else:
            warnings.append("no usable close prices")

        spy_frame, spy_warnings = _load_price_history("SPY", lookback_days)
        if spy_frame is not None:
            spy_prices = _price_series(spy_frame)
            if prices is not None and spy_prices is not None:
                for periods, key in [(63, "relative_strength_spy_63d"), (126, "relative_strength_spy_126d")]:
                    ticker_ret = _period_return(prices, periods)
                    spy_ret = _period_return(spy_prices, periods)
                    indicators[key] = (ticker_ret - spy_ret) if ticker_ret is not None and spy_ret is not None else None
        elif not any("SPY" in warning for warning in spy_warnings):
            warnings.extend(spy_warnings[:1])

    missing = [key for key, value in indicators.items() if value is None]
    if price_rows < min(lookback_days, 60):
        warnings.append("insufficient price history for full indicator set")
    payload = {
        "ticker": ticker,
        "as_of_date": as_of_date,
        "lookback_days": lookback_days,
        "indicators": indicators,
        "scores": _score_technical_indicators(indicators),
        "data_quality": {
            "price_rows": price_rows,
            "missing_fields": missing,
            "warnings": warnings,
        },
    }
    if persist:
        try:
            _persist_technical_indicators(payload, db_path)
        except Exception as exc:
            payload["data_quality"]["warnings"].append(f"persistence failed: {exc}")
    return payload


def _strategy_position_frame(frame: Any, strategy_name: str) -> tuple[Any | None, list[str]]:
    warnings: list[str] = []
    if pd is None or frame is None:
        return None, ["price data unavailable"]
    if strategy_name not in STRATEGY_NAMES:
        return None, [f"unsupported strategy: {strategy_name}"]
    data = frame.copy()
    prices = _price_series(data)
    if prices is None or len(prices) < 2:
        return None, ["insufficient price history"]
    data = data.loc[prices.index].copy()
    data["price"] = prices.astype(float)
    data["return"] = data["price"].pct_change()
    data["sma_50"] = data["price"].rolling(50).mean()
    data["sma_200"] = data["price"].rolling(200).mean()
    data["return_63d"] = data["price"].pct_change(63)
    data["return_126d"] = data["price"].pct_change(126)
    data["rsi_14"] = [
        _rsi(data["price"].iloc[: idx + 1], 14)
        for idx in range(len(data))
    ]

    if strategy_name == "moving_average_trend":
        if len(data) < 200:
            warnings.append("fewer than 200 rows; moving-average trend strategy cannot fully initialize")
        signal = ((data["price"] > data["sma_50"]) & (data["sma_50"] > data["sma_200"])).astype(float)
    elif strategy_name == "momentum":
        if len(data) < 127:
            warnings.append("fewer than 127 rows; momentum strategy cannot fully initialize")
        signal = ((data["return_63d"] > 0.0) & (data["return_126d"] > 0.0)).astype(float)
    else:
        if len(data) < 15:
            warnings.append("fewer than 15 rows; RSI strategy cannot fully initialize")
        signal = []
        active = 0.0
        for rsi_value in data["rsi_14"]:
            if rsi_value is None or pd.isna(rsi_value):
                signal.append(active)
                continue
            if rsi_value < 30.0:
                active = 1.0
            elif rsi_value >= 50.0:
                active = 0.0
            signal.append(active)
        signal = pd.Series(signal, index=data.index)

    data["raw_signal"] = signal
    data["position"] = data["raw_signal"].shift(1).fillna(0.0)
    data["strategy_return"] = data["position"] * data["return"].fillna(0.0)
    return data, warnings


def _extract_trades(data: Any) -> list[dict[str, Any]]:
    if pd is None or data is None or data.empty:
        return []
    trades: list[dict[str, Any]] = []
    position = data["position"].fillna(0.0)
    in_trade = False
    entry_price = None
    entry_date = None
    for idx, pos in position.items():
        row = data.loc[idx]
        date = str(row.get("bar_date") or idx)[:10]
        price = _finite_float(row.get("price"))
        if pos > 0 and not in_trade:
            in_trade = True
            entry_price = price
            entry_date = date
        elif pos <= 0 and in_trade:
            exit_price = price
            trade_return = None
            if entry_price and exit_price:
                trade_return = (exit_price / entry_price) - 1.0
            trades.append(
                {
                    "entry_date": entry_date,
                    "exit_date": date,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "return": trade_return,
                }
            )
            in_trade = False
            entry_price = None
            entry_date = None
    if in_trade:
        row = data.iloc[-1]
        exit_price = _finite_float(row.get("price"))
        trade_return = (exit_price / entry_price) - 1.0 if entry_price and exit_price else None
        trades.append(
            {
                "entry_date": entry_date,
                "exit_date": str(row.get("bar_date") or data.index[-1])[:10],
                "entry_price": entry_price,
                "exit_price": exit_price,
                "return": trade_return,
            }
        )
    return trades


def _backtest_metrics(data: Any, trades: list[dict[str, Any]], spy_frame: Any | None = None) -> dict[str, Any]:
    metrics = {
        "total_return": None,
        "benchmark_return": None,
        "annualized_return": None,
        "annualized_volatility": None,
        "sharpe": None,
        "max_drawdown": None,
        "win_rate": None,
        "trades_count": len(trades),
    }
    if pd is None or data is None or len(data) < 2:
        return metrics
    strat_returns = data["strategy_return"].fillna(0.0)
    equity = (1.0 + strat_returns).cumprod()
    total_return = _finite_float(equity.iloc[-1] - 1.0)
    prices = data["price"].dropna()
    benchmark_return = (prices.iloc[-1] / prices.iloc[0]) - 1.0 if len(prices) >= 2 and prices.iloc[0] else None
    ann_vol = _finite_float(strat_returns.std() * math.sqrt(252))
    periods = max(len(strat_returns), 1)
    ann_return = ((1.0 + total_return) ** (252.0 / periods) - 1.0) if total_return is not None and total_return > -1.0 else None
    sharpe = (ann_return / ann_vol) if ann_return is not None and ann_vol and ann_vol > 0 else None
    wins = [t for t in trades if t.get("return") is not None and t["return"] > 0]
    closed = [t for t in trades if t.get("return") is not None]
    metrics.update(
        {
            "total_return": total_return,
            "benchmark_return": benchmark_return,
            "annualized_return": ann_return,
            "annualized_volatility": ann_vol,
            "sharpe": sharpe,
            "max_drawdown": _max_drawdown(equity),
            "win_rate": (len(wins) / len(closed)) if closed else None,
            "trades_count": len(trades),
        }
    )
    if spy_frame is not None:
        spy_prices = _price_series(spy_frame)
        if spy_prices is not None and len(spy_prices) >= 2 and spy_prices.iloc[0]:
            metrics["spy_return"] = (spy_prices.iloc[-1] / spy_prices.iloc[0]) - 1.0
    return metrics


def _persist_backtest(payload: dict[str, Any], db_path: str | None) -> None:
    conn = _connect_analysis_db(db_path)
    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO strategy_backtest_runs (
                backtest_id, strategy_name, ticker, start_date, end_date, params_json,
                metrics_json, trades_json, data_quality_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload["backtest_id"],
                payload["strategy_name"],
                payload["ticker"],
                payload.get("start_date"),
                payload.get("end_date"),
                _json_dumps(payload.get("params") or {}),
                _json_dumps(payload["metrics"]),
                _json_dumps(payload.get("trades") or []),
                _json_dumps(payload["data_quality"]),
                _now_utc(),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def run_strategy_backtest(
    ticker: str,
    strategy_name: str,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    params: dict[str, Any] | None = None,
    db_path: str | None = None,
    persist: bool = True
) -> dict[str, Any]:
    ticker = str(ticker or "").strip().upper()
    strategy_name = str(strategy_name or "").strip()
    params = dict(params or {})
    frame, warnings = _load_price_history(ticker, None)
    if frame is not None and pd is not None:
        frame = frame.copy()
        if start_date:
            frame = frame[frame["bar_date"].astype(str) >= start_date]
        if end_date:
            frame = frame[frame["bar_date"].astype(str) <= end_date]
    data, strategy_warnings = _strategy_position_frame(frame, strategy_name)
    warnings.extend(strategy_warnings)
    trades = _extract_trades(data)
    spy_frame, _ = _load_price_history("SPY", None)
    metrics = _backtest_metrics(data, trades, spy_frame)
    effective_start = None
    effective_end = None
    if data is not None and pd is not None and not data.empty:
        effective_start = str(data.iloc[0].get("bar_date") or start_date)
        effective_end = str(data.iloc[-1].get("bar_date") or end_date)
    payload = {
        "backtest_id": str(uuid.uuid4()),
        "ticker": ticker,
        "strategy_name": strategy_name,
        "start_date": effective_start or start_date,
        "end_date": effective_end or end_date,
        "params": params,
        "metrics": metrics,
        "trades": trades,
        "data_quality": {
            "warnings": warnings,
        },
    }
    if persist:
        try:
            _persist_backtest(payload, db_path)
        except Exception as exc:
            payload["data_quality"]["warnings"].append(f"persistence failed: {exc}")
    return payload


def _latest_backtest_for_signal(ticker: str, strategy_name: str, db_path: str | None) -> dict[str, Any] | None:
    path = db_path or os.getenv("SQLITE_DB", SQLITE_DB)
    if not Path(path).is_file():
        return None
    try:
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT backtest_id, metrics_json, data_quality_json
            FROM strategy_backtest_runs
            WHERE ticker = ? AND strategy_name = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (ticker, strategy_name),
        ).fetchone()
    except sqlite3.Error:
        return None
    finally:
        try:
            conn.close()
        except Exception:
            pass
    if not row:
        return None
    try:
        return {
            "backtest_id": row["backtest_id"],
            "metrics": json.loads(row["metrics_json"] or "{}"),
            "data_quality": json.loads(row["data_quality_json"] or "{}"),
        }
    except Exception:
        return None


def _signal_from_indicators(strategy_name: str, technical: dict[str, Any]) -> tuple[str, float, str, dict[str, list[str]]]:
    indicators = technical.get("indicators", {})
    rationale = {"supporting": [], "contradicting": [], "warnings": list(technical.get("data_quality", {}).get("warnings", []))}
    if strategy_name == "moving_average_trend":
        latest = _finite_float(indicators.get("latest_close"))
        sma50 = _finite_float(indicators.get("sma_50"))
        sma200 = _finite_float(indicators.get("sma_200"))
        if latest is None or sma50 is None or sma200 is None:
            rationale["warnings"].append("moving average inputs unavailable")
            return "unclear", 0.0, "medium_term", rationale
        if latest > sma50 > sma200:
            rationale["supporting"].append("close above SMA50 and SMA50 above SMA200")
            return "bullish", 0.8, "medium_term", rationale
        rationale["contradicting"].append("moving average trend condition not met")
        return "neutral", 0.35, "medium_term", rationale
    if strategy_name == "momentum":
        ret63 = _finite_float(indicators.get("return_63d"))
        ret126 = _finite_float(indicators.get("return_126d"))
        rs63 = _finite_float(indicators.get("relative_strength_spy_63d"))
        if ret63 is None or ret126 is None:
            rationale["warnings"].append("momentum return inputs unavailable")
            return "unclear", 0.0, "medium_term", rationale
        if ret63 > 0 and ret126 > 0:
            strength = 0.65 + (0.15 if rs63 is not None and rs63 > 0 else 0.0)
            rationale["supporting"].append("63d and 126d returns are positive")
            if rs63 is not None and rs63 > 0:
                rationale["supporting"].append("outperforming SPY over 63d")
            return "bullish", strength, "medium_term", rationale
        rationale["contradicting"].append("momentum returns are not both positive")
        return "neutral", 0.30, "medium_term", rationale
    if strategy_name == "rsi_mean_reversion":
        rsi_value = _finite_float(indicators.get("rsi_14"))
        if rsi_value is None:
            rationale["warnings"].append("RSI unavailable")
            return "unclear", 0.0, "near_term", rationale
        if rsi_value < 30.0:
            rationale["supporting"].append("RSI below 30")
            return "bullish", _clamp((30.0 - rsi_value) / 30.0 + 0.45), "near_term", rationale
        if rsi_value > 70.0:
            rationale["contradicting"].append("RSI above 70")
            return "bearish", _clamp((rsi_value - 70.0) / 30.0 + 0.35), "near_term", rationale
        return "neutral", 0.35, "near_term", rationale
    rationale["warnings"].append(f"unsupported strategy: {strategy_name}")
    return "unclear", 0.0, "near_term", rationale


def _persist_signal(payload: dict[str, Any], db_path: str | None) -> None:
    conn = _connect_analysis_db(db_path)
    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO strategy_signals (
                signal_id, ticker, strategy_name, signal_date, signal_direction,
                signal_strength, confidence, horizon, metrics_json, rationale_json,
                backtest_id, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload["signal_id"],
                payload["ticker"],
                payload["strategy_name"],
                payload["signal_date"],
                payload["signal_direction"],
                payload.get("signal_strength"),
                payload.get("confidence"),
                payload.get("horizon"),
                _json_dumps(payload.get("metrics") or {}),
                _json_dumps(payload.get("rationale") or {}),
                payload.get("backtest_id"),
                _now_utc(),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def generate_strategy_signal(
    ticker: str,
    strategy_name: str,
    *,
    db_path: str | None = None,
    persist: bool = True
) -> dict[str, Any]:
    ticker = str(ticker or "").strip().upper()
    strategy_name = str(strategy_name or "").strip()
    technical = compute_technical_indicators(ticker, db_path=db_path, persist=False)
    direction, strength, horizon, rationale = _signal_from_indicators(strategy_name, technical)
    backtest = _latest_backtest_for_signal(ticker, strategy_name, db_path)
    backtest_id = backtest.get("backtest_id") if backtest else None
    missing_count = len(technical.get("data_quality", {}).get("missing_fields", []))
    total_count = len(TECHNICAL_INDICATOR_KEYS)
    data_confidence = _clamp(1.0 - (missing_count / max(total_count, 1)))
    backtest_confidence = 0.0
    if backtest:
        sharpe = _finite_float(backtest.get("metrics", {}).get("sharpe"))
        total_return = _finite_float(backtest.get("metrics", {}).get("total_return"))
        if sharpe is not None:
            backtest_confidence += _clamp((sharpe + 0.5) / 2.0) * 0.5
        if total_return is not None:
            backtest_confidence += _clamp((total_return + 0.10) / 0.50) * 0.5
    confidence = round(_clamp(0.75 * data_confidence + 0.25 * backtest_confidence), 4)
    payload = {
        "signal_id": str(uuid.uuid4()),
        "ticker": ticker,
        "strategy_name": strategy_name,
        "signal_date": technical.get("as_of_date") or _today_utc(),
        "signal_direction": direction,
        "signal_strength": round(float(strength), 4),
        "confidence": confidence,
        "horizon": horizon,
        "metrics": technical.get("indicators", {}),
        "rationale": rationale,
        "backtest_id": backtest_id,
    }
    if persist:
        try:
            _persist_signal(payload, db_path)
        except Exception as exc:
            payload["rationale"]["warnings"].append(f"persistence failed: {exc}")
    return payload


def _print_json(payload: Any) -> None:
    if isinstance(payload, ResolvedTicker):
        payload = asdict(payload)
    print(_json_dumps(payload))


def _run_full(args: argparse.Namespace) -> dict[str, Any]:
    ticker_or_query = args.ticker
    resolution = resolve_single_ticker(ticker_or_query)
    ticker = resolution.ticker or str(ticker_or_query).strip().upper()
    result: dict[str, Any] = {
        "resolution": asdict(resolution),
        "financial": collect_financial_metrics(ticker, db_path=args.db_path, persist=not args.no_persist),
        "technical": compute_technical_indicators(ticker, lookback_days=args.lookback_days, db_path=args.db_path, persist=not args.no_persist),
        "signals": {},
    }
    if args.include_backtests or os.getenv("ENABLE_SINGLE_TICKER_BACKTESTS", "0").strip().lower() in {"1", "true", "yes", "on"}:
        result["backtests"] = {}
    for strategy in sorted(STRATEGY_NAMES):
        if "backtests" in result:
            result["backtests"][strategy] = run_strategy_backtest(ticker, strategy, db_path=args.db_path, persist=not args.no_persist)
        result["signals"][strategy] = generate_strategy_signal(ticker, strategy, db_path=args.db_path, persist=not args.no_persist)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-ticker structured analysis")
    parser.add_argument("--db-path", default=None)
    parser.add_argument("--no-persist", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("resolve")
    p.add_argument("ticker")

    p = sub.add_parser("financial")
    p.add_argument("ticker")

    p = sub.add_parser("technical")
    p.add_argument("ticker")
    p.add_argument("--lookback-days", type=int, default=252)

    p = sub.add_parser("backtest")
    p.add_argument("ticker")
    p.add_argument("--strategy", default="moving_average_trend")
    p.add_argument("--start-date", default=None)
    p.add_argument("--end-date", default=None)

    p = sub.add_parser("signal")
    p.add_argument("ticker")
    p.add_argument("--strategy", default="moving_average_trend")

    p = sub.add_parser("full")
    p.add_argument("ticker")
    p.add_argument("--lookback-days", type=int, default=252)
    p.add_argument("--include-backtests", action="store_true")

    args = parser.parse_args()
    try:
        if args.command == "resolve":
            _print_json(resolve_single_ticker(args.ticker))
        elif args.command == "financial":
            _print_json(collect_financial_metrics(args.ticker, db_path=args.db_path, persist=not args.no_persist))
        elif args.command == "technical":
            _print_json(compute_technical_indicators(args.ticker, lookback_days=args.lookback_days, db_path=args.db_path, persist=not args.no_persist))
        elif args.command == "backtest":
            _print_json(run_strategy_backtest(args.ticker, args.strategy, start_date=args.start_date, end_date=args.end_date, db_path=args.db_path, persist=not args.no_persist))
        elif args.command == "signal":
            _print_json(generate_strategy_signal(args.ticker, args.strategy, db_path=args.db_path, persist=not args.no_persist))
        elif args.command == "full":
            _print_json(_run_full(args))
    except Exception as exc:  # expected CLI data issues should surface as JSON
        _print_json({"error": str(exc), "warnings": ["command failed"]})


if __name__ == "__main__":
    main()
