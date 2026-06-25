import json
import sqlite3
import sys
import tempfile
import time
import types
import unittest
from unittest import mock
from pathlib import Path

try:
    from rag_trial.chat import chatter
    CHATTER_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - runtime environment guard
    chatter = types.ModuleType("chatter")
    sys.modules["chatter"] = chatter
    CHATTER_IMPORT_ERROR = exc

from rag_trial.evaluation import evaluation_suite
from rag_trial.analysis import macro_extract
import single_ticker_analysis as sta
from rag_trial.db.create_sql_db import create_database, ensure_migrations


def _single_ticker_price_frame(values, *, start="2025-01-01", ticker="AAA"):
    import pandas as pd

    dates = pd.date_range(start, periods=len(values), freq="D")
    return pd.DataFrame(
        {
            "ticker": ticker,
            "bar_date": dates.strftime("%Y-%m-%d"),
            "close": [float(v) for v in values],
            "adj_close": [float(v) for v in values],
            "volume": [1000.0 + i for i in range(len(values))],
        }
    )


class TestRagPipelineUpgrades(unittest.TestCase):
    def _seed_signal_discovery_db(self, conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            INSERT INTO articles (
                article_id, url, source, source_provider, source_trust_tier, content_class,
                article_quality_score, published_at, title, raw_text, status, processing_state
            ) VALUES (
                'article-1', 'https://example.com/oil-signal', 'Reuters', 'wire', 'tier_1', 'news_report',
                0.95, '2026-04-24T08:00:00+00:00', 'Oil supply risk rises after OPEC cuts', 'full text', 'published', 'processed'
            );

            INSERT INTO chunks (
                chunk_id, article_id, chunk_index, period_key, published_date, text, embedding_json, processing_state
            ) VALUES (
                'chunk-1', 'article-1', 0, '2026-04-24', '2026-04-24',
                'Oil prices jumped after OPEC extended cuts and traders warned of tighter supply.',
                '[1.0, 0.0, 0.0]',
                'processed'
            );

            INSERT INTO macro_extraction_runs (
                run_id, article_id, chunk_id, model_provider, model_name, prompt_version, schema_version,
                created_at, success, raw_json
            ) VALUES (
                'run-1', 'article-1', 'chunk-1', 'test-provider', 'test-model', 'v1', 'v1',
                '2026-04-24T08:05:00+00:00', 1, '{}'
            );

            INSERT INTO macro_events (
                macro_event_id, run_id, article_id, chunk_id, event_type, summary, region, time_horizon,
                event_time_start, event_time_end, confidence, verification_status, support_score, novelty_hint,
                urgency, market_surprise
            ) VALUES (
                'event-1', 'run-1', 'article-1', 'chunk-1', 'commodity_supply_disruption',
                'OPEC supply cuts tightened the oil market.', 'global', 'near_term',
                '2026-04-24', '2026-04-24', 0.81, 'verified', 0.87, 'new', 'high', 'high'
            );

            INSERT INTO event_clusters (
                cluster_id, event_type, primary_shock_type, region, canonical_summary, summary_embedding_json,
                first_event_time, last_event_time, cluster_window_days, member_count, unique_source_count,
                asset_targets_json, cluster_status, created_at, updated_at
            ) VALUES (
                'cluster-1', 'commodity_supply_disruption', 'commodity_supply_disruption', 'global',
                'Oil supply disruption after OPEC cuts tightened balances.',
                '[1.0, 0.0, 0.0]',
                '2026-04-24', '2026-04-24', 7, 1, 1,
                '["oil","brent"]', 'active', '2026-04-24T08:06:00+00:00', '2026-04-24T08:06:00+00:00'
            );

            INSERT INTO cluster_members (
                cluster_id, macro_event_id, similarity_score, match_reasons_json, event_time, article_id, chunk_id, source, created_at
            ) VALUES (
                'cluster-1', 'event-1', 0.95, '["same_story"]', '2026-04-24', 'article-1', 'chunk-1', 'Reuters', '2026-04-24T08:06:00+00:00'
            );

            INSERT INTO event_cluster_scores (
                score_id, cluster_id, score_date, novelty_score, source_quality_score, velocity_score,
                asset_impact_score, confidence_score, recency_score, signal_score,
                supporting_event_count, supporting_source_count, created_at, updated_at
            ) VALUES (
                'score-1', 'cluster-1', '2026-04-24', 0.92, 0.88, 0.81,
                0.79, 0.84, 0.95, 0.90,
                1, 1, '2026-04-24T08:07:00+00:00', '2026-04-24T08:07:00+00:00'
            );

            INSERT INTO signal_alerts (
                signal_id, cluster_id, score_id, signal_date, rank, signal_score,
                headline, summary, novelty_hint, urgency, market_surprise, top_assets_json,
                status, created_at, updated_at
            ) VALUES (
                'signal-1', 'cluster-1', 'score-1', '2026-04-24', 1, 0.90,
                'Oil supply risk rises after OPEC cuts',
                'A fresh supply disruption narrative is pushing oil higher.',
                'new', 'high', 'high', '["oil","brent"]',
                'active', '2026-04-24T08:08:00+00:00', '2026-04-24T08:08:00+00:00'
            );
            """
        )
        conn.commit()

    def test_enforce_enums_returns_cleaned_events_and_audit_rows(self):
        events, audits = macro_extract._enforce_enums(
            [
                {
                    "event_type": "growth_upside_surprize",
                    "summary": "Example summary",
                    "region": "global",
                    "time_horizon": "near_term",
                    "shock_types": ["growth_upside_surprize", "unknown_value"],
                    "channels": [
                        {
                            "channel_name": "growth_differentials",
                            "direction": "upward",
                            "strength": "strong",
                        }
                    ],
                    "asset_impacts": [
                        {
                            "target_type": "ticker",
                            "target_id": "NVDA",
                            "direction": "up",
                            "strength": "very_strong",
                            "horizon": "near_term",
                            "rationale": "example",
                        }
                    ],
                    "evidence_spans": ["quoted phrase"],
                    "confidence": 0.8,
                    "novelty_hint": "fresh",
                    "urgency": "High",
                    "market_surprise": "medium",
                }
            ]
        )

        self.assertEqual(events[0]["event_type"], "growth_upside_surprise")
        self.assertEqual(events[0]["shock_types"], ["growth_upside_surprise"])
        self.assertEqual(events[0]["channels"], [])
        self.assertEqual(events[0]["asset_impacts"][0]["strength"], "strong")
        self.assertIsNone(events[0]["novelty_hint"])
        self.assertEqual(events[0]["urgency"], "high")
        self.assertEqual(events[0]["market_surprise"], "medium")
        actions = {(row["field_label"], row["action"]) for row in audits}
        self.assertIn(("event_type", "snapped"), actions)
        self.assertIn(("shock_types", "dropped"), actions)
        self.assertIn(("novelty_hint", "dropped"), actions)

    def test_extract_verifications_parses_optional_signal_tags(self):
        rows = macro_extract._extract_verifications(
            {
                "verifications": [
                    {
                        "candidate_index": 0,
                        "verification_status": "VERIFIED",
                        "support_score": 0.82,
                        "confidence_calibrated": 0.71,
                        "rejection_reason": "",
                        "novelty_hint": "new",
                        "urgency": "high",
                        "market_surprise": "medium",
                        "verifier_notes": "two direct anchors",
                    }
                ]
            }
        )

        self.assertEqual(rows[0]["verification_status"], "verified")
        self.assertEqual(rows[0]["novelty_hint"], "new")
        self.assertEqual(rows[0]["urgency"], "high")
        self.assertEqual(rows[0]["market_surprise"], "medium")
        self.assertEqual(rows[0]["verifier_notes"], "two direct anchors")

    def test_verification_rows_store_trace_data_for_weak_or_rejected_candidates(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "macro_trace.db"
            create_database(str(db_path))
            conn = sqlite3.connect(db_path)

            chunk = {
                "article_id": "article-1",
                "chunk_id": "chunk-1",
                "text": "Oil prices jumped after OPEC extended cuts and traders flagged a supply shock.",
                "content_class": "news_report",
            }
            candidates = [
                {
                    "event_type": "commodity_supply_disruption",
                    "summary": "OPEC supply cuts lifted oil prices.",
                    "region": "global",
                    "time_horizon": "near_term",
                    "shock_types": ["commodity_supply_disruption"],
                    "channels": [],
                    "asset_impacts": [],
                    "evidence_spans": ["Oil prices jumped", "OPEC extended cuts"],
                    "confidence": 0.78,
                    "novelty_hint": "new",
                    "urgency": "high",
                    "market_surprise": "medium",
                }
            ]
            candidate_ids = macro_extract._write_candidate_rows(
                conn,
                run_id="run-1",
                chunk=chunk,
                candidates=candidates,
            )
            rows = macro_extract._write_verification_rows(
                conn,
                run_id="run-1",
                chunk=chunk,
                candidates=candidates,
                verifications=[
                    {
                        "candidate_index": 0,
                        "verification_status": "weak",
                        "support_score": 0.6,
                        "confidence_calibrated": 0.55,
                        "rejection_reason": "",
                        "novelty_hint": "continuation",
                        "urgency": "high",
                        "market_surprise": "high",
                        "verifier_notes": "support exists but context is still thin",
                    }
                ],
                candidate_ids=candidate_ids,
            )

            stored = conn.execute(
                """
                SELECT verification_status, verifier_notes_json
                FROM macro_event_verifications
                WHERE verification_id = ?
                """,
                (macro_extract._md5("run-1::verification::0"),),
            ).fetchone()
            conn.close()

        self.assertEqual(rows[0]["verification_status"], "weak")
        self.assertEqual(rows[0]["novelty_hint"], "continuation")
        self.assertEqual(rows[0]["market_surprise"], "high")
        self.assertEqual(stored[0], "weak")
        notes = json.loads(stored[1])
        self.assertEqual(notes["novelty_hint"], "continuation")
        self.assertEqual(notes["urgency"], "high")
        self.assertEqual(notes["market_surprise"], "high")
        self.assertEqual(notes["matched_spans"], ["Oil prices jumped", "OPEC extended cuts"])

    def test_write_normalized_persists_verification_metadata(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "macro_normalized.db"
            create_database(str(db_path))
            conn = sqlite3.connect(db_path)
            macro_extract._write_normalized(
                conn,
                "run-2",
                {"article_id": "article-1", "chunk_id": "chunk-2"},
                [
                    {
                        "event_type": "commodity_supply_disruption",
                        "summary": "OPEC supply cuts lifted oil prices.",
                        "region": "global",
                        "time_horizon": "near_term",
                        "shock_types": ["commodity_supply_disruption"],
                        "channels": [],
                        "asset_impacts": [],
                        "evidence_spans": ["OPEC extended cuts"],
                        "confidence": 0.74,
                        "verification_status": "verified",
                        "support_score": 0.88,
                        "novelty_hint": "new",
                        "urgency": "high",
                        "market_surprise": "medium",
                    }
                ],
            )
            row = conn.execute(
                """
                SELECT verification_status, support_score, novelty_hint, urgency, market_surprise
                FROM macro_events
                WHERE macro_event_id = ?
                """,
                (macro_extract._md5("run-2::0"),),
            ).fetchone()
            conn.close()

        self.assertEqual(row, ("verified", 0.88, "new", "high", "medium"))

    def test_answer_grounding_detects_expected_sections(self):
        grounding = evaluation_suite._answer_grounding(
            "Answer: Nvidia rallied [S1]\nEvidence: Demand was strong [S1][S2]\nTheory: None.",
            {"chunk-1": "S1", "chunk-2": "S2"},
        )
        self.assertTrue(grounding["has_inline_citations"])
        self.assertTrue(grounding["has_answer_section"])
        self.assertTrue(grounding["has_evidence_section"])
        self.assertTrue(grounding["has_theory_section"])
        self.assertEqual(grounding["citation_count"], 2)

    def test_answer_grounding_accepts_financial_and_market_tags(self):
        grounding = evaluation_suite._answer_grounding(
            "Answer: Revenue rose year-over-year [F]\nEvidence: Price action stayed positive [M]\nTheory: None.",
            {"chunk-1": "S1"},
        )
        self.assertTrue(grounding["has_inline_citations"])
        self.assertTrue(grounding["has_financial_citation"])
        self.assertTrue(grounding["has_market_citation"])
        self.assertEqual(grounding["citation_count"], 2)
        self.assertEqual(grounding["used_citations"], ["F", "M"])

    def test_build_citation_map_and_provenance(self):
        if CHATTER_IMPORT_ERROR is not None:
            self.skipTest(f"chatter import failed: {CHATTER_IMPORT_ERROR}")
        chunks = [
            {
                "chunk_uid": "chunk-1",
                "source": "CNBC",
                "title": "First title",
                "retrieval_kind": "entity_mentions",
                "text": "first",
            },
            {
                "chunk_uid": "chunk-2",
                "source": "Reuters",
                "title": "Second title",
                "retrieval_kind": "sqlite_semantic",
                "text": "second",
            },
        ]
        citation_map = chatter.build_citation_map(chunks)
        self.assertEqual(citation_map, {"chunk-1": "S1", "chunk-2": "S2"})
        provenance = chatter.format_provenance(chunks)
        self.assertIn("[S1]", provenance)
        self.assertIn("chunk=chunk-2", provenance)
        structured = chatter.ensure_structured_answer("A plain answer", chunks)
        self.assertIn("Answer:", structured)
        self.assertIn("Evidence:", structured)
        self.assertIn("Theory:", structured)
        self.assertIn("[S1]", structured)

    def test_classify_query_route_detects_signal_discovery_intent(self):
        if CHATTER_IMPORT_ERROR is not None:
            self.skipTest(f"chatter import failed: {CHATTER_IMPORT_ERROR}")
        route = chatter.classify_query_route(
            query="What are the top knowledge arbitrage signals right now?",
            summary_mode=False,
            causal_intent=False,
            market_data_intent=False,
            target=None,
        )
        self.assertEqual(route, "signal_discovery")

    def test_classify_query_route_uses_single_ticker_financial_for_finance_intent(self):
        if CHATTER_IMPORT_ERROR is not None:
            self.skipTest(f"chatter import failed: {CHATTER_IMPORT_ERROR}")
        query = "Is QCOM fundamentally sound based on its financial statements?"
        target = chatter.QueryTarget(
            query_type=chatter.QUERY_TYPE_SINGLE,
            canonical_name="QCOM",
            display_name="QUALCOMM",
            ticker="QCOM",
            entity_type="ORG",
            confidence=0.99,
        )
        route = chatter.classify_query_route(
            query=query,
            summary_mode=False,
            causal_intent=False,
            market_data_intent=chatter.is_market_data_intent(query),
            financial_intent=chatter.is_single_ticker_financial_intent(query),
            explicit_latest_news_intent=chatter.is_explicit_latest_news_query(query),
            target=target,
        )
        self.assertEqual(route, "single_ticker_financial")
        self.assertNotEqual(route, "live_market_data")

    def test_classify_query_route_does_not_use_fundamentals_route_for_price_trend(self):
        if CHATTER_IMPORT_ERROR is not None:
            self.skipTest(f"chatter import failed: {CHATTER_IMPORT_ERROR}")
        query = "What is QCOM price trend?"
        target = chatter.QueryTarget(
            query_type=chatter.QUERY_TYPE_SINGLE,
            canonical_name="QCOM",
            display_name="QUALCOMM",
            ticker="QCOM",
            entity_type="ORG",
            confidence=0.99,
        )
        route = chatter.classify_query_route(
            query=query,
            summary_mode=False,
            causal_intent=False,
            market_data_intent=chatter.is_market_data_intent(query),
            financial_intent=chatter.is_single_ticker_financial_intent(query),
            explicit_latest_news_intent=chatter.is_explicit_latest_news_query(query),
            target=target,
        )
        self.assertFalse(chatter.is_single_ticker_financial_intent(query))
        self.assertEqual(route, "live_market_data")
        self.assertNotEqual(route, "single_ticker_financial")

    def test_classify_query_route_detects_market_tickers_today_intent(self):
        if CHATTER_IMPORT_ERROR is not None:
            self.skipTest(f"chatter import failed: {CHATTER_IMPORT_ERROR}")

        for query in (
            "What are the market tickers for today?",
            "today's market tickers",
        ):
            route = chatter.classify_query_route(
                query=query,
                summary_mode=False,
                causal_intent=False,
                market_data_intent=chatter.is_market_data_intent(query),
                target=None,
            )
            self.assertEqual(route, "market_tickers_today")

    def test_resolve_query_target_detects_unmapped_plain_ticker_with_financial_context(self):
        if CHATTER_IMPORT_ERROR is not None:
            self.skipTest(f"chatter import failed: {CHATTER_IMPORT_ERROR}")
        query = "Is MXL fundamentally sound based on its financial statements?"
        target = chatter.resolve_query_target(
            query=query,
            alias_to_ticker={},
            ticker_to_canonical={},
            driver=None,
            sqlite_conn=None,
            alias_to_fin_entity={},
        )

        self.assertEqual(target.query_type, chatter.QUERY_TYPE_SINGLE)
        self.assertEqual(target.ticker, "MXL")
        self.assertEqual(target.canonical_name, "MXL")
        self.assertEqual(target.resolution_mode, "ticker_token_unmapped")
        self.assertGreater(target.confidence, 0.6)

    def test_resolve_query_target_detects_unmapped_compound_ticker_with_financial_context(self):
        if CHATTER_IMPORT_ERROR is not None:
            self.skipTest(f"chatter import failed: {CHATTER_IMPORT_ERROR}")
        query = "Is DX-Y.NYB fundamentally sound based on its financial statements?"
        target = chatter.resolve_query_target(
            query=query,
            alias_to_ticker={},
            ticker_to_canonical={},
            driver=None,
            sqlite_conn=None,
            alias_to_fin_entity={},
        )

        self.assertEqual(target.query_type, chatter.QUERY_TYPE_SINGLE)
        self.assertEqual(target.ticker, "DX-Y.NYB")
        self.assertEqual(target.canonical_name, "DX-Y.NYB")
        self.assertEqual(target.resolution_mode, "ticker_token_compound_unmapped")
        self.assertGreater(target.confidence, 0.6)

    def test_resolve_query_target_does_not_map_macro_acronym_to_unmapped_ticker(self):
        if CHATTER_IMPORT_ERROR is not None:
            self.skipTest(f"chatter import failed: {CHATTER_IMPORT_ERROR}")
        query = "What does GDP trend look like this quarter?"
        target = chatter.resolve_query_target(
            query=query,
            alias_to_ticker={},
            ticker_to_canonical={},
            driver=None,
            sqlite_conn=None,
            alias_to_fin_entity={},
        )

        self.assertEqual(target.query_type, chatter.QUERY_TYPE_GENERAL)
        self.assertIsNone(target.ticker)
        self.assertEqual(target.resolution_mode, "unresolved")

    def test_compute_price_trend_handles_period_index_for_ytd(self):
        if CHATTER_IMPORT_ERROR is not None:
            self.skipTest(f"chatter import failed: {CHATTER_IMPORT_ERROR}")

        import pandas as pd

        hist = pd.DataFrame(
            {
                "Close": [100.0, 110.0, 120.0],
                "Volume": [1000, 1200, 1400],
            },
            index=pd.PeriodIndex(
                ["2026-01-02", "2026-04-27", "2026-04-28"],
                freq="D",
            ),
        )

        trend = chatter._compute_price_trend(hist)

        self.assertEqual(trend["latest_close"], (120.0, "2026-04-28", None))
        self.assertEqual(trend["return_ytd"][1], "2026-04-28")
        self.assertIsNone(trend["return_ytd"][2])
        self.assertAlmostEqual(trend["return_ytd"][0], 20.0)

    def test_fetch_financial_context_requests_enough_history_for_one_year_return(self):
        if CHATTER_IMPORT_ERROR is not None:
            self.skipTest(f"chatter import failed: {CHATTER_IMPORT_ERROR}")

        import pandas as pd
        from datetime import date

        start_dates: list[str] = []

        class _FakeToolkit:
            def __init__(self, tickers, api_key, start_date, end_date, quarterly=None):  # noqa: ARG002
                start_dates.append(start_date)
                self.tickers = tickers
                self.technicals = types.SimpleNamespace(
                    collect_all_indicators=lambda: pd.DataFrame()
                )

            def get_profile(self):
                return pd.DataFrame({"QCOM": {"Company Name": "QUALCOMM"}})

            def get_historical_data(self):
                return pd.DataFrame()

            def get_income_statement(self):
                return pd.DataFrame()

            def get_balance_sheet_statement(self):
                return pd.DataFrame()

            def get_cash_flow_statement(self):
                return pd.DataFrame()

        fake_financetoolkit = types.ModuleType("financetoolkit")
        fake_financetoolkit.Toolkit = _FakeToolkit

        with mock.patch.dict("os.environ", {"FMP_API_KEY": "unit-test-key"}, clear=False), \
             mock.patch.dict(sys.modules, {"financetoolkit": fake_financetoolkit}):
            context = chatter.fetch_financial_context(
                ticker="QCOM",
                date_start=None,
                date_end=None,
                include_technicals=False,
            )

        self.assertIn("FINANCIAL DATA [F]", context)
        self.assertGreaterEqual((date.today() - date.fromisoformat(start_dates[0])).days, 399)

    def test_fetch_financial_context_uses_ratio_api_for_debt_to_equity(self):
        if CHATTER_IMPORT_ERROR is not None:
            self.skipTest(f"chatter import failed: {CHATTER_IMPORT_ERROR}")

        import pandas as pd

        class _FakeRatios:
            def get_debt_to_equity_ratio(self, rounding=None):  # noqa: ARG002
                return pd.DataFrame({"2025": {"QCOM": 0.42}})

        class _FakeToolkit:
            def __init__(self, tickers, api_key, start_date, end_date, quarterly=None):  # noqa: ARG002
                self.tickers = tickers
                self.ratios = _FakeRatios()
                self.technicals = types.SimpleNamespace(
                    collect_all_indicators=lambda: pd.DataFrame()
                )

            def get_profile(self):
                return pd.DataFrame({"QCOM": {"Company Name": "QUALCOMM"}})

            def get_historical_data(self):
                return pd.DataFrame()

            def get_income_statement(self):
                return pd.DataFrame()

            def get_balance_sheet_statement(self):
                return pd.DataFrame()

            def get_cash_flow_statement(self):
                return pd.DataFrame()

        fake_financetoolkit = types.ModuleType("financetoolkit")
        fake_financetoolkit.Toolkit = _FakeToolkit

        with mock.patch.dict("os.environ", {"FMP_API_KEY": "unit-test-key"}, clear=False), \
             mock.patch.dict(sys.modules, {"financetoolkit": fake_financetoolkit}):
            context = chatter.fetch_financial_context(
                ticker="QCOM",
                date_start=None,
                date_end=None,
                include_technicals=False,
            )

        self.assertIn("debt_to_equity_ratio: ok", context)
        self.assertIn("Debt-to-Equity [2025]: 0.42", context)
        self.assertNotIn("Debt-to-Equity: unavailable", context)

    def test_fetch_financial_context_reads_ticker_row_multiindex_statements(self):
        if CHATTER_IMPORT_ERROR is not None:
            self.skipTest(f"chatter import failed: {CHATTER_IMPORT_ERROR}")

        import pandas as pd

        def _statement_frame(values: dict[str, list[float]]):
            index = pd.MultiIndex.from_product(
                [["QCOM"], list(values.keys())],
                names=["Ticker", "Metric"],
            )
            rows = [values[metric] for metric in values]
            return pd.DataFrame(rows, index=index, columns=["2024", "2025"])

        class _FakeRatios:
            def get_debt_to_equity_ratio(self, rounding=None):  # noqa: ARG002
                return pd.DataFrame()

        class _FakeToolkit:
            def __init__(self, tickers, api_key, start_date, end_date, quarterly=None):  # noqa: ARG002
                self.tickers = tickers
                self.ratios = _FakeRatios()
                self.technicals = types.SimpleNamespace(
                    collect_all_indicators=lambda: pd.DataFrame()
                )

            def get_profile(self):
                return pd.DataFrame(
                    {"QCOM": {"Company Name": "QUALCOMM", "Market Capitalization": 300_000_000_000.0}}
                )

            def get_historical_data(self):
                return pd.DataFrame()

            def get_income_statement(self):
                return _statement_frame(
                    {
                        "Revenue": [180_000_000_000.0, 200_000_000_000.0],
                        "Net Income": [18_000_000_000.0, 24_000_000_000.0],
                        "Gross Profit": [90_000_000_000.0, 110_000_000_000.0],
                        "Operating Income": [30_000_000_000.0, 36_000_000_000.0],
                    }
                )

            def get_balance_sheet_statement(self):
                return _statement_frame(
                    {
                        "Total Assets": [450_000_000_000.0, 500_000_000_000.0],
                        "Total Stockholders Equity": [200_000_000_000.0, 220_000_000_000.0],
                        "Total Current Assets": [100_000_000_000.0, 120_000_000_000.0],
                        "Total Current Liabilities": [50_000_000_000.0, 60_000_000_000.0],
                        "Cash and Cash Equivalents": [25_000_000_000.0, 30_000_000_000.0],
                        "Total Debt": [70_000_000_000.0, 80_000_000_000.0],
                    }
                )

            def get_cash_flow_statement(self):
                return _statement_frame(
                    {
                        "Operating Cash Flow": [28_000_000_000.0, 32_000_000_000.0],
                        "Capital Expenditure": [-8_000_000_000.0, -10_000_000_000.0],
                    }
                )

        fake_financetoolkit = types.ModuleType("financetoolkit")
        fake_financetoolkit.Toolkit = _FakeToolkit

        with mock.patch.dict("os.environ", {"FMP_API_KEY": "unit-test-key"}, clear=False), \
             mock.patch.dict(sys.modules, {"financetoolkit": fake_financetoolkit}):
            context = chatter.fetch_financial_context(
                ticker="QCOM",
                date_start=None,
                date_end=None,
                include_technicals=False,
            )

        self.assertIn("Revenue [2025]: $200.00B", context)
        self.assertIn("Net Income [2025]: $24.00B", context)
        self.assertIn("Total Assets [2025]: $500.00B", context)
        self.assertIn("Operating Cash Flow [2025]: $32.00B", context)
        self.assertIn("Revenue YoY Growth [2024→2025]: 11.1%", context)
        self.assertNotIn("\n  Revenue: unavailable", context)

    def test_fetch_financial_context_accepts_deduped_ticker_list_and_batches_toolkit(self):
        if CHATTER_IMPORT_ERROR is not None:
            self.skipTest(f"chatter import failed: {CHATTER_IMPORT_ERROR}")

        import pandas as pd

        init_calls: list[tuple[list[str], object]] = []

        class _FakeRatios:
            def get_debt_to_equity_ratio(self, rounding=None):  # noqa: ARG002
                return pd.DataFrame()

        class _FakeToolkit:
            def __init__(self, tickers, api_key, start_date, end_date, quarterly=None):  # noqa: ARG002
                init_calls.append((list(tickers), quarterly))
                self.ratios = _FakeRatios()
                self.technicals = types.SimpleNamespace(
                    collect_all_indicators=lambda: pd.DataFrame()
                )

            def get_profile(self):
                return pd.DataFrame(
                    {
                        "AAPL": {"Company Name": "Apple Inc."},
                        "MSFT": {"Company Name": "Microsoft Corporation"},
                    }
                )

            def get_historical_data(self):
                return pd.DataFrame()

            def get_income_statement(self):
                return pd.DataFrame()

            def get_balance_sheet_statement(self):
                return pd.DataFrame()

            def get_cash_flow_statement(self):
                return pd.DataFrame()

        fake_financetoolkit = types.ModuleType("financetoolkit")
        fake_financetoolkit.Toolkit = _FakeToolkit

        with mock.patch.dict("os.environ", {"FMP_API_KEY": "unit-test-key"}, clear=False), \
             mock.patch.dict(sys.modules, {"financetoolkit": fake_financetoolkit}):
            context = chatter.fetch_financial_context(
                ticker=["AAPL", "MSFT", "AAPL"],
                date_start=None,
                date_end=None,
                include_technicals=False,
            )

        self.assertIn("MULTI-TICKER FINANCIAL DATA [F]", context)
        self.assertIn("[AAPL]", context)
        self.assertIn("Company Name: Apple Inc.", context)
        self.assertIn("[MSFT]", context)
        self.assertIn("Company Name: Microsoft Corporation", context)
        self.assertTrue(init_calls)
        self.assertTrue(all(tickers == ["AAPL", "MSFT"] for tickers, _ in init_calls))

    def test_single_ticker_financial_route_skips_retrieval_and_uses_financial_context(self):
        if CHATTER_IMPORT_ERROR is not None:
            self.skipTest(f"chatter import failed: {CHATTER_IMPORT_ERROR}")

        query = "Is QCOM fundamentally sound based on its financial statements?"
        financial_block = (
            "FINANCIAL DATA [F] - cite any fact from this block as [F], NOT as [Sx] or [M]:\n"
            "2. COMPANY PROFILE\n"
            "  Company Name: QUALCOMM Incorporated\n"
            "(Source: FinancialModelingPrep via FinanceToolkit | Ticker: QCOM)\n"
            "[PRICE HISTORY | QCOM]\n"
            "  2026-04-24: 182.14"
        )
        news_block = (
            "NEWS DATA [N]\n"
            "Cite facts from this block as [N1], [N2], etc. Do not cite as [Sx], [M], or [F].\n"
            "Source: Google News RSS | Query: QUALCOMM Incorporated | URL: https://news.google.com/rss/search?q=QUALCOMM+Incorporated\n"
            "\n"
            "[N1] Title: Qualcomm posts strong earnings"
        )

        with mock.patch.object(chatter, "fetch_financial_context", return_value=financial_block) as fetch_mock, \
             mock.patch.object(
                 chatter,
                 "fetch_single_ticker_news_context",
                 return_value=(news_block, "QUALCOMM Incorporated", 1),
             ) as fetch_news_mock, \
             mock.patch.object(chatter, "retrieve", side_effect=AssertionError("retrieve() must not be called")), \
             mock.patch.object(chatter, "decompose_query", side_effect=AssertionError("decompose_query() must not be called")), \
             mock.patch.object(chatter, "DEBUG_SKIP_GENERATION", False), \
             mock.patch.object(
                 chatter,
                 "generate_answer",
                 return_value=(
                    "Answer: QCOM has usable reported fundamentals and adequate liquidity [F]\n"
                    "Fundamental Assessment: Mixed\n"
                    "Fundamental Score (0-10): 6\n"
                    "Score Rationale: The score is near 5 because the available [F] data is usable but not enough to establish a clearly sound profile [F]\n"
                     "Key Fundamental Drivers:\n"
                     "- Reported fundamentals are available for QCOM [F]\n"
                     "- Liquidity is described as adequate in the financial data [F]\n"
                     "- Valuation evidence is limited in the supplied block [F]\n"
                     "Risks / Gaps:\n"
                     "- Price history is secondary and does not drive the fundamental score [F]\n"
                     "- Recent news is a caveat rather than a fundamental score input [N1]"
                 ),
             ) as generate_mock:
            result = chatter.run_query_once(
                query=query,
                embed_model=object(),
                reranker=None,
                gen_client=object(),
                driver=None,
                sqlite_conn=None,
                alias_to_ticker={},
                ticker_to_canonical={"QCOM": "QUALCOMM"},
                alias_to_fin_entity={},
                base_system_prompt="base",
                base_causal_system_prompt="causal",
                base_daily_summary_prompt="daily",
                base_single_ticker_financial_prompt="financial",
                memory=chatter.ConversationMemory(),
                skip_generation=False,
            )

        self.assertEqual(result["route_type"], "single_ticker_financial")
        self.assertEqual(result["decision"], "answer")
        self.assertEqual(result["chunks"], [])
        self.assertIn("[F]", result["answer"])
        self.assertTrue(result["retrieval_trace"]["finance_context_present"])
        self.assertTrue(result["retrieval_trace"]["news_context_present"])
        self.assertEqual(result["retrieval_trace"]["news_query"], "QUALCOMM Incorporated")
        self.assertEqual(result["retrieval_trace"]["news_item_count"], 1)
        self.assertEqual(result["retrieval_trace"]["selected_chunk_count"], 0)
        self.assertIn("Fundamental Assessment:", result["answer"])
        self.assertIn("Fundamental Score (0-10):", result["answer"])
        self.assertNotIn("Outlook:", result["answer"])
        self.assertNotIn("Outlook Confidence (0-1):", result["answer"])
        fetch_mock.assert_called_once()
        fetch_news_mock.assert_called_once()
        self.assertTrue(fetch_mock.call_args.kwargs.get("include_technicals"))
        generate_mock.assert_called_once()
        _, context_arg, _, _ = generate_mock.call_args.args
        self.assertIn("FINANCIAL DATA [F]", context_arg)
        self.assertIn("NEWS DATA [N]", context_arg)
        self.assertIn("TARGET TICKER : QCOM", context_arg)

    def test_single_ticker_financial_route_fails_closed_without_financial_data(self):
        if CHATTER_IMPORT_ERROR is not None:
            self.skipTest(f"chatter import failed: {CHATTER_IMPORT_ERROR}")

        query = "Is SPAX.PVT fundamentally sound based on its financial statements?"
        with mock.patch.object(chatter, "fetch_financial_context", return_value="") as fetch_mock, \
             mock.patch.object(chatter, "fetch_single_ticker_news_context") as fetch_news_mock, \
             mock.patch.object(chatter, "retrieve", side_effect=AssertionError("retrieve() must not be called")):
            result = chatter.run_query_once(
                query=query,
                embed_model=object(),
                reranker=None,
                gen_client=object(),
                driver=None,
                sqlite_conn=None,
                alias_to_ticker={},
                ticker_to_canonical={},
                alias_to_fin_entity={},
                base_system_prompt="base",
                base_causal_system_prompt="causal",
                base_daily_summary_prompt="daily",
                base_single_ticker_financial_prompt="financial",
                memory=chatter.ConversationMemory(),
                skip_generation=False,
            )

        self.assertEqual(result["route_type"], "single_ticker_financial")
        self.assertEqual(result["decision"], "abstain")
        self.assertEqual(result["chunks"], [])
        self.assertFalse(result["retrieval_trace"]["finance_context_present"])
        self.assertFalse(result["retrieval_trace"]["news_context_present"])
        self.assertEqual(result["retrieval_trace"]["news_item_count"], 0)
        self.assertEqual(result["retrieval_trace"]["selected_chunk_count"], 0)
        self.assertIn("Insufficient financial data [F]", result["answer"])
        self.assertIn("Fundamental Assessment: Mixed", result["answer"])
        self.assertIn("Fundamental Score (0-10): 5", result["answer"])
        self.assertNotIn("Outlook:", result["answer"])
        fetch_mock.assert_called_once()
        fetch_news_mock.assert_not_called()

    def test_single_ticker_fundamental_enforcement_inserts_and_normalizes_score(self):
        if CHATTER_IMPORT_ERROR is not None:
            self.skipTest(f"chatter import failed: {CHATTER_IMPORT_ERROR}")

        answer = (
            "Answer: Fundamentals are supported by positive free cash flow and strong liquidity [F]\n"
            "Fundamental Assessment: Unsound\n"
            "Fundamental Score (0-10): 14\n"
            "8\n"
            "Score Rationale: The score reflects positive free cash flow and strong liquidity [F]\n"
            "Key Fundamental Drivers:\n"
            "- Positive free cash flow supports the profile [F]\n"
            "- Strong liquidity supports the profile [F]\n"
            "- Moderate leverage supports the profile [F]\n"
            "Risks / Gaps:\n"
            "- Price trend is secondary and cannot determine the score [F]\n"
            "- Missing valuation detail limits confidence [F]"
        )

        enforced = chatter._remove_orphan_single_ticker_value_lines(
            chatter._enforce_single_ticker_fundamental_sections(answer)
        )

        self.assertIn("Fundamental Assessment: Sound", enforced)
        self.assertIn("Fundamental Score (0-10): 10", enforced)
        self.assertNotIn("\n8\n", enforced)
        self.assertEqual(chatter._validate_single_ticker_output(enforced), [])

    def test_single_ticker_fundamental_enforcement_converts_legacy_outlook_lines(self):
        if CHATTER_IMPORT_ERROR is not None:
            self.skipTest(f"chatter import failed: {CHATTER_IMPORT_ERROR}")

        answer = (
            "Answer: Fundamentals are mixed with missing metrics [F]\n"
            "Outlook: Neutral\n"
            "Outlook Confidence (0-1): 0.58\n"
            "Score Rationale: The score is close to 5 because available fundamentals are mixed [F]\n"
            "Key Fundamental Drivers:\n"
            "- Revenue is available in the financial data [F]\n"
            "- Cash flow is available in the financial data [F]\n"
            "- Liquidity is available in the financial data [F]\n"
            "Risks / Gaps:\n"
            "- Some fundamental metrics are missing [F]\n"
            "- Price trend is secondary to the score [F]"
        )

        enforced = chatter._enforce_single_ticker_fundamental_sections(answer)

        self.assertIn("Fundamental Assessment:", enforced)
        self.assertIn("Fundamental Score (0-10): 6", enforced)
        self.assertNotIn("Outlook:", enforced)
        self.assertNotIn("Outlook Confidence (0-1):", enforced)
        self.assertEqual(chatter._validate_single_ticker_output(enforced), [])

    def test_market_tickers_today_route_writes_filtered_analyst_context_to_file(self):
        if CHATTER_IMPORT_ERROR is not None:
            self.skipTest(f"chatter import failed: {CHATTER_IMPORT_ERROR}")

        screened_tickers = ["AAPL", "MSFT", "TSLA", "NVDA"]
        expected_analyst_path = str(chatter.QUERY_CONTEXT_DIR / "query_analyst_context.json")

        def _grades_consensus(symbol):
            return {
                "AAPL": [{"symbol": "AAPL", "consensus": "Buy"}],
                "MSFT": [{"symbol": "MSFT", "consensus": " Strong Buy "}],
                "TSLA": [{"symbol": "TSLA", "consensus": "Hold"}],
                "NVDA": [{"symbol": "NVDA", "consensus": "Strong Sell"}],
            }[symbol]

        def _endpoint_payload(endpoint):
            return lambda symbol, **kwargs: [{"symbol": symbol, "endpoint": endpoint}]

        with mock.patch.object(chatter, "fetch_today_market_screen_tickers", return_value=screened_tickers), \
             mock.patch.object(chatter, "get_grades_consensus", side_effect=_grades_consensus) as consensus_mock, \
             mock.patch.object(chatter, "_acquire_financial_data_requests"), \
             mock.patch.object(chatter, "get_price_target_consensus", side_effect=_endpoint_payload("price_target_consensus")), \
             mock.patch.object(chatter, "get_price_target_summary", side_effect=_endpoint_payload("price_target_summary")), \
             mock.patch.object(chatter, "get_ratings_snapshot", side_effect=_endpoint_payload("ratings_snapshot")), \
             mock.patch.object(chatter, "_dump_analyst_context_only", return_value=expected_analyst_path) as dump_mock, \
             mock.patch.object(chatter, "fetch_financial_context") as fetch_mock, \
             mock.patch.object(chatter, "fetch_single_ticker_news_context", side_effect=AssertionError("news fetch must not be called")), \
             mock.patch.object(chatter, "retrieve", side_effect=AssertionError("retrieve() must not be called")), \
             mock.patch.object(chatter, "decompose_query", side_effect=AssertionError("decompose_query() must not be called")), \
             mock.patch.object(chatter, "generate_answer", side_effect=AssertionError("generate_answer() must not be called")):
            result = chatter.run_query_once(
                query="What are the market tickers for today?",
                embed_model=object(),
                reranker=None,
                gen_client=object(),
                driver=None,
                sqlite_conn=None,
                alias_to_ticker={},
                ticker_to_canonical={},
                alias_to_fin_entity={},
                base_system_prompt="base",
                base_causal_system_prompt="causal",
                base_daily_summary_prompt="daily",
                base_single_ticker_financial_prompt="financial",
                memory=chatter.ConversationMemory(),
                skip_generation=False,
            )

        self.assertEqual(result["route_type"], "market_tickers_today")
        self.assertIn("Analyst context written to query_analyst_context.json", result["answer"])
        self.assertIn(expected_analyst_path, result["answer"])
        self.assertEqual(result["decision"], "answer")
        self.assertFalse(result["retrieval_trace"]["finance_context_present"])
        self.assertTrue(result["retrieval_trace"]["analyst_context_present"])
        self.assertEqual(result["retrieval_trace"]["analyst_context_path"], expected_analyst_path)
        self.assertEqual(result["retrieval_trace"]["screened_ticker_count"], 4)
        self.assertEqual(result["retrieval_trace"]["filtered_ticker_count"], 2)
        self.assertEqual(result["retrieval_trace"]["filtered_out_count"], 2)
        self.assertEqual(result["retrieval_trace"]["tickers"], ["AAPL", "MSFT"])
        dump_mock.assert_called_once()
        analyst_context = dump_mock.call_args.kwargs["analyst_context"]
        self.assertEqual(
            [item["symbol"] for item in analyst_context["tickers"]],
            ["AAPL", "MSFT"],
        )
        self.assertEqual(analyst_context["ticker_counts"]["screened"], 4)
        self.assertEqual(analyst_context["ticker_counts"]["passed_consensus_filter"], 2)
        self.assertEqual(analyst_context["ticker_counts"]["failed_or_filtered"], 2)
        self.assertEqual(
            [item["symbol"] for item in analyst_context["filtered_out"]],
            ["TSLA", "NVDA"],
        )
        self.assertEqual(
            analyst_context["tickers"][0]["grades_consensus"],
            [{"symbol": "AAPL", "consensus": "Buy"}],
        )
        self.assertNotIn("grades", analyst_context["tickers"][0])
        self.assertNotIn("grades_historical", analyst_context["tickers"][0])
        self.assertNotIn("ratings_historical", analyst_context["tickers"][0])
        self.assertNotIn("analyst_estimates_window", analyst_context)
        self.assertNotIn("analyst_estimates", analyst_context["tickers"][0])
        consensus_mock.assert_has_calls([mock.call(symbol) for symbol in screened_tickers], any_order=True)
        fetch_mock.assert_not_called()

    def test_dump_analyst_context_writes_json_grouped_by_ticker_without_filtered_out(self):
        if CHATTER_IMPORT_ERROR is not None:
            self.skipTest(f"chatter import failed: {CHATTER_IMPORT_ERROR}")

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = chatter._dump_analyst_context_only(
                base_dir=tmp_dir,
                analyst_context={
                    "query": "market tickers today",
                    "route": "market_tickers_today",
                    "filtered_out": [{"symbol": "TSLA", "reason": "consensus_not_buy"}],
                    "ticker_counts": {
                        "screened": 2,
                        "passed_consensus_filter": 1,
                        "failed_or_filtered": 1,
                    },
                    "tickers": [
                        {
                            "symbol": "NVDA",
                            "grades": [{"analyst": "legacy"}],
                            "grades_historical": [{"analyst": "legacy"}],
                            "grades_consensus": [{"consensus": "Buy"}],
                            "ratings_historical": [{"analyst": "legacy"}],
                            "analyst_estimates": [{"revenueAvg": 1}],
                        }
                    ],
                },
            )

            with open(path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)

        self.assertTrue(path.endswith("query_analyst_context.json"))
        self.assertNotIn("filtered_out", payload)
        self.assertEqual(payload["ticker_order"], ["NVDA"])
        self.assertIn("NVDA", payload["tickers"])
        self.assertNotIn("grades", payload["tickers"]["NVDA"])
        self.assertNotIn("grades_historical", payload["tickers"]["NVDA"])
        self.assertNotIn("ratings_historical", payload["tickers"]["NVDA"])
        self.assertEqual(payload["tickers"]["NVDA"]["analyst_estimates"], [{"revenueAvg": 1}])

    def test_market_tickers_today_route_fails_closed_without_screen_tickers(self):
        if CHATTER_IMPORT_ERROR is not None:
            self.skipTest(f"chatter import failed: {CHATTER_IMPORT_ERROR}")

        with mock.patch.object(chatter, "fetch_today_market_screen_tickers", return_value=[]), \
             mock.patch.object(chatter, "fetch_financial_context") as fetch_mock, \
             mock.patch.object(chatter, "get_grades_consensus") as consensus_mock, \
             mock.patch.object(chatter, "decompose_query", side_effect=AssertionError("decompose_query() must not be called")), \
             mock.patch.object(chatter, "retrieve", side_effect=AssertionError("retrieve() must not be called")), \
             mock.patch.object(chatter, "generate_answer", side_effect=AssertionError("generate_answer() must not be called")):
            result = chatter.run_query_once(
                query="today's market tickers",
                embed_model=object(),
                reranker=None,
                gen_client=object(),
                driver=None,
                sqlite_conn=None,
                alias_to_ticker={},
                ticker_to_canonical={},
                alias_to_fin_entity={},
                base_system_prompt="base",
                base_causal_system_prompt="causal",
                base_daily_summary_prompt="daily",
                base_single_ticker_financial_prompt="financial",
                memory=chatter.ConversationMemory(),
                skip_generation=False,
            )

        self.assertEqual(result["route_type"], "market_tickers_today")
        self.assertEqual(result["decision"], "abstain")
        self.assertFalse(result["retrieval_trace"]["finance_context_present"])
        self.assertFalse(result["retrieval_trace"]["analyst_context_present"])
        self.assertEqual(result["retrieval_trace"]["screened_ticker_count"], 0)
        self.assertIn("No market tickers were retrieved", result["answer"])
        fetch_mock.assert_not_called()
        consensus_mock.assert_not_called()

    def test_market_tickers_today_route_fails_closed_without_buy_consensus(self):
        if CHATTER_IMPORT_ERROR is not None:
            self.skipTest(f"chatter import failed: {CHATTER_IMPORT_ERROR}")

        def _grades_consensus(symbol):
            return {
                "TSLA": [{"symbol": "TSLA", "consensus": "Hold"}],
                "NVDA": [{"symbol": "NVDA", "consensus": "Sell"}],
            }[symbol]

        with mock.patch.object(chatter, "fetch_today_market_screen_tickers", return_value=["TSLA", "NVDA"]), \
             mock.patch.object(chatter, "get_grades_consensus", side_effect=_grades_consensus) as consensus_mock, \
             mock.patch.object(chatter, "_acquire_financial_data_requests"), \
             mock.patch.object(chatter, "_dump_analyst_context_only", side_effect=AssertionError("dump must not be called")), \
             mock.patch.object(chatter, "fetch_financial_context") as fetch_mock, \
             mock.patch.object(chatter, "decompose_query", side_effect=AssertionError("decompose_query() must not be called")), \
             mock.patch.object(chatter, "retrieve", side_effect=AssertionError("retrieve() must not be called")), \
             mock.patch.object(chatter, "generate_answer", side_effect=AssertionError("generate_answer() must not be called")):
            result = chatter.run_query_once(
                query="today's market tickers",
                embed_model=object(),
                reranker=None,
                gen_client=object(),
                driver=None,
                sqlite_conn=None,
                alias_to_ticker={},
                ticker_to_canonical={},
                alias_to_fin_entity={},
                base_system_prompt="base",
                base_causal_system_prompt="causal",
                base_daily_summary_prompt="daily",
                base_single_ticker_financial_prompt="financial",
                memory=chatter.ConversationMemory(),
                skip_generation=False,
            )

        self.assertEqual(result["route_type"], "market_tickers_today")
        self.assertEqual(result["decision"], "abstain")
        self.assertFalse(result["retrieval_trace"]["analyst_context_present"])
        self.assertEqual(result["retrieval_trace"]["screened_ticker_count"], 2)
        self.assertEqual(result["retrieval_trace"]["filtered_ticker_count"], 0)
        self.assertEqual(result["retrieval_trace"]["filtered_out_count"], 2)
        self.assertEqual(result["retrieval_trace"]["tickers"], [])
        self.assertIn("No market tickers passed the analyst consensus filter", result["answer"])
        consensus_mock.assert_has_calls([mock.call("TSLA"), mock.call("NVDA")], any_order=True)
        fetch_mock.assert_not_called()

    def test_fetch_today_market_screen_tickers_parallel_merges_in_screener_order(self):
        if CHATTER_IMPORT_ERROR is not None:
            self.skipTest(f"chatter import failed: {CHATTER_IMPORT_ERROR}")

        import pandas as pd

        def _fetch_screener(screener, count=100):
            if screener == "first":
                time.sleep(0.02)
                return pd.DataFrame({"symbol": ["AAA", "BBB", "CCC"]})
            return pd.DataFrame({"symbol": ["CCC", "DDD"]})

        with mock.patch.object(chatter, "SCREENERS", ["first", "second"]), \
             mock.patch.object(chatter, "MARKET_TICKERS_SCREENER_WORKERS", 2), \
             mock.patch.object(chatter, "MARKET_TICKERS_MAX_SCREENED", 0), \
             mock.patch.object(chatter, "fetch_yahoo_screener", side_effect=_fetch_screener):
            self.assertEqual(
                chatter.fetch_today_market_screen_tickers(count=100),
                ["AAA", "BBB", "CCC", "DDD"],
            )

    def test_build_market_ticker_analyst_context_preserves_order_with_parallel_calls(self):
        if CHATTER_IMPORT_ERROR is not None:
            self.skipTest(f"chatter import failed: {CHATTER_IMPORT_ERROR}")

        def _grades_consensus(symbol):
            if symbol == "AAPL":
                time.sleep(0.02)
            return {
                "AAPL": [{"symbol": "AAPL", "consensus": "Buy"}],
                "MSFT": [{"symbol": "MSFT", "consensus": "Strong Buy"}],
                "TSLA": [{"symbol": "TSLA", "consensus": "Hold"}],
                "NVDA": [{"symbol": "NVDA", "consensus": "Buy"}],
            }[symbol]

        def _endpoint_payload(endpoint):
            return lambda symbol, **kwargs: [{"symbol": symbol, "endpoint": endpoint}]

        with mock.patch.object(chatter, "MARKET_TICKERS_ANALYST_WORKERS", 4), \
             mock.patch.object(chatter, "_acquire_financial_data_requests"), \
             mock.patch.object(chatter, "get_grades_consensus", side_effect=_grades_consensus), \
             mock.patch.object(chatter, "get_price_target_consensus", side_effect=_endpoint_payload("price_target_consensus")), \
             mock.patch.object(chatter, "get_price_target_summary", side_effect=_endpoint_payload("price_target_summary")), \
             mock.patch.object(chatter, "get_ratings_snapshot", side_effect=_endpoint_payload("ratings_snapshot")):
            context = chatter.build_market_ticker_analyst_context(
                query="market tickers today",
                tickers=["AAPL", "MSFT", "TSLA", "NVDA"],
                timestamp="20260511_120000",
            )

        self.assertEqual([item["symbol"] for item in context["tickers"]], ["AAPL", "MSFT", "NVDA"])
        self.assertEqual([item["symbol"] for item in context["filtered_out"]], ["TSLA"])
        self.assertNotIn("analyst_estimates", context["tickers"][0])

    def test_build_market_ticker_analyst_context_uses_shared_limiter_for_fmp_calls(self):
        if CHATTER_IMPORT_ERROR is not None:
            self.skipTest(f"chatter import failed: {CHATTER_IMPORT_ERROR}")

        def _grades_consensus(symbol):
            return {
                "AAPL": [{"symbol": "AAPL", "consensus": "Buy"}],
                "TSLA": [{"symbol": "TSLA", "consensus": "Hold"}],
            }[symbol]

        def _endpoint_payload(endpoint):
            return lambda symbol, **kwargs: [{"symbol": symbol, "endpoint": endpoint}]

        with mock.patch.object(chatter, "MARKET_TICKERS_ANALYST_WORKERS", 1), \
             mock.patch.object(chatter, "_acquire_financial_data_requests") as acquire_mock, \
             mock.patch.object(chatter, "get_grades_consensus", side_effect=_grades_consensus), \
             mock.patch.object(chatter, "get_price_target_consensus", side_effect=_endpoint_payload("price_target_consensus")), \
             mock.patch.object(chatter, "get_price_target_summary", side_effect=_endpoint_payload("price_target_summary")), \
             mock.patch.object(chatter, "get_ratings_snapshot", side_effect=_endpoint_payload("ratings_snapshot")):
            context = chatter.build_market_ticker_analyst_context(
                query="market tickers today",
                tickers=["AAPL", "TSLA"],
                timestamp="20260511_120000",
            )

        self.assertEqual([item["symbol"] for item in context["tickers"]], ["AAPL"])
        self.assertEqual(acquire_mock.call_count, 5)

    def test_extract_company_name_from_financial_context(self):
        if CHATTER_IMPORT_ERROR is not None:
            self.skipTest(f"chatter import failed: {CHATTER_IMPORT_ERROR}")

        block = (
            "FINANCIAL DATA [F]\n"
            "2. COMPANY PROFILE\n"
            "  Company Name: Nokia Oyj\n"
            "  Sector: Technology\n"
        )
        self.assertEqual(
            chatter._extract_company_name_from_financial_context(block),
            "Nokia Oyj",
        )

    def test_fetch_single_ticker_news_context_builds_gdelt_query_from_company_name(self):
        if CHATTER_IMPORT_ERROR is not None:
            self.skipTest(f"chatter import failed: {CHATTER_IMPORT_ERROR}")

        captured: dict[str, object] = {}

        class _Resp:
            url = "https://api.gdeltproject.org/api/v2/doc/doc?query=%22Nokia%20Oyj%22"

            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "articles": [
                        {
                            "title": "Nokia signs new 5G deal",
                            "domain": "example.com",
                            "seendate": "20260504T100000Z",
                            "url": "https://example.com/nokia",
                        }
                    ]
                }

        def _fake_get(url, params, timeout):
            captured["url"] = url
            captured["params"] = params
            captured["timeout"] = timeout
            return _Resp()

        fake_requests = types.SimpleNamespace(get=_fake_get)
        with mock.patch.dict(sys.modules, {"requests": fake_requests}):
            block, query, item_count = chatter.fetch_single_ticker_news_context(
                "Nokia Oyj",
                max_items=1,
            )

        self.assertEqual(captured.get("url"), chatter.GDELT_DOC_SEARCH_URL)
        params = captured.get("params")
        self.assertIsInstance(params, dict)
        self.assertIn('"Nokia Oyj"', str(params.get("query")))
        self.assertIn("sourcelang:english", str(params.get("query")))
        self.assertEqual(params.get("mode"), "artlist")
        self.assertEqual(params.get("format"), "json")
        self.assertEqual(params.get("sort"), "datedesc")
        self.assertEqual(params.get("timespan"), "1month")
        self.assertEqual(params.get("maxrecords"), 1)
        self.assertEqual(query, "Nokia Oyj")
        self.assertEqual(item_count, 1)
        self.assertIn("NEWS DATA [N]", block)
        self.assertIn("[N1] Title:", block)
        self.assertIn("Publisher: example.com", block)
        self.assertIn("Published: 20260504T100000Z", block)
        self.assertIn("Link: https://example.com/nokia", block)

    def test_fetch_single_ticker_news_context_caps_gdelt_articles_at_50(self):
        if CHATTER_IMPORT_ERROR is not None:
            self.skipTest(f"chatter import failed: {CHATTER_IMPORT_ERROR}")

        captured: dict[str, object] = {}

        class _Resp:
            url = "https://api.gdeltproject.org/api/v2/doc/doc?maxrecords=50"

            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "articles": [
                        {
                            "title": f"Nokia article {idx}",
                            "domain": f"publisher{idx}.com",
                            "seendate": "20260504T100000Z",
                            "url": f"https://example.com/nokia/{idx}",
                        }
                        for idx in range(60)
                    ]
                }

        def _fake_get(url, params, timeout):
            captured["params"] = params
            return _Resp()

        fake_requests = types.SimpleNamespace(get=_fake_get)
        with mock.patch.dict(sys.modules, {"requests": fake_requests}):
            block, _query, item_count = chatter.fetch_single_ticker_news_context(
                "Nokia Oyj",
                max_items=75,
            )

        params = captured.get("params")
        self.assertIsInstance(params, dict)
        self.assertEqual(params.get("maxrecords"), 50)
        self.assertEqual(item_count, 50)
        self.assertIn("[N50] Title: Nokia article 49", block)
        self.assertNotIn("[N51] Title:", block)

    def test_single_ticker_financial_route_answers_when_news_fetch_fails_softly(self):
        if CHATTER_IMPORT_ERROR is not None:
            self.skipTest(f"chatter import failed: {CHATTER_IMPORT_ERROR}")

        query = "Is QCOM fundamentally sound based on its financial statements?"
        financial_block = (
            "FINANCIAL DATA [F] - cite any fact from this block as [F], NOT as [Sx] or [M]:\n"
            "2. COMPANY PROFILE\n"
            "  Company Name: QUALCOMM Incorporated\n"
            "(Source: FinancialModelingPrep via FinanceToolkit | Ticker: QCOM)\n"
            "[PRICE HISTORY | QCOM]\n"
            "  2026-04-24: 182.14"
        )

        with mock.patch.object(chatter, "fetch_financial_context", return_value=financial_block), \
             mock.patch.object(
                 chatter,
                 "fetch_single_ticker_news_context",
                 return_value=("", "QUALCOMM Incorporated", 0),
             ), \
             mock.patch.object(chatter, "retrieve", side_effect=AssertionError("retrieve() must not be called")), \
             mock.patch.object(chatter, "decompose_query", side_effect=AssertionError("decompose_query() must not be called")), \
             mock.patch.object(chatter, "DEBUG_SKIP_GENERATION", False), \
             mock.patch.object(
                 chatter,
                 "generate_answer",
                 return_value=(
                    "Answer: QCOM has usable reported fundamentals but not enough evidence for a clearly sound profile [F]\n"
                    "Fundamental Assessment: Mixed\n"
                    "Fundamental Score (0-10): 6\n"
                    "Score Rationale: The score is near 5 because the supplied financial data is usable but limited [F]\n"
                     "Key Fundamental Drivers:\n"
                     "- Reported fundamentals are available for QCOM [F]\n"
                     "- Company profile data identifies the target [F]\n"
                     "- Price history is present but secondary to fundamentals [F]\n"
                     "Risks / Gaps:\n"
                     "- News context is unavailable and should only be treated as a caveat [F]\n"
                     "- The supplied block does not provide enough fundamental detail for a sound score [F]"
                 ),
             ):
            result = chatter.run_query_once(
                query=query,
                embed_model=object(),
                reranker=None,
                gen_client=object(),
                driver=None,
                sqlite_conn=None,
                alias_to_ticker={},
                ticker_to_canonical={"QCOM": "QUALCOMM"},
                alias_to_fin_entity={},
                base_system_prompt="base",
                base_causal_system_prompt="causal",
                base_daily_summary_prompt="daily",
                base_single_ticker_financial_prompt="financial",
                memory=chatter.ConversationMemory(),
                skip_generation=False,
            )

        self.assertEqual(result["route_type"], "single_ticker_financial")
        self.assertEqual(result["decision"], "answer")
        self.assertTrue(result["retrieval_trace"]["finance_context_present"])
        self.assertFalse(result["retrieval_trace"]["news_context_present"])
        self.assertEqual(result["retrieval_trace"]["news_item_count"], 0)
        self.assertIn("Fundamental Assessment:", result["answer"])
        self.assertIn("Fundamental Score (0-10):", result["answer"])
        self.assertNotIn("Outlook:", result["answer"])

    def test_single_ticker_prompt_template_allows_news_citations_and_forbids_sx(self):
        if CHATTER_IMPORT_ERROR is not None:
            self.skipTest(f"chatter import failed: {CHATTER_IMPORT_ERROR}")

        raw_template = chatter.SINGLE_TICKER_FINANCIAL_PROMPT_TEMPLATE
        template = "\n".join(raw_template) if isinstance(raw_template, list) else str(raw_template)
        self.assertIn("NEWS DATA [N]", template)
        self.assertIn("[N1]/[N2]/...", template)
        self.assertIn("Do not use outside knowledge", template)
        self.assertIn("[Sx] citations", template)

    def test_fetch_market_context_delegates_to_financial_context_for_mapped_hops(self):
        if CHATTER_IMPORT_ERROR is not None:
            self.skipTest(f"chatter import failed: {CHATTER_IMPORT_ERROR}")

        def _fake_financial_context(**kwargs):
            return (
                "FINANCIAL DATA [F]\n"
                f"Source: FinancialModelingPrep via FinanceToolkit | Ticker: {kwargs['ticker']}\n"
                "11. TECHNICAL INDICATORS\n"
                "  RSI: 52.3"
            )

        with mock.patch.object(
            chatter,
            "fetch_financial_context",
            side_effect=_fake_financial_context,
        ) as fetch_mock:
            context = chatter.fetch_market_context(
                hops=["oil", "brent", "oil", "not-a-market-hop"],
                date_start="2026-01-01",
                date_end="2026-04-28",
                lookback_days=7,
                query="How could oil affect USD?",
            )

        self.assertIn("QUERY: How could oil affect USD?", context)
        self.assertIn("SUPPLEMENTAL FINANCIAL DATA [F]", context)
        self.assertIn("FINANCIAL DATA [F]", context)
        self.assertIn("[OIL | CL=F]", context)
        self.assertIn("[BRENT | BZ=F]", context)
        self.assertNotIn("MARKET DATA [M]", context)
        self.assertNotIn("Use only the FINANCIAL DATA [F] block below.", context)
        self.assertEqual([call.kwargs["ticker"] for call in fetch_mock.call_args_list], ["CL=F", "BZ=F"])
        for call in fetch_mock.call_args_list:
            self.assertTrue(call.kwargs["include_technicals"])
            self.assertEqual(call.kwargs["lookback_days"], 365)

    def test_fetch_market_context_returns_empty_for_unmapped_or_empty_financial_data(self):
        if CHATTER_IMPORT_ERROR is not None:
            self.skipTest(f"chatter import failed: {CHATTER_IMPORT_ERROR}")

        with mock.patch.object(chatter, "fetch_financial_context") as fetch_mock:
            self.assertEqual(
                chatter.fetch_market_context(
                    hops=["not-a-market-hop"],
                    date_start=None,
                    date_end=None,
                ),
                "",
            )
            fetch_mock.assert_not_called()

        with mock.patch.object(chatter, "fetch_financial_context", return_value="") as fetch_mock:
            self.assertEqual(
                chatter.fetch_market_context(
                    hops=["oil"],
                    date_start=None,
                    date_end=None,
                ),
                "",
            )
            fetch_mock.assert_called_once()

    def test_fetch_yahoo_screener_returns_empty_when_yfinance_missing(self):
        if CHATTER_IMPORT_ERROR is not None:
            self.skipTest(f"chatter import failed: {CHATTER_IMPORT_ERROR}")

        with mock.patch.dict(sys.modules, {"yfinance": None}):
            df = chatter.fetch_yahoo_screener("most_actives", count=100)

        self.assertTrue(hasattr(df, "empty"))
        self.assertTrue(df.empty)

    def test_fetch_financial_context_includes_numbered_technicals_section_when_enabled(self):
        if CHATTER_IMPORT_ERROR is not None:
            self.skipTest(f"chatter import failed: {CHATTER_IMPORT_ERROR}")

        class _FakeTable:
            def __init__(self, empty: bool):
                self.empty = empty

        class _FakeTechnicals:
            def collect_all_indicators(self):
                return _FakeTable(empty=False)

        class _FakeToolkit:
            def __init__(self, tickers, api_key, start_date, end_date):  # noqa: ARG002
                self.tickers = tickers
                self.technicals = _FakeTechnicals()
                self.ratios = types.SimpleNamespace(
                    collect_profitability_ratios=lambda: _FakeTable(empty=True),
                    collect_liquidity_ratios=lambda: _FakeTable(empty=True),
                    collect_valuation_ratios=lambda: _FakeTable(empty=True),
                )

            def get_profile(self):
                return _FakeTable(empty=True)

            def get_historical_data(self):
                return _FakeTable(empty=True)

            def get_income_statement(self):
                return _FakeTable(empty=True)

            def get_balance_sheet_statement(self):
                return _FakeTable(empty=True)

            def get_cash_flow_statement(self):
                return _FakeTable(empty=True)

        fake_financetoolkit = types.ModuleType("financetoolkit")
        fake_financetoolkit.Toolkit = _FakeToolkit
        fake_pandas = types.ModuleType("pandas")
        fake_pandas.notna = lambda value: value is not None

        with mock.patch.dict("os.environ", {"FMP_API_KEY": "unit-test-key"}, clear=False), \
             mock.patch.dict(sys.modules, {"financetoolkit": fake_financetoolkit, "pandas": fake_pandas}):
            with_technicals = chatter.fetch_financial_context(
                ticker="QCOM",
                date_start=None,
                date_end=None,
                include_technicals=True,
            )
            without_technicals = chatter.fetch_financial_context(
                ticker="QCOM",
                date_start=None,
                date_end=None,
                include_technicals=False,
            )

        self.assertIn("11. TECHNICAL INDICATORS", with_technicals)
        self.assertIn("RSI: unavailable", with_technicals)
        self.assertIn("indicators: ok", with_technicals)
        self.assertEqual(without_technicals, "")

    def test_fetch_financial_context_includes_full_company_description(self):
        if CHATTER_IMPORT_ERROR is not None:
            self.skipTest(f"chatter import failed: {CHATTER_IMPORT_ERROR}")

        try:
            import pandas as pd
        except ImportError as exc:
            self.skipTest(f"pandas unavailable: {exc}")

        long_description = (
            "Fastly, Inc. operates an edge cloud platform for processing, serving, "
            "and securing customer applications. "
            + "The platform helps developers deliver digital experiences at the edge. " * 8
            + "This final sentence verifies the description was not truncated."
        )

        class _FakeToolkit:
            def __init__(self, tickers, api_key, start_date, end_date, quarterly=None):  # noqa: ARG002
                self.tickers = tickers
                self.technicals = types.SimpleNamespace(
                    collect_all_indicators=lambda: pd.DataFrame()
                )

            def get_profile(self):
                return pd.DataFrame(
                    {"FSLY": {
                        "Company Name": "Fastly, Inc.",
                        "Description": long_description,
                    }}
                )

            def get_historical_data(self):
                return pd.DataFrame()

            def get_income_statement(self):
                return pd.DataFrame()

            def get_balance_sheet_statement(self):
                return pd.DataFrame()

            def get_cash_flow_statement(self):
                return pd.DataFrame()

        fake_financetoolkit = types.ModuleType("financetoolkit")
        fake_financetoolkit.Toolkit = _FakeToolkit

        with mock.patch.dict("os.environ", {"FMP_API_KEY": "unit-test-key"}, clear=False), \
             mock.patch.dict(sys.modules, {"financetoolkit": fake_financetoolkit}):
            context = chatter.fetch_financial_context(
                ticker="FSLY",
                date_start=None,
                date_end=None,
                include_technicals=False,
            )

        self.assertIn(f"Description: {long_description}", context)
        self.assertIn("This final sentence verifies the description was not truncated.", context)

    def test_signal_discovery_route_returns_structured_answer_and_logs_scores(self):
        if CHATTER_IMPORT_ERROR is not None:
            self.skipTest(f"chatter import failed: {CHATTER_IMPORT_ERROR}")

        class DummyEmbedModel:
            def encode(self, texts, normalize_embeddings=True):
                if isinstance(texts, str):
                    texts = [texts]
                vectors = []
                for text in texts:
                    lowered = str(text).lower()
                    if "oil" in lowered or "opec" in lowered or "supply" in lowered:
                        vectors.append([1.0, 0.0, 0.0])
                    else:
                        vectors.append([0.0, 1.0, 0.0])
                return vectors

        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "signals.db"
            create_database(str(db_path))
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            try:
                self._seed_signal_discovery_db(conn)

                result = chatter.run_query_once(
                    query="What are the top knowledge arbitrage signals?",
                    embed_model=DummyEmbedModel(),
                    reranker=None,
                    gen_client=object(),
                    driver=None,
                    sqlite_conn=conn,
                    alias_to_ticker={},
                    ticker_to_canonical={},
                    alias_to_fin_entity={},
                    base_system_prompt="Answer with grounded evidence.",
                    base_causal_system_prompt="Answer with grounded evidence.",
                    base_daily_summary_prompt="Answer with grounded evidence.",
                    base_single_ticker_financial_prompt="Answer with grounded evidence.",
                    memory=chatter.ConversationMemory(),
                    skip_generation=True,
                )

                retrieval_row = conn.execute(
                    """
                    SELECT selected, signal_id, cluster_id, score_trace_json
                    FROM retrieval_candidates
                    WHERE run_id = ?
                    ORDER BY final_score DESC
                    LIMIT 1
                    """,
                    (result["run_id"],),
                ).fetchone()
                qa_row = conn.execute(
                    """
                    SELECT route_type, retrieval_trace_json, selected_signal_alerts_json, answer_decision_json
                    FROM qa_runs
                    WHERE run_id = ?
                    """,
                    (result["run_id"],),
                ).fetchone()
            finally:
                conn.close()

        self.assertEqual(result["route_type"], "signal_discovery")
        self.assertIn("Answer:", result["answer"])
        self.assertIn("Evidence:", result["answer"])
        self.assertIn("Theory:", result["answer"])
        self.assertTrue(result["selected_signals"])
        self.assertEqual(retrieval_row["signal_id"], "signal-1")
        self.assertEqual(retrieval_row["cluster_id"], "cluster-1")
        self.assertEqual(retrieval_row["selected"], 1)
        score_trace = json.loads(retrieval_row["score_trace_json"])
        self.assertIn("signal_score", score_trace)
        self.assertIn("recency_score", score_trace)
        self.assertEqual(qa_row["route_type"], "signal_discovery")
        self.assertIn("signal-1", qa_row["selected_signal_alerts_json"])
        self.assertIn("selected_signal_count", qa_row["answer_decision_json"])

    def test_create_database_includes_upgrade_tables(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "upgrade.db"
            create_database(str(db_path))
            conn = sqlite3.connect(db_path)
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
            candidate_columns = {
                row[1] for row in conn.execute("PRAGMA table_info(macro_event_candidates)").fetchall()
            }
            verification_columns = {
                row[1] for row in conn.execute("PRAGMA table_info(macro_event_verifications)").fetchall()
            }
            retrieval_columns = {
                row[1] for row in conn.execute("PRAGMA table_info(retrieval_candidates)").fetchall()
            }
            conn.close()
        tables = {row[0] for row in rows}
        self.assertIn("event_clusters", tables)
        self.assertIn("cluster_members", tables)
        self.assertIn("event_cluster_scores", tables)
        self.assertIn("market_reactions", tables)
        self.assertIn("signal_alerts", tables)
        self.assertIn("source_quality", tables)
        self.assertIn("macro_processing_audit", tables)
        self.assertIn("macro_enum_audit", tables)
        self.assertIn("macro_event_index", candidate_columns)
        self.assertIn("candidate_json", candidate_columns)
        self.assertIn("evidence_span_valid", verification_columns)
        self.assertIn("cluster_id", retrieval_columns)
        self.assertIn("signal_id", retrieval_columns)

    def test_ensure_migrations_backfills_legacy_candidate_and_observability_columns(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "legacy.db"
            conn = sqlite3.connect(db_path)
            conn.executescript(
                """
                CREATE TABLE macro_event_candidates (
                    candidate_id TEXT PRIMARY KEY,
                    run_id TEXT,
                    article_id TEXT NOT NULL,
                    chunk_id TEXT,
                    event_type TEXT,
                    summary TEXT,
                    evidence_text TEXT,
                    evidence_span_json TEXT,
                    confidence_raw REAL,
                    confidence_candidate REAL,
                    extraction_pass TEXT,
                    source_trust_tier TEXT,
                    content_class TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE macro_event_verifications (
                    verification_id TEXT PRIMARY KEY,
                    candidate_id TEXT NOT NULL,
                    run_id TEXT,
                    article_id TEXT NOT NULL,
                    chunk_id TEXT,
                    verification_status TEXT NOT NULL,
                    support_score REAL,
                    confidence_calibrated REAL,
                    rejection_reason TEXT,
                    verifier_notes_json TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE qa_runs (
                    run_id TEXT PRIMARY KEY,
                    query TEXT NOT NULL,
                    route_type TEXT,
                    resolved_target_json TEXT,
                    retrieval_trace_json TEXT,
                    selected_chunks_json TEXT,
                    selected_macro_events_json TEXT,
                    answer_confidence REAL,
                    decision TEXT,
                    latency REAL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE retrieval_candidates (
                    run_id TEXT NOT NULL,
                    candidate_id TEXT NOT NULL,
                    candidate_kind TEXT NOT NULL,
                    article_id TEXT,
                    chunk_id TEXT,
                    macro_event_id TEXT,
                    semantic_score REAL,
                    cross_encoder_score REAL,
                    keyword_overlap_score REAL,
                    target_match_score REAL,
                    source_quality_score REAL,
                    recency_score REAL,
                    graph_relevance_score REAL,
                    event_support_score REAL,
                    duplicate_penalty REAL,
                    ambiguity_penalty REAL,
                    final_score REAL,
                    score_trace_json TEXT,
                    selected INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (run_id, candidate_id)
                );
                """
            )
            conn.commit()
            conn.close()

            ensure_migrations(str(db_path))

            conn = sqlite3.connect(db_path)
            candidate_columns = {
                row[1] for row in conn.execute("PRAGMA table_info(macro_event_candidates)").fetchall()
            }
            verification_columns = {
                row[1] for row in conn.execute("PRAGMA table_info(macro_event_verifications)").fetchall()
            }
            qa_run_columns = {
                row[1] for row in conn.execute("PRAGMA table_info(qa_runs)").fetchall()
            }
            retrieval_columns = {
                row[1] for row in conn.execute("PRAGMA table_info(retrieval_candidates)").fetchall()
            }
            conn.close()

        self.assertTrue(
            {
                "macro_event_index",
                "region",
                "time_horizon",
                "initial_confidence",
                "confidence_initial",
                "evidence_spans_json",
                "candidate_json",
                "raw_candidate_json",
                "novelty_hint",
                "urgency",
                "market_surprise",
            }.issubset(candidate_columns)
        )
        self.assertTrue(
            {
                "macro_event_index",
                "confidence_initial",
                "evidence_span_valid",
                "evidence_spans_count",
                "matched_spans_count",
            }.issubset(verification_columns)
        )
        self.assertTrue(
            {"route_decision_json", "selected_signals_json", "answer_meta_json"}.issubset(
                qa_run_columns
            )
        )
        self.assertTrue({"cluster_id", "signal_id"}.issubset(retrieval_columns))

    def test_v2_metrics_aggregate_deterministically(self):
        results = [
            {
                "id": "case-a",
                "passed": True,
                "chunk_score": {"recall": 1.0},
                "v2": {
                    "target_resolution_eval": {"available": True, "passed": True},
                    "abstention_eval": {"available": True, "passed": True},
                    "source_trust_eval": {"available": True, "passed": True},
                    "cluster_recall_eval": {"available": True, "hits": 2, "expected_count": 2, "recall": 1.0},
                    "signal_ranking_eval": {"available": True, "ndcg": 1.0},
                    "novelty_detection_eval": {"available": True, "matches": 2, "expected_count": 2, "accuracy": 1.0},
                    "verifier_precision_eval": {"available": True, "hits": 2, "predicted_verified": 2},
                    "contradiction_eval": {"available": True, "has_contradiction": False},
                    "unsupported_mechanism_eval": {"available": True, "unsupported_count": 0, "mechanism_count": 2},
                    "signal_market_feedback_eval": {"available": True, "hits": 1, "misses": 0, "considered": 1, "hit_rate": 1.0},
                    "observability_eval": {"available": True, "passed": True},
                    "slices": {
                        "query_type_route": "latest_news",
                        "ambiguity": "low",
                        "recency_sensitivity": "high",
                        "source_quality": "high",
                    },
                },
            },
            {
                "id": "case-b",
                "passed": False,
                "chunk_score": {"recall": 0.4},
                "v2": {
                    "target_resolution_eval": {"available": True, "passed": False},
                    "abstention_eval": {"available": True, "passed": False},
                    "source_trust_eval": {"available": True, "passed": False},
                    "cluster_recall_eval": {"available": True, "hits": 1, "expected_count": 2, "recall": 0.5},
                    "signal_ranking_eval": {"available": True, "ndcg": 0.5},
                    "novelty_detection_eval": {"available": True, "matches": 0, "expected_count": 1, "accuracy": 0.0},
                    "verifier_precision_eval": {"available": True, "hits": 0, "predicted_verified": 1},
                    "contradiction_eval": {"available": True, "has_contradiction": True},
                    "unsupported_mechanism_eval": {"available": True, "unsupported_count": 1, "mechanism_count": 2},
                    "signal_market_feedback_eval": {"available": True, "hits": 0, "misses": 1, "considered": 1, "hit_rate": 0.0},
                    "observability_eval": {"available": True, "passed": False},
                    "slices": {
                        "query_type_route": "macro_causal",
                        "ambiguity": "high",
                        "recency_sensitivity": "low",
                        "source_quality": "low",
                    },
                },
            },
        ]

        metrics = evaluation_suite._compute_v2_metrics(results)
        self.assertEqual(metrics["target_resolution_accuracy"]["value"], 0.5)
        self.assertEqual(metrics["abstention_correctness"]["value"], 0.5)
        self.assertEqual(metrics["source_trust_compliance"]["value"], 0.5)
        self.assertEqual(metrics["cluster_recall"]["value"], 0.75)
        self.assertEqual(metrics["signal_ranking_quality"]["value"], 0.75)
        self.assertEqual(metrics["novelty_detection"]["value"], 0.6667)
        self.assertEqual(metrics["verifier_precision"]["value"], 0.6667)
        self.assertEqual(metrics["contradiction_rate"]["value"], 0.5)
        self.assertEqual(metrics["unsupported_mechanism_rate"]["value"], 0.25)
        self.assertEqual(metrics["signal_hit_rate"]["value"], 0.5)
        self.assertEqual(metrics["observability_coverage"]["value"], 0.5)

    def test_v2_slice_report_groups_cases(self):
        results = [
            {
                "id": "case-a",
                "passed": True,
                "chunk_score": {"recall": 1.0},
                "v2": {
                    "slices": {
                        "query_type_route": "latest_news",
                        "ambiguity": "low",
                        "recency_sensitivity": "high",
                        "source_quality": "high",
                    }
                },
            },
            {
                "id": "case-b",
                "passed": False,
                "chunk_score": {"recall": 0.5},
                "v2": {
                    "slices": {
                        "query_type_route": "latest_news",
                        "ambiguity": "high",
                        "recency_sensitivity": "low",
                        "source_quality": "low",
                    }
                },
            },
        ]
        report = evaluation_suite._build_slice_report(results)
        self.assertIn("query_type_route", report)
        self.assertEqual(report["query_type_route"]["latest_news"]["cases"], 2)
        self.assertEqual(report["ambiguity"]["high"]["cases"], 1)
        self.assertEqual(report["ambiguity"]["low"]["cases"], 1)
        self.assertEqual(report["recency_sensitivity"]["high"]["cases"], 1)
        self.assertEqual(report["source_quality"]["low"]["cases"], 1)

    def test_release_gate_is_explicit_and_transparent(self):
        metrics = {
            "target_resolution_accuracy": {"value": 0.9},
            "abstention_correctness": {"value": 0.85},
            "source_trust_compliance": {"value": 0.95},
            "cluster_recall": {"value": 0.92},
            "signal_ranking_quality": {"value": 0.81},
            "novelty_detection": {"value": 0.88},
            "verifier_precision": {"value": 0.75},
            "contradiction_rate": {"value": 0.3},
            "unsupported_mechanism_rate": {"value": 0.1},
            "signal_hit_rate": {"value": 0.45},
            "observability_coverage": {"value": 1.0},
        }
        thresholds = {
            "case_pass_rate_min": 0.8,
            "target_resolution_accuracy_min": 0.8,
            "abstention_correctness_min": 0.8,
            "source_trust_compliance_min": 0.9,
            "cluster_recall_min": 0.9,
            "signal_ranking_quality_min": 0.8,
            "novelty_detection_min": 0.8,
            "verifier_precision_min": 0.7,
            "contradiction_rate_max": 0.2,
            "unsupported_mechanism_rate_max": 0.2,
            "signal_hit_rate_min": 0.5,
            "observability_coverage_min": 0.95,
        }
        gate = evaluation_suite._evaluate_release_gate(
            metrics=metrics,
            case_passed=8,
            case_total=10,
            thresholds=thresholds,
            require_all_metrics=False,
        )
        self.assertEqual(gate["status"], "fail")
        failed_metrics = {item["metric"] for item in gate["checks"] if not item["passed"]}
        self.assertIn("contradiction_rate", failed_metrics)
        self.assertIn("signal_hit_rate", failed_metrics)

    def test_log_observability_persists_route_and_score_details(self):
        if CHATTER_IMPORT_ERROR is not None:
            self.skipTest(f"chatter import failed: {CHATTER_IMPORT_ERROR}")

        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "observability.db"
            create_database(str(db_path))
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            chatter._log_observability(
                conn=conn,
                run_id="run-obs",
                query="What are the top knowledge arbitrage signals today?",
                route_type="signal_discovery",
                target=None,
                candidates=[
                    {
                        "candidate_id": "signal-1",
                        "retrieval_kind": "signal",
                        "cluster_id": "cluster-1",
                        "signal_id": "signal-1",
                        "semantic_score": 0.91,
                        "keyword_overlap_score": 0.83,
                        "target_match_score": 0.6,
                        "source_quality_score": 0.94,
                        "recency_score": 0.97,
                        "graph_relevance_score": 0.72,
                        "event_support_score": 0.88,
                        "duplicate_penalty": 0.0,
                        "ambiguity_penalty": 0.0,
                        "final_score": 0.89,
                        "signal_score": 0.91,
                        "score_components": {
                            "semantic_score": 0.91,
                            "keyword_overlap_score": 0.83,
                            "target_match_score": 0.6,
                            "source_quality_score": 0.94,
                            "recency_score": 0.97,
                            "graph_relevance_score": 0.72,
                            "event_support_score": 0.88,
                            "duplicate_penalty": 0.0,
                            "ambiguity_penalty": 0.0,
                        },
                    }
                ],
                selected_chunks=[],
                selected_signals=[{"signal_id": "signal-1", "cluster_id": "cluster-1", "signal_score": 0.91}],
                answer_confidence=82.0,
                decision="answer",
                latency_ms=12.5,
                retrieval_trace={"route_type": "signal_discovery"},
                route_reason={"route_seed": "broad_exploration", "final_route": "signal_discovery"},
                answer_meta={"decision": "answer", "answer_confidence": 82.0, "route_type": "signal_discovery"},
            )
            qa_row = conn.execute(
                "SELECT route_decision_json, selected_signals_json, answer_meta_json, answer_decision_json FROM qa_runs WHERE run_id = ?",
                ("run-obs",),
            ).fetchone()
            candidate_row = conn.execute(
                "SELECT cluster_id, signal_id, score_trace_json FROM retrieval_candidates WHERE run_id = ?",
                ("run-obs",),
            ).fetchone()
            conn.close()

        route_decision = json.loads(qa_row["route_decision_json"])
        selected_signals = json.loads(qa_row["selected_signals_json"])
        answer_meta = json.loads(qa_row["answer_meta_json"])
        answer_decision = json.loads(qa_row["answer_decision_json"])
        score_trace = json.loads(candidate_row["score_trace_json"])
        self.assertEqual(route_decision["final_route"], "signal_discovery")
        self.assertEqual(selected_signals[0]["signal_id"], "signal-1")
        self.assertEqual(answer_meta["decision"], "answer")
        self.assertEqual(answer_decision["decision"], "answer")
        self.assertEqual(candidate_row["cluster_id"], "cluster-1")
        self.assertEqual(candidate_row["signal_id"], "signal-1")
        self.assertIn("semantic_score", score_trace)
        self.assertIn("signal_score", score_trace)

    def test_evaluate_case_signal_discovery_surfaces_metrics_and_observability(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "signal_eval.db"
            create_database(str(db_path))
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            try:
                conn.execute(
                    """
                    INSERT INTO articles (
                        article_id, url, title, source, published_at, raw_text,
                        source_provider, source_trust_tier, content_class, article_quality_score, processing_state
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "article-1",
                        "https://example.com/a1",
                        "Rates shock shifts global bond risk",
                        "Reuters",
                        "2026-04-24T08:00:00Z",
                        "Direct evidence for a macro signal.",
                        "reuters",
                        "tier_1",
                        "news_report",
                        0.93,
                        "ingested",
                    ),
                )
                conn.execute(
                    """
                    INSERT INTO chunks (
                        chunk_id, article_id, chunk_index, text, published_date, processing_state
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "chunk-1",
                        "article-1",
                        0,
                        "US yields jumped after a stronger inflation print surprised traders.",
                        "2026-04-24",
                        "chunked",
                    ),
                )
                conn.execute(
                    """
                    INSERT INTO macro_events (
                        macro_event_id, run_id, article_id, chunk_id, event_type, summary, region,
                        time_horizon, confidence, verification_status, support_score, novelty_hint, urgency, market_surprise
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "event-1",
                        "run-macro-1",
                        "article-1",
                        "chunk-1",
                        "inflation_upside_surprise",
                        "Inflation surprise pushed yields higher.",
                        "us",
                        "near_term",
                        0.84,
                        "verified",
                        0.91,
                        "new",
                        "high",
                        "high",
                    ),
                )
                conn.commit()

                def _mock_run_query_once(**kwargs):
                    sql_conn = kwargs["sqlite_conn"]
                    sql_conn.execute(
                    """
                    INSERT INTO qa_runs (
                        run_id, query, route_type, route_decision_json, resolved_target_json, retrieval_trace_json,
                        selected_chunks_json, selected_macro_events_json, selected_signals_json,
                        answer_confidence, decision, answer_meta_json, answer_decision_json, latency, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "run-signal",
                        "What are the top knowledge arbitrage signals today?",
                        "signal_discovery",
                        json.dumps({"route_seed": "broad_exploration", "final_route": "signal_discovery"}),
                        json.dumps({}),
                        json.dumps({"route_type": "signal_discovery"}),
                        json.dumps([{"chunk_uid": "chunk-1"}]),
                        json.dumps([{"macro_event_id": "event-1"}]),
                        json.dumps(
                            [
                                {"signal_id": "signal-1", "cluster_id": "cluster-1", "signal_score": 0.91},
                                {"signal_id": "signal-2", "cluster_id": "cluster-2", "signal_score": 0.73},
                            ]
                        ),
                        82.0,
                        "answer",
                        json.dumps({"decision": "answer", "answer_confidence": 82.0}),
                        json.dumps({"decision": "answer", "answer_confidence": 82.0}),
                        15.0,
                        "2026-04-24T08:01:00Z",
                    ),
                )
                    for candidate_id, cluster_id, signal_id, final_score, selected in (
                        ("cand-1", "cluster-1", "signal-1", 0.91, 1),
                        ("cand-2", "cluster-2", "signal-2", 0.73, 1),
                    ):
                        sql_conn.execute(
                        """
                        INSERT INTO retrieval_candidates (
                            run_id, candidate_id, candidate_kind, chunk_id, macro_event_id, cluster_id, signal_id,
                            semantic_score, keyword_overlap_score, target_match_score, source_quality_score,
                            recency_score, graph_relevance_score, event_support_score,
                            duplicate_penalty, ambiguity_penalty, final_score, score_trace_json, selected, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            "run-signal",
                            candidate_id,
                            "signal",
                            "chunk-1",
                            "event-1",
                            cluster_id,
                            signal_id,
                            0.9,
                            0.8,
                            0.7,
                            0.95,
                            0.98,
                            0.74,
                            0.88,
                            0.0,
                            0.0,
                            final_score,
                            json.dumps(
                                {
                                    "semantic_score": 0.9,
                                    "keyword_overlap_score": 0.8,
                                    "target_match_score": 0.7,
                                    "source_quality_score": 0.95,
                                    "recency_score": 0.98,
                                    "graph_relevance_score": 0.74,
                                    "event_support_score": 0.88,
                                    "duplicate_penalty": 0.0,
                                    "ambiguity_penalty": 0.0,
                                }
                            ),
                            selected,
                            "2026-04-24T08:01:00Z",
                        ),
                    )
                    sql_conn.commit()
                    return {
                        "run_id": "run-signal",
                        "query": "What are the top knowledge arbitrage signals today?",
                        "answer": (
                            "Answer: 1. Inflation surprise in US rates [S1] 2. Follow-through in duration hedging [S1]\n"
                            "Evidence: Reuters reports the upside inflation shock and yield move [S1].\n"
                            "Theory: The evidence suggests a near-term rates spillover, but conviction should stay bounded."
                        ),
                        "chunks": [
                            {
                                "chunk_uid": "chunk-1",
                                "article_id": "article-1",
                                "source": "Reuters",
                                "title": "Rates shock shifts global bond risk",
                            }
                        ],
                        "urls": ["https://example.com/a1"],
                        "logs": [],
                        "citation_map": {"chunk-1": "S1"},
                        "provenance": "Why this answer: [S1] Reuters.",
                        "target": None,
                        "resolved_target": {"canonical_name": None, "query_type": "general"},
                        "resolved_target_json": {"canonical_name": None, "query_type": "general"},
                        "route_type": "signal_discovery",
                        "retrieval_trace": {"route_type": "signal_discovery"},
                        "answer_confidence": 82.0,
                        "decision": "answer",
                        "answer_meta": {"decision": "answer", "answer_confidence": 82.0, "route_type": "signal_discovery"},
                        "selected_macro_events": [
                            {
                                "macro_event_id": "event-1",
                                "verification_status": "verified",
                                "support_score": 0.91,
                                "confidence_calibrated": 0.84,
                                "novelty_hint": "new",
                            }
                        ],
                        "selected_signals": [
                            {
                                "signal_id": "signal-1",
                                "cluster_id": "cluster-1",
                                "signal_score": 0.91,
                                "novelty_hint": "new",
                                "market_feedback": {"outcome_label": "hit"},
                            },
                            {
                                "signal_id": "signal-2",
                                "cluster_id": "cluster-2",
                                "signal_score": 0.73,
                                "novelty_hint": "stale",
                                "market_feedback": {"outcome_label": "miss"},
                            },
                        ],
                        "contradiction_signals": False,
                    }

                runtime = {
                    "embed_model": object(),
                    "reranker": None,
                    "gen_client": None,
                    "driver": object(),
                    "sqlite_conn": conn,
                    "alias_to_ticker": {},
                    "ticker_to_canonical": {},
                    "alias_to_fin_entity": {},
                    "base_system_prompt": "",
                    "base_causal_system_prompt": "",
                    "base_daily_summary_prompt": "",
                    "base_single_ticker_financial_prompt": "",
                }
                case = {
                    "id": "signal-route",
                    "query": "What are the top knowledge arbitrage signals today?",
                    "expected_chunks": ["chunk-1"],
                    "expected_entities": [],
                    "expected_macro_event_ids": ["event-1"],
                    "expected_macro_event_types": ["inflation_upside_surprise"],
                    "expected_route_type": "signal_discovery",
                    "expected_cluster_ids": ["cluster-1", "cluster-2"],
                    "expected_signal_ids": ["signal-1", "signal-2"],
                    "expected_novelty_hints": {"signal-1": "new", "signal-2": "stale"},
                    "expected_answer_grounding": {
                        "require_inline_citations": True,
                        "require_evidence_section": True,
                        "require_theory_section": True,
                        "min_cited_sources": 1,
                    },
                    "min_cluster_recall": 1.0,
                    "min_signal_ranking_quality": 1.0,
                    "min_novelty_detection": 1.0,
                    "min_signal_hit_rate": 0.5,
                    "require_observability": True,
                }

                with mock.patch.object(
                    evaluation_suite.chatter,
                    "run_query_once",
                    side_effect=_mock_run_query_once,
                    create=True,
                ):
                    result = evaluation_suite.evaluate_case(case, runtime, skip_generation=False)
            finally:
                conn.close()

        self.assertEqual(result["v2"]["route_type"], "signal_discovery")
        self.assertTrue(result["grounding"]["has_answer_section"])
        self.assertTrue(result["grounding"]["has_evidence_section"])
        self.assertTrue(result["grounding"]["has_theory_section"])
        self.assertEqual(result["v2"]["cluster_recall_eval"]["recall"], 1.0)
        self.assertEqual(result["v2"]["signal_ranking_eval"]["ndcg"], 1.0)
        self.assertEqual(result["v2"]["novelty_detection_eval"]["accuracy"], 1.0)
        self.assertEqual(result["v2"]["signal_market_feedback_eval"]["hit_rate"], 0.5)
        self.assertTrue(result["v2"]["observability_eval"]["passed"])
        self.assertTrue(result["passed"])

    def test_backwards_compatible_decision_and_policy_fallbacks(self):
        decision = evaluation_suite._extract_decision(
            result={},
            answer="Answer: insufficient evidence to answer confidently.",
            answer_confidence=None,
        )
        self.assertEqual(decision, "abstain")
        abstention_eval = evaluation_suite._evaluate_abstention(
            {"expected_abstain": True},
            decision,
            "Answer: insufficient evidence.",
        )
        self.assertTrue(abstention_eval["passed"])

        source_eval = evaluation_suite._evaluate_source_trust(
            {"source_trust_policy": {"disallow_tiers": ["blocked"]}},
            {
                "chunk-1": {
                    "source": "Example",
                    "source_trust_tier": "blocked",
                    "content_class": "navigation_page",
                    "article_quality_score": 0.2,
                }
            },
        )
        self.assertTrue(source_eval["available"])
        self.assertFalse(source_eval["passed"])

        target_eval = evaluation_suite._evaluate_target_resolution(
            {"expected_target": {"canonical_name": "NVDA", "query_type": "single_entity"}},
            {
                "canonical_name": "NVDA",
                "query_type": "single_entity",
                "ticker": "NVDA",
                "ambiguity_score": 0.1,
                "needs_disambiguation": False,
                "resolution_mode": "direct",
                "candidates": [],
            },
        )
        self.assertTrue(target_eval["available"])
        self.assertTrue(target_eval["passed"])


class TestSingleTickerStructuredAnalysis(unittest.TestCase):
    def test_single_ticker_resolution_direct_company_and_ambiguous(self):
        direct = sta.resolve_single_ticker("NVDA")
        self.assertEqual(direct.ticker, "NVDA")
        self.assertGreaterEqual(direct.confidence, 0.95)

        company = sta.resolve_single_ticker("Advanced Micro Devices")
        self.assertEqual(company.ticker, "AMD")
        self.assertFalse(company.needs_disambiguation)

        fake_rows = [
            {"ticker": "AAA", "company_name": "Acme Data Inc.", "canonical_name": "Acme Data Inc.", "exchange": None, "aliases": ["Acme"]},
            {"ticker": "BBB", "company_name": "Acme Devices Inc.", "canonical_name": "Acme Devices Inc.", "exchange": None, "aliases": ["Acme"]},
        ]
        with mock.patch.object(sta, "_load_ticker_rows", return_value=fake_rows):
            ambiguous = sta.resolve_single_ticker("Acme")
        self.assertIsNone(ambiguous.ticker)
        self.assertTrue(ambiguous.needs_disambiguation)

        no_match = sta.resolve_single_ticker("not a listed company")
        self.assertIsNone(no_match.ticker)
        self.assertTrue(no_match.warnings)

    def test_single_ticker_tables_are_created_idempotently(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "analysis.db"
            create_database(str(db_path))
            create_database(str(db_path))
            conn = sqlite3.connect(db_path)
            rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            idx_rows = conn.execute("SELECT name FROM sqlite_master WHERE type='index'").fetchall()
            conn.close()

        tables = {row[0] for row in rows}
        indexes = {row[0] for row in idx_rows}
        self.assertIn("single_ticker_financial_metrics", tables)
        self.assertIn("single_ticker_technical_indicators", tables)
        self.assertIn("strategy_backtest_runs", tables)
        self.assertIn("strategy_signals", tables)
        self.assertIn("idx_single_ticker_financial_ticker_date", indexes)
        self.assertIn("idx_strategy_signals_ticker_date", indexes)

    def test_single_ticker_indicator_helpers_and_short_history(self):
        import pandas as pd

        prices = pd.Series([1, 2, 3, 4, 5], dtype="float64")
        self.assertEqual(sta._simple_moving_average(prices, 3), 4.0)
        self.assertLess(sta._rsi(pd.Series([100 - i for i in range(30)], dtype="float64"), 14), 30.0)

        short_frame = _single_ticker_price_frame([10, 11, 12, 13, 14])

        def fake_loader(ticker, lookback_days=None):
            if ticker == "SPY":
                return None, ["SPY unavailable"]
            return short_frame, []

        with mock.patch.object(sta, "_load_price_history", side_effect=fake_loader):
            result = sta.compute_technical_indicators("AAA", persist=False)

        self.assertEqual(result["data_quality"]["price_rows"], 5)
        self.assertIsNone(result["indicators"]["sma_20"])
        self.assertIn("insufficient price history", " ".join(result["data_quality"]["warnings"]))

    def test_single_ticker_strategy_signals_on_synthetic_data(self):
        uptrend = _single_ticker_price_frame(range(1, 260))
        downtrend = _single_ticker_price_frame([100 - i for i in range(40)])

        def uptrend_loader(ticker, lookback_days=None):
            if ticker == "SPY":
                return _single_ticker_price_frame(range(1, 260), ticker="SPY"), []
            return uptrend, []

        with mock.patch.object(sta, "_load_price_history", side_effect=uptrend_loader):
            ma_signal = sta.generate_strategy_signal("AAA", "moving_average_trend", persist=False)
            momentum_signal = sta.generate_strategy_signal("AAA", "momentum", persist=False)

        self.assertEqual(ma_signal["signal_direction"], "bullish")
        self.assertEqual(momentum_signal["signal_direction"], "bullish")

        def oversold_loader(ticker, lookback_days=None):
            if ticker == "SPY":
                return None, ["SPY unavailable"]
            return downtrend, []

        with mock.patch.object(sta, "_load_price_history", side_effect=oversold_loader):
            rsi_signal = sta.generate_strategy_signal("AAA", "rsi_mean_reversion", persist=False)

        self.assertEqual(rsi_signal["signal_direction"], "bullish")

    def test_single_ticker_backtest_handles_short_data_and_shifts_signal(self):
        short_frame = _single_ticker_price_frame([10, 11, 12, 13, 14])

        def fake_loader(ticker, lookback_days=None):
            if ticker == "SPY":
                return None, ["SPY unavailable"]
            return short_frame, []

        with mock.patch.object(sta, "_load_price_history", side_effect=fake_loader):
            result = sta.run_strategy_backtest("AAA", "moving_average_trend", persist=False)

        self.assertIn("metrics", result)
        self.assertTrue(result["data_quality"]["warnings"])

        data, warnings = sta._strategy_position_frame(_single_ticker_price_frame(range(1, 260)), "momentum")
        self.assertFalse(warnings)
        first_signal_idx = data.index[data["raw_signal"] > 0][0]
        self.assertEqual(float(data["position"].iloc[0]), 0.0)
        self.assertEqual(float(data.loc[first_signal_idx, "position"]), 0.0)
        self.assertEqual(float(data.loc[first_signal_idx + 1, "position"]), 1.0)

    def test_single_ticker_persistence_writes_valid_json_fields(self):
        metrics = {key: None for key in sta.FINANCIAL_METRIC_KEYS}
        metrics.update(
            {
                "revenue_ttm": 100.0,
                "revenue_growth_yoy": 0.10,
                "gross_margin": 0.60,
                "operating_margin": 0.20,
                "net_margin": 0.15,
                "free_cash_flow": 12.0,
                "fcf_margin": 0.12,
                "total_debt": 20.0,
                "cash_and_equivalents": 30.0,
                "current_ratio": 1.8,
                "debt_to_equity": 0.3,
            }
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = str(Path(tmp_dir) / "persist.db")
            with mock.patch.object(sta, "_fetch_financial_metrics_from_provider", return_value=(metrics, "unit-test", [])):
                result = sta.collect_financial_metrics("AAA", db_path=db_path, persist=True)
            conn = sqlite3.connect(db_path)
            row = conn.execute(
                "SELECT metrics_json, score_json, data_quality_json FROM single_ticker_financial_metrics WHERE ticker='AAA'"
            ).fetchone()
            conn.close()

        self.assertIsNotNone(row)
        self.assertEqual(json.loads(row[0])["revenue_ttm"], 100.0)
        self.assertIn("financial_quality_score", json.loads(row[1]))
        self.assertEqual(result["data_quality"]["available_metrics_count"], 11)
        self.assertIsInstance(json.loads(row[2])["warnings"], list)

    def test_single_ticker_cli_resolve_prints_valid_json(self):
        import io
        from contextlib import redirect_stdout

        stdout = io.StringIO()
        with mock.patch("sys.argv", ["single_ticker_analysis.py", "--no-persist", "resolve", "NVDA"]):
            with redirect_stdout(stdout):
                sta.main()
        self.assertEqual(json.loads(stdout.getvalue())["ticker"], "NVDA")

    @unittest.skipIf(CHATTER_IMPORT_ERROR is not None, f"chatter import failed: {CHATTER_IMPORT_ERROR}")
    def test_chatter_single_ticker_analysis_hook_disabled_and_soft_failure(self):
        with mock.patch.object(chatter, "ENABLE_SINGLE_TICKER_ANALYSIS", False):
            self.assertEqual(chatter._build_single_ticker_analysis_context("NVDA", query="NVDA"), "")

        with mock.patch.object(chatter, "ENABLE_SINGLE_TICKER_ANALYSIS", True), \
             mock.patch.object(chatter, "ENABLE_SINGLE_TICKER_BACKTESTS", False), \
             mock.patch.object(sta, "collect_financial_metrics", side_effect=RuntimeError("financial boom")), \
             mock.patch.object(sta, "compute_technical_indicators", side_effect=RuntimeError("technical boom")), \
             mock.patch.object(sta, "generate_strategy_signal", side_effect=RuntimeError("signal boom")):
            block = chatter._build_single_ticker_analysis_context("NVDA", query="NVDA")

        self.assertTrue(block.startswith("[SINGLE TICKER ANALYSIS]"))
        payload = json.loads(block.split("\n", 1)[1])
        self.assertTrue(any("financial boom" in warning for warning in payload["warnings"]))
        self.assertTrue(any("technical boom" in warning for warning in payload["warnings"]))
        self.assertTrue(any("signal boom" in warning for warning in payload["warnings"]))


if __name__ == "__main__":
    unittest.main()
