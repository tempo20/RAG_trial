import json
import sqlite3
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    from rag_trial.chat import chatter
    from rag_trial.chat.convo_memory import ConversationMemory
    CHATTER_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - environment guard
    chatter = types.ModuleType("chatter")
    sys.modules["chatter"] = chatter
    ConversationMemory = object  # type: ignore[assignment]
    CHATTER_IMPORT_ERROR = exc

from rag_trial.db.create_sql_db import create_database


class _DummyEmbedModel:
    def encode(self, texts, normalize_embeddings=True):  # noqa: D401
        return [[1.0, 0.0, 0.0] for _ in texts]


@unittest.skipIf(CHATTER_IMPORT_ERROR is not None, f"chatter import failed: {CHATTER_IMPORT_ERROR}")
class TestKnowledgeArbitrageRoute(unittest.TestCase):
    def test_signal_discovery_route_preserves_contract_and_logs_scores(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "qa.db"
            create_database(str(db_path))
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row

            target = chatter.QueryTarget(
                query_type=chatter.QUERY_TYPE_GENERAL,
                canonical_name=None,
                display_name="general news",
                ticker=None,
                entity_type=None,
                confidence=0.0,
                resolution_mode="signal_discovery",
            )
            evidence_chunk = {
                "candidate_id": "signal::1::evidence::0",
                "chunk_uid": "chunk-1",
                "article_id": "article-1",
                "macro_event_id": "macro-1",
                "retrieval_kind": "signal_cluster_evidence",
                "source": "Reuters",
                "title": "Oil supply shock",
                "url": "https://example.com/oil",
                "period_key": "2026-04",
                "published_date": "2026-04-24",
                "text": "Oil prices jumped after OPEC extended supply cuts.",
                "macro_summary": "Oil prices jumped after supply cuts tightened the market.",
                "event_type": "commodity_supply_disruption",
                "verification_status": "verified",
                "support_score": 0.91,
                "cluster_id": "cluster-1",
                "signal_id": "signal-1",
                "final_score": 0.81,
                "score": 0.81,
                "score_components": {
                    "semantic_score": 0.8,
                    "keyword_overlap_score": 0.7,
                    "target_match_score": 0.4,
                    "source_quality_score": 0.9,
                    "recency_score": 0.95,
                    "graph_relevance_score": 0.9,
                    "event_support_score": 0.91,
                    "duplicate_penalty": 0.0,
                    "ambiguity_penalty": 0.0,
                    "signal_score": 0.84,
                },
            }
            selected_signals = [
                {
                    "candidate_id": "signal::1",
                    "signal_id": "signal-1",
                    "cluster_id": "cluster-1",
                    "headline": "Oil supply cuts are tightening the market",
                    "summary": "Oil supply cuts tightened the market and pushed prices higher.",
                    "signal_score": 0.84,
                    "final_score": 0.81,
                    "novelty_hint": "new",
                    "urgency": "high",
                    "market_surprise": "medium",
                    "evidence_chunks": [evidence_chunk],
                    "score_components": evidence_chunk["score_components"],
                }
            ]
            retrieval_trace = {
                "route_type": "signal_discovery",
                "candidate_count": 1,
                "ranked_count": 1,
                "ranked_candidates": selected_signals,
                "signal_ids": ["signal-1"],
            }

            with patch.object(chatter, "resolve_coreference", return_value=("What are the top knowledge arbitrage signals today?", False)), \
                patch.object(chatter, "is_market_data_intent", return_value=False), \
                patch.object(chatter, "is_causal_analysis_intent", return_value=False), \
                patch.object(chatter, "is_summary_query", return_value=False), \
                patch.object(chatter, "infer_summary_date_range", return_value=(None, None)), \
                patch.object(chatter, "extract_source_filter", return_value=None), \
                patch.object(chatter, "decompose_query", return_value=[{"query": "What are the top knowledge arbitrage signals today?", "time_start": None, "time_end": None}]), \
                patch.object(chatter, "resolve_temporal_carryover", side_effect=lambda items, memory: items), \
                patch.object(chatter, "retrieve_top_signals", return_value=(selected_signals, target, retrieval_trace)):
                result = chatter.run_query_once(
                    query="What are the top knowledge arbitrage signals today?",
                    embed_model=_DummyEmbedModel(),
                    reranker=None,
                    gen_client=None,
                    driver=None,
                    sqlite_conn=conn,
                    alias_to_ticker={},
                    ticker_to_canonical={},
                    alias_to_fin_entity={},
                    base_system_prompt="",
                    base_causal_system_prompt="",
                    base_daily_summary_prompt="",
                    base_single_ticker_financial_prompt="",
                    memory=ConversationMemory(),
                    skip_generation=True,
                )

            stored = conn.execute(
                "SELECT score_trace_json, selected FROM retrieval_candidates WHERE signal_id = 'signal-1'"
            ).fetchone()
            qa_run = conn.execute(
                "SELECT route_type, selected_signals_json FROM qa_runs WHERE run_id = ?",
                (result["run_id"],),
            ).fetchone()
            conn.close()

        self.assertIn("Answer:", result["answer"])
        self.assertIn("Evidence:", result["answer"])
        self.assertIn("Theory:", result["answer"])
        self.assertEqual(result["route_type"], "signal_discovery")
        self.assertTrue(result["selected_signals"])
        self.assertEqual(qa_run["route_type"], "signal_discovery")
        self.assertTrue(json.loads(qa_run["selected_signals_json"]))
        self.assertEqual(stored["selected"], 1)
        score_trace = json.loads(stored["score_trace_json"])
        self.assertIn("semantic_score", score_trace)
        self.assertIn("keyword_overlap_score", score_trace)
        self.assertIn("signal_score", score_trace)


if __name__ == "__main__":
    unittest.main()
