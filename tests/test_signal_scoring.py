import unittest
import sqlite3
import tempfile
from pathlib import Path
from unittest import mock

from rag_trial.db.create_sql_db import create_database
from rag_trial.analysis.signal_scoring import compute_cluster_score, compute_novelty_score, run_signal_scoring


def _insert_cluster_fixture(
    conn: sqlite3.Connection,
    *,
    cluster_id: str,
    macro_event_id: str,
    article_id: str,
    chunk_id: str,
    source: str,
    source_trust_tier: str,
    article_quality_score: float,
    support_score: float,
    confidence: float,
    novelty_hint: str,
    urgency: str,
    market_surprise: str,
    event_date: str,
    summary: str,
    target_id: str,
    impact_strength: str,
    impact_confidence: float,
) -> None:
    conn.execute(
        """
        INSERT INTO articles (
            article_id, url, title, source, source_trust_tier, article_quality_score, raw_text, published_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            article_id,
            f"https://example.com/{article_id}",
            f"title-{article_id}",
            source,
            source_trust_tier,
            article_quality_score,
            "text",
            f"{event_date}T00:00:00+00:00",
        ),
    )
    conn.execute(
        """
        INSERT INTO chunks (
            chunk_id, article_id, chunk_index, text, published_date, period_key
        ) VALUES (?, ?, 0, ?, ?, ?)
        """,
        (chunk_id, article_id, summary, event_date, event_date[:7]),
    )
    conn.execute(
        """
        INSERT INTO macro_extraction_runs (
            run_id, article_id, chunk_id, model_provider, model_name, prompt_version, schema_version,
            created_at, success, raw_json
        ) VALUES (?, ?, ?, 'test', 'test', 'v1', 'v1', ?, 1, '{}')
        """,
        (f"run-{macro_event_id}", article_id, chunk_id, f"{event_date}T01:00:00+00:00"),
    )
    conn.execute(
        """
        INSERT INTO macro_events (
            macro_event_id, run_id, article_id, chunk_id, event_type, summary, region, time_horizon,
            event_time_start, event_time_end, confidence, verification_status, support_score, novelty_hint,
            urgency, market_surprise
        ) VALUES (?, ?, ?, ?, 'commodity_supply_disruption', ?, 'global', 'near_term', ?, ?, ?, 'verified', ?, ?, ?, ?)
        """,
        (
            macro_event_id,
            f"run-{macro_event_id}",
            article_id,
            chunk_id,
            summary,
            event_date,
            event_date,
            confidence,
            support_score,
            novelty_hint,
            urgency,
            market_surprise,
        ),
    )
    conn.execute(
        """
        INSERT INTO event_clusters (
            cluster_id, event_type, primary_shock_type, region, canonical_summary, summary_embedding_json,
            first_event_time, last_event_time, cluster_window_days, member_count, unique_source_count,
            asset_targets_json, cluster_status, created_at, updated_at
        ) VALUES (?, 'commodity_supply_disruption', 'commodity_supply_disruption', 'global', ?, '[1.0, 0.0, 0.0]',
                  ?, ?, 7, 1, 1, ?, 'active', ?, ?)
        """,
        (
            cluster_id,
            summary,
            event_date,
            event_date,
            f"[\"ticker:{target_id}\"]",
            f"{event_date}T01:30:00+00:00",
            f"{event_date}T01:30:00+00:00",
        ),
    )
    conn.execute(
        """
        INSERT INTO cluster_members (
            cluster_id, macro_event_id, similarity_score, match_reasons_json, event_time, article_id, chunk_id, source, created_at
        ) VALUES (?, ?, 0.95, '["same_story"]', ?, ?, ?, ?, ?)
        """,
        (
            cluster_id,
            macro_event_id,
            event_date,
            article_id,
            chunk_id,
            source,
            f"{event_date}T01:40:00+00:00",
        ),
    )
    conn.execute(
        """
        INSERT INTO asset_impacts (
            impact_id, macro_event_id, target_type, target_id, direction, strength, horizon, confidence
        ) VALUES (?, ?, 'ticker', ?, 'up', ?, 'near_term', ?)
        """,
        (
            f"impact-{macro_event_id}",
            macro_event_id,
            target_id,
            impact_strength,
            impact_confidence,
        ),
    )


class TestSignalScoring(unittest.TestCase):
    def test_stale_repeated_story_gets_low_novelty(self) -> None:
        cluster = {
            "cluster_id": "c-stale",
            "first_event_time": "2026-03-01T00:00:00+00:00",
            "last_event_time": "2026-04-20T00:00:00+00:00",
            "member_count": 4,
        }
        members = [
            {"novelty_hint": "stale", "event_time": "2026-04-20T00:00:00+00:00"},
            {"novelty_hint": "continuation", "event_time": "2026-04-19T00:00:00+00:00"},
        ]
        novelty = compute_novelty_score(cluster, members, prior_alert_count=2)
        self.assertLess(novelty, 0.35)

    def test_one_source_weak_event_scores_lower_than_multi_source_fresh_event(self) -> None:
        weak_cluster = {
            "cluster_id": "weak",
            "first_event_time": "2026-04-20T00:00:00+00:00",
            "last_event_time": "2026-04-20T00:00:00+00:00",
            "member_count": 1,
            "unique_source_count": 1,
        }
        weak_members = [
            {
                "source": "ExampleWire",
                "source_trust_tier": "tier_3",
                "article_quality_score": 0.35,
                "support_score": 0.35,
                "confidence": 0.35,
                "novelty_hint": "continuation",
                "event_time": "2026-04-20T00:00:00+00:00",
            }
        ]
        weak_impacts = [
            {"target_type": "ticker", "target_id": "AAA", "strength": "weak", "confidence": 0.3}
        ]

        strong_cluster = {
            "cluster_id": "strong",
            "first_event_time": "2026-04-23T00:00:00+00:00",
            "last_event_time": "2026-04-24T00:00:00+00:00",
            "member_count": 3,
            "unique_source_count": 3,
        }
        strong_members = [
            {
                "source": "Reuters",
                "source_trust_tier": "tier_1",
                "article_quality_score": 0.9,
                "support_score": 0.9,
                "confidence": 0.88,
                "novelty_hint": "new",
                "event_time": "2026-04-24T00:00:00+00:00",
            },
            {
                "source": "Bloomberg",
                "source_trust_tier": "tier_1",
                "article_quality_score": 0.92,
                "support_score": 0.86,
                "confidence": 0.85,
                "novelty_hint": "new",
                "event_time": "2026-04-24T00:00:00+00:00",
            },
            {
                "source": "FT",
                "source_trust_tier": "tier_1",
                "article_quality_score": 0.88,
                "support_score": 0.84,
                "confidence": 0.82,
                "novelty_hint": "new",
                "event_time": "2026-04-23T00:00:00+00:00",
            },
        ]
        strong_impacts = [
            {"target_type": "ticker", "target_id": "AAA", "strength": "strong", "confidence": 0.9},
            {"target_type": "commodity", "target_id": "oil", "strength": "moderate", "confidence": 0.8},
        ]

        weak_score = compute_cluster_score(weak_cluster, weak_members, weak_impacts, prior_alert_count=1)
        strong_score = compute_cluster_score(strong_cluster, strong_members, strong_impacts, prior_alert_count=0)

        self.assertLess(weak_score["signal_score"], 0.5)
        self.assertGreater(strong_score["signal_score"], weak_score["signal_score"])
        self.assertGreater(strong_score["signal_score"], 0.7)

    def test_rescoring_same_day_is_idempotent_for_novelty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "signals_idempotent.db"
            create_database(str(db_path))
            conn = sqlite3.connect(db_path)
            _insert_cluster_fixture(
                conn,
                cluster_id="cluster-1",
                macro_event_id="event-1",
                article_id="article-1",
                chunk_id="chunk-1",
                source="Reuters",
                source_trust_tier="tier_1",
                article_quality_score=0.95,
                support_score=0.9,
                confidence=0.9,
                novelty_hint="new",
                urgency="high",
                market_surprise="high",
                event_date="2026-04-24",
                summary="Oil supply shock tightened balances.",
                target_id="XOM",
                impact_strength="strong",
                impact_confidence=0.9,
            )
            conn.commit()
            conn.close()

            fixed_now = "2026-04-24T09:00:00+00:00"
            with mock.patch("rag_trial.analysis.signal_scoring._now_utc", return_value=fixed_now):
                first_summary = run_signal_scoring(str(db_path), limit=10)
                conn = sqlite3.connect(db_path)
                first_row = conn.execute(
                    """
                    SELECT novelty_score, signal_score
                    FROM event_cluster_scores
                    WHERE cluster_id = 'cluster-1' AND score_date = '2026-04-24'
                    """
                ).fetchone()
                conn.close()

                second_summary = run_signal_scoring(str(db_path), limit=10)
                conn = sqlite3.connect(db_path)
                second_row = conn.execute(
                    """
                    SELECT novelty_score, signal_score
                    FROM event_cluster_scores
                    WHERE cluster_id = 'cluster-1' AND score_date = '2026-04-24'
                    """
                ).fetchone()
                conn.close()

        self.assertEqual(first_summary["signals_deactivated"], 0)
        self.assertEqual(second_summary["signals_deactivated"], 0)
        self.assertIsNotNone(first_row)
        self.assertIsNotNone(second_row)
        self.assertEqual(first_row[0], second_row[0])
        self.assertEqual(first_row[1], second_row[1])

    def test_signal_limit_deactivates_stale_same_day_alerts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "signals_limit.db"
            create_database(str(db_path))
            conn = sqlite3.connect(db_path)
            _insert_cluster_fixture(
                conn,
                cluster_id="cluster-strong",
                macro_event_id="event-strong",
                article_id="article-strong",
                chunk_id="chunk-strong",
                source="Reuters",
                source_trust_tier="tier_1",
                article_quality_score=0.97,
                support_score=0.93,
                confidence=0.92,
                novelty_hint="new",
                urgency="high",
                market_surprise="high",
                event_date="2026-04-24",
                summary="Strong and fresh supply shock.",
                target_id="XOM",
                impact_strength="strong",
                impact_confidence=0.92,
            )
            _insert_cluster_fixture(
                conn,
                cluster_id="cluster-weak",
                macro_event_id="event-weak",
                article_id="article-weak",
                chunk_id="chunk-weak",
                source="ExampleWire",
                source_trust_tier="tier_3",
                article_quality_score=0.3,
                support_score=0.35,
                confidence=0.35,
                novelty_hint="stale",
                urgency="low",
                market_surprise="low",
                event_date="2026-04-18",
                summary="Older continuation with weaker support.",
                target_id="CVX",
                impact_strength="weak",
                impact_confidence=0.3,
            )
            conn.commit()
            conn.close()

            fixed_now = "2026-04-24T09:00:00+00:00"
            with mock.patch("rag_trial.analysis.signal_scoring._now_utc", return_value=fixed_now):
                initial_summary = run_signal_scoring(str(db_path), limit=2)
                limited_summary = run_signal_scoring(str(db_path), limit=1)

            conn = sqlite3.connect(db_path)
            rows = conn.execute(
                """
                SELECT cluster_id, status
                FROM signal_alerts
                WHERE signal_date = '2026-04-24'
                ORDER BY cluster_id
                """
            ).fetchall()
            conn.close()

        self.assertEqual(initial_summary["signals_written"], 2)
        self.assertEqual(limited_summary["signals_written"], 1)
        self.assertEqual(limited_summary["signals_deactivated"], 1)
        statuses = {row[0]: row[1] for row in rows}
        self.assertEqual(statuses.get("cluster-strong"), "active")
        self.assertEqual(statuses.get("cluster-weak"), "inactive")


if __name__ == "__main__":
    unittest.main()
