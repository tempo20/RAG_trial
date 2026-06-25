import sqlite3
import tempfile
import unittest
from pathlib import Path

from rag_trial.db.create_sql_db import create_database
from rag_trial.analysis.event_cluster import run_event_clustering


def _fake_embedding_fn(texts: list[str]) -> list[list[float]]:
    out: list[list[float]] = []
    for text in texts:
        lowered = text.lower()
        if "oil" in lowered:
            out.append([1.0, 0.0, 0.0])
        else:
            out.append([0.0, 1.0, 0.0])
    return out


class TestEventClustering(unittest.TestCase):
    def test_same_story_across_three_articles_becomes_one_cluster(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "cluster.db"
            create_database(str(db_path))
            conn = sqlite3.connect(db_path)
            conn.executemany(
                """
                INSERT INTO articles (article_id, url, title, source, raw_text, published_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    ("a1", "https://example.com/1", "Oil shock one", "Reuters", "text", "2026-04-20"),
                    ("a2", "https://example.com/2", "Oil shock two", "Bloomberg", "text", "2026-04-21"),
                    ("a3", "https://example.com/3", "Oil shock three", "FT", "text", "2026-04-22"),
                ],
            )
            conn.executemany(
                """
                INSERT INTO chunks (chunk_id, article_id, chunk_index, text, published_date, period_key)
                VALUES (?, ?, 0, ?, ?, '2026-04')
                """,
                [
                    ("c1", "a1", "Oil prices jumped after an OPEC supply cut.", "2026-04-20"),
                    ("c2", "a2", "OPEC supply cuts pushed oil higher again.", "2026-04-21"),
                    ("c3", "a3", "Oil rallied as supply disruption fears persisted.", "2026-04-22"),
                ],
            )
            conn.execute(
                "INSERT INTO macro_extraction_runs (run_id, created_at, success) VALUES ('run-1', '2026-04-20T00:00:00+00:00', 1)"
            )
            conn.executemany(
                """
                INSERT INTO macro_events (
                    macro_event_id, run_id, article_id, chunk_id, event_type, summary,
                    region, time_horizon, event_time_start, confidence, verification_status, support_score
                )
                VALUES (?, 'run-1', ?, ?, 'commodity_supply_disruption', ?, 'global', 'near_term', ?, 0.82, 'verified', 0.85)
                """,
                [
                    ("m1", "a1", "c1", "Oil prices jumped after OPEC cuts tightened supply.", "2026-04-20"),
                    ("m2", "a2", "c2", "OPEC cuts tightened supply and lifted oil prices.", "2026-04-21"),
                    ("m3", "a3", "c3", "Supply disruption fears kept oil prices elevated.", "2026-04-22"),
                ],
            )
            conn.executemany(
                "INSERT INTO macro_event_shock_types (macro_event_id, shock_type) VALUES (?, 'commodity_supply_disruption')",
                [("m1",), ("m2",), ("m3",)],
            )
            conn.executemany(
                """
                INSERT INTO asset_impacts (
                    impact_id, macro_event_id, target_type, target_id, direction, strength, horizon, confidence
                )
                VALUES (?, ?, 'commodity', 'oil', 'up', 'strong', 'near_term', 0.8)
                """,
                [("i1", "m1"), ("i2", "m2"), ("i3", "m3")],
            )
            conn.commit()
            conn.close()

            summary = run_event_clustering(
                str(db_path),
                window_days=7,
                similarity_threshold=0.82,
                embedding_fn=_fake_embedding_fn,
            )

            conn = sqlite3.connect(db_path)
            cluster_count = conn.execute("SELECT COUNT(*) FROM event_clusters").fetchone()[0]
            member_count = conn.execute("SELECT COUNT(*) FROM cluster_members").fetchone()[0]
            cluster_member_size = conn.execute(
                "SELECT member_count FROM event_clusters LIMIT 1"
            ).fetchone()[0]
            conn.close()

        self.assertEqual(summary["clusters_created"], 1)
        self.assertEqual(cluster_count, 1)
        self.assertEqual(member_count, 3)
        self.assertEqual(cluster_member_size, 3)


if __name__ == "__main__":
    unittest.main()
