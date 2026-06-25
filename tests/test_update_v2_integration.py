from __future__ import annotations

import importlib
import os
import shutil
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
import types
import unittest
import uuid
from pathlib import Path
from unittest import mock

from rag_trial.db.create_sql_db import create_database

try:
    from neo4j import GraphDatabase
except Exception as exc:  # pragma: no cover - dependency guard
    GraphDatabase = None
    NEO4J_IMPORT_ERROR = exc
else:
    NEO4J_IMPORT_ERROR = None


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_neo4j(uri: str, user: str, password: str, timeout_s: int = 120) -> bool:
    if GraphDatabase is None:
        return False
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        driver = None
        try:
            driver = GraphDatabase.driver(uri, auth=(user, password))
            driver.verify_connectivity()
            return True
        except Exception:
            time.sleep(2.0)
        finally:
            if driver is not None:
                driver.close()
    return False


def _acquire_neo4j() -> tuple[dict[str, str] | None, str | None]:
    env_uri = os.getenv("TEST_NEO4J_URI")
    env_user = os.getenv("TEST_NEO4J_USER", "neo4j")
    env_password = os.getenv("TEST_NEO4J_PASSWORD")
    if env_uri and env_password:
        if _wait_for_neo4j(env_uri, env_user, env_password, timeout_s=20):
            return {"uri": env_uri, "user": env_user, "password": env_password}, None
        return None, "configured TEST_NEO4J_URI is not reachable"

    if shutil.which("docker") is None:
        return None, "docker not available and TEST_NEO4J_URI is not configured"

    image = os.getenv("TEST_NEO4J_IMAGE", "neo4j:5-community")
    container_name = f"ragtrial-neo4j-e2e-{uuid.uuid4().hex[:8]}"
    bolt_port = _free_port()
    http_port = _free_port()
    password = "test-password-123"
    run_result = subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--rm",
            "--name",
            container_name,
            "-e",
            f"NEO4J_AUTH=neo4j/{password}",
            "-p",
            f"{bolt_port}:7687",
            "-p",
            f"{http_port}:7474",
            image,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if run_result.returncode != 0:
        msg = run_result.stderr.strip() or run_result.stdout.strip() or "docker run failed"
        return None, msg

    uri = f"bolt://127.0.0.1:{bolt_port}"
    if not _wait_for_neo4j(uri, "neo4j", password):
        subprocess.run(
            ["docker", "rm", "-f", container_name],
            capture_output=True,
            text=True,
            check=False,
        )
        return None, "ephemeral Neo4j container failed to become ready"

    return {
        "uri": uri,
        "user": "neo4j",
        "password": password,
        "container_name": container_name,
    }, None


def _release_neo4j(config: dict[str, str] | None) -> None:
    if not config:
        return
    container_name = config.get("container_name")
    if not container_name:
        return
    subprocess.run(
        ["docker", "rm", "-f", container_name],
        capture_output=True,
        text=True,
        check=False,
    )


def _build_tgrag_stub() -> types.ModuleType:
    module = types.ModuleType("tgrag_setup")

    def run_sqlite_pass(**kwargs) -> None:
        _ = kwargs
        return None

    module.run_sqlite_pass = run_sqlite_pass
    return module


def _build_macro_stub(db_path: Path) -> types.ModuleType:
    module = types.ModuleType("macro_extract")

    def run_extraction(limit: int | None = None) -> dict[str, int | None]:
        _ = limit
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute(
            """
            INSERT OR REPLACE INTO articles (
                article_id, url, title, source, source_provider, source_trust_tier,
                article_quality_score, raw_text, published_at, processing_state
            ) VALUES (
                'article-1', 'https://example.com/e2e', 'E2E macro article', 'Reuters', 'wire', 'tier_1',
                0.96, 'macro text', '2026-04-24T08:00:00+00:00', 'processed'
            )
            """
        )
        conn.execute(
            """
            INSERT OR REPLACE INTO chunks (
                chunk_id, article_id, chunk_index, text, published_date, period_key, processing_state
            ) VALUES (
                'chunk-1', 'article-1', 0, 'Oil supply shock tightened balances.', '2026-04-24', '2026-04', 'processed'
            )
            """
        )
        conn.execute(
            """
            INSERT OR REPLACE INTO macro_extraction_runs (
                run_id, article_id, chunk_id, model_provider, model_name, prompt_version, schema_version,
                created_at, success, raw_json
            ) VALUES (
                'run-e2e', 'article-1', 'chunk-1', 'test', 'test', 'v1', 'v1',
                '2026-04-24T08:01:00+00:00', 1, '{}'
            )
            """
        )
        conn.execute(
            """
            INSERT OR REPLACE INTO macro_events (
                macro_event_id, run_id, article_id, chunk_id, event_type, summary, region, time_horizon,
                event_time_start, event_time_end, confidence, verification_status, support_score, novelty_hint,
                urgency, market_surprise
            ) VALUES (
                'event-e2e-1', 'run-e2e', 'article-1', 'chunk-1', 'commodity_supply_disruption',
                'Oil supply shock tightened balances.', 'global', 'near_term',
                '2026-04-24', '2026-04-24', 0.88, 'verified', 0.9, 'new', 'high', 'high'
            )
            """
        )
        conn.execute(
            """
            INSERT OR REPLACE INTO macro_event_shock_types (macro_event_id, shock_type)
            VALUES ('event-e2e-1', 'commodity_supply_disruption')
            """
        )
        conn.execute(
            """
            INSERT OR REPLACE INTO asset_impacts (
                impact_id, macro_event_id, target_type, target_id, direction, strength, horizon, confidence, rationale
            ) VALUES (
                'impact-e2e-1', 'event-e2e-1', 'ticker', 'XOM', 'up', 'strong', 'near_term', 0.9, 'supply shock'
            )
            """
        )
        conn.commit()
        conn.close()
        return {"events_written": 1}

    module.run_extraction = run_extraction
    return module


@unittest.skipIf(NEO4J_IMPORT_ERROR is not None, f"neo4j import failed: {NEO4J_IMPORT_ERROR}")
class TestUpdateV2Integration(unittest.TestCase):
    def test_update_v2_full_orchestration_with_real_neo4j(self) -> None:
        neo4j_config, skip_reason = _acquire_neo4j()
        if neo4j_config is None:
            self.skipTest(skip_reason or "no Neo4j instance available")

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            db_path = tmp_path / "my_database.db"
            create_database(str(db_path))

            tgrag_stub = _build_tgrag_stub()
            macro_stub = _build_macro_stub(db_path)
            original_cwd = Path.cwd()
            os.chdir(tmp_path)
            try:
                with mock.patch.dict(
                    os.environ,
                    {
                        "NEO4J_URI": neo4j_config["uri"],
                        "NEO4J_USER": neo4j_config["user"],
                        "NEO4J_PASSWORD": neo4j_config["password"],
                        "SQLITE_DB": str(db_path),
                    },
                    clear=False,
                ):
                    sys.modules.pop("graph_schema", None)
                    sys.modules.pop("neo4j_sync", None)
                    with mock.patch.dict(
                        sys.modules,
                        {
                            "tgrag_setup": tgrag_stub,
                            "macro_extract": macro_stub,
                        },
                    ):
                        from rag_trial.ingestion import update_v2

                        importlib.reload(update_v2)
                        with mock.patch.object(
                            sys,
                            "argv",
                            [
                                "update_v2.py",
                                "--no-scrape",
                                "--cluster-window-days",
                                "7",
                                "--signal-limit",
                                "1",
                                "--wipe-neo4j",
                            ],
                        ):
                            update_v2.main()

                conn = sqlite3.connect(db_path)
                signal_row = conn.execute(
                    """
                    SELECT signal_id, cluster_id, status
                    FROM signal_alerts
                    WHERE status = 'active'
                    LIMIT 1
                    """
                ).fetchone()
                cluster_count = conn.execute("SELECT COUNT(*) FROM event_clusters").fetchone()[0]
                score_count = conn.execute("SELECT COUNT(*) FROM event_cluster_scores").fetchone()[0]
                conn.close()

                self.assertIsNotNone(signal_row)
                self.assertGreaterEqual(cluster_count, 1)
                self.assertGreaterEqual(score_count, 1)

                driver = GraphDatabase.driver(
                    neo4j_config["uri"],
                    auth=(neo4j_config["user"], neo4j_config["password"]),
                )
                try:
                    with driver.session() as session:
                        macro_nodes = session.run(
                            "MATCH (m:MacroEvent) RETURN count(m) AS n"
                        ).single()["n"]
                        cluster_nodes = session.run(
                            "MATCH (c:EventCluster) RETURN count(c) AS n"
                        ).single()["n"]
                        signal_nodes = session.run(
                            "MATCH (s:Signal) RETURN count(s) AS n"
                        ).single()["n"]
                finally:
                    driver.close()

                self.assertGreaterEqual(int(macro_nodes), 1)
                self.assertGreaterEqual(int(cluster_nodes), 1)
                self.assertGreaterEqual(int(signal_nodes), 1)
            finally:
                os.chdir(original_cwd)
                _release_neo4j(neo4j_config)


if __name__ == "__main__":
    unittest.main()
