from __future__ import annotations

import sqlite3


SCHEMA_SQL = """
    CREATE TABLE IF NOT EXISTS articles (
        article_id TEXT PRIMARY KEY,
        url TEXT NOT NULL,
        title TEXT,
        source TEXT,
        source_rss TEXT,
        source_provider TEXT,
        published_at TEXT,
        scraped_at_utc TEXT,
        content_hash TEXT,
        status TEXT,
        raw_text TEXT NOT NULL,
        source_trust_tier TEXT DEFAULT 'tier_3',
        content_class TEXT DEFAULT 'news_report',
        article_quality_score REAL,
        quality_flags_json TEXT,
        processing_state TEXT NOT NULL DEFAULT 'ingested',
        neo4j_synced_at TEXT
    );

    CREATE TABLE IF NOT EXISTS chunks (
        chunk_id TEXT PRIMARY KEY,
        article_id TEXT NOT NULL,
        chunk_index INTEGER NOT NULL,
        text TEXT NOT NULL,
        token_count INTEGER,
        published_date TEXT,
        period_key TEXT,
        embedding_json TEXT,
        topic_labels_json TEXT,
        theme_id TEXT,
        theme_label TEXT,
        theme_confidence REAL,
        processing_state TEXT NOT NULL DEFAULT 'chunked',
        neo4j_synced_at TEXT,
        FOREIGN KEY(article_id) REFERENCES articles(article_id)
    );

    CREATE TABLE IF NOT EXISTS entity_mentions (
        mention_id TEXT PRIMARY KEY,
        chunk_id TEXT NOT NULL,
        article_id TEXT NOT NULL,
        canonical_entity_id TEXT NOT NULL,
        display_name TEXT,
        entity_type TEXT,
        ticker TEXT,
        mention_text TEXT,
        confidence REAL,
        FOREIGN KEY(chunk_id) REFERENCES chunks(chunk_id),
        FOREIGN KEY(article_id) REFERENCES articles(article_id)
    );

    CREATE TABLE IF NOT EXISTS theme_mentions (
        theme_mention_id TEXT PRIMARY KEY,
        chunk_id TEXT NOT NULL,
        article_id TEXT NOT NULL,
        canonical_theme TEXT NOT NULL,
        mention_text TEXT,
        confidence REAL,
        FOREIGN KEY(chunk_id) REFERENCES chunks(chunk_id),
        FOREIGN KEY(article_id) REFERENCES articles(article_id)
    );

    CREATE TABLE IF NOT EXISTS macro_extraction_runs (
        run_id TEXT PRIMARY KEY,
        article_id TEXT,
        chunk_id TEXT,
        model_provider TEXT,
        model_name TEXT,
        prompt_version TEXT,
        schema_version TEXT,
        created_at TEXT NOT NULL,
        success INTEGER NOT NULL,
        raw_json TEXT,
        error_text TEXT
    );

    CREATE TABLE IF NOT EXISTS macro_events (
        macro_event_id TEXT PRIMARY KEY,
        run_id TEXT NOT NULL,
        article_id TEXT NOT NULL,
        chunk_id TEXT,
        event_type TEXT NOT NULL,
        summary TEXT NOT NULL,
        region TEXT,
        time_horizon TEXT,
        event_time_start TEXT,
        event_time_end TEXT,
        confidence REAL,
        verification_status TEXT,
        support_score REAL,
        novelty_hint TEXT,
        urgency TEXT,
        market_surprise TEXT,
        FOREIGN KEY(run_id) REFERENCES macro_extraction_runs(run_id),
        FOREIGN KEY(article_id) REFERENCES articles(article_id),
        FOREIGN KEY(chunk_id) REFERENCES chunks(chunk_id)
    );

    CREATE TABLE IF NOT EXISTS macro_event_shock_types (
        macro_event_id TEXT NOT NULL,
        shock_type TEXT NOT NULL,
        PRIMARY KEY (macro_event_id, shock_type),
        FOREIGN KEY(macro_event_id) REFERENCES macro_events(macro_event_id)
    );

    CREATE TABLE IF NOT EXISTS macro_channels (
        macro_channel_id TEXT PRIMARY KEY,
        macro_event_id TEXT NOT NULL,
        channel_name TEXT NOT NULL,
        direction TEXT NOT NULL,
        strength TEXT NOT NULL,
        confidence REAL,
        FOREIGN KEY(macro_event_id) REFERENCES macro_events(macro_event_id)
    );

    CREATE TABLE IF NOT EXISTS asset_impacts (
        impact_id TEXT PRIMARY KEY,
        macro_event_id TEXT NOT NULL,
        target_type TEXT NOT NULL,
        target_id TEXT NOT NULL,
        direction TEXT NOT NULL,
        strength TEXT NOT NULL,
        horizon TEXT,
        confidence REAL,
        rationale TEXT,
        FOREIGN KEY(macro_event_id) REFERENCES macro_events(macro_event_id)
    );

    CREATE TABLE IF NOT EXISTS evidence_spans (
        evidence_id TEXT PRIMARY KEY,
        run_id TEXT NOT NULL,
        article_id TEXT NOT NULL,
        chunk_id TEXT,
        parent_kind TEXT NOT NULL,
        parent_id TEXT NOT NULL,
        evidence_text TEXT NOT NULL,
        FOREIGN KEY(run_id) REFERENCES macro_extraction_runs(run_id),
        FOREIGN KEY(article_id) REFERENCES articles(article_id),
        FOREIGN KEY(chunk_id) REFERENCES chunks(chunk_id)
    );

    CREATE TABLE IF NOT EXISTS macro_processing_audit (
        audit_id TEXT PRIMARY KEY,
        run_id TEXT,
        article_id TEXT NOT NULL,
        chunk_id TEXT NOT NULL,
        created_at TEXT NOT NULL,
        stage TEXT NOT NULL,
        status TEXT NOT NULL,
        failure_reason TEXT,
        queue_name TEXT,
        chunk_macro_score INTEGER,
        was_hard_include INTEGER NOT NULL DEFAULT 0,
        event_count INTEGER,
        suspicious INTEGER NOT NULL DEFAULT 0,
        review_reasons_json TEXT,
        raw_response_excerpt TEXT,
        FOREIGN KEY(run_id) REFERENCES macro_extraction_runs(run_id),
        FOREIGN KEY(article_id) REFERENCES articles(article_id),
        FOREIGN KEY(chunk_id) REFERENCES chunks(chunk_id)
    );

    CREATE TABLE IF NOT EXISTS macro_enum_audit (
        audit_id TEXT PRIMARY KEY,
        run_id TEXT NOT NULL,
        macro_event_index INTEGER,
        parent_kind TEXT NOT NULL,
        field_label TEXT NOT NULL,
        raw_value TEXT,
        normalized_value TEXT,
        action TEXT NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY(run_id) REFERENCES macro_extraction_runs(run_id)
    );

    CREATE TABLE IF NOT EXISTS macro_event_candidates (
        candidate_id TEXT PRIMARY KEY,
        run_id TEXT,
        article_id TEXT NOT NULL,
        chunk_id TEXT,
        macro_event_index INTEGER,
        event_type TEXT,
        summary TEXT,
        region TEXT,
        time_horizon TEXT,
        evidence_text TEXT,
        evidence_span_json TEXT,
        evidence_spans_json TEXT,
        confidence_raw REAL,
        confidence_candidate REAL,
        initial_confidence REAL,
        confidence_initial REAL,
        candidate_json TEXT,
        raw_candidate_json TEXT,
        novelty_hint TEXT,
        urgency TEXT,
        market_surprise TEXT,
        extraction_pass TEXT,
        source_trust_tier TEXT,
        content_class TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY(run_id) REFERENCES macro_extraction_runs(run_id),
        FOREIGN KEY(article_id) REFERENCES articles(article_id),
        FOREIGN KEY(chunk_id) REFERENCES chunks(chunk_id)
    );

    CREATE TABLE IF NOT EXISTS macro_event_verifications (
        verification_id TEXT PRIMARY KEY,
        candidate_id TEXT NOT NULL,
        run_id TEXT,
        article_id TEXT NOT NULL,
        chunk_id TEXT,
        macro_event_index INTEGER,
        verification_status TEXT NOT NULL,
        support_score REAL,
        confidence_initial REAL,
        confidence_calibrated REAL,
        evidence_span_valid INTEGER,
        evidence_spans_count INTEGER,
        matched_spans_count INTEGER,
        rejection_reason TEXT,
        verifier_notes_json TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY(candidate_id) REFERENCES macro_event_candidates(candidate_id),
        FOREIGN KEY(run_id) REFERENCES macro_extraction_runs(run_id),
        FOREIGN KEY(article_id) REFERENCES articles(article_id),
        FOREIGN KEY(chunk_id) REFERENCES chunks(chunk_id)
    );

    CREATE TABLE IF NOT EXISTS event_clusters (
        cluster_id TEXT PRIMARY KEY,
        event_type TEXT,
        primary_shock_type TEXT,
        region TEXT,
        canonical_summary TEXT NOT NULL,
        summary_embedding_json TEXT,
        first_event_time TEXT,
        last_event_time TEXT,
        cluster_window_days INTEGER,
        member_count INTEGER NOT NULL DEFAULT 0,
        unique_source_count INTEGER NOT NULL DEFAULT 0,
        asset_targets_json TEXT,
        cluster_status TEXT NOT NULL DEFAULT 'active',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS cluster_members (
        cluster_id TEXT NOT NULL,
        macro_event_id TEXT NOT NULL,
        similarity_score REAL,
        match_reasons_json TEXT,
        event_time TEXT,
        article_id TEXT,
        chunk_id TEXT,
        source TEXT,
        created_at TEXT NOT NULL,
        PRIMARY KEY (cluster_id, macro_event_id),
        FOREIGN KEY(cluster_id) REFERENCES event_clusters(cluster_id),
        FOREIGN KEY(macro_event_id) REFERENCES macro_events(macro_event_id),
        FOREIGN KEY(article_id) REFERENCES articles(article_id),
        FOREIGN KEY(chunk_id) REFERENCES chunks(chunk_id)
    );

    CREATE TABLE IF NOT EXISTS event_cluster_scores (
        score_id TEXT PRIMARY KEY,
        cluster_id TEXT NOT NULL,
        score_date TEXT NOT NULL,
        novelty_score REAL,
        source_quality_score REAL,
        velocity_score REAL,
        asset_impact_score REAL,
        confidence_score REAL,
        recency_score REAL,
        signal_score REAL,
        supporting_event_count INTEGER,
        supporting_source_count INTEGER,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        FOREIGN KEY(cluster_id) REFERENCES event_clusters(cluster_id)
    );

    CREATE TABLE IF NOT EXISTS signal_alerts (
        signal_id TEXT PRIMARY KEY,
        cluster_id TEXT NOT NULL,
        score_id TEXT,
        signal_date TEXT NOT NULL,
        rank INTEGER,
        signal_score REAL NOT NULL,
        headline TEXT,
        summary TEXT,
        novelty_hint TEXT,
        urgency TEXT,
        market_surprise TEXT,
        top_assets_json TEXT,
        status TEXT NOT NULL DEFAULT 'active',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        FOREIGN KEY(cluster_id) REFERENCES event_clusters(cluster_id),
        FOREIGN KEY(score_id) REFERENCES event_cluster_scores(score_id)
    );

    CREATE TABLE IF NOT EXISTS market_reactions (
        reaction_id TEXT PRIMARY KEY,
        signal_id TEXT,
        cluster_id TEXT,
        macro_event_id TEXT,
        target_type TEXT,
        target_id TEXT,
        event_time TEXT NOT NULL,
        predicted_direction TEXT,
        return_1d REAL,
        return_3d REAL,
        return_5d REAL,
        outcome_label TEXT,
        data_availability_status TEXT NOT NULL,
        notes_json TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY(signal_id) REFERENCES signal_alerts(signal_id),
        FOREIGN KEY(cluster_id) REFERENCES event_clusters(cluster_id),
        FOREIGN KEY(macro_event_id) REFERENCES macro_events(macro_event_id)
    );

    CREATE TABLE IF NOT EXISTS source_quality (
        source_name TEXT PRIMARY KEY,
        source_provider TEXT,
        source_trust_tier TEXT,
        quality_score REAL,
        reliability_label TEXT,
        article_count INTEGER,
        last_article_at TEXT,
        notes_json TEXT,
        updated_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS qa_runs (
        run_id TEXT PRIMARY KEY,
        query TEXT NOT NULL,
        route_type TEXT,
        route_reason TEXT,
        route_decision_json TEXT,
        resolved_target_json TEXT,
        retrieval_trace_json TEXT,
        selected_chunks_json TEXT,
        selected_macro_events_json TEXT,
        selected_signals_json TEXT,
        selected_signal_alerts_json TEXT,
        answer_confidence REAL,
        decision TEXT,
        answer_meta_json TEXT,
        answer_decision_json TEXT,
        latency REAL,
        created_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS retrieval_candidates (
        run_id TEXT NOT NULL,
        candidate_id TEXT NOT NULL,
        candidate_kind TEXT NOT NULL,
        article_id TEXT,
        chunk_id TEXT,
        macro_event_id TEXT,
        cluster_id TEXT,
        signal_id TEXT,
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
        PRIMARY KEY (run_id, candidate_id),
        FOREIGN KEY(run_id) REFERENCES qa_runs(run_id),
        FOREIGN KEY(article_id) REFERENCES articles(article_id),
        FOREIGN KEY(chunk_id) REFERENCES chunks(chunk_id),
        FOREIGN KEY(macro_event_id) REFERENCES macro_events(macro_event_id),
        FOREIGN KEY(cluster_id) REFERENCES event_clusters(cluster_id),
        FOREIGN KEY(signal_id) REFERENCES signal_alerts(signal_id)
    );
"""


INDEX_SQL = """
    CREATE INDEX IF NOT EXISTS idx_articles_published_at ON articles(published_at);
    CREATE INDEX IF NOT EXISTS idx_articles_source_trust_tier ON articles(source_trust_tier);
    CREATE INDEX IF NOT EXISTS idx_articles_source_provider ON articles(source_provider);
    CREATE INDEX IF NOT EXISTS idx_articles_content_class ON articles(content_class);
    CREATE INDEX IF NOT EXISTS idx_articles_processing_state ON articles(processing_state);
    CREATE INDEX IF NOT EXISTS idx_articles_state_published ON articles(processing_state, published_at);

    CREATE INDEX IF NOT EXISTS idx_chunks_article_id ON chunks(article_id);
    CREATE INDEX IF NOT EXISTS idx_chunks_period_key ON chunks(period_key);
    CREATE INDEX IF NOT EXISTS idx_chunks_published_date ON chunks(published_date);
    CREATE INDEX IF NOT EXISTS idx_chunks_processing_state ON chunks(processing_state);
    CREATE INDEX IF NOT EXISTS idx_chunks_theme_id ON chunks(theme_id);

    CREATE INDEX IF NOT EXISTS idx_entity_mentions_chunk_id ON entity_mentions(chunk_id);
    CREATE INDEX IF NOT EXISTS idx_entity_mentions_article_id ON entity_mentions(article_id);
    CREATE INDEX IF NOT EXISTS idx_theme_mentions_chunk_id ON theme_mentions(chunk_id);
    CREATE INDEX IF NOT EXISTS idx_theme_mentions_article_id ON theme_mentions(article_id);

    CREATE INDEX IF NOT EXISTS idx_macro_events_run_id ON macro_events(run_id);
    CREATE INDEX IF NOT EXISTS idx_macro_events_article_id ON macro_events(article_id);
    CREATE INDEX IF NOT EXISTS idx_macro_events_chunk_id ON macro_events(chunk_id);
    CREATE INDEX IF NOT EXISTS idx_macro_events_event_type ON macro_events(event_type);
    CREATE INDEX IF NOT EXISTS idx_macro_events_event_time_start ON macro_events(event_time_start);
    CREATE INDEX IF NOT EXISTS idx_macro_events_event_time_end ON macro_events(event_time_end);
    CREATE INDEX IF NOT EXISTS idx_macro_events_verification_status ON macro_events(verification_status);

    CREATE INDEX IF NOT EXISTS idx_asset_impacts_macro_event_id ON asset_impacts(macro_event_id);
    CREATE INDEX IF NOT EXISTS idx_asset_impacts_target ON asset_impacts(target_type, target_id);
    CREATE INDEX IF NOT EXISTS idx_evidence_spans_run_id ON evidence_spans(run_id);
    CREATE INDEX IF NOT EXISTS idx_evidence_spans_parent ON evidence_spans(parent_kind, parent_id);

    CREATE INDEX IF NOT EXISTS idx_macro_processing_audit_chunk_id ON macro_processing_audit(chunk_id);
    CREATE INDEX IF NOT EXISTS idx_macro_processing_audit_run_id ON macro_processing_audit(run_id);
    CREATE INDEX IF NOT EXISTS idx_macro_processing_audit_queue_name ON macro_processing_audit(queue_name);
    CREATE INDEX IF NOT EXISTS idx_macro_processing_audit_status ON macro_processing_audit(status);
    CREATE INDEX IF NOT EXISTS idx_macro_enum_audit_run_id ON macro_enum_audit(run_id);
    CREATE INDEX IF NOT EXISTS idx_macro_enum_audit_action ON macro_enum_audit(action);

    CREATE INDEX IF NOT EXISTS idx_macro_event_candidates_run_id ON macro_event_candidates(run_id);
    CREATE INDEX IF NOT EXISTS idx_macro_event_candidates_article_id ON macro_event_candidates(article_id);
    CREATE INDEX IF NOT EXISTS idx_macro_event_candidates_chunk_id ON macro_event_candidates(chunk_id);
    CREATE INDEX IF NOT EXISTS idx_macro_event_candidates_event_type ON macro_event_candidates(event_type);
    CREATE INDEX IF NOT EXISTS idx_macro_event_candidates_created_at ON macro_event_candidates(created_at);

    CREATE INDEX IF NOT EXISTS idx_macro_event_verifications_candidate_id ON macro_event_verifications(candidate_id);
    CREATE INDEX IF NOT EXISTS idx_macro_event_verifications_article_id ON macro_event_verifications(article_id);
    CREATE INDEX IF NOT EXISTS idx_macro_event_verifications_status ON macro_event_verifications(verification_status);
    CREATE INDEX IF NOT EXISTS idx_macro_event_verifications_created_at ON macro_event_verifications(created_at);

    CREATE INDEX IF NOT EXISTS idx_event_clusters_event_type ON event_clusters(event_type);
    CREATE INDEX IF NOT EXISTS idx_event_clusters_primary_shock_type ON event_clusters(primary_shock_type);
    CREATE INDEX IF NOT EXISTS idx_event_clusters_region ON event_clusters(region);
    CREATE INDEX IF NOT EXISTS idx_event_clusters_last_event_time ON event_clusters(last_event_time);
    CREATE INDEX IF NOT EXISTS idx_cluster_members_cluster_id ON cluster_members(cluster_id);
    CREATE INDEX IF NOT EXISTS idx_cluster_members_macro_event_id ON cluster_members(macro_event_id);
    CREATE INDEX IF NOT EXISTS idx_cluster_members_event_time ON cluster_members(event_time);

    CREATE INDEX IF NOT EXISTS idx_event_cluster_scores_cluster_id ON event_cluster_scores(cluster_id);
    CREATE INDEX IF NOT EXISTS idx_event_cluster_scores_score_date ON event_cluster_scores(score_date);
    CREATE INDEX IF NOT EXISTS idx_event_cluster_scores_signal_score ON event_cluster_scores(signal_score DESC);

    CREATE INDEX IF NOT EXISTS idx_signal_alerts_cluster_id ON signal_alerts(cluster_id);
    CREATE INDEX IF NOT EXISTS idx_signal_alerts_score_id ON signal_alerts(score_id);
    CREATE INDEX IF NOT EXISTS idx_signal_alerts_signal_date ON signal_alerts(signal_date);
    CREATE INDEX IF NOT EXISTS idx_signal_alerts_signal_score ON signal_alerts(signal_score DESC);

    CREATE INDEX IF NOT EXISTS idx_market_reactions_cluster_id ON market_reactions(cluster_id);
    CREATE INDEX IF NOT EXISTS idx_market_reactions_macro_event_id ON market_reactions(macro_event_id);
    CREATE INDEX IF NOT EXISTS idx_market_reactions_signal_id ON market_reactions(signal_id);
    CREATE INDEX IF NOT EXISTS idx_market_reactions_target ON market_reactions(target_type, target_id);
    CREATE INDEX IF NOT EXISTS idx_market_reactions_event_time ON market_reactions(event_time);

    CREATE INDEX IF NOT EXISTS idx_source_quality_provider ON source_quality(source_provider);
    CREATE INDEX IF NOT EXISTS idx_source_quality_score ON source_quality(quality_score DESC);
    CREATE INDEX IF NOT EXISTS idx_source_quality_last_article_at ON source_quality(last_article_at);

    CREATE INDEX IF NOT EXISTS idx_qa_runs_created_at ON qa_runs(created_at);
    CREATE INDEX IF NOT EXISTS idx_qa_runs_route_type ON qa_runs(route_type);
    CREATE INDEX IF NOT EXISTS idx_qa_runs_decision ON qa_runs(decision);

    CREATE INDEX IF NOT EXISTS idx_retrieval_candidates_run_id ON retrieval_candidates(run_id);
    CREATE INDEX IF NOT EXISTS idx_retrieval_candidates_selected ON retrieval_candidates(selected);
    CREATE INDEX IF NOT EXISTS idx_retrieval_candidates_macro_event_id ON retrieval_candidates(macro_event_id);
    CREATE INDEX IF NOT EXISTS idx_retrieval_candidates_cluster_id ON retrieval_candidates(cluster_id);
    CREATE INDEX IF NOT EXISTS idx_retrieval_candidates_signal_id ON retrieval_candidates(signal_id);
    CREATE INDEX IF NOT EXISTS idx_retrieval_candidates_run_rank ON retrieval_candidates(run_id, final_score DESC);
"""


REQUIRED_COLUMNS: dict[str, dict[str, str]] = {
    "articles": {
        "source_provider": "TEXT",
        "source_trust_tier": "TEXT DEFAULT 'tier_3'",
        "content_class": "TEXT DEFAULT 'news_report'",
        "article_quality_score": "REAL",
        "quality_flags_json": "TEXT",
        "processing_state": "TEXT NOT NULL DEFAULT 'ingested'",
        "neo4j_synced_at": "TEXT",
    },
    "chunks": {
        "topic_labels_json": "TEXT",
        "theme_id": "TEXT",
        "theme_label": "TEXT",
        "theme_confidence": "REAL",
        "processing_state": "TEXT NOT NULL DEFAULT 'chunked'",
        "neo4j_synced_at": "TEXT",
    },
    "macro_events": {
        "event_time_start": "TEXT",
        "event_time_end": "TEXT",
        "verification_status": "TEXT",
        "support_score": "REAL",
        "novelty_hint": "TEXT",
        "urgency": "TEXT",
        "market_surprise": "TEXT",
    },
    "macro_event_candidates": {
        "macro_event_index": "INTEGER",
        "region": "TEXT",
        "time_horizon": "TEXT",
        "evidence_spans_json": "TEXT",
        "initial_confidence": "REAL",
        "confidence_initial": "REAL",
        "candidate_json": "TEXT",
        "raw_candidate_json": "TEXT",
        "novelty_hint": "TEXT",
        "urgency": "TEXT",
        "market_surprise": "TEXT",
    },
    "macro_event_verifications": {
        "macro_event_index": "INTEGER",
        "confidence_initial": "REAL",
        "evidence_span_valid": "INTEGER",
        "evidence_spans_count": "INTEGER",
        "matched_spans_count": "INTEGER",
    },
    "qa_runs": {
        "route_reason": "TEXT",
        "route_decision_json": "TEXT",
        "selected_signals_json": "TEXT",
        "selected_signal_alerts_json": "TEXT",
        "answer_meta_json": "TEXT",
        "answer_decision_json": "TEXT",
    },
    "retrieval_candidates": {
        "cluster_id": "TEXT",
        "signal_id": "TEXT",
    },
}


def connect_sqlite(db_path: str = "my_database.db", *, fk: bool = True) -> sqlite3.Connection:
    """Open a SQLite connection with consistent settings."""
    conn = sqlite3.connect(db_path)
    if fk:
        conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def _table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    if not _table_exists(conn, table_name):
        return set()
    return {
        str(row[1])
        for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    }


def _ensure_required_columns(conn: sqlite3.Connection) -> None:
    """Backfill columns that may be missing in legacy databases."""
    for table, columns in REQUIRED_COLUMNS.items():
        if not _table_exists(conn, table):
            continue
        existing = _table_columns(conn, table)
        for column, column_type in columns.items():
            if column in existing:
                continue
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")
            print(f"[migration] added {table}.{column}")


def _ensure_schema_objects(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_SQL)
    _ensure_required_columns(conn)
    conn.executescript(INDEX_SQL)


def create_database(db_path: str = "my_database.db") -> None:
    conn = connect_sqlite(db_path)
    _ensure_schema_objects(conn)
    conn.commit()
    conn.close()
    print(f"Database created successfully at: {db_path}")


def ensure_migrations(db_path: str = "my_database.db") -> None:
    """Apply schema migrations needed for existing databases."""
    conn = connect_sqlite(db_path)
    _ensure_schema_objects(conn)
    conn.commit()
    conn.close()


if __name__ == "__main__":
    create_database()
