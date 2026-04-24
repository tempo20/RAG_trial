from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from create_sql_db import connect_sqlite, create_database


EmbeddingFn = Callable[[list[str]], list[list[float]]]


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _md5(value: str) -> str:
    return hashlib.md5(value.encode("utf-8")).hexdigest()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    text = text.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S"):
            try:
                dt = datetime.strptime(text, fmt)
                break
            except ValueError:
                dt = None
        if dt is None:
            return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _summary_tokens(text: str) -> set[str]:
    cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in (text or ""))
    return {token for token in cleaned.split() if len(token) > 2}


def _cosine_similarity(left: list[float] | None, right: list[float] | None) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / (left_norm * right_norm)))


def _lexical_similarity(left: str, right: str) -> float:
    left_tokens = _summary_tokens(left)
    right_tokens = _summary_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = len(left_tokens & right_tokens)
    union = len(left_tokens | right_tokens)
    return overlap / union if union else 0.0


def _region_compatible(left: str | None, right: str | None) -> bool:
    lval = (left or "").strip().lower()
    rval = (right or "").strip().lower()
    if not lval or not rval:
        return True
    if lval == rval:
        return True
    return "global" in {lval, rval}


def _event_time(record: dict[str, Any]) -> datetime | None:
    return (
        _parse_dt(record.get("event_time_start"))
        or _parse_dt(record.get("event_time_end"))
        or _parse_dt(record.get("published_date"))
        or _parse_dt(record.get("event_time"))
    )


def _time_window_ok(event: dict[str, Any], cluster: dict[str, Any], window_days: int) -> bool:
    event_dt = _event_time(event)
    if event_dt is None:
        return True
    cluster_end = _parse_dt(cluster.get("last_event_time")) or _parse_dt(cluster.get("event_time"))
    if cluster_end is None:
        return True
    return abs((event_dt - cluster_end).days) <= int(window_days)


def _same_type_or_shock_overlap(event: dict[str, Any], cluster: dict[str, Any]) -> bool:
    event_type = (event.get("event_type") or "").strip().lower()
    cluster_type = (cluster.get("event_type") or "").strip().lower()
    if event_type and cluster_type and event_type == cluster_type:
        return True
    event_shocks = {str(item).strip().lower() for item in event.get("shock_types", []) if str(item).strip()}
    cluster_shocks = {str(item).strip().lower() for item in cluster.get("shock_types", []) if str(item).strip()}
    primary = (cluster.get("primary_shock_type") or "").strip().lower()
    if primary:
        cluster_shocks.add(primary)
    return bool(event_shocks & cluster_shocks)


def _asset_overlap(event: dict[str, Any], cluster: dict[str, Any]) -> set[str]:
    left = {str(item).strip() for item in event.get("asset_targets", []) if str(item).strip()}
    right = {str(item).strip() for item in cluster.get("asset_targets", []) if str(item).strip()}
    return left & right


def _default_embedding_fn(texts: list[str]) -> list[list[float]]:
    from tgrag_setup import embed_texts

    return embed_texts(texts)


def compute_event_similarity(event_a: dict[str, Any], event_b: dict[str, Any]) -> float:
    """Compute a deterministic similarity score between two event-like records."""
    embedding_score = _cosine_similarity(
        event_a.get("embedding"),
        event_b.get("embedding"),
    )
    lexical_score = _lexical_similarity(
        str(event_a.get("summary") or ""),
        str(event_b.get("summary") or ""),
    )
    base_similarity = max(embedding_score, lexical_score)
    bonus = 0.0
    if (event_a.get("event_type") or "").strip().lower() == (event_b.get("event_type") or "").strip().lower():
        bonus += 0.03
    if _same_type_or_shock_overlap(event_a, event_b):
        bonus += 0.03
    if _region_compatible(event_a.get("region"), event_b.get("region")):
        bonus += 0.02
    if _asset_overlap(event_a, event_b):
        bonus += 0.05
    return min(1.0, base_similarity + bonus)


def _load_lookup_rows(conn, sql: str, ids: list[str]) -> dict[str, list[str]]:
    if not ids:
        return {}
    placeholders = ",".join("?" for _ in ids)
    rows = conn.execute(sql.format(placeholders=placeholders), ids).fetchall()
    lookup: dict[str, list[str]] = {}
    for left, right in rows:
        lookup.setdefault(str(left), []).append(str(right))
    return lookup


def load_unclustered_macro_events(
    conn,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Load verified macro events that have not yet been assigned to a cluster."""
    sql = """
        SELECT
            m.macro_event_id,
            m.run_id,
            m.article_id,
            m.chunk_id,
            m.event_type,
            m.summary,
            m.region,
            m.time_horizon,
            m.event_time_start,
            m.event_time_end,
            m.confidence,
            m.support_score,
            m.novelty_hint,
            m.urgency,
            m.market_surprise,
            c.published_date,
            a.source
        FROM macro_events m
        LEFT JOIN cluster_members cm ON cm.macro_event_id = m.macro_event_id
        LEFT JOIN chunks c ON c.chunk_id = m.chunk_id
        LEFT JOIN articles a ON a.article_id = m.article_id
        WHERE cm.macro_event_id IS NULL
        ORDER BY coalesce(m.event_time_start, m.event_time_end, c.published_date) DESC,
                 m.macro_event_id DESC
    """
    params: list[Any] = []
    if limit is not None and limit > 0:
        sql += " LIMIT ?"
        params.append(limit)
    rows = [dict(row) for row in conn.execute(sql, params).fetchall()]
    event_ids = [row["macro_event_id"] for row in rows]
    shock_lookup = _load_lookup_rows(
        conn,
        """
        SELECT macro_event_id, shock_type
        FROM macro_event_shock_types
        WHERE macro_event_id IN ({placeholders})
        ORDER BY macro_event_id, shock_type
        """,
        event_ids,
    )
    asset_lookup = _load_lookup_rows(
        conn,
        """
        SELECT macro_event_id, target_type || ':' || target_id
        FROM asset_impacts
        WHERE macro_event_id IN ({placeholders})
        ORDER BY macro_event_id, impact_id
        """,
        event_ids,
    )
    for row in rows:
        row["shock_types"] = shock_lookup.get(row["macro_event_id"], [])
        row["asset_targets"] = asset_lookup.get(row["macro_event_id"], [])
        row["event_time"] = (
            row.get("event_time_start")
            or row.get("event_time_end")
            or row.get("published_date")
        )
    return rows


def _load_cluster_candidates(
    conn,
    event: dict[str, Any],
    *,
    window_days: int,
) -> list[dict[str, Any]]:
    sql = """
        SELECT
            ec.cluster_id,
            ec.event_type,
            ec.primary_shock_type,
            ec.region,
            ec.canonical_summary,
            ec.summary_embedding_json,
            ec.first_event_time,
            ec.last_event_time,
            ec.asset_targets_json
        FROM event_clusters ec
        WHERE ec.cluster_status = 'active'
        ORDER BY ec.updated_at DESC, ec.cluster_id
    """
    rows = [dict(row) for row in conn.execute(sql).fetchall()]
    cluster_ids = [row["cluster_id"] for row in rows]
    shock_lookup = _load_lookup_rows(
        conn,
        """
        SELECT cm.cluster_id, mst.shock_type
        FROM cluster_members cm
        JOIN macro_event_shock_types mst
          ON mst.macro_event_id = cm.macro_event_id
        WHERE cm.cluster_id IN ({placeholders})
        ORDER BY cm.cluster_id, mst.shock_type
        """,
        cluster_ids,
    )
    for row in rows:
        row["summary"] = row.get("canonical_summary")
        row["embedding"] = None
        raw_embedding = row.get("summary_embedding_json")
        if raw_embedding:
            try:
                parsed = json.loads(raw_embedding)
                if isinstance(parsed, list):
                    row["embedding"] = [float(item) for item in parsed]
            except (TypeError, ValueError, json.JSONDecodeError):
                row["embedding"] = None
        try:
            row["asset_targets"] = json.loads(row.get("asset_targets_json") or "[]")
        except json.JSONDecodeError:
            row["asset_targets"] = []
        row["shock_types"] = shock_lookup.get(row["cluster_id"], [])
    filtered: list[dict[str, Any]] = []
    for row in rows:
        if not _time_window_ok(event, row, window_days):
            continue
        if not _region_compatible(event.get("region"), row.get("region")):
            continue
        if not _same_type_or_shock_overlap(event, row):
            continue
        filtered.append(row)
    return filtered


def find_or_create_cluster(
    conn,
    event: dict[str, Any],
    *,
    window_days: int = 7,
    similarity_threshold: float = 0.82,
) -> tuple[str, bool, float, list[str]]:
    """Find a matching cluster or create a new one for the supplied event."""
    candidates = _load_cluster_candidates(conn, event, window_days=window_days)
    best_cluster: dict[str, Any] | None = None
    best_similarity = -1.0
    best_reasons: list[str] = []
    for candidate in candidates:
        similarity = compute_event_similarity(event, candidate)
        if similarity < similarity_threshold or similarity <= best_similarity:
            continue
        reasons = []
        if (event.get("event_type") or "").strip().lower() == (candidate.get("event_type") or "").strip().lower():
            reasons.append("same_event_type")
        if set(event.get("shock_types", [])) & set(candidate.get("shock_types", [])):
            reasons.append("shock_overlap")
        if _asset_overlap(event, candidate):
            reasons.append("asset_overlap")
        if _region_compatible(event.get("region"), candidate.get("region")):
            reasons.append("region_compatible")
        best_cluster = candidate
        best_similarity = similarity
        best_reasons = reasons
    if best_cluster is not None:
        return best_cluster["cluster_id"], False, best_similarity, best_reasons

    cluster_id = _md5(f"cluster::{event['macro_event_id']}")
    now = _now_utc()
    event_time = event.get("event_time")
    conn.execute(
        """
        INSERT OR IGNORE INTO event_clusters (
            cluster_id,
            event_type,
            primary_shock_type,
            region,
            canonical_summary,
            summary_embedding_json,
            first_event_time,
            last_event_time,
            cluster_window_days,
            member_count,
            unique_source_count,
            asset_targets_json,
            cluster_status,
            created_at,
            updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?)
        """,
        (
            cluster_id,
            event.get("event_type"),
            (event.get("shock_types") or [None])[0],
            event.get("region"),
            event.get("summary"),
            json.dumps(event.get("embedding")) if event.get("embedding") else None,
            event_time,
            event_time,
            int(window_days),
            0,
            0,
            json.dumps(event.get("asset_targets", []), ensure_ascii=True),
            now,
            now,
        ),
    )
    return cluster_id, True, 1.0, ["new_cluster"]


def write_cluster_members(
    conn,
    *,
    cluster_id: str,
    event: dict[str, Any],
    similarity_score: float,
    match_reasons: list[str],
) -> None:
    """Persist cluster membership for a macro event."""
    conn.execute(
        """
        INSERT OR REPLACE INTO cluster_members (
            cluster_id,
            macro_event_id,
            similarity_score,
            match_reasons_json,
            event_time,
            article_id,
            chunk_id,
            source,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            cluster_id,
            event["macro_event_id"],
            round(_safe_float(similarity_score), 4),
            json.dumps(match_reasons, ensure_ascii=True),
            event.get("event_time"),
            event.get("article_id"),
            event.get("chunk_id"),
            event.get("source"),
            _now_utc(),
        ),
    )


def _load_cluster_member_events(conn, cluster_id: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT
            cm.cluster_id,
            cm.macro_event_id,
            cm.similarity_score,
            cm.event_time,
            cm.source,
            m.article_id,
            m.chunk_id,
            m.event_type,
            m.summary,
            m.region,
            m.time_horizon,
            m.event_time_start,
            m.event_time_end,
            m.confidence,
            m.support_score,
            m.novelty_hint,
            m.urgency,
            m.market_surprise,
            c.published_date
        FROM cluster_members cm
        JOIN macro_events m ON m.macro_event_id = cm.macro_event_id
        LEFT JOIN chunks c ON c.chunk_id = m.chunk_id
        WHERE cm.cluster_id = ?
        ORDER BY coalesce(m.support_score, m.confidence, 0) DESC,
                 coalesce(m.event_time_start, m.event_time_end, c.published_date) DESC
        """,
        (cluster_id,),
    ).fetchall()
    members = [dict(row) for row in rows]
    event_ids = [row["macro_event_id"] for row in members]
    shock_lookup = _load_lookup_rows(
        conn,
        """
        SELECT macro_event_id, shock_type
        FROM macro_event_shock_types
        WHERE macro_event_id IN ({placeholders})
        ORDER BY macro_event_id, shock_type
        """,
        event_ids,
    )
    asset_lookup = _load_lookup_rows(
        conn,
        """
        SELECT macro_event_id, target_type || ':' || target_id
        FROM asset_impacts
        WHERE macro_event_id IN ({placeholders})
        ORDER BY macro_event_id, impact_id
        """,
        event_ids,
    )
    for row in members:
        row["shock_types"] = shock_lookup.get(row["macro_event_id"], [])
        row["asset_targets"] = asset_lookup.get(row["macro_event_id"], [])
        row["event_time"] = row.get("event_time") or row.get("event_time_start") or row.get("event_time_end") or row.get("published_date")
    return members


def update_cluster_metadata(
    conn,
    *,
    cluster_id: str,
    embedding_fn: EmbeddingFn | None = None,
) -> dict[str, Any]:
    """Recompute and persist metadata for a cluster."""
    members = _load_cluster_member_events(conn, cluster_id)
    if not members:
        return {}
    representative = max(
        members,
        key=lambda row: (
            _safe_float(row.get("support_score"), -1.0),
            _safe_float(row.get("confidence"), -1.0),
            _event_time(row) or datetime.min.replace(tzinfo=timezone.utc),
        ),
    )
    datetimes = [dt for dt in (_event_time(row) for row in members) if dt is not None]
    first_event_time = min(datetimes).isoformat() if datetimes else representative.get("event_time")
    last_event_time = max(datetimes).isoformat() if datetimes else representative.get("event_time")
    unique_sources = sorted({str(row.get("source") or "").strip() for row in members if str(row.get("source") or "").strip()})
    all_assets = sorted({asset for row in members for asset in row.get("asset_targets", [])})
    event_type_counts = Counter(str(row.get("event_type") or "").strip() for row in members if str(row.get("event_type") or "").strip())
    shock_counts = Counter(shock for row in members for shock in row.get("shock_types", []))
    embed = representative.get("embedding")
    if embed is None and representative.get("summary"):
        embedder = embedding_fn or _default_embedding_fn
        embed = embedder([str(representative["summary"])])[0]
    row = {
        "event_type": event_type_counts.most_common(1)[0][0] if event_type_counts else representative.get("event_type"),
        "primary_shock_type": shock_counts.most_common(1)[0][0] if shock_counts else (representative.get("shock_types") or [None])[0],
        "region": representative.get("region"),
        "canonical_summary": representative.get("summary"),
        "summary_embedding_json": json.dumps(embed, ensure_ascii=True) if embed else None,
        "first_event_time": first_event_time,
        "last_event_time": last_event_time,
        "member_count": len(members),
        "unique_source_count": len(unique_sources),
        "asset_targets_json": json.dumps(all_assets, ensure_ascii=True),
        "updated_at": _now_utc(),
    }
    conn.execute(
        """
        UPDATE event_clusters
        SET event_type = ?,
            primary_shock_type = ?,
            region = ?,
            canonical_summary = ?,
            summary_embedding_json = ?,
            first_event_time = ?,
            last_event_time = ?,
            member_count = ?,
            unique_source_count = ?,
            asset_targets_json = ?,
            updated_at = ?
        WHERE cluster_id = ?
        """,
        (
            row["event_type"],
            row["primary_shock_type"],
            row["region"],
            row["canonical_summary"],
            row["summary_embedding_json"],
            row["first_event_time"],
            row["last_event_time"],
            row["member_count"],
            row["unique_source_count"],
            row["asset_targets_json"],
            row["updated_at"],
            cluster_id,
        ),
    )
    return row


def run_event_clustering(
    db_path: str = "my_database.db",
    *,
    window_days: int = 7,
    similarity_threshold: float = 0.82,
    limit: int | None = None,
    embedding_fn: EmbeddingFn | None = None,
) -> dict[str, Any]:
    """Cluster unassigned macro events into reusable event clusters."""
    create_database(db_path)
    conn = connect_sqlite(db_path)
    conn.row_factory = sqlite3.Row
    events = load_unclustered_macro_events(conn, limit=limit)
    if not events:
        conn.close()
        return {
            "events_considered": 0,
            "members_written": 0,
            "clusters_created": 0,
            "clusters_updated": 0,
        }

    summaries = [str(event.get("summary") or "") for event in events]
    if any(summary.strip() for summary in summaries):
        embedder = embedding_fn or _default_embedding_fn
        embeddings = embedder(summaries)
        for event, embedding in zip(events, embeddings):
            event["embedding"] = embedding

    touched_clusters: set[str] = set()
    created_clusters = 0
    written_members = 0
    for event in events:
        cluster_id, created, similarity, reasons = find_or_create_cluster(
            conn,
            event,
            window_days=window_days,
            similarity_threshold=similarity_threshold,
        )
        if created:
            created_clusters += 1
        write_cluster_members(
            conn,
            cluster_id=cluster_id,
            event=event,
            similarity_score=similarity,
            match_reasons=reasons,
        )
        written_members += 1
        touched_clusters.add(cluster_id)

    updated_clusters = 0
    for cluster_id in touched_clusters:
        update_cluster_metadata(conn, cluster_id=cluster_id, embedding_fn=embedding_fn)
        updated_clusters += 1

    conn.commit()
    conn.close()
    return {
        "events_considered": len(events),
        "members_written": written_members,
        "clusters_created": created_clusters,
        "clusters_updated": updated_clusters,
        "window_days": int(window_days),
        "similarity_threshold": float(similarity_threshold),
    }


if __name__ == "__main__":
    summary = run_event_clustering()
    for key, value in summary.items():
        print(f"{key}: {value}")
