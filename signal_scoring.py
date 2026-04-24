from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any

from create_sql_db import connect_sqlite, create_database


SIGNAL_SCORE_WEIGHTS = {
    "novelty_score": 0.25,
    "source_quality_score": 0.20,
    "velocity_score": 0.20,
    "asset_impact_score": 0.15,
    "confidence_score": 0.10,
    "recency_score": 0.10,
}

TRUST_TIER_SCORES = {
    "tier_1": 0.95,
    "tier_2": 0.75,
    "tier_3": 0.55,
    "blocked": 0.05,
}

IMPACT_STRENGTH_SCORES = {
    "weak": 0.35,
    "moderate": 0.65,
    "strong": 0.90,
}


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


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        try:
            dt = datetime.strptime(text, "%Y-%m-%d")
        except ValueError:
            return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _load_cluster_rows(conn) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT
            ec.cluster_id,
            ec.event_type,
            ec.primary_shock_type,
            ec.region,
            ec.canonical_summary,
            ec.first_event_time,
            ec.last_event_time,
            ec.member_count,
            ec.unique_source_count,
            ec.asset_targets_json
        FROM event_clusters ec
        WHERE ec.cluster_status = 'active'
        ORDER BY ec.last_event_time DESC, ec.cluster_id
        """
    ).fetchall()
    return [dict(row) for row in rows]


def _load_cluster_member_rows(conn, cluster_ids: list[str]) -> dict[str, list[dict[str, Any]]]:
    if not cluster_ids:
        return {}
    placeholders = ",".join("?" for _ in cluster_ids)
    rows = conn.execute(
        f"""
        SELECT
            cm.cluster_id,
            cm.macro_event_id,
            cm.source,
            cm.event_time,
            m.confidence,
            m.support_score,
            m.novelty_hint,
            m.urgency,
            m.market_surprise,
            a.source_trust_tier,
            a.article_quality_score
        FROM cluster_members cm
        JOIN macro_events m ON m.macro_event_id = cm.macro_event_id
        LEFT JOIN articles a ON a.article_id = m.article_id
        WHERE cm.cluster_id IN ({placeholders})
        ORDER BY cm.cluster_id, cm.event_time DESC
        """,
        cluster_ids,
    ).fetchall()
    lookup: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        lookup[str(row["cluster_id"])].append(dict(row))
    return dict(lookup)


def _load_cluster_asset_impacts(conn, cluster_ids: list[str]) -> dict[str, list[dict[str, Any]]]:
    if not cluster_ids:
        return {}
    placeholders = ",".join("?" for _ in cluster_ids)
    rows = conn.execute(
        f"""
        SELECT
            cm.cluster_id,
            ai.target_type,
            ai.target_id,
            ai.direction,
            ai.strength,
            ai.horizon,
            ai.confidence
        FROM cluster_members cm
        JOIN asset_impacts ai ON ai.macro_event_id = cm.macro_event_id
        WHERE cm.cluster_id IN ({placeholders})
        ORDER BY cm.cluster_id, ai.target_type, ai.target_id
        """,
        cluster_ids,
    ).fetchall()
    lookup: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        lookup[str(row["cluster_id"])].append(dict(row))
    return dict(lookup)


def _load_previous_alert_counts(conn, score_date: str) -> dict[str, int]:
    rows = conn.execute(
        """
        SELECT
            sa.cluster_id,
            COUNT(DISTINCT sa.signal_date) AS n
        FROM signal_alerts sa
        JOIN event_clusters ec ON ec.cluster_id = sa.cluster_id
        WHERE sa.signal_date < ?
          AND date(sa.signal_date) >= date(?, '-' || coalesce(ec.cluster_window_days, 7) || ' days')
        GROUP BY sa.cluster_id
        """,
        (score_date, score_date),
    ).fetchall()
    return {str(row["cluster_id"]): int(row["n"]) for row in rows}


def compute_novelty_score(cluster: dict[str, Any], members: list[dict[str, Any]], prior_alert_count: int) -> float:
    """Score how new a cluster looks relative to prior surfaced signals."""
    latest_dt = _parse_dt(cluster.get("last_event_time"))
    first_dt = _parse_dt(cluster.get("first_event_time")) or latest_dt
    age_days = max(0.0, (latest_dt - first_dt).total_seconds() / 86400.0) if latest_dt and first_dt else 0.0
    freshness = math.exp(-age_days / 7.0)
    hint_penalty = 0.0
    hints = Counter(str(item.get("novelty_hint") or "").strip().lower() for item in members if str(item.get("novelty_hint") or "").strip())
    dominant_hint = hints.most_common(1)[0][0] if hints else ""
    if dominant_hint == "continuation":
        hint_penalty = 0.20
    elif dominant_hint == "stale":
        hint_penalty = 0.40
    prior_penalty = min(0.45, 0.15 * max(0, prior_alert_count))
    return _clamp01(0.15 + (0.85 * freshness) - hint_penalty - prior_penalty)


def compute_velocity_score(cluster: dict[str, Any], members: list[dict[str, Any]]) -> float:
    """Score how quickly a cluster is accumulating corroboration."""
    if not members:
        return 0.0
    latest_dt = _parse_dt(cluster.get("last_event_time")) or max((_parse_dt(item.get("event_time")) for item in members), default=None)
    if latest_dt is None:
        return _clamp01(len(members) / 4.0)
    recent_cutoff = latest_dt.timestamp() - (48 * 3600)
    recent_members = [
        item
        for item in members
        if (_parse_dt(item.get("event_time")) or latest_dt).timestamp() >= recent_cutoff
    ]
    recent_sources = {str(item.get("source") or "").strip().lower() for item in recent_members if str(item.get("source") or "").strip()}
    density = _clamp01(len(recent_members) / 4.0)
    source_breadth = _clamp01(len(recent_sources) / 4.0)
    member_scale = _clamp01(_safe_float(cluster.get("member_count"), 0.0) / 6.0)
    return _clamp01((0.45 * density) + (0.35 * source_breadth) + (0.20 * member_scale))


def compute_source_quality_score(cluster: dict[str, Any], members: list[dict[str, Any]]) -> float:
    """Score the quality of sources supporting a cluster."""
    if not members:
        return 0.0
    quality_scores: list[float] = []
    for item in members:
        tier = str(item.get("source_trust_tier") or "").strip().lower()
        trust_score = TRUST_TIER_SCORES.get(tier, 0.60)
        raw_quality = _safe_float(item.get("article_quality_score"), 0.65)
        quality_score = raw_quality / 100.0 if raw_quality > 1.0 else raw_quality
        quality_scores.append(_clamp01((0.65 * trust_score) + (0.35 * quality_score)))
    return _clamp01(sum(quality_scores) / len(quality_scores))


def compute_asset_impact_score(cluster: dict[str, Any], impacts: list[dict[str, Any]]) -> float:
    """Score the breadth and strength of asset impacts tied to a cluster."""
    if not impacts:
        return 0.0
    unique_targets = {
        f"{item.get('target_type')}:{item.get('target_id')}"
        for item in impacts
        if item.get("target_type") and item.get("target_id")
    }
    target_breadth = _clamp01(len(unique_targets) / 4.0)
    strength_scores = [
        IMPACT_STRENGTH_SCORES.get(str(item.get("strength") or "").strip().lower(), 0.40)
        for item in impacts
    ]
    confidence_scores = [_clamp01(_safe_float(item.get("confidence"), 0.5)) for item in impacts]
    avg_strength = sum(strength_scores) / len(strength_scores)
    avg_confidence = sum(confidence_scores) / len(confidence_scores)
    return _clamp01((0.55 * avg_strength) + (0.25 * avg_confidence) + (0.20 * target_breadth))


def compute_confidence_score(cluster: dict[str, Any], members: list[dict[str, Any]]) -> float:
    """Score the verified support strength of the cluster."""
    if not members:
        return _clamp01(_safe_float(cluster.get("confidence"), 0.0))
    support_scores = [
        _clamp01(_safe_float(item.get("support_score"), _safe_float(item.get("confidence"), 0.0)))
        for item in members
    ]
    return _clamp01(sum(support_scores) / len(support_scores))


def compute_recency_score(cluster: dict[str, Any], now: datetime | None = None) -> float:
    """Score how recent the latest event in a cluster is."""
    latest_dt = _parse_dt(cluster.get("last_event_time"))
    if latest_dt is None:
        return 0.0
    now = now or datetime.now(timezone.utc)
    age_days = max(0.0, (now - latest_dt).total_seconds() / 86400.0)
    return _clamp01(math.exp(-age_days / 5.0))


def compute_cluster_score(
    cluster: dict[str, Any],
    members: list[dict[str, Any]],
    impacts: list[dict[str, Any]],
    prior_alert_count: int = 0,
) -> dict[str, Any]:
    """Compute all deterministic cluster scoring components."""
    novelty_score = compute_novelty_score(cluster, members, prior_alert_count)
    source_quality_score = compute_source_quality_score(cluster, members)
    velocity_score = compute_velocity_score(cluster, members)
    asset_impact_score = compute_asset_impact_score(cluster, impacts)
    confidence_score = compute_confidence_score(cluster, members)
    recency_score = compute_recency_score(cluster)
    signal_score = (
        SIGNAL_SCORE_WEIGHTS["novelty_score"] * novelty_score
        + SIGNAL_SCORE_WEIGHTS["source_quality_score"] * source_quality_score
        + SIGNAL_SCORE_WEIGHTS["velocity_score"] * velocity_score
        + SIGNAL_SCORE_WEIGHTS["asset_impact_score"] * asset_impact_score
        + SIGNAL_SCORE_WEIGHTS["confidence_score"] * confidence_score
        + SIGNAL_SCORE_WEIGHTS["recency_score"] * recency_score
    )
    return {
        "cluster_id": cluster["cluster_id"],
        "novelty_score": round(novelty_score, 4),
        "source_quality_score": round(source_quality_score, 4),
        "velocity_score": round(velocity_score, 4),
        "asset_impact_score": round(asset_impact_score, 4),
        "confidence_score": round(confidence_score, 4),
        "recency_score": round(recency_score, 4),
        "signal_score": round(_clamp01(signal_score), 4),
        "supporting_event_count": len(members),
        "supporting_source_count": len(
            {
                str(item.get("source") or "").strip().lower()
                for item in members
                if str(item.get("source") or "").strip()
            }
        ),
    }


def _derive_signal_direction(impacts: list[dict[str, Any]]) -> str:
    directions = Counter(str(item.get("direction") or "").strip().lower() for item in impacts if str(item.get("direction") or "").strip())
    if not directions:
        return "mixed"
    return directions.most_common(1)[0][0]


def _derive_cluster_hints(members: list[dict[str, Any]]) -> dict[str, str | None]:
    out: dict[str, str | None] = {"novelty_hint": None, "urgency": None, "market_surprise": None}
    for key in out:
        counter = Counter(str(item.get(key) or "").strip().lower() for item in members if str(item.get(key) or "").strip())
        out[key] = counter.most_common(1)[0][0] if counter else None
    return out


def write_cluster_scores(
    conn,
    score_rows: list[dict[str, Any]],
    cluster_lookup: dict[str, dict[str, Any]],
    impacts_lookup: dict[str, list[dict[str, Any]]],
    member_lookup: dict[str, list[dict[str, Any]]],
    *,
    limit: int | None = None,
    score_date: str | None = None,
) -> dict[str, Any]:
    """Persist cluster scores and surface top signals into SQLite."""
    if not score_rows:
        return {"scores_written": 0, "signals_written": 0, "signals_deactivated": 0}
    scored_at = _now_utc()
    score_date = score_date or scored_at[:10]
    ranked = sorted(score_rows, key=lambda row: row["signal_score"], reverse=True)
    for row in ranked:
        score_id = _md5(f"score::{row['cluster_id']}::{score_date}")
        conn.execute(
            """
            INSERT OR REPLACE INTO event_cluster_scores (
                score_id,
                cluster_id,
                score_date,
                novelty_score,
                source_quality_score,
                velocity_score,
                asset_impact_score,
                confidence_score,
                recency_score,
                signal_score,
                supporting_event_count,
                supporting_source_count,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                score_id,
                row["cluster_id"],
                score_date,
                row["novelty_score"],
                row["source_quality_score"],
                row["velocity_score"],
                row["asset_impact_score"],
                row["confidence_score"],
                row["recency_score"],
                row["signal_score"],
                row["supporting_event_count"],
                row["supporting_source_count"],
                scored_at,
                scored_at,
            ),
        )
        row["score_id"] = score_id

    top_rows = ranked[:limit] if limit is not None and limit > 0 else ranked
    for rank, row in enumerate(top_rows, start=1):
        cluster = cluster_lookup[row["cluster_id"]]
        impacts = impacts_lookup.get(row["cluster_id"], [])
        members = member_lookup.get(row["cluster_id"], [])
        hints = _derive_cluster_hints(members)
        signal_id = _md5(f"signal::{row['cluster_id']}::{score_date}")
        row["signal_id"] = signal_id
        top_assets = sorted(
            {
                f"{impact.get('target_type')}:{impact.get('target_id')}"
                for impact in impacts
                if impact.get("target_type") and impact.get("target_id")
            }
        )
        conn.execute(
            """
            INSERT OR REPLACE INTO signal_alerts (
                signal_id,
                cluster_id,
                score_id,
                signal_date,
                rank,
                signal_score,
                headline,
                summary,
                novelty_hint,
                urgency,
                market_surprise,
                top_assets_json,
                status,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?)
            """,
            (
                signal_id,
                row["cluster_id"],
                row["score_id"],
                score_date,
                rank,
                row["signal_score"],
                cluster.get("canonical_summary"),
                cluster.get("canonical_summary"),
                hints.get("novelty_hint"),
                hints.get("urgency"),
                hints.get("market_surprise"),
                json.dumps(top_assets, ensure_ascii=True),
                scored_at,
                scored_at,
            ),
        )
    signals_deactivated = 0
    if limit is not None and limit > 0:
        top_signal_ids = [str(row["signal_id"]) for row in top_rows if row.get("signal_id")]
        if top_signal_ids:
            placeholders = ",".join("?" for _ in top_signal_ids)
            params: list[Any] = [scored_at, score_date, *top_signal_ids]
            result = conn.execute(
                f"""
                UPDATE signal_alerts
                SET status = 'inactive',
                    updated_at = ?
                WHERE signal_date = ?
                  AND status = 'active'
                  AND signal_id NOT IN ({placeholders})
                """,
                params,
            )
        else:
            result = conn.execute(
                """
                UPDATE signal_alerts
                SET status = 'inactive',
                    updated_at = ?
                WHERE signal_date = ?
                  AND status = 'active'
                """,
                (scored_at, score_date),
            )
        signals_deactivated = max(0, int(result.rowcount or 0))
    return {
        "scores_written": len(ranked),
        "signals_written": len(top_rows),
        "signals_deactivated": signals_deactivated,
    }


def _refresh_source_quality(conn, member_lookup: dict[str, list[dict[str, Any]]]) -> None:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for members in member_lookup.values():
        for item in members:
            source_name = str(item.get("source") or "").strip()
            if source_name:
                by_source[source_name].append(item)
    for source_name, rows in by_source.items():
        trust_counts = Counter(str(row.get("source_trust_tier") or "").strip().lower() for row in rows if str(row.get("source_trust_tier") or "").strip())
        trust_tier = trust_counts.most_common(1)[0][0] if trust_counts else None
        quality_values = []
        for row in rows:
            raw_quality = _safe_float(row.get("article_quality_score"), 0.65)
            quality_values.append(raw_quality / 100.0 if raw_quality > 1.0 else raw_quality)
        quality_score = _clamp01(sum(quality_values) / len(quality_values)) if quality_values else 0.0
        conn.execute(
            """
            INSERT OR REPLACE INTO source_quality (
                source_name,
                source_provider,
                source_trust_tier,
                quality_score,
                reliability_label,
                article_count,
                last_article_at,
                notes_json,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                source_name,
                None,
                trust_tier,
                quality_score,
                "high" if quality_score >= 0.75 else "medium" if quality_score >= 0.5 else "low",
                len(rows),
                max((row.get("event_time") or "") for row in rows),
                None,
                _now_utc(),
            ),
        )


def run_signal_scoring(
    db_path: str = "my_database.db",
    *,
    limit: int | None = None,
) -> dict[str, Any]:
    """Score event clusters deterministically and publish ranked signals."""
    create_database(db_path)
    conn = connect_sqlite(db_path)
    conn.row_factory = sqlite3.Row
    clusters = _load_cluster_rows(conn)
    if not clusters:
        conn.close()
        return {"clusters_scored": 0, "signals_written": 0, "signals_deactivated": 0}
    cluster_ids = [row["cluster_id"] for row in clusters]
    member_lookup = _load_cluster_member_rows(conn, cluster_ids)
    impacts_lookup = _load_cluster_asset_impacts(conn, cluster_ids)
    score_date = _now_utc()[:10]
    prior_alert_counts = _load_previous_alert_counts(conn, score_date)
    _refresh_source_quality(conn, member_lookup)
    score_rows: list[dict[str, Any]] = []
    for cluster in clusters:
        score_rows.append(
            compute_cluster_score(
                cluster,
                member_lookup.get(cluster["cluster_id"], []),
                impacts_lookup.get(cluster["cluster_id"], []),
                prior_alert_counts.get(cluster["cluster_id"], 0),
            )
        )
    write_summary = write_cluster_scores(
        conn,
        score_rows,
        {row["cluster_id"]: row for row in clusters},
        impacts_lookup,
        member_lookup,
        limit=limit,
        score_date=score_date,
    )
    conn.commit()
    conn.close()
    return {
        "clusters_scored": len(score_rows),
        **write_summary,
    }


if __name__ == "__main__":
    summary = run_signal_scoring()
    for key, value in summary.items():
        print(f"{key}: {value}")
