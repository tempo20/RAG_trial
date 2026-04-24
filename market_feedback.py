from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from create_sql_db import connect_sqlite, create_database


DEFAULT_PARQUET = Path("hist_data.parquet")


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _load_long_frame(parquet_path: Path):
    try:
        from hist_to_db import long_frame_from_parquet
    except Exception as exc:  # pragma: no cover - environment-dependent
        return None, f"market_data_loader_unavailable:{exc}"
    try:
        frame = long_frame_from_parquet(parquet_path)
    except Exception as exc:  # pragma: no cover - environment-dependent
        return None, f"market_data_load_failed:{exc}"
    return frame, None


def load_unchecked_asset_impacts(conn, limit: int | None = None) -> list[dict[str, Any]]:
    """Load signal-linked asset impacts that do not yet have market feedback rows."""
    sql = """
        SELECT
            sa.signal_id,
            sa.cluster_id,
            cm.macro_event_id,
            ai.impact_id,
            ai.target_type,
            ai.target_id,
            ai.direction AS predicted_direction,
            coalesce(m.event_time_start, m.event_time_end, sa.signal_date) AS event_time
        FROM signal_alerts sa
        JOIN cluster_members cm ON cm.cluster_id = sa.cluster_id
        JOIN asset_impacts ai ON ai.macro_event_id = cm.macro_event_id
        JOIN macro_events m ON m.macro_event_id = cm.macro_event_id
        LEFT JOIN market_reactions mr
          ON mr.signal_id = sa.signal_id
         AND mr.macro_event_id = cm.macro_event_id
         AND mr.target_type = ai.target_type
         AND mr.target_id = ai.target_id
        WHERE mr.reaction_id IS NULL
        ORDER BY sa.signal_score DESC, sa.signal_date DESC, ai.impact_id
    """
    params: list[Any] = []
    if limit is not None and limit > 0:
        sql += " LIMIT ?"
        params.append(limit)
    return [dict(row) for row in conn.execute(sql, params).fetchall()]


def get_price_window(long_frame, ticker: str, event_time: str) -> tuple[list[dict[str, Any]] | None, str]:
    """Return price rows from the local historical dataset for the requested ticker."""
    event_dt = _parse_dt(event_time)
    if event_dt is None:
        return None, "invalid_event_time"
    ticker = str(ticker or "").strip().upper()
    if not ticker:
        return None, "missing_target_id"
    data = long_frame[long_frame["ticker"].astype(str).str.upper() == ticker]
    if data.empty:
        return None, "ticker_not_found"
    data = data.sort_values("bar_date").reset_index(drop=True)
    data["bar_dt"] = data["bar_date"].astype(str)
    eligible = data[data["bar_dt"] >= event_dt.strftime("%Y-%m-%d")]
    if eligible.empty:
        return None, "no_forward_window"
    return eligible.to_dict(orient="records"), "ok"


def compute_forward_returns(price_rows: list[dict[str, Any]]) -> dict[str, float | None]:
    """Compute 1d/3d/5d forward returns from a list of ordered market bars."""
    if not price_rows:
        return {"return_1d": None, "return_3d": None, "return_5d": None}
    start_row = price_rows[0]
    start_price = start_row.get("adj_close") or start_row.get("close")
    try:
        start_price = float(start_price)
    except (TypeError, ValueError):
        return {"return_1d": None, "return_3d": None, "return_5d": None}

    def _forward_at(offset: int) -> float | None:
        if offset >= len(price_rows):
            return None
        end_price = price_rows[offset].get("adj_close") or price_rows[offset].get("close")
        try:
            end_price = float(end_price)
        except (TypeError, ValueError):
            return None
        if start_price == 0.0:
            return None
        return round((end_price / start_price) - 1.0, 6)

    return {
        "return_1d": _forward_at(1),
        "return_3d": _forward_at(3),
        "return_5d": _forward_at(5),
    }


def compare_predicted_vs_actual(predicted_direction: str | None, returns: dict[str, float | None]) -> str:
    """Classify the forward return outcome against the predicted direction."""
    observed = [value for value in returns.values() if value is not None]
    if not observed:
        return "unclear"
    avg_return = sum(observed) / len(observed)
    direction = str(predicted_direction or "").strip().lower()
    if direction in {"up", "positive"}:
        return "hit" if avg_return > 0 else "miss"
    if direction in {"down", "negative"}:
        return "hit" if avg_return < 0 else "miss"
    return "unclear"


def write_market_reaction(conn, payload: dict[str, Any]) -> None:
    """Persist one market feedback evaluation row."""
    conn.execute(
        """
        INSERT OR REPLACE INTO market_reactions (
            reaction_id,
            signal_id,
            cluster_id,
            macro_event_id,
            target_type,
            target_id,
            event_time,
            predicted_direction,
            return_1d,
            return_3d,
            return_5d,
            outcome_label,
            data_availability_status,
            notes_json,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            payload["reaction_id"],
            payload.get("signal_id"),
            payload.get("cluster_id"),
            payload.get("macro_event_id"),
            payload.get("target_type"),
            payload.get("target_id"),
            payload.get("event_time"),
            payload.get("predicted_direction"),
            payload.get("return_1d"),
            payload.get("return_3d"),
            payload.get("return_5d"),
            payload.get("outcome_label"),
            payload.get("data_availability_status"),
            payload.get("notes_json"),
            _now_utc(),
        ),
    )


def run_market_feedback(
    db_path: str = "my_database.db",
    *,
    parquet_path: str | Path = DEFAULT_PARQUET,
    limit: int | None = None,
) -> dict[str, Any]:
    """Evaluate surfaced signal impacts against local historical market data."""
    parquet_path = Path(parquet_path)
    create_database(db_path)
    conn = connect_sqlite(db_path)
    conn.row_factory = sqlite3.Row
    pending = load_unchecked_asset_impacts(conn, limit=limit)
    if not pending:
        conn.close()
        return {"evaluated": 0, "skipped": 0, "status": "no_pending_impacts"}
    if not parquet_path.exists():
        conn.close()
        return {
            "evaluated": 0,
            "skipped": len(pending),
            "status": "historical_data_missing",
            "parquet_path": str(parquet_path),
        }
    long_frame, error = _load_long_frame(parquet_path)
    if long_frame is None:
        conn.close()
        return {
            "evaluated": 0,
            "skipped": len(pending),
            "status": error,
            "parquet_path": str(parquet_path),
        }
    evaluated = 0
    skipped = 0
    for row in pending:
        reaction_id = f"{row['signal_id']}::{row['macro_event_id']}::{row['target_type']}::{row['target_id']}"
        if str(row.get("target_type") or "").strip().lower() != "ticker":
            write_market_reaction(
                conn,
                {
                    **row,
                    "reaction_id": reaction_id,
                    "return_1d": None,
                    "return_3d": None,
                    "return_5d": None,
                    "outcome_label": "unclear",
                    "data_availability_status": "unsupported_target_type",
                    "notes_json": json.dumps({"reason": "only ticker targets supported by local parquet"}, ensure_ascii=True),
                },
            )
            skipped += 1
            continue
        price_rows, status = get_price_window(long_frame, row.get("target_id"), row.get("event_time"))
        if price_rows is None:
            write_market_reaction(
                conn,
                {
                    **row,
                    "reaction_id": reaction_id,
                    "return_1d": None,
                    "return_3d": None,
                    "return_5d": None,
                    "outcome_label": "unclear",
                    "data_availability_status": status,
                    "notes_json": json.dumps({"reason": status}, ensure_ascii=True),
                },
            )
            skipped += 1
            continue
        returns = compute_forward_returns(price_rows)
        outcome_label = compare_predicted_vs_actual(row.get("predicted_direction"), returns)
        write_market_reaction(
            conn,
            {
                **row,
                "reaction_id": reaction_id,
                **returns,
                "outcome_label": outcome_label,
                "data_availability_status": status,
                "notes_json": None,
            },
        )
        evaluated += 1
    conn.commit()
    conn.close()
    return {
        "evaluated": evaluated,
        "skipped": skipped,
        "status": "ok",
        "parquet_path": str(parquet_path),
    }


if __name__ == "__main__":
    summary = run_market_feedback()
    for key, value in summary.items():
        print(f"{key}: {value}")
