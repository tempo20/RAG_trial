"""
conversation_memory.py — Persistent conversation memory for T-GRAG Chatbot

Responsibilities:
  1. TurnMemory      — typed record of one completed turn
  2. ConversationMemory — session-scoped store with entity/temporal carryover
  3. resolve_coreference — rewrite pronouns/references using last resolved entity
  4. resolve_temporal_carryover — inherit last time range when query has no time anchor
  5. save_memory / load_memory — JSON persistence across restarts
  6. summarize_session — LLM-based compression once turn count exceeds threshold

Design constraints respected:
  - No Neo4j writes (no schema pollution)
  - QueryTarget is stored by value (canonical_name + metadata), not by reference,
    so the driver can be closed without invalidating memory
  - Persistence is a plain JSON file; no additional infrastructure required
  - summarize_session is opt-in and only called when MAX_TURNS_BEFORE_SUMMARY is hit,
    keeping latency impact isolated to that single turn
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MEMORY_PATH = Path("conversation_memory.json")

# After this many turns the oldest turns are compressed into a text summary
# to prevent the system-prompt injection from growing unbounded.
MAX_TURNS_BEFORE_SUMMARY = int(8)

# How many recent turns are always kept verbatim (not compressed)
RECENT_TURNS_KEPT = int(3)

# How many recent turns to inject into the system prompt
CONTEXT_TURNS_INJECTED = int(3)

# Pronoun / vague-reference patterns that trigger coreference resolution
_COREF_RE = re.compile(
    r"\b(they|it|the company|their|its|the firm|the stock|the brand|"
    r"this company|this stock|that company|that stock|"
    r"the entity|this entity)\b",
    re.IGNORECASE,
)

# Phrases that indicate the user wants to inherit the last time window
_TEMPORAL_INHERIT_RE = re.compile(
    r"\b(same period|same time|that period|that time|then|"
    r"during that|over that period|in that period|at that time)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class TurnMemory:
    """
    Immutable record of one completed question-answer turn.

    We store QueryTarget fields by value (not the object itself) so the
    dataclass is trivially serialisable to JSON without custom encoders.
    """
    # Input
    query: str
    turn_index: int
    timestamp: str                      # ISO-8601 UTC

    # Resolved entity (mirrors QueryTarget fields we care about for carryover)
    canonical_name: Optional[str]       # e.g. "AAPL" or "elon musk"
    display_name: Optional[str]         # e.g. "Apple Inc." or "Elon Musk"
    ticker: Optional[str]               # non-None only for ORG
    entity_type: Optional[str]          # ORG / PER / LOC / None
    entity_confidence: float

    # Temporal anchors resolved for this turn
    date_start: Optional[str]           # YYYY-MM-DD or None
    date_end: Optional[str]             # YYYY-MM-DD or None

    # Output
    answer_summary: str                 # first 300 chars of the generated answer
    chunk_uids: list[str]               # chunk_uids retrieved (for dedup / audit)
    source_urls: list[str]              # URLs surfaced to user


@dataclass
class ConversationMemory:
    """
    Session-scoped memory store.

    Attributes
    ----------
    turns : list[TurnMemory]
        Full verbatim turn history (may be partially replaced by summary).
    summary : str
        LLM-generated compression of older turns. Empty until first compression.
    session_id : str
        Stable identifier written to the JSON file.
    """
    turns: list[TurnMemory] = field(default_factory=list)
    summary: str = ""
    session_id: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def turn_count(self) -> int:
        return len(self.turns)

    @property
    def last_turn(self) -> Optional[TurnMemory]:
        return self.turns[-1] if self.turns else None

    @property
    def last_entity(self) -> tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Returns (canonical_name, display_name, entity_type) from the most
        recent turn that resolved a non-None entity.
        Walks backwards so a general-query turn doesn't erase entity context.
        """
        for t in reversed(self.turns):
            if t.canonical_name:
                return t.canonical_name, t.display_name, t.entity_type
        return None, None, None

    @property
    def last_date_range(self) -> tuple[Optional[str], Optional[str]]:
        """
        Returns (date_start, date_end) from the most recent turn that
        had an explicit temporal filter.
        """
        for t in reversed(self.turns):
            if t.date_start or t.date_end:
                return t.date_start, t.date_end
        return None, None

    # ------------------------------------------------------------------
    # Turn recording
    # ------------------------------------------------------------------

    def record_turn(
        self,
        *,
        query: str,
        target,                         # QueryTarget or None
        date_start: Optional[str],
        date_end: Optional[str],
        answer: str,
        chunks: list[dict],
        source_urls: list[str],
    ) -> TurnMemory:
        """
        Build a TurnMemory from a completed retrieval+generation cycle and
        append it to the turn list. target may be None for general queries.
        """
        turn = TurnMemory(
            query=query,
            turn_index=self.turn_count,
            timestamp=datetime.now(timezone.utc).isoformat(),
            canonical_name=target.canonical_name if target else None,
            display_name=target.display_name if target else None,
            ticker=target.ticker if target else None,
            entity_type=target.entity_type if target else None,
            entity_confidence=target.confidence if target else 0.0,
            date_start=date_start,
            date_end=date_end,
            answer_summary=answer[:300],
            chunk_uids=[c["chunk_uid"] for c in chunks if c.get("chunk_uid")],
            source_urls=source_urls,
        )
        self.turns.append(turn)
        return turn

    # ------------------------------------------------------------------
    # System-prompt injection
    # ------------------------------------------------------------------

    def context_for_prompt(self, max_turns: int = CONTEXT_TURNS_INJECTED) -> str:
        """
        Build a compact block to prepend to the system prompt so the LLM
        is aware of conversational context.

        Format:
            [CONVERSATION HISTORY]
            <summary if present>
            Turn N: Asked about <entity> (<period>): <answer_summary>
            ...
        """
        lines: list[str] = []

        if self.summary:
            lines.append(f"Earlier in this session: {self.summary}")

        recent = self.turns[-max_turns:] if self.turns else []
        for t in recent:
            entity_label = t.display_name or "general query"
            period_label = (
                f"{t.date_start} -> {t.date_end}"
                if (t.date_start or t.date_end)
                else "no time filter"
            )
            lines.append(
                f"Turn {t.turn_index + 1}: "
                f"Asked about {entity_label} ({period_label}): "
                f"{t.answer_summary}"
            )

        if not lines:
            return ""

        return "[CONVERSATION HISTORY]\n" + "\n".join(lines)

    # ------------------------------------------------------------------
    # Compression
    # ------------------------------------------------------------------

    def maybe_compress(self, gen_client: Any, gen_model: str, max_tokens: int = 256) -> bool:
        """
        If turn count exceeds MAX_TURNS_BEFORE_SUMMARY, compress the oldest
        turns into self.summary and drop them from self.turns.

        Returns True if compression happened.
        """
        if self.turn_count <= MAX_TURNS_BEFORE_SUMMARY:
            return False

        compress_count = self.turn_count - RECENT_TURNS_KEPT
        old_turns = self.turns[:compress_count]
        self.turns = self.turns[compress_count:]

        # Build a text block of the old turns for the LLM to compress
        old_text_lines = []
        for t in old_turns:
            entity_label = t.display_name or "general"
            period = (
                f"{t.date_start}→{t.date_end}" if (t.date_start or t.date_end) else "—"
            )
            old_text_lines.append(
                f"Q: {t.query}\nEntity: {entity_label} | Period: {period}\nA: {t.answer_summary}"
            )
        old_text = "\n---\n".join(old_text_lines)

        prior = f"Prior summary: {self.summary}\n\n" if self.summary else ""

        prompt = (
            f"{prior}"
            "Summarise the following financial news conversation turns in 3-5 sentences. "
            "Preserve: which companies/entities were discussed, which time periods were queried, "
            "and the key facts surfaced. Be concise.\n\n"
            f"{old_text}"
        )

        try:
            out = gen_client.messages.create(
                model=gen_model,
                max_tokens=max_tokens,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            parts = [
                getattr(b, "text", "") for b in (getattr(out, "content", []) or [])
            ]
            self.summary = "".join(parts).strip()
        except Exception as e:
            # Compression is best-effort; don't crash the chatbot
            print(f"  [memory] Compression failed: {e}")
            # Restore old turns so nothing is lost
            self.turns = old_turns + self.turns
            return False

        return True


# ---------------------------------------------------------------------------
# Coreference resolution
# ---------------------------------------------------------------------------

def resolve_coreference(query: str, memory: ConversationMemory) -> tuple[str, bool]:
    """
    Replace pronoun/vague-reference tokens with the last resolved entity's
    display name.

    Returns (rewritten_query, was_rewritten).
    If no entity in memory or no pronoun found, returns (query, False).
    """
    if not _COREF_RE.search(query):
        return query, False

    _, display_name, _ = memory.last_entity
    if not display_name:
        return query, False

    rewritten = _COREF_RE.sub(display_name, query)
    return rewritten, (rewritten != query)


# ---------------------------------------------------------------------------
# Temporal carryover
# ---------------------------------------------------------------------------

def resolve_temporal_carryover(
    sub_queries: list[dict],
    memory: ConversationMemory,
) -> list[dict]:
    """
    If the current query has no time anchors but the user's phrasing suggests
    they want to stay in the same period, inherit the last known date range.

    Two triggers:
      a) The raw query contains a phrase like "same period", "during that", etc.
      b) ALL sub-queries have null time_start and time_end AND the memory has
         a recent date range (passive carryover — disabled by default, see note).

    Note on passive carryover: automatically inheriting a date range for every
    timeless query would cause surprising behaviour (e.g. "tell me about NVDA"
    after a date-scoped Apple query would silently scope NVDA to Apple's period).
    We only do it when the user explicitly signals temporal continuity via phrase.
    """
    last_start, last_end = memory.last_date_range
    if not (last_start or last_end):
        return sub_queries  # nothing to inherit

    # Check if any sub-query already has a time anchor
    has_anchor = any(sq.get("time_start") or sq.get("time_end") for sq in sub_queries)
    if has_anchor:
        return sub_queries

    # Check if the query text signals temporal continuity
    combined_text = " ".join(sq.get("query", "") for sq in sub_queries)
    if not _TEMPORAL_INHERIT_RE.search(combined_text):
        return sub_queries

    # Apply inherited range
    return [
        {**sq, "time_start": last_start, "time_end": last_end}
        for sq in sub_queries
    ]


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_memory(memory: ConversationMemory, path: Path = MEMORY_PATH) -> None:
    payload = {
        "session_id": memory.session_id,
        "summary": memory.summary,
        "turns": [asdict(t) for t in memory.turns],
    }
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_memory(path: Path = MEMORY_PATH) -> ConversationMemory:
    """
    Load memory from JSON if it exists, otherwise return a fresh instance.
    Gracefully handles missing or corrupt files.
    """
    if not path.is_file():
        return ConversationMemory()

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        turns = [TurnMemory(**t) for t in payload.get("turns", [])]
        return ConversationMemory(
            turns=turns,
            summary=payload.get("summary", ""),
            session_id=payload.get("session_id", ""),
        )
    except Exception as e:
        print(f"  [memory] Could not load {path}: {e}. Starting fresh.")
        return ConversationMemory()