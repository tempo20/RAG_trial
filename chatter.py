"""
Temporal-Graph RAG Chatbot

Connects to an already-populated Neo4j graph (built by tgrag_setup.py)
and runs interactive retrieval + generation in a terminal loop.

Features:
  - Temporal Query Decomposition (TQD): splits multi-temporal queries
  - Three-layer retrieval: temporal filter -> entity match -> semantic ranking
  - Sub-answer aggregation for comparative / multi-period questions

Usage:
    python chatter.py
"""

import json
import os
import re
import time
import warnings
from datetime import datetime, timezone

from dotenv import load_dotenv
load_dotenv()

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore", category=FutureWarning)

import dateparser
import numpy as np
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase
from transformers import pipeline

from tgrag_setup import EMBED_MODEL_NAME, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

GEN_MODEL_NAME = "Qwen/Qwen3-0.6B"

SOURCE_KEYWORDS = {
    "bbc": "BBC",
    "bloomberg": "Bloomberg",
    "cnbc": "CNBC",
    "marketwatch": "MarketWatch",
    "nasdaq": "Nasdaq",
    "stockbiz": "Stockbiz",
    "marketbeat": "MarketBeat",
    "cbs": "CBS MoneyWatch",
}


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def extract_source_filter(query: str) -> str | None:
    q = query.lower()
    for keyword, label in SOURCE_KEYWORDS.items():
        if keyword in q:
            return label
    return None


def strip_think_tags(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    return text.strip()


# Temporal Query Decomposition 

TIME_WORDS = re.compile(
    # NOTE: ordering matters: put full date patterns before month-only.
    r"\b(yesterday|last\s+week|this\s+week|last\s+month|this\s+month|"
    r"today|ago|before\s+\w+day|after\s+\w+day|since|until|between|"
    r"(?:january|february|march|april|may|june|july|august|"
    r"september|october|november|december)\s+\d{1,2}(?:st|nd|rd|th)?(?:,)?\s+\d{4}|"
    r"(?:january|february|march|april|may|june|july|august|"
    r"september|october|november|december)\s+\d{4}|"
    r"\d{4}[-/]\d{2}[-/]\d{2}|"
    r"\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|"
    r"\d{4}[-/]\d{2}|\d{1,2}/\d{1,2})\b",
    re.IGNORECASE,
)

MULTI_TIME_WORDS = re.compile(
    r"\b(compare|vs\.?|versus|differ)",
    re.IGNORECASE,
)

FROM_TO_PATTERN = re.compile(
    r"from\s+(.+?)\s+to\s+(.+?)[\?\.\!,]?\s*$",
    re.IGNORECASE,
)

DECOMPOSE_PROMPT = """\
Analyze this question and decompose it into separate sub-queries for each time period.

Rules:
- Today is {today}.
- Use ISO format YYYY-MM-DD for dates.
- Each sub-query should target one time period.

Output ONLY valid JSON (no markdown):
{{"sub_queries": [{{"query": "...", "time_start": "YYYY-MM-DD or null", "time_end": "YYYY-MM-DD or null"}}]}}

Question: {query}"""


def resolve_date(date_str: str | None) -> int | None:
    """Convert a date string to unix timestamp via dateparser."""
    if not date_str or date_str.lower() == "null":
        return None
    dt = dateparser.parse(
        date_str, settings={"RETURN_AS_TIMEZONE_AWARE": True}
    )
    if dt:
        return int(dt.timestamp())
    return None


def ts_to_datestr(ts: int | None) -> str | None:
    """Convert unix timestamp seconds to YYYY-MM-DD (UTC)."""
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).date().isoformat()


def _resolve_time_phrase(phrase: str) -> tuple[str | None, str | None]:
    """Resolve a single time phrase to (start_date, end_date) strings."""
    now = datetime.now(timezone.utc)
    p = phrase.strip().lower()

    if p == "yesterday":
        dt = dateparser.parse("yesterday", settings={"RETURN_AS_TIMEZONE_AWARE": True})
        d = dt.strftime("%Y-%m-%d") if dt else None
        return d, d
    if "last week" in p:
        s = dateparser.parse("7 days ago", settings={"RETURN_AS_TIMEZONE_AWARE": True})
        e = dateparser.parse("1 day ago", settings={"RETURN_AS_TIMEZONE_AWARE": True})
        return (
            s.strftime("%Y-%m-%d") if s else None,
            e.strftime("%Y-%m-%d") if e else None,
        )
    if "this week" in p:
        days_back = now.weekday()
        s = dateparser.parse(
            f"{days_back} days ago", settings={"RETURN_AS_TIMEZONE_AWARE": True}
        )
        return s.strftime("%Y-%m-%d") if s else None, now.strftime("%Y-%m-%d")
    if "last month" in p:
        s = dateparser.parse("1 month ago", settings={"RETURN_AS_TIMEZONE_AWARE": True})
        return s.strftime("%Y-%m-%d") if s else None, now.strftime("%Y-%m-%d")
    if "this month" in p:
        return now.strftime("%Y-%m-01"), now.strftime("%Y-%m-%d")
    if p == "today":
        d = now.strftime("%Y-%m-%d")
        return d, d

    dt = dateparser.parse(p, settings={"RETURN_AS_TIMEZONE_AWARE": True})
    if dt:
        d = dt.strftime("%Y-%m-%d")
        return d, d
    return None, None


def _extract_single_time_range(query: str) -> dict:
    """Use dateparser directly to resolve time expressions without LLM."""
    time_matches = TIME_WORDS.findall(query)
    if not time_matches:
        return {"query": query, "time_start": None, "time_end": None}

    # TIME_WORDS can yield multiple overlapping candidates. Prefer the "most specific"
    # match (typically contains a 4-digit year and is longer).
    def _match_score(m: str) -> tuple[int, int]:
        year_bonus = 1 if re.search(r"\d{4}", m) else 0
        return (year_bonus, len(m))

    best = max(time_matches, key=_match_score)
    start, end = _resolve_time_phrase(best)
    return {"query": query, "time_start": start, "time_end": end}


def decompose_query(query: str, pipe) -> list[dict]:
    """Resolve temporal references. Uses regex whenever possible, LLM only as last resort."""
    has_time = TIME_WORDS.search(query)

    if not has_time:
        return [{"query": query, "time_start": None, "time_end": None}]

    # Handle "from X to Y" pattern with regex (no LLM needed)
    ft = FROM_TO_PATTERN.search(query)
    if ft:
        period_a, period_b = ft.group(1).strip(), ft.group(2).strip()
        a_start, a_end = _resolve_time_phrase(period_a)
        b_start, b_end = _resolve_time_phrase(period_b)
        if a_start or b_start:
            return [
                {"query": query, "time_start": a_start, "time_end": a_end},
                {"query": query, "time_start": b_start, "time_end": b_end},
            ]

    # Common multi-period phrasing without requiring "vs/compare".
    # Example: "over last week and this week"
    if re.search(r"\blast\s+week\b", query, flags=re.IGNORECASE) and re.search(
        r"\bthis\s+week\b", query, flags=re.IGNORECASE
    ):
        last_start, last_end = _resolve_time_phrase("last week")
        this_start, this_end = _resolve_time_phrase("this week")
        if last_start or this_start:
            return [
                {"query": query, "time_start": last_start, "time_end": last_end},
                {"query": query, "time_start": this_start, "time_end": this_end},
            ]
    if re.search(r"\blast\s+month\b", query, flags=re.IGNORECASE) and re.search(
        r"\bthis\s+month\b", query, flags=re.IGNORECASE
    ):
        last_start, last_end = _resolve_time_phrase("last month")
        this_start, this_end = _resolve_time_phrase("this month")
        if last_start or this_start:
            return [
                {"query": query, "time_start": last_start, "time_end": last_end},
                {"query": query, "time_start": this_start, "time_end": this_end},
            ]

    has_multi = MULTI_TIME_WORDS.search(query)
    if not has_multi:
        return [_extract_single_time_range(query)]

    # True multi-period query (compare/vs) -- use LLM to decompose
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    prompt = DECOMPOSE_PROMPT.format(today=today, query=query)
    messages = [{"role": "user", "content": prompt}]
    out = pipe(messages, max_new_tokens=512, do_sample=False)
    raw = out[0]["generated_text"][-1]["content"]
    raw = strip_think_tags(raw)

    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            sub_queries = data.get("sub_queries", [])
            if isinstance(sub_queries, list):
                valid = []
                for sq in sub_queries:
                    if isinstance(sq, dict) and "query" in sq:
                        valid.append({
                            "query": sq["query"],
                            "time_start": sq.get("time_start"),
                            "time_end": sq.get("time_end"),
                        })
                if valid:
                    return valid
        except (json.JSONDecodeError, TypeError):
            pass

    return [_extract_single_time_range(query)]


# Three-Layer Retrieval 

def three_layer_retrieve(
    query: str,
    embed_model: SentenceTransformer,
    driver,
    top_k: int = 3,
    expanded_k: int = 6,
    recency_half_life_days: float = 7.0,
    source_filter: str | None = None,
    time_start: int | None = None,
    time_end: int | None = None,
):
    qvec = embed_model.encode([query], normalize_embeddings=True)[0]
    now_ts = int(datetime.now(timezone.utc).timestamp())
    half_life_seconds = recency_half_life_days * 86400.0
    market_texts: list[str] = []

    with driver.session() as session:
        # Layer 1: Temporal subgraph filter 
        time_clauses = ""
        params: dict = {"source_filter": source_filter}
        if time_start is not None:
            time_clauses += " AND c.published_ts >= $ts_start"
            params["ts_start"] = time_start
        if time_end is not None:
            time_clauses += " AND c.published_ts <= $ts_end"
            params["ts_end"] = time_end

        rows = session.run(
            f"""
            MATCH (c:Chunk)<-[:HAS_CHUNK]-(a:Article)
            WHERE ($source_filter IS NULL OR c.source = $source_filter)
            {time_clauses}
            RETURN c.chunk_uid AS chunk_uid,
                   c.text AS text,
                   c.embedding AS embedding,
                   c.published_ts AS published_ts,
                   c.source AS source,
                   a.title AS title,
                   a.url AS url,
                   c.article_id AS article_id,
                   c.chunk_id AS chunk_id
            """,
            params,
        ).data()

        # Layer 2: Entity-based coarse retrieval 
        query_lower = query.lower()
        entity_rows = session.run(
            "MATCH (e:Entity) RETURN e.canonical_name AS cname, e.name AS name"
        ).data()

        matched_entities: list[str] = []
        for e in entity_rows:
            cname = e["cname"]
            name_lower = e["name"].lower()
            if cname in query_lower or name_lower in query_lower:
                matched_entities.append(cname)
                continue
            for word in cname.split():
                if len(word) > 3 and re.search(
                    r"\b" + re.escape(word) + r"\b", query_lower
                ):
                    matched_entities.append(cname)
                    break

        if matched_entities:
            related = session.run(
                """
                MATCH (e:Entity)-[:RELATED_TO]-(e2:Entity)
                WHERE e.canonical_name IN $enames
                RETURN DISTINCT e2.canonical_name AS cname
                """,
                {"enames": matched_entities},
            ).data()
            for r in related:
                if r["cname"] not in matched_entities:
                    matched_entities.append(r["cname"])

        entity_chunk_uids: set[str] = set()
        if matched_entities:
            ent_params: dict = {
                "enames": matched_entities,
                "source_filter": source_filter,
            }
            ent_time = ""
            if time_start is not None:
                ent_time += " AND c.published_ts >= $ts_start"
                ent_params["ts_start"] = time_start
            if time_end is not None:
                ent_time += " AND c.published_ts <= $ts_end"
                ent_params["ts_end"] = time_end

            ent_chunks = session.run(
                f"""
                MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
                WHERE e.canonical_name IN $enames
                  AND ($source_filter IS NULL OR c.source = $source_filter)
                  {ent_time}
                RETURN DISTINCT c.chunk_uid AS chunk_uid
                """,
                ent_params,
            ).data()
            entity_chunk_uids = {r["chunk_uid"] for r in ent_chunks}

        # Market bar retrieval:
        # - preferred: resolve tickers via Entity -> ALIASES_TICKER
        # - fallback: if the user explicitly mentions a ticker like "AAPL", fetch it directly
        tickers: list[str] = []
        if matched_entities:
            tickers_rows = session.run(
                """
                MATCH (e:Entity)-[:ALIASES_TICKER]->(i:Instrument)
                WHERE e.canonical_name IN $enames
                RETURN DISTINCT i.ticker AS ticker
                """,
                {"enames": matched_entities},
            ).data()
            tickers = [r["ticker"] for r in tickers_rows if r.get("ticker")]

            def _fmt_num(x) -> str:
                if x is None:
                    return "NA"
                try:
                    return f"{float(x):.2f}"
                except Exception:
                    return str(x)

            # Fallback: if we couldn't find any Instrument tickers via ALIASES_TICKER
            # (common when the graph only has company-name entities like "NVIDIA"),
            # try a simple consonant-based abbreviation match against Instrument tickers.
            if not tickers and matched_entities:
                vowels = set("aeiou")

                def consonant_abbr(s: str) -> str:
                    letters = [ch for ch in (s or "").lower() if ch.isalpha()]
                    cons = [ch for ch in letters if ch not in vowels]
                    return "".join(cons).upper()

                candidates: set[str] = set()
                for en in matched_entities:
                    abbr = consonant_abbr(en)
                    if 2 <= len(abbr) <= 6:
                        candidates.add(abbr)

                if candidates:
                    cand_rows = session.run(
                        """
                        MATCH (i:Instrument)
                        WHERE i.ticker IN $cands
                        RETURN DISTINCT i.ticker AS ticker
                        """,
                        {"cands": sorted(candidates)},
                    ).data()
                    tickers = [r["ticker"] for r in cand_rows if r.get("ticker")]
        if not tickers:
            # Extract possible tickers from the query. Keep it conservative: 1-5 uppercase letters.
            explicit = sorted({t.upper() for t in re.findall(r"\b[A-Z]{1,5}\b", query)})
            if explicit:
                cand_rows = session.run(
                    """
                    MATCH (i:Instrument)
                    WHERE i.ticker IN $cands
                    RETURN DISTINCT i.ticker AS ticker
                    """,
                    {"cands": explicit},
                ).data()
                tickers = [r["ticker"] for r in cand_rows if r.get("ticker")]

            dstart = ts_to_datestr(time_start)
            dend = ts_to_datestr(time_end)
            is_exact_day = dstart is not None and dend is not None and dstart == dend

            for ticker in tickers:
                bars = session.run(
                    """
                    MATCH (i:Instrument {ticker: $ticker})<-[:FOR_INSTRUMENT]-(b:MarketBar)
                    WHERE ($dstart IS NULL OR b.bar_date >= date($dstart))
                      AND ($dend IS NULL OR b.bar_date <= date($dend))
                    RETURN b.bar_date AS bar_date,
                           b.open AS open,
                           b.high AS high,
                           b.low AS low,
                           b.close AS close
                    ORDER BY b.bar_date DESC
                    LIMIT $limit
                    """,
                    {
                        "ticker": ticker,
                        "dstart": dstart,
                        "dend": dend,
                        "limit": 7,
                    },
                ).data()

                # If the user asked for an exact date and that date had no trading bar
                # (weekend/holiday/missing series), fall back to the nearest *previous*
                # trading day only.
                if not bars and is_exact_day:
                    prev = session.run(
                        """
                        MATCH (i:Instrument {ticker: $ticker})<-[:FOR_INSTRUMENT]-(b:MarketBar)
                        WHERE b.bar_date < date($dstart)
                        RETURN b.bar_date AS bar_date,
                               b.open AS open,
                               b.high AS high,
                               b.low AS low,
                               b.close AS close
                        ORDER BY b.bar_date DESC
                        LIMIT 1
                        """,
                        {"ticker": ticker, "dstart": dstart},
                    ).data()
                    bars = prev

                if not bars:
                    continue

                # Sort ascending for start/end summary.
                bars_sorted = sorted(bars, key=lambda r: r["bar_date"])
                start_close = bars_sorted[0].get("close")
                end_close = bars_sorted[-1].get("close")

                pct_change = None
                if start_close is not None and end_close is not None:
                    try:
                        pct_change = (
                            (float(end_close) - float(start_close)) / float(start_close) * 100.0
                        )
                    except Exception:
                        pct_change = None

                lows = [b.get("low") for b in bars_sorted if b.get("low") is not None]
                highs = [b.get("high") for b in bars_sorted if b.get("high") is not None]
                min_low = min(float(x) for x in lows) if lows else None
                max_high = max(float(x) for x in highs) if highs else None

                line_items = []
                for b in bars:
                    bd = b["bar_date"]
                    bd_s = bd.isoformat() if hasattr(bd, "isoformat") else str(bd)
                    line_items.append(
                        f"{bd_s}: O={_fmt_num(b.get('open'))} C={_fmt_num(b.get('close'))}"
                    )

                start_bd = bars_sorted[0]["bar_date"]
                end_bd = bars_sorted[-1]["bar_date"]
                start_bd_s = start_bd.isoformat() if hasattr(start_bd, "isoformat") else str(start_bd)
                end_bd_s = end_bd.isoformat() if hasattr(end_bd, "isoformat") else str(end_bd)

                summary = (
                    f"Market bars for {ticker} in window: {start_bd_s} to {end_bd_s} "
                    f"(start close={_fmt_num(start_close)}, end close={_fmt_num(end_close)}"
                )
                if is_exact_day and start_bd_s != dstart:
                    summary += ", used nearest previous trading day"
                if pct_change is not None:
                    summary += f", pct_change={pct_change:.2f}%"
                if min_low is not None:
                    summary += f", min low={_fmt_num(min_low)}"
                if max_high is not None:
                    summary += f", max high={_fmt_num(max_high)}"
                summary += ")"

                market_texts.append(summary + "\n" + "\n".join(line_items))

    # Layer 3: Semantic fine-grained retrieval 
    scored = []
    for r in rows:
        emb = np.array(r["embedding"], dtype=np.float32)
        sim = cosine_sim(qvec, emb)

        ts = r.get("published_ts")
        if ts is None:
            recency_weight = 0.85
        else:
            age = max(0, now_ts - int(ts))
            recency_weight = float(
                np.exp(-np.log(2) * age / half_life_seconds)
            )

        entity_boost = 1.0 if r["chunk_uid"] in entity_chunk_uids else 0.0
        score = (0.70 * sim) + (0.20 * recency_weight) + (0.10 * entity_boost)

        r["semantic_sim"] = sim
        r["recency_weight"] = recency_weight
        r["entity_match"] = bool(entity_boost)
        r["score"] = score
        scored.append(r)

    scored.sort(key=lambda x: x["score"], reverse=True)
    seeds = scored[:top_k]

    # Graph expansion via NEXT_CHUNK neighbours
    expanded = {s["chunk_uid"]: s for s in seeds}
    with driver.session() as session:
        for s in seeds:
            neighbors = session.run(
                """
                MATCH (c:Chunk {chunk_uid: $chunk_uid})
                OPTIONAL MATCH (c)-[:NEXT_CHUNK]->(n1:Chunk)
                OPTIONAL MATCH (c)<-[:NEXT_CHUNK]-(p1:Chunk)
                WITH collect(DISTINCT n1) + collect(DISTINCT p1) AS nearby
                UNWIND nearby AS n
                WITH n
                WHERE n IS NOT NULL
                  AND ($source_filter IS NULL OR n.source = $source_filter)
                RETURN collect(n) AS nearby
                """,
                {"chunk_uid": s["chunk_uid"], "source_filter": source_filter},
            ).single()

            if neighbors and neighbors["nearby"]:
                for n in neighbors["nearby"]:
                    uid = n.get("chunk_uid")
                    if uid and uid not in expanded:
                        expanded[uid] = {
                            "chunk_uid": uid,
                            "text": n.get("text", ""),
                            "source": n.get("source"),
                            "title": s["title"],
                            "url": s["url"],
                            "article_id": n.get("article_id"),
                            "chunk_id": n.get("chunk_id"),
                            "score": s["score"] * 0.95,
                        }

    expanded_list = sorted(
        expanded.values(), key=lambda x: x.get("score", 0), reverse=True
    )
    return expanded_list[:expanded_k], market_texts


# Generation 

SYSTEM_PROMPT_TEMPLATE = (
    "You are a finance assistant. Answer the user's question using only the "
    "provided context (news chunks and, when available, market data snippets). "
    "The articles in your database span from {date_min} to {date_max}. "
    "If there is not enough evidence, say so clearly."
)

def generate_answer(query: str, context: str, pipe, system_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer directly and concisely."},
    ]
    out = pipe(messages, max_new_tokens=512, do_sample=False)
    raw = out[0]["generated_text"][-1]["content"]
    return strip_think_tags(raw)


# Main loop 

def main():
    print("Loading embedding model...")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    print("Loading generation model...")
    pipe = pipeline("text-generation", model=GEN_MODEL_NAME)

    print("Connecting to Neo4j...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()

    with driver.session() as session:
        date_range = session.run(
            """
            MATCH (a:Article)
            WHERE a.published IS NOT NULL
            RETURN min(a.published) AS earliest, max(a.published) AS latest
            """
        ).single()
    date_min = date_range["earliest"] or "unknown"
    date_max = date_range["latest"] or "unknown"
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        date_min=date_min, date_max=date_max
    )

    print(f"\n--- TG-RAG Chatbot ready ---")
    print(f"Articles from {date_min} to {date_max}")
    print("Type your question (or 'quit' to exit).\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not query or query.lower() in {"quit", "exit", "q"}:
            print("Bye!")
            break

        t0 = time.perf_counter()

        source_filter = extract_source_filter(query)
        if source_filter:
            print(f"  [source filter: {source_filter}]")

        # Step 1: Temporal Query Decomposition 
        sub_queries = decompose_query(query, pipe)
        if len(sub_queries) > 1:
            print(f"  [decomposed into {len(sub_queries)} sub-queries]")

        # Step 2: Three-layer retrieval per sub-query
        all_contexts: list[str] = []
        for sq in sub_queries:
            ts_start = resolve_date(sq["time_start"])
            ts_end = resolve_date(sq["time_end"])

            if sq["time_start"] or sq["time_end"]:
                print(
                    f"  [time filter: "
                    f"{sq.get('time_start', '?')} -> "
                    f"{sq.get('time_end', '?')}]"
                )

            retrieved_chunks, market_texts = three_layer_retrieve(
                query=sq["query"],
                embed_model=embed_model,
                driver=driver,
                top_k=3,
                expanded_k=6,
                recency_half_life_days=5,
                source_filter=source_filter,
                time_start=ts_start,
                time_end=ts_end,
            )

            if retrieved_chunks:
                period_label = ""
                if sq.get("time_start") and sq.get("time_end"):
                    period_label = f"[{sq['time_start']} to {sq['time_end']}] "
                for x in retrieved_chunks:
                    if x.get("text"):
                        all_contexts.append(f"{period_label}{x['text']}")

            # Always include market snippets when available, even if we had no news chunks.
            if market_texts:
                period_label = ""
                if sq.get("time_start") and sq.get("time_end"):
                    period_label = f"[{sq['time_start']} to {sq['time_end']}] "
                for t in market_texts:
                    all_contexts.append(f"{period_label}{t}")

        # Step 3: Single generation from merged context
        if not all_contexts:
            print("Assistant: No relevant chunks found.\n")
            continue

        context = "\n\n".join(all_contexts)
        final = generate_answer(query, context, pipe, system_prompt)
        elapsed = time.perf_counter() - t0
        print(f"\nAssistant: {final}")
        print(f"  [{elapsed:.1f}s]\n")

    driver.close()


if __name__ == "__main__":
    main()
