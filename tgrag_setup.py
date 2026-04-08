"""
tgrag_setup.py — T-GRAG Knowledge Graph Construction

Implements the Temporal Knowledge Graph Generator from the T-GRAG paper:
  "T-GRAG: A Dynamic GraphRAG Framework for Resolving Temporal Conflicts
   and Redundancy in Knowledge Retrieval" (MM '25)

Pipeline (mirrors Algorithm 1, Step 1):
  1. Load scraped articles from JSON
  2. Filter duplicates (content hash) and already-ingested articles
  3. Assign each article a Period (temporal bucket)
  4. Split articles into Chunks (fixed text blocks d^{t_i}_j)
  5. For each Chunk, extract:
       - KnowledgeNode per entity found  (e^{T_i}_i, one node per entity × period)
       - Knowledge units per entity      (k^{t_i}, atomic facts)
       - Typed RELATED_TO edges between entity KnowledgeNodes in same period
  6. Embed KnowledgeNodes (coarse, R_node) and Knowledge units (fine, R_knowledge)
     and Chunks (source text extractor)
  7. Write everything to Neo4j
  8. Link SAME_ENTITY_AS chains across periods

Key design decisions vs. original implementation:
  - REMOVED: Entity/Instrument/MarketBar nodes (hist_to_db.py is retired)
  - REMOVED: :ALIASES_TICKER, :FOR_INSTRUMENT, :MENTIONS_INSTRUMENT edges
  - REMOVED: CO_OCCURS_CHUNK / CO_OCCURS_ARTICLE edges (replaced by typed RELATED_TO)
  - NEW: KnowledgeNode is the temporal entity snapshot — one per (entity, period)
  - NEW: Knowledge nodes are atomic facts extracted per chunk per entity
  - NEW: period_key indexed on Chunk, KnowledgeNode for fast R_time subgraph pull
  - NEW: SAME_ENTITY_AS chain links the same entity across periods (temporal evolution)
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from dotenv import load_dotenv
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

from graph_schema import (
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    PERIOD_GRANULARITY,
    period_key_for, knowledge_node_uid,
    ensure_schema, wipe_tgrag_graph,
    UPSERT_PERIOD, UPSERT_ARTICLE, UPSERT_CHUNK, UPSERT_CHUNK_EMBEDDING,
    UPSERT_KNOWLEDGE_NODE, UPSERT_KNOWLEDGE_NODE_EMBEDDING,
    UPSERT_KNOWLEDGE, UPSERT_KNOWLEDGE_EMBEDDING,
    UPSERT_CHUNK_TO_KN_EDGE, UPSERT_RELATION, LINK_SAME_ENTITY,
)

load_dotenv()

# Config

ARTICLES_JSON = Path(os.getenv("ARTICLES_JSON", "cnbc_articles.json"))
TICKER_MAP_PATH = Path(os.getenv("TICKER_MAP_PATH", "ticker_company_map.csv"))

# Chunking
CHUNK_TOKEN_TARGET = int(os.getenv("CHUNK_TOKEN_TARGET", "400"))   # ~400 tokens per chunk
CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", "50"))

# Embedding model — must match PERIOD_GRANULARITY vector index dim
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "dunzhang/stella_en_1.5B_v5")
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "32"))

# NER / extraction model
NER_MODEL_NAME = os.getenv("NER_MODEL_NAME", "dslim/bert-large-NER")

# Minimum article quality gate
MIN_ARTICLE_WORDS = int(os.getenv("MIN_ARTICLE_WORDS", "80"))

# Neo4j write batch size
NEO4J_BATCH_SIZE = int(os.getenv("NEO4J_BATCH_SIZE", "200"))


# Helpers

def _parse_published(raw: str | None) -> datetime | None:
    if not raw:
        return None
    for fmt in [
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S",
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S GMT",
        "%Y-%m-%d",
    ]:
        try:
            return datetime.strptime(raw.strip(), fmt)
        except ValueError:
            continue
    return None


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _article_id(url: str, published: datetime | None) -> str:
    """Stable ID: hash of URL + date (not time, to survive re-scrapes)."""
    date_str = published.strftime("%Y-%m-%d") if published else "nodate"
    raw = f"{url}::{date_str}"
    return hashlib.md5(raw.encode()).hexdigest()


# Loading

def load_articles(json_path: Path = ARTICLES_JSON) -> list[dict]:
    with open(json_path, encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("articles", [])


def filter_articles(
    articles: list[dict],
    skip_ids: set[str],
    min_words: int = MIN_ARTICLE_WORDS,
) -> list[dict]:
    """
    Apply quality gates and skip already-ingested articles.
    Returns enriched article dicts with: article_id, published_dt,
    period_key, content_hash.
    """
    seen_hashes: set[str] = set()
    out = []

    for art in articles:
        # Status gate
        if art.get("status") not in ("ok", None):
            continue

        text = (art.get("text") or "").strip()
        if not text:
            continue

        # Minimum length gate
        if len(text.split()) < min_words:
            continue

        # Content-hash deduplication (catches same article at different URLs)
        chash = _content_hash(text)
        if chash in seen_hashes:
            continue
        seen_hashes.add(chash)

        published_dt = _parse_published(art.get("published"))
        # Fall back to scrape time if no published date
        if published_dt is None:
            scraped = art.get("scraped_at") or art.get("scraped_at_utc")
            published_dt = _parse_published(scraped)
        if published_dt is None:
            published_dt = datetime.now(timezone.utc)

        article_id = _article_id(art.get("url", ""), published_dt)

        if article_id in skip_ids:
            continue

        period_key = period_key_for(published_dt, PERIOD_GRANULARITY)

        out.append({
            **art,
            "article_id":   article_id,
            "published_dt": published_dt,
            "period_key":   period_key,
            "content_hash": chash,
            "text":         text,
        })

    return out


def get_existing_article_ids(driver) -> set[str]:
    with driver.session() as session:
        result = session.run("MATCH (a:Article) RETURN a.article_id AS id")
        return {r["id"] for r in result}


# Chunking

def _approx_tokens(text: str) -> int:
    """Rough token count: ~0.75 words per token (good enough for chunking)."""
    return int(len(text.split()) / 0.75)


def chunk_text(
    text: str,
    target_tokens: int = CHUNK_TOKEN_TARGET,
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
) -> list[str]:
    """
    Split text into overlapping chunks of approximately target_tokens each.
    Splits on sentence boundaries where possible.
    """
    # Split into sentences (simple regex; replace with spaCy sentencizer if needed)
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for sent in sentences:
        sent_tokens = _approx_tokens(sent)

        if current_tokens + sent_tokens > target_tokens and current:
            chunks.append(" ".join(current))
            # Overlap: keep last N tokens worth of sentences
            overlap: list[str] = []
            overlap_count = 0
            for s in reversed(current):
                t = _approx_tokens(s)
                if overlap_count + t > overlap_tokens:
                    break
                overlap.insert(0, s)
                overlap_count += t
            current = overlap
            current_tokens = overlap_count

        current.append(sent)
        current_tokens += sent_tokens

    if current:
        chunks.append(" ".join(current))

    return [c for c in chunks if c.strip()]


def build_chunks(articles: list[dict]) -> list[dict]:
    """
    For each article, produce a list of chunk dicts ready for Neo4j.
    Each chunk carries article_id, period_key, published_date for denormalized
    fast lookups (avoids traversal through Article during R_time filtering).
    """
    all_chunks = []
    for art in articles:
        text_blocks = chunk_text(art["text"])
        for i, block in enumerate(text_blocks):
            chunk_uid = f"{art['article_id']}::chunk::{i}"
            all_chunks.append({
                "chunk_uid":      chunk_uid,
                "text":           block,
                "chunk_index":    i,
                "article_id":     art["article_id"],
                "published_date": art["published_dt"].strftime("%Y-%m-%d"),
                "period_key":     art["period_key"],
                "token_count":    _approx_tokens(block),
                # Populated later by embed_chunks()
                "embedding":      None,
            })
    return all_chunks


# Entity Extraction

def load_extraction_model(model_name: str = NER_MODEL_NAME):
    """Load a token-classification NER pipeline."""
    from transformers import pipeline as hf_pipeline
    device = 0 if torch.cuda.is_available() else -1
    print(f"[NER] Loading {model_name} on {'GPU' if device == 0 else 'CPU'}")
    return hf_pipeline(
        "ner",
        model=model_name,
        aggregation_strategy="simple",
        device=device,
    )


def load_ticker_company_map(path: Path = TICKER_MAP_PATH) -> dict[str, str]:
    """
    Load ticker -> canonical company name map.
    CSV must have columns: ticker, company_name
    Returns {company_name_lower: ticker}
    """
    if not path.is_file():
        return {}
    import csv
    mapping = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ticker = (row.get("ticker") or "").strip().upper()
            name = (row.get("company_name") or "").strip()
            if ticker and name:
                mapping[name.lower()] = ticker
    return mapping


def _canonicalize(name: str) -> str:
    """Normalize entity name for KnowledgeNode identity."""
    return re.sub(r'\s+', ' ', name.strip().lower())


def _map_ticker(canonical: str, ticker_lookup: dict[str, str]) -> str | None:
    """Exact then prefix match against ticker company map."""
    direct = ticker_lookup.get(canonical)
    if direct:
        return direct
    # Try longest prefix match
    for company, ticker in ticker_lookup.items():
        if canonical.startswith(company) or company.startswith(canonical):
            return ticker
    return None


# Entity types we care about — filters NER output to relevant classes
RELEVANT_ENTITY_TYPES = frozenset({
    "ORG", "PER", "GPE", "LOC", "PRODUCT", "EVENT",
    # BERT-NER uses B-/I- prefixes which aggregation_strategy handles,
    # but some models emit bare type strings:
    "B-ORG", "I-ORG", "B-PER", "I-PER",
})


def extract_entities_from_chunks(
    chunks: list[dict],
    pipe,
    ticker_lookup: dict[str, str],
) -> list[dict]:
    """
    Run NER over chunk texts. Returns a list of entity mention dicts:
    {
        chunk_uid, canonical_name, entity_type, raw_name,
        ticker (or None), period_key
    }

    Deduplication: same canonical_name × chunk_uid → one mention.
    """
    texts = [c["text"] for c in chunks]
    mentions = []
    seen: set[tuple[str, str]] = set()  # (chunk_uid, canonical_name)

    # Process in batches
    batch_size = 16
    for batch_start in range(0, len(chunks), batch_size):
        batch_chunks = chunks[batch_start: batch_start + batch_size]
        batch_texts = [c["text"] for c in batch_chunks]

        try:
            batch_results = pipe(batch_texts)
        except Exception as e:
            print(f"[NER] Batch {batch_start} failed: {e}")
            batch_results = [[] for _ in batch_chunks]

        for chunk, ner_result in zip(batch_chunks, batch_results):
            for ent in ner_result:
                etype = ent.get("entity_group") or ent.get("entity", "")
                # Strip B-/I- prefixes if model didn't aggregate
                etype = re.sub(r'^[BI]-', '', etype)
                if etype not in RELEVANT_ENTITY_TYPES:
                    continue

                raw_name = ent.get("word", "").strip()
                if not raw_name or len(raw_name) < 2:
                    continue

                canonical = _canonicalize(raw_name)
                key = (chunk["chunk_uid"], canonical)
                if key in seen:
                    continue
                seen.add(key)

                mentions.append({
                    "chunk_uid":      chunk["chunk_uid"],
                    "canonical_name": canonical,
                    "raw_name":       raw_name,
                    "entity_type":    etype,
                    "ticker":         _map_ticker(canonical, ticker_lookup),
                    "period_key":     chunk["period_key"],
                    "article_id":     chunk["article_id"],
                })

    return mentions


def build_knowledge_nodes(
    mentions: list[dict],
) -> dict[str, dict]:
    """
    Aggregate mentions into KnowledgeNode specs.
    One KnowledgeNode per (canonical_name × period_key).

    Returns {kn_uid: kn_dict}
    """
    nodes: dict[str, dict] = {}
    for m in mentions:
        kn_uid = knowledge_node_uid(m["canonical_name"], m["period_key"])
        if kn_uid not in nodes:
            nodes[kn_uid] = {
                "kn_uid":         kn_uid,
                "canonical_name": m["canonical_name"],
                "entity_type":    m["entity_type"],
                "period_key":     m["period_key"],
                "ticker":         m["ticker"],
                # description is the aggregated knowledge text — built below
                "description":    "",
                "source_chunks":  [],  # chunk_uids that mention this entity in this period
                "embedding":      None,
            }
        kn = nodes[kn_uid]
        # Accumulate which chunks mention this KN (for SOURCED_FROM edges)
        if m["chunk_uid"] not in kn["source_chunks"]:
            kn["source_chunks"].append(m["chunk_uid"])
        # First occurrence wins for entity_type if already set
        if not kn["entity_type"] and m["entity_type"]:
            kn["entity_type"] = m["entity_type"]

    return nodes


def build_knowledge_units(
    chunks: list[dict],
    mentions: list[dict],
    knowledge_nodes: dict[str, dict],
) -> list[dict]:
    """
    Extract atomic Knowledge units (k^{t_i}) from chunks for each entity
    mentioned in that chunk.

    Strategy: for each (chunk, entity) pair, produce one Knowledge unit
    whose text is the chunk's most entity-relevant sentence(s).
    This approximates T-GRAG's fine-grained knowledge granularity
    without requiring a full LLM extraction pass.
    """
    chunk_by_uid = {c["chunk_uid"]: c for c in chunks}
    knowledge_units: list[dict] = []
    seen_k: set[str] = set()

    for m in mentions:
        chunk = chunk_by_uid.get(m["chunk_uid"])
        if not chunk:
            continue

        kn_uid = knowledge_node_uid(m["canonical_name"], m["period_key"])
        if kn_uid not in knowledge_nodes:
            continue

        # Extract the sentence(s) in the chunk that mention the entity
        relevant_text = _extract_relevant_sentences(
            chunk["text"], m["raw_name"], m["canonical_name"]
        )
        if not relevant_text:
            relevant_text = chunk["text"][:500]  # fallback: first 500 chars

        knowledge_uid = hashlib.md5(
            f"{kn_uid}::{m['chunk_uid']}".encode()
        ).hexdigest()[:16]

        if knowledge_uid in seen_k:
            continue
        seen_k.add(knowledge_uid)

        knowledge_units.append({
            "knowledge_uid": knowledge_uid,
            "text":          relevant_text,
            "kn_uid":        kn_uid,
            "chunk_uid":     m["chunk_uid"],
            "period_key":    m["period_key"],
            "fact_type":     m["entity_type"],
            "embedding":     None,
        })

        # Also accumulate into KnowledgeNode description (for R_node coarse embedding)
        kn = knowledge_nodes[kn_uid]
        if relevant_text not in kn["description"]:
            kn["description"] = (kn["description"] + " " + relevant_text).strip()

    return knowledge_units


def _extract_relevant_sentences(
    text: str, raw_name: str, canonical_name: str
) -> str:
    """
    Return sentences from text that mention the entity by name.
    Matches case-insensitively on either the raw or canonical form.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    patterns = [re.escape(raw_name), re.escape(canonical_name)]
    pattern = re.compile("|".join(patterns), re.IGNORECASE)
    matched = [s for s in sentences if pattern.search(s)]
    return " ".join(matched[:3])  # cap at 3 sentences to keep knowledge atomic


def build_relations(
    mentions: list[dict],
    knowledge_nodes: dict[str, dict],
) -> list[dict]:
    """
    Build typed RELATED_TO edges between KnowledgeNodes co-occurring in the
    same chunk.

    Unlike the old CO_OCCURS_ARTICLE approach, edges are:
    - Scoped to the same chunk (lower noise, ~chunk-local co-occurrence)
    - Scoped to the same period_key (no cross-period edge pollution)
    - typed as "CO_OCCURS" (extensible: LLM relation extraction can add
      typed relations like "REPORTS_ON", "SUBSIDIARY_OF" later)
    """
    # Build chunk -> list of kn_uids index
    chunk_to_kns: dict[str, list[str]] = {}
    for m in mentions:
        kn_uid = knowledge_node_uid(m["canonical_name"], m["period_key"])
        if kn_uid not in knowledge_nodes:
            continue
        chunk_to_kns.setdefault(m["chunk_uid"], [])
        if kn_uid not in chunk_to_kns[m["chunk_uid"]]:
            chunk_to_kns[m["chunk_uid"]].append(kn_uid)

    relations = []
    seen_edges: set[tuple[str, str]] = set()

    for chunk_uid, kn_uids in chunk_to_kns.items():
        # Get period from first KN (all KNs from same chunk share same period)
        period_key = knowledge_nodes[kn_uids[0]]["period_key"] if kn_uids else None
        if not period_key:
            continue

        # Only build edges if <= 10 entities in chunk (>10 = low-signal dense article)
        if len(kn_uids) > 10:
            continue

        for i, src in enumerate(kn_uids):
            for tgt in kn_uids[i + 1:]:
                # Canonical edge: always smaller uid -> larger uid
                edge = (min(src, tgt), max(src, tgt))
                if edge in seen_edges:
                    continue
                seen_edges.add(edge)
                relations.append({
                    "src_uid":    src,
                    "tgt_uid":    tgt,
                    "rel_type":   "CO_OCCURS",
                    "period_key": period_key,
                    "description": f"Co-occur in chunk {chunk_uid}",
                })

    return relations


# Embedding

_embed_model: SentenceTransformer | None = None


def get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        print(f"[embed] Loading {EMBED_MODEL_NAME}")
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    return _embed_model


def embed_texts(texts: list[str], batch_size: int = EMBED_BATCH_SIZE) -> list[list[float]]:
    model = get_embed_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # cosine sim = dot product after normalization
    )
    return [e.tolist() for e in embeddings]


def embed_chunks(chunks: list[dict]) -> list[dict]:
    print(f"[embed] Embedding {len(chunks)} chunks ...")
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)
    for c, emb in zip(chunks, embeddings):
        c["embedding"] = emb
    return chunks


def embed_knowledge_nodes(knowledge_nodes: dict[str, dict]) -> dict[str, dict]:
    nodes = list(knowledge_nodes.values())
    print(f"[embed] Embedding {len(nodes)} KnowledgeNodes ...")
    # Embed on description (aggregated knowledge text)
    texts = [
        (n["description"] or n["canonical_name"])
        for n in nodes
    ]
    embeddings = embed_texts(texts)
    for node, emb in zip(nodes, embeddings):
        node["embedding"] = emb
    return knowledge_nodes


def embed_knowledge_units(knowledge_units: list[dict]) -> list[dict]:
    print(f"[embed] Embedding {len(knowledge_units)} Knowledge units ...")
    texts = [k["text"] for k in knowledge_units]
    embeddings = embed_texts(texts)
    for k, emb in zip(knowledge_units, embeddings):
        k["embedding"] = emb
    return knowledge_units


# Neo4j writing

def _batched(lst: list, n: int):
    for i in range(0, len(lst), n):
        yield lst[i: i + n]


def populate_neo4j(
    articles: list[dict],
    chunks: list[dict],
    knowledge_nodes: dict[str, dict],
    knowledge_units: list[dict],
    relations: list[dict],
    mentions: list[dict],
    reset: bool = False,
    embedding_dim: int = 768,
) -> None:
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()

    with driver.session() as session:
        if reset:
            print("[neo4j] Wiping existing graph ...")
            wipe_tgrag_graph(session)

        ensure_schema(session, embedding_dim=embedding_dim)

        # Periods 
        period_keys = sorted({a["period_key"] for a in articles})
        print(f"[neo4j] Upserting {len(period_keys)} Period nodes ...")
        for pk in period_keys:
            session.run(UPSERT_PERIOD, {
                "key":         pk,
                "granularity": PERIOD_GRANULARITY,
                "label":       pk,
            })

        # Articles 
        print(f"[neo4j] Upserting {len(articles)} Article nodes ...")
        for art in articles:
            session.run(UPSERT_ARTICLE, {
                "article_id":    art["article_id"],
                "url":           art.get("url", ""),
                "title":         art.get("title", ""),
                "source":        art.get("source", ""),
                "published_date": art["published_dt"].strftime("%Y-%m-%d"),
                "scraped_at":    datetime.now(timezone.utc).isoformat(),
                "period_key":    art["period_key"],
                "content_hash":  art["content_hash"],
            })

        # Chunks 
        print(f"[neo4j] Upserting {len(chunks)} Chunk nodes ...")
        for batch in _batched(chunks, NEO4J_BATCH_SIZE):
            for c in batch:
                session.run(UPSERT_CHUNK, {
                    "chunk_uid":      c["chunk_uid"],
                    "text":           c["text"],
                    "chunk_index":    c["chunk_index"],
                    "article_id":     c["article_id"],
                    "published_date": c["published_date"],
                    "period_key":     c["period_key"],
                    "token_count":    c["token_count"],
                })

        print(f"[neo4j] Writing Chunk embeddings ...")
        for batch in _batched(chunks, NEO4J_BATCH_SIZE):
            for c in batch:
                if c.get("embedding"):
                    session.run(UPSERT_CHUNK_EMBEDDING, {
                        "chunk_uid": c["chunk_uid"],
                        "embedding": c["embedding"],
                    })

        # KnowledgeNodes 
        kn_list = list(knowledge_nodes.values())
        print(f"[neo4j] Upserting {len(kn_list)} KnowledgeNode nodes ...")
        for batch in _batched(kn_list, NEO4J_BATCH_SIZE):
            for kn in batch:
                session.run(UPSERT_KNOWLEDGE_NODE, {
                    "kn_uid":         kn["kn_uid"],
                    "canonical_name": kn["canonical_name"],
                    "entity_type":    kn["entity_type"],
                    "period_key":     kn["period_key"],
                    "description":    kn["description"],
                })

        print(f"[neo4j] Writing KnowledgeNode embeddings ...")
        for batch in _batched(kn_list, NEO4J_BATCH_SIZE):
            for kn in batch:
                if kn.get("embedding"):
                    session.run(UPSERT_KNOWLEDGE_NODE_EMBEDDING, {
                        "kn_uid":    kn["kn_uid"],
                        "embedding": kn["embedding"],
                    })

        # Chunk -> KnowledgeNode edges (SOURCED_FROM) 
        print(f"[neo4j] Writing Chunk->KnowledgeNode edges ...")
        for kn in kn_list:
            for chunk_uid in kn.get("source_chunks", []):
                session.run(UPSERT_CHUNK_TO_KN_EDGE, {
                    "chunk_uid": chunk_uid,
                    "kn_uid":    kn["kn_uid"],
                })

        # Knowledge units 
        print(f"[neo4j] Upserting {len(knowledge_units)} Knowledge nodes ...")
        for batch in _batched(knowledge_units, NEO4J_BATCH_SIZE):
            for k in batch:
                session.run(UPSERT_KNOWLEDGE, {
                    "knowledge_uid": k["knowledge_uid"],
                    "text":          k["text"],
                    "kn_uid":        k["kn_uid"],
                    "chunk_uid":     k["chunk_uid"],
                    "period_key":    k["period_key"],
                    "fact_type":     k["fact_type"],
                })

        print(f"[neo4j] Writing Knowledge embeddings ...")
        for batch in _batched(knowledge_units, NEO4J_BATCH_SIZE):
            for k in batch:
                if k.get("embedding"):
                    session.run(UPSERT_KNOWLEDGE_EMBEDDING, {
                        "knowledge_uid": k["knowledge_uid"],
                        "embedding":     k["embedding"],
                    })

        # Relations 
        print(f"[neo4j] Upserting {len(relations)} RELATED_TO edges ...")
        for batch in _batched(relations, NEO4J_BATCH_SIZE):
            for r in batch:
                session.run(UPSERT_RELATION, {
                    "src_uid":     r["src_uid"],
                    "tgt_uid":     r["tgt_uid"],
                    "rel_type":    r["rel_type"],
                    "period_key":  r["period_key"],
                    "description": r["description"],
                    
                })

        # SAME_ENTITY_AS cross-period chains 
        print("[neo4j] Linking SAME_ENTITY_AS chains across periods ...")
        result = session.run(LINK_SAME_ENTITY).single()
        linked = result["linked"] if result else 0
        print(f"  Linked {linked} cross-period entity pairs")

    driver.close()
    print("[neo4j] Write complete.")


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------

def run_setup(
    reset: bool = False,
    skip_entities: bool = False,
) -> None:
    """
    Full T-GRAG setup pipeline. Called by update.py.
    """
    # 1. Load and filter articles
    print("[setup] Loading articles ...")
    raw_articles = load_articles()

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()
    skip_ids: set[str] = set() if reset else get_existing_article_ids(driver)
    driver.close()

    articles = filter_articles(raw_articles, skip_ids)
    print(f"[setup] {len(articles)} new articles after filtering (skipped {len(skip_ids)} existing)")

    if not articles:
        print("[setup] No new articles to process. Done.")
        return

    # 2. Chunk
    print("[setup] Chunking articles ...")
    chunks = build_chunks(articles)
    print(f"[setup] Produced {len(chunks)} chunks")

    # 3. Entity extraction
    knowledge_nodes: dict[str, dict] = {}
    knowledge_units: list[dict] = []
    relations: list[dict] = []
    mentions: list[dict] = []

    if not skip_entities:
        pipe = load_extraction_model()
        ticker_lookup = load_ticker_company_map()

        print("[setup] Extracting entities ...")
        mentions = extract_entities_from_chunks(chunks, pipe, ticker_lookup)
        print(f"[setup] {len(mentions)} entity mentions extracted")

        knowledge_nodes = build_knowledge_nodes(mentions)
        print(f"[setup] {len(knowledge_nodes)} KnowledgeNodes (entity × period)")

        knowledge_units = build_knowledge_units(chunks, mentions, knowledge_nodes)
        print(f"[setup] {len(knowledge_units)} Knowledge units")

        relations = build_relations(mentions, knowledge_nodes)
        print(f"[setup] {len(relations)} RELATED_TO edges")

        del pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 4. Embed
    chunks = embed_chunks(chunks)
    if knowledge_nodes:
        knowledge_nodes = embed_knowledge_nodes(knowledge_nodes)
    if knowledge_units:
        knowledge_units = embed_knowledge_units(knowledge_units)

    # Infer embedding dim from first chunk
    embedding_dim = len(chunks[0]["embedding"]) if chunks and chunks[0].get("embedding") else 768

    # 5. Write to Neo4j
    populate_neo4j(
        articles=articles,
        chunks=chunks,
        knowledge_nodes=knowledge_nodes,
        knowledge_units=knowledge_units,
        relations=relations,
        mentions=mentions,
        reset=reset,
        embedding_dim=embedding_dim,
    )

    print("[setup] Pipeline complete.")