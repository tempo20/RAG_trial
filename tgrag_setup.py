"""
tgrag_setup.py - T-GRAG Knowledge Graph Construction

Implements the Temporal Knowledge Graph Generator from the T-GRAG paper:
  "T-GRAG: A Dynamic GraphRAG Framework for Resolving Temporal Conflicts
   and Redundancy in Knowledge Retrieval" (MM '25)

Pipeline (mirrors Algorithm 1, Step 1):
  1. Load scraped articles from JSON
  2. Filter duplicates (content hash + MinHash near-dup) and already-ingested articles
  3. Assign each article a Period (temporal bucket)
  4. Split articles into Chunks (fixed text blocks d^{t_i}_j)
  5. For each Chunk, extract:
       - KnowledgeNode per entity found  (e^{T_i}_i, one node per entity - period)
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
  - NEW: KnowledgeNode is the temporal entity snapshot - one per (entity, period)
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
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from dotenv import load_dotenv
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
try:
    from datasketch import MinHash, MinHashLSH
except ImportError:  # pragma: no cover - exercised in dependency-light envs
    MinHash = None
    MinHashLSH = None
try:
    from rapidfuzz import fuzz
except ImportError: 
    fuzz = None

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
FIN_ENTITY_MAP_PATH = Path(os.getenv("FIN_ENTITY_MAP_PATH", "financial_entity_map.csv"))

# Chunking
CHUNK_TOKEN_TARGET = int(os.getenv("CHUNK_TOKEN_TARGET", "400"))   # ~400 tokens per chunk
CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", "50"))

# Embedding model - must match PERIOD_GRANULARITY vector index dim
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "dunzhang/stella_en_1.5B_v5")
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "32"))

# NER / extraction model
NER_MODEL_NAME = os.getenv("NER_MODEL_NAME", "dbmdz/bert-large-cased-finetuned-conll03-english")
ENABLE_FIN_ENTITY_LINKER = os.getenv("ENABLE_FIN_ENTITY_LINKER", "1").strip().lower() in {"1", "true", "yes", "on"}
FIN_LINK_FUZZY_THRESHOLD = float(os.getenv("FIN_LINK_FUZZY_THRESHOLD", "92"))
FIN_LINK_ENABLE_FUZZY = os.getenv("FIN_LINK_ENABLE_FUZZY", "1").strip().lower() in {"1", "true", "yes", "on"}

# Minimum article quality gate
MIN_ARTICLE_WORDS = int(os.getenv("MIN_ARTICLE_WORDS", "80"))

# Deduplication (ingestion-authoritative)
ENABLE_MINHASH_DEDUP = os.getenv("ENABLE_MINHASH_DEDUP", "1").strip().lower() in {"1", "true", "yes", "on"}
MINHASH_LSH_THRESHOLD = float(os.getenv("MINHASH_LSH_THRESHOLD", "0.86"))
MINHASH_NUM_PERM = int(os.getenv("MINHASH_NUM_PERM", "128"))
MINHASH_SHINGLE_SIZE = int(os.getenv("MINHASH_SHINGLE_SIZE", "5"))

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


def _dedup_tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9$_.-]+", text.lower())


def _token_shingles(tokens: list[str], shingle_size: int) -> set[str]:
    if not tokens:
        return set()
    shingle_size = max(1, shingle_size)
    if len(tokens) < shingle_size:
        return {" ".join(tokens)}
    return {
        " ".join(tokens[i:i + shingle_size])
        for i in range(len(tokens) - shingle_size + 1)
    }


def _build_minhash(text: str, num_perm: int, shingle_size: int):
    if MinHash is None:
        return None
    shingles = _token_shingles(_dedup_tokens(text), shingle_size)
    if not shingles:
        return None
    sig = MinHash(num_perm=max(16, num_perm))
    for sh in shingles:
        sig.update(sh.encode("utf-8"))
    return sig


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
    period_key, content_hash. Exact dedup + near dedup are both applied here.
    """
    seen_hashes: set[str] = set()
    candidates = []

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

        candidates.append({
            **art,
            "article_id":   article_id,
            "published_dt": published_dt,
            "period_key":   period_key,
            "content_hash": chash,
            "text":         text,
        })

    if not candidates:
        return []

    if not ENABLE_MINHASH_DEDUP:
        return candidates

    if MinHashLSH is None:
        print("[dedup] datasketch not installed; MinHash near-dedup disabled")
        return candidates

    # Deterministic canonical selection: keep oldest-then-URL article in each near-dup cluster.
    ordered = sorted(
        candidates,
        key=lambda a: (a["published_dt"].isoformat(), a.get("url", "")),
    )
    lsh = MinHashLSH(
        threshold=MINHASH_LSH_THRESHOLD,
        num_perm=max(16, MINHASH_NUM_PERM),
    )

    out = []
    dropped_near_dupes = 0

    for i, art in enumerate(ordered):
        signature = _build_minhash(
            text=art["text"],
            num_perm=MINHASH_NUM_PERM,
            shingle_size=MINHASH_SHINGLE_SIZE,
        )
        # If a signature can't be built, keep article (never drop on uncertain dedup state).
        if signature is None:
            out.append(art)
            continue

        if lsh.query(signature):
            dropped_near_dupes += 1
            continue

        lsh.insert(f"doc::{i}", signature)
        out.append(art)

    if dropped_near_dupes:
        print(f"[dedup] MinHash near-duplicate filter dropped {dropped_near_dupes} articles")

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


_LEGAL_SUFFIXES = re.compile(
    r"""[\s,\.]*\b(
        incorporated|corporation|international|
        holdings|technologies|technology|solutions|
        pharmaceuticals|financial|services|group|
        limited|enterprises|partners|associates|
        inc|corp|ltd|llc|llp|plc|co|sa|ag|nv|bv|se
    )\b[\s\.]*$""",
    re.IGNORECASE | re.VERBOSE,
)

def _strip_legal(name: str) -> str:
    """Iteratively strip trailing legal suffixes until stable."""
    prev = None
    current = name.strip().rstrip(".,")
    while current != prev:
        prev = current
        current = _LEGAL_SUFFIXES.sub("", current).strip().rstrip(".,")
    return current


def _canonicalize(name: str) -> str:
    """
    Normalize entity name for KnowledgeNode identity.
    Lowercases, collapses whitespace, and strips punctuation noise.
    Does NOT strip legal suffixes - that's done separately so the
    raw_name is preserved for display.
    """
    name = name.strip()
    name = re.sub(r"[''`]", "", name)          # smart/straight apostrophes
    name = re.sub(r"[^\w\s&\-]", " ", name)   # keep alphanumeric, &, hyphen
    name = re.sub(r"\s+", " ", name)
    return name.strip().lower()


def load_ticker_company_map(path: Path = TICKER_MAP_PATH) -> tuple[dict[str, str], dict[str, str]]:
    """
    Load ticker_company_map.csv.

    Returns two dicts:
      alias_to_ticker  : {lowercased_alias_or_name: TICKER}
                         built from company_name + all aliases column entries
                         ALSO includes the legal-stripped form of each alias
      ticker_to_canonical : {TICKER: canonical display name}
                            the company_name column value
    """
    if not path.is_file():
        return {}, {}

    import csv
    alias_to_ticker: dict[str, str] = {}
    ticker_to_canonical: dict[str, str] = {}

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ticker = (row.get("ticker") or "").strip().upper()
            name = (row.get("company_name") or "").strip()
            aliases_raw = (row.get("aliases") or "").strip()
            if not ticker or not name:
                continue

            ticker_to_canonical[ticker] = name

            # Build the full set of surface forms to map - ticker
            surface_forms = [name] + [a.strip() for a in aliases_raw.split(";") if a.strip()]
            for form in surface_forms:
                key = _canonicalize(form)
                alias_to_ticker[key] = ticker
                # Also register the legal-stripped form
                stripped_key = _canonicalize(_strip_legal(form))
                if stripped_key and stripped_key != key:
                    alias_to_ticker[stripped_key] = ticker

    return alias_to_ticker, ticker_to_canonical


# -- Entity type scope --------------------------------------------------------
# dbmdz CoNLL NER emits: PER, ORG, LOC, MISC.
# ORG/MISC are linker-driven; PER/LOC keep heuristic fallback filters.
RELEVANT_ENTITY_TYPES = frozenset({"ORG", "PER", "LOC", "MISC"})

# Countries - used to filter LOC down to only countries.
def _slugify(value: str) -> str:
    value = _canonicalize(value)
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-")
    return value


def _prefer_entity_match(existing: dict[str, Any], incoming: dict[str, Any]) -> bool:
    """Prefer ticker-backed mappings, then longer display names."""
    existing_ticker = bool(existing.get("ticker"))
    incoming_ticker = bool(incoming.get("ticker"))
    if incoming_ticker and not existing_ticker:
        return True
    if existing_ticker and not incoming_ticker:
        return False
    return len(str(incoming.get("display_name") or "")) > len(str(existing.get("display_name") or ""))


def load_financial_entity_map(
    fin_map_path: Path = FIN_ENTITY_MAP_PATH,
    ticker_map_path: Path = TICKER_MAP_PATH,
) -> dict[str, dict[str, Any]]:
    """
    Build alias -> financial entity mapping from:
      1) ticker_company_map.csv (listed companies)
      2) financial_entity_map.csv (non-ticker institutions + optional ticker overrides)
    """
    alias_to_entity: dict[str, dict[str, Any]] = {}

    def register_aliases(surface_forms: list[str], entity: dict[str, Any]) -> None:
        for form in surface_forms:
            key = _canonicalize(form)
            if not key:
                continue
            existing = alias_to_entity.get(key)
            if existing and not _prefer_entity_match(existing, entity):
                continue
            alias_to_entity[key] = dict(entity)

            stripped = _canonicalize(_strip_legal(form))
            if stripped and stripped != key:
                existing = alias_to_entity.get(stripped)
                if existing and not _prefer_entity_match(existing, entity):
                    continue
                alias_to_entity[stripped] = dict(entity)

    alias_to_ticker, ticker_to_canonical = load_ticker_company_map(ticker_map_path)
    for alias, ticker in alias_to_ticker.items():
        entity = {
            "canonical_name": ticker,
            "display_name": ticker_to_canonical.get(ticker, ticker),
            "entity_type": "ORG",
            "ticker": ticker,
        }
        register_aliases([alias, ticker], entity)

    if fin_map_path.is_file():
        with open(fin_map_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                display_name = (row.get("display_name") or "").strip()
                canonical_id = (row.get("canonical_id") or "").strip()
                entity_type = (row.get("entity_type") or "ORG").strip().upper() or "ORG"
                ticker = (row.get("ticker") or "").strip().upper() or None
                aliases_raw = (row.get("aliases") or "").strip()
                if not display_name and not canonical_id and not ticker:
                    continue

                canonical_name = ticker or _slugify(canonical_id or display_name)
                if not canonical_name:
                    continue

                entity = {
                    "canonical_name": canonical_name,
                    "display_name": display_name or canonical_name,
                    "entity_type": entity_type,
                    "ticker": ticker,
                }
                surface_forms = [display_name, canonical_id]
                if ticker:
                    surface_forms.append(ticker)
                if aliases_raw:
                    surface_forms.extend(a.strip() for a in aliases_raw.split(";") if a.strip())
                register_aliases(surface_forms, entity)

    return alias_to_entity


def _fuzzy_link_candidate(
    mention_canonical: str,
    alias_to_entity: dict[str, dict[str, Any]],
    threshold: float,
    enable_fuzzy: bool = True,  # ADD THIS — caller controls, not global flag
) -> tuple[str, dict[str, Any], float] | None:
    if not mention_canonical or not enable_fuzzy or fuzz is None:
        return None

    first_char = mention_canonical[0]
    candidates: list[tuple[str, dict[str, Any]]] = [
        (alias, entity)
        for alias, entity in alias_to_entity.items()
        if alias and alias[0] == first_char and abs(len(alias) - len(mention_canonical)) <= max(4, len(mention_canonical) // 2)
    ]
    if not candidates:
        return None

    ranked: list[tuple[float, str, dict[str, Any]]] = []
    for alias, entity in candidates:
        score = float(fuzz.token_set_ratio(mention_canonical, alias))
        ranked.append((score, alias, entity))
    ranked.sort(key=lambda x: x[0], reverse=True)

    best_score, _, best_entity = ranked[0]
    second_score = ranked[1][0] if len(ranked) > 1 else -1.0
    if best_score < threshold:
        return None
    if second_score >= threshold and (best_score - second_score) < 2.0:
        return None
    return ranked[0][1], best_entity, best_score


def link_financial_entity(
    mention_text: str,
    ner_label: str,
    alias_to_entity: dict[str, dict[str, Any]],
    fuzzy_threshold: float = FIN_LINK_FUZZY_THRESHOLD,
) -> dict[str, Any] | None:
    """
    Resolve a mention to a financial entity.
    Deterministic stages:
      1. Exact alias match
      2. Constrained fuzzy match (if enabled and available)
      3. Label-aware acceptance: fuzzy only for ORG/MISC mentions
    """
    mention_canonical = _canonicalize(mention_text)
    if not mention_canonical:
        return None

    exact = alias_to_entity.get(mention_canonical)
    if exact:
        return {
            **exact,
            "link_method": "exact",
            "link_score": 100.0,
        }

    if ner_label not in {"ORG", "MISC"}:
        return None

    fuzzy_match = _fuzzy_link_candidate(
    mention_canonical, alias_to_entity, fuzzy_threshold,
    enable_fuzzy=FIN_LINK_ENABLE_FUZZY,  # module default, but now overridable
)
    if not fuzzy_match:
        return None

    _, entity, score = fuzzy_match
    return {
        **entity,
        "link_method": "fuzzy",
        "link_score": score,
    }


# Sourced from ISO 3166-1 common names. Extend as needed.
_COUNTRIES: frozenset[str] = frozenset({
    "afghanistan", "albania", "algeria", "angola", "argentina", "australia",
    "austria", "bangladesh", "belgium", "bolivia", "brazil", "cambodia",
    "cameroon", "canada", "chile", "china", "colombia", "congo", "croatia",
    "cuba", "czech republic", "czechia", "denmark", "ecuador", "egypt",
    "ethiopia", "finland", "france", "germany", "ghana", "greece", "guatemala",
    "hungary", "india", "indonesia", "iran", "iraq", "ireland", "israel",
    "italy", "japan", "jordan", "kazakhstan", "kenya", "kuwait", "laos",
    "lebanon", "libya", "malaysia", "mexico", "morocco", "mozambique",
    "myanmar", "nepal", "netherlands", "new zealand", "nigeria", "north korea",
    "norway", "pakistan", "panama", "peru", "philippines", "poland",
    "portugal", "qatar", "romania", "russia", "saudi arabia", "senegal",
    "serbia", "singapore", "somalia", "south africa", "south korea", "spain",
    "sri lanka", "sudan", "sweden", "switzerland", "syria", "taiwan",
    "tanzania", "thailand", "tunisia", "turkey", "ukraine", "united arab emirates",
    "uae", "united kingdom", "uk", "united states", "usa", "us", "america",
    "uruguay", "uzbekistan", "venezuela", "vietnam", "yemen", "zimbabwe",
    # Common adjective/demonym forms that NER may emit
    "american", "chinese", "european", "british", "russian", "japanese",
    "german", "french", "indian", "korean", "iranian", "israeli",
})

# Minimum token count for a PER entity to be retained.
# Filters out single-word names that are rarely prominent figures.
_MIN_PER_TOKENS = 2


def _is_country(canonical: str) -> bool:
    return canonical in _COUNTRIES


def _is_prominent_person(canonical: str) -> bool:
    """
    Heuristic: require at least two tokens (first + last name).
    Single-word PER mentions are almost always noise or pronouns
    that the NER model mis-tagged.
    """
    return len(canonical.split()) >= _MIN_PER_TOKENS


def extract_entities_from_chunks(
    chunks: list[dict],
    pipe,
    alias_to_fin_entity: dict[str, dict[str, Any]],
) -> list[dict]:
    """
    Run NER over chunk texts, then resolve ORG/MISC mentions via
    a financial entity linker. PER/LOC fallback to heuristic filters.
    """
    mentions = []
    seen: set[tuple[str, str]] = set()  # (chunk_uid, canonical_name)

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
                etype = re.sub(r"^[BI]-", "", etype)
                if etype not in RELEVANT_ENTITY_TYPES:
                    continue

                raw_name = ent.get("word", "").strip()
                if not raw_name or len(raw_name) < 2:
                    continue

                base_canonical = _canonicalize(raw_name)
                ticker: str | None = None
                resolved_type = etype
                link_method = "ner_heuristic"
                link_score: float | None = None

                linked = None
                if ENABLE_FIN_ENTITY_LINKER:
                    linked = link_financial_entity(raw_name, etype, alias_to_fin_entity)

                if linked:
                    canonical_name = str(linked["canonical_name"])
                    display_name = str(linked["display_name"])
                    ticker = linked.get("ticker")
                    resolved_type = str(linked.get("entity_type") or etype)
                    link_method = str(linked.get("link_method") or "exact")
                    link_score = float(linked.get("link_score")) if linked.get("link_score") is not None else None
                elif etype == "PER":
                    if not _is_prominent_person(base_canonical):
                        continue
                    canonical_name = base_canonical
                    display_name = raw_name
                elif etype == "LOC":
                    if not _is_country(base_canonical):
                        continue
                    canonical_name = base_canonical
                    display_name = raw_name
                else:
                    continue

                key = (chunk["chunk_uid"], canonical_name)
                if key in seen:
                    continue
                seen.add(key)

                mentions.append({
                    "chunk_uid":      chunk["chunk_uid"],
                    "canonical_name": canonical_name,
                    "display_name":   display_name,
                    "raw_name":       raw_name,
                    "entity_type":    resolved_type,
                    "ticker":         ticker,
                    "link_method":    link_method,
                    "link_score":     link_score,
                    "period_key":     chunk["period_key"],
                    "article_id":     chunk["article_id"],
                })

    return mentions


def build_knowledge_nodes(
    mentions: list[dict],
) -> dict[str, dict]:
    """
    Aggregate mentions into KnowledgeNode specs.
    One KnowledgeNode per (canonical_name - period_key).

    Returns {kn_uid: kn_dict}
    """
    nodes: dict[str, dict] = {}
    for m in mentions:
        kn_uid = knowledge_node_uid(m["canonical_name"], m["period_key"])
        if kn_uid not in nodes:
            nodes[kn_uid] = {
                "kn_uid":         kn_uid,
                "canonical_name": m["canonical_name"],
                "display_name":   m.get("display_name", m["canonical_name"]),
                "entity_type":    m["entity_type"],
                "period_key":     m["period_key"],
                "ticker":         m["ticker"],
                # description is the aggregated knowledge text - built below
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
    chunk["text"], m["raw_name"], m["canonical_name"], m.get("display_name")
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
    text: str, raw_name: str, canonical_name: str, display_name: str | None = None
) -> str:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    search_terms = [raw_name]
    if display_name and display_name != raw_name:
        search_terms.append(display_name)
    # Only use canonical_name for text search if it looks like actual text, not a ticker
    if canonical_name and not canonical_name.isupper():
        search_terms.append(canonical_name)
    pattern = re.compile("|".join(re.escape(t) for t in search_terms), re.IGNORECASE)
    matched = [s for s in sentences if pattern.search(s)]
    return " ".join(matched[:3])


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
                    "display_name":   kn.get("display_name", kn["canonical_name"]),
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
        alias_to_fin_entity = load_financial_entity_map()

        print("[setup] Extracting entities ...")
        mentions = extract_entities_from_chunks(chunks, pipe, alias_to_fin_entity)
        print(f"[setup] {len(mentions)} entity mentions extracted")

        knowledge_nodes = build_knowledge_nodes(mentions)
        print(f"[setup] {len(knowledge_nodes)} KnowledgeNodes (entity - period)")

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
