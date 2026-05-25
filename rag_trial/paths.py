from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
RUNTIME_DIR = PROJECT_ROOT / "runtime"

ARTICLES_JSON_PATH = DATA_DIR / "articles" / "cnbc_articles.json"
HIST_DATA_PARQUET_PATH = DATA_DIR / "market" / "hist_data.parquet"
TICKER_MAP_PATH = DATA_DIR / "reference" / "ticker_company_map.csv"
FIN_ENTITY_MAP_PATH = DATA_DIR / "reference" / "financial_entity_map.csv"

PROMPT_TEMPLATES_PATH = CONFIG_DIR / "prompt_templates.json"
GOLD_EVAL_CASES_PATH = PROJECT_ROOT / "tests" / "fixtures" / "gold_eval_cases.json"

SQLITE_DB_PATH = RUNTIME_DIR / "db" / "my_database.db"
TA_SQLITE_CACHE_DB_PATH = RUNTIME_DIR / "db" / "ta_cache.db"
CONVERSATION_MEMORY_PATH = RUNTIME_DIR / "memory" / "conversation_memory.json"
QUERY_CONTEXT_DIR = RUNTIME_DIR / "query_context"


def env_path(name: str, default: Path) -> Path:
    return Path(os.getenv(name, str(default)))


def env_str_path(name: str, default: Path) -> str:
    return os.getenv(name, str(default))
