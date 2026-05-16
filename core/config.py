"""
core/config.py
--------------
Unified configuration for the recipe recommender system.

Design principles:
  - load_settings() resolves paths and reads env vars — never touches Elasticsearch.
  - get_es_client() is called explicitly by components that need it (search/, indexer).
    This prevents import-time ConnectionErrors when Docker is not running.
  - All paths can be overridden via environment variables or a .env file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()  # no-op if .env is absent


# ---------------------------------------------------------------------------
# Settings dataclass
# ---------------------------------------------------------------------------

@dataclass
class Settings:
    # ── Elasticsearch ───────────────────────────────────────────────────────
    es_host: str = "http://localhost:9200"
    es_index: str = "recipes"
    es_timeout: int = 30

    # ── Phase 2 model artifacts ─────────────────────────────────────────────
    embeddings_path: Path = Path("data/final_residual_v2_embeddings.pt")
    model_path: Path = Path("data/best_model_residual_v2_all_features_mse.pth")
    column_mapping_path: Path = Path("data/column_mapping.json")
    recipes_path: Path = Path("data/PROCESSED_search_recipes.parquet")

    # ── Phase 1 gold dataset (offline / pipeline use only) ──────────────────
    gold_reviews_path: Path = Path("data/gold_labeled_reviews.parquet")

    # ── User data ───────────────────────────────────────────────────────────
    user_profiles_path: Path = Path("data/user_profiles.parquet")

    # ── Model hyperparameters (must match the trained checkpoint) ───────────
    hidden_dim: int = 128

    # ── Recommender ─────────────────────────────────────────────────────────
    n_neighbors: int = 20          # session-based NN candidate pool
    recency_half_life_days: int = 365  # for exponential decay in affinity vector
    min_reviews_for_cf: int = 5    # threshold to activate LensKit item-item CF

    # ── Search / reranker ───────────────────────────────────────────────────
    default_top_k: int = 10
    candidate_pool_size: int = 100  # ES retrieval size before reranking

    # ── Tag metadata ────────────────────────────────────────────────────────
    # 17 culinary tags produced by Phase 1. Stored here so schema.py and the
    # affinity vector builder both reference a single source of truth.
    culinary_tags: tuple[str, ...] = field(default_factory=lambda: (
        "family_hit",
        "delicious_tasty",
        "healthy_nutritious",
        "quick_easy",
        "comfort_food",
        "gourmet_fancy",
        "budget_friendly",
        "meal_prep",
        "vegetarian",
        "vegan",
        "gluten_free",
        "dairy_free",
        "spicy",
        "sweet_dessert",
        "savory",
        "breakfast_brunch",
        "entertaining",
    ))


# ---------------------------------------------------------------------------
# Factory — call this once at application startup
# ---------------------------------------------------------------------------

def load_settings() -> Settings:
    """
    Build a Settings instance from environment variables.

    Environment variables (all optional — fall back to dataclass defaults):
      ES_HOST, ES_INDEX, ES_TIMEOUT
      EMBEDDINGS_PATH, MODEL_PATH, COLUMN_MAPPING_PATH, RECIPES_PATH
      GOLD_REVIEWS_PATH, USER_PROFILES_PATH
      HIDDEN_DIM, N_NEIGHBORS, DEFAULT_TOP_K, CANDIDATE_POOL_SIZE
      RECENCY_HALF_LIFE_DAYS, MIN_REVIEWS_FOR_CF
    """
    def _path(env_key: str, default: Path) -> Path:
        raw = os.getenv(env_key)
        return Path(raw) if raw else default

    return Settings(
        es_host=os.getenv("ES_HOST", "http://localhost:9200"),
        es_index=os.getenv("ES_INDEX", "recipes"),
        es_timeout=int(os.getenv("ES_TIMEOUT", "30")),
        embeddings_path=_path("EMBEDDINGS_PATH", Path("data/final_residual_v2_embeddings.pt")),
        model_path=_path("MODEL_PATH", Path("data/best_model_residual_v2_all_features_mse.pth")),
        column_mapping_path=_path("COLUMN_MAPPING_PATH", Path("data/column_mapping.json")),
        recipes_path=_path("RECIPES_PATH", Path("data/PROCESSED_search_recipes.parquet")),
        gold_reviews_path=_path("GOLD_REVIEWS_PATH", Path("data/gold_labeled_reviews.parquet")),
        user_profiles_path=_path("USER_PROFILES_PATH", Path("data/user_profiles.parquet")),
        hidden_dim=int(os.getenv("HIDDEN_DIM", "128")),
        n_neighbors=int(os.getenv("N_NEIGHBORS", "20")),
        recency_half_life_days=int(os.getenv("RECENCY_HALF_LIFE_DAYS", "365")),
        min_reviews_for_cf=int(os.getenv("MIN_REVIEWS_FOR_CF", "5")),
        default_top_k=int(os.getenv("DEFAULT_TOP_K", "10")),
        candidate_pool_size=int(os.getenv("CANDIDATE_POOL_SIZE", "100")),
    )


# ---------------------------------------------------------------------------
# Lazy Elasticsearch client — import and call only when Docker is running
# ---------------------------------------------------------------------------

def get_es_client(settings: Optional[Settings] = None):
    """
    Return a connected Elasticsearch client.

    Called explicitly by search/indexer.py and search/engine.py.
    Never called at import time — components that don't need ES
    (recommender/, pipeline/) import config.py safely without triggering
    a ConnectionError.

    Raises:
        elasticsearch.ConnectionError: if Elasticsearch is not reachable.
    """
    from elasticsearch import Elasticsearch  # local import — keeps ES optional

    cfg = settings or load_settings()
    client = Elasticsearch(
        cfg.es_host,
        request_timeout=cfg.es_timeout,
    )
    # Validate connection eagerly so the caller gets a clear error immediately.
    client.info()
    return client
