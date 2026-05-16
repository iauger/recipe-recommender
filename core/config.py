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

    # ── Phase 2 raw training inputs (pipeline/ use only — not needed at runtime) ──
    raw_recipes_path: Path = Path("data/modeling_recipe.parquet")
    raw_reviews_path: Path = Path("data/modeling_reviews.parquet")

    # ── Phase 2 model artifacts ─────────────────────────────────────────────
    embeddings_path: Path = Path("data/final_residual_v2_embeddings.pt")
    model_path: Path = Path("data/best_model_residual_v2_all_features_mse.pth")
    column_mapping_path: Path = Path("data/column_mapping.json")
    recipes_path: Path = Path("data/PROCESSED_search_recipes.parquet")

    # ── Phase 2 pipeline artifacts (used by pipeline/ for training) ────────
    # processed_recipes_path: scaled + encoded DataFrame consumed by RecipeDataset.
    # Distinct from recipes_path (the ES-ready search parquet).
    processed_recipes_path: Path = Path("data/PROCESSED_recipes.parquet")

    # ── Phase 2 supplementary artifacts ────────────────────────────────────
    umap_projection_path: Path = Path("data/final_residual_v2_umap_projection.npy")

    # ── Phase 1 gold dataset (offline / pipeline use only) ──────────────────
    gold_reviews_path: Path = Path("data/gold_labeled_reviews_20260310_135905.parquet")

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
    # 17 review-sentiment tags produced by Phase 1 (DSCI 632 Word2Vec classifier).
    # Stored here as the single source of truth for schema.py, the affinity vector
    # builder, and any UI display logic.
    #
    # Polarity guide (relevant for UI display and affinity interpretation):
    #   POSITIVE  — high pred/intensity = desirable signal
    #   NEGATIVE  — high pred/intensity = undesirable signal (affinity formula
    #               still works correctly — (rating - mean) * intensity yields
    #               a negative affinity when negative tags co-occur with low ratings)
    #   NEUTRAL   — directionally ambiguous; context-dependent
    #
    # tag                        polarity   prevalence (% reviews)
    # ─────────────────────────────────────────────────────────────
    # family_hit                 POSITIVE   45.0%
    # delicious_tasty            POSITIVE   38.4%
    # would_make_again           POSITIVE   36.7%
    # easy_quick                 POSITIVE   30.1%
    # substitution_modification  NEUTRAL    28.9%
    # ingredient_issue           NEGATIVE   24.0%
    # crispy_crunchy             POSITIVE   12.3%
    # moist_tender               POSITIVE   10.3%
    # would_not_make_again       NEGATIVE    9.1%
    # too_spicy                  NEGATIVE    8.6%
    # time_consuming_complex     NEGATIVE    7.6%
    # bland_lacks_flavor         NEGATIVE    6.4%
    # dry                        NEGATIVE    4.4%
    # too_sweet                  NEGATIVE    3.7%
    # mushy_soggy                NEGATIVE    3.6%
    # too_acidic                 NEGATIVE    2.6%
    # too_salty                  NEGATIVE    2.0%
    culinary_tags: tuple[str, ...] = field(default_factory=lambda: (
        "family_hit",
        "delicious_tasty",
        "would_make_again",
        "easy_quick",
        "substitution_modification",
        "ingredient_issue",
        "crispy_crunchy",
        "moist_tender",
        "would_not_make_again",
        "too_spicy",
        "time_consuming_complex",
        "bland_lacks_flavor",
        "dry",
        "too_sweet",
        "mushy_soggy",
        "too_acidic",
        "too_salty",
    ))

    # Negative-polarity tags — used by the affinity builder to flag signals
    # that should penalise recommendations rather than boost them.
    negative_tags: frozenset[str] = field(default_factory=lambda: frozenset({
        "ingredient_issue",
        "would_not_make_again",
        "too_spicy",
        "time_consuming_complex",
        "bland_lacks_flavor",
        "dry",
        "too_sweet",
        "mushy_soggy",
        "too_acidic",
        "too_salty",
    }))


# ---------------------------------------------------------------------------
# Factory — call this once at application startup
# ---------------------------------------------------------------------------

def load_settings() -> Settings:
    """
    Build a Settings instance from environment variables.

    Environment variables (all optional — fall back to dataclass defaults):
      ES_HOST, ES_INDEX, ES_TIMEOUT
      RAW_RECIPES_PATH, RAW_REVIEWS_PATH
      EMBEDDINGS_PATH, MODEL_PATH, COLUMN_MAPPING_PATH, RECIPES_PATH, UMAP_PROJECTION_PATH
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
        raw_recipes_path=_path("RAW_RECIPES_PATH", Path("data/modeling_recipe.parquet")),
        raw_reviews_path=_path("RAW_REVIEWS_PATH", Path("data/modeling_reviews.parquet")),
        embeddings_path=_path("EMBEDDINGS_PATH", Path("data/final_residual_v2_embeddings.pt")),
        model_path=_path("MODEL_PATH", Path("data/best_model_residual_v2_all_features_mse.pth")),
        column_mapping_path=_path("COLUMN_MAPPING_PATH", Path("data/column_mapping.json")),
        recipes_path=_path("RECIPES_PATH", Path("data/PROCESSED_search_recipes.parquet")),
        processed_recipes_path=_path("PROCESSED_RECIPES_PATH", Path("data/PROCESSED_recipes.parquet")),
        umap_projection_path=_path("UMAP_PROJECTION_PATH", Path("data/final_residual_v2_umap_projection.npy")),
        gold_reviews_path=_path("GOLD_REVIEWS_PATH", Path("data/gold_labeled_reviews_20260310_135905.parquet")),
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
