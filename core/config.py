"""
Unified configuration for the recipe recommender system.

load_settings() reads env vars and resolves paths; get_es_client() is a
separate explicit call so components that don't need ES can import safely.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()  # no-op if .env is absent


@dataclass
class Settings:
    # Elasticsearch
    es_host: str = "http://localhost:9200"
    es_index: str = "recipes"
    es_timeout: int = 30

    # Phase 2 raw training inputs (pipeline/ use only)
    raw_recipes_path: Path = Path("data/modeling_recipe.parquet")
    raw_reviews_path: Path = Path("data/modeling_reviews.parquet")

    # Phase 2 model artifacts
    embeddings_path: Path = Path("data/final_residual_v2_embeddings.pt")
    model_path: Path = Path("data/best_model_residual_v2_all_features_mse.pth")
    column_mapping_path: Path = Path("data/column_mapping.json")
    recipes_path: Path = Path("data/PROCESSED_search_recipes.parquet")

    # Scaled + encoded DataFrame for RecipeDataset — distinct from the ES-ready recipes_path.
    processed_recipes_path: Path = Path("data/PROCESSED_recipes.parquet")

    # Phase 2 supplementary artifacts
    umap_projection_path: Path = Path("data/final_residual_v2_umap_projection.npy")

    # Food.com raw files (name-resolution fallback for display only)
    food_raw_recipes_path: Path = Path("data/RAW_recipes.csv")

    # Phase 1 gold dataset (offline / pipeline use only)
    gold_reviews_path: Path = Path("data/gold_labeled_reviews.parquet")

    # User data
    user_profiles_path: Path = Path("data/user_profiles.parquet")

    # Model hyperparameters (must match the trained checkpoint)
    hidden_dim: int = 128

    # Recommender 
    n_neighbors: int = 20          # session-based NN candidate pool
    recency_half_life_days: int = 180  # for exponential decay in affinity vector
    min_reviews_for_cf: int = 5    # threshold to activate LensKit item-item CF

    # Search / reranker
    default_top_k: int = 10
    candidate_pool_size: int = 100  # ES retrieval size before reranking

    # Tag metadata 
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

    # Tags that penalise recommendations rather than boost them.
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


def load_settings() -> Settings:
    """Build Settings from environment variables, falling back to dataclass defaults."""
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
        food_raw_recipes_path=_path("FOOD_RAW_RECIPES_PATH", Path("data/RAW_recipes.csv")),
        gold_reviews_path=_path("GOLD_REVIEWS_PATH", Path("data/gold_labeled_reviews_20260310_135905.parquet")),
        user_profiles_path=_path("USER_PROFILES_PATH", Path("data/user_profiles.parquet")),
        hidden_dim=int(os.getenv("HIDDEN_DIM", "128")),
        n_neighbors=int(os.getenv("N_NEIGHBORS", "20")),
        recency_half_life_days=int(os.getenv("RECENCY_HALF_LIFE_DAYS", "180")),
        min_reviews_for_cf=int(os.getenv("MIN_REVIEWS_FOR_CF", "5")),
        default_top_k=int(os.getenv("DEFAULT_TOP_K", "10")),
        candidate_pool_size=int(os.getenv("CANDIDATE_POOL_SIZE", "100")),
    )


def get_es_client(settings: Optional[Settings] = None):
    """Return a connected Elasticsearch client; raises ConnectionError if ES is unreachable."""
    from elasticsearch import Elasticsearch  # local import — keeps ES optional

    cfg = settings or load_settings()
    client = Elasticsearch(
        cfg.es_host,
        request_timeout=cfg.es_timeout,
    )
    client.info()  # fail fast so the caller gets a clear error
    return client
