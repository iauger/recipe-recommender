"""
pipeline/preprocessing.py
--------------------------
Preprocessing utilities: load, aggregate, scale, encode, and export
recipe and review-derived features for model training and search indexing.

Ported from Phase 2 (CS 615 / RecipeFeedback-ResNet src/preprocessing.py).

Path mapping to unified core/config.py:
    Phase 2 name               → unified Settings field
    ─────────────────────────────────────────────────────
    raw_recipes_path           → settings.raw_recipes_path
    raw_reviews_path           → settings.raw_reviews_path
    raw_labeled_reviews_path   → settings.gold_reviews_path
    processed_recipes_path     → settings.processed_recipes_path
    processed_search_path      → settings.recipes_path
    models_dir/column_mapping  → settings.column_mapping_path
"""

from __future__ import annotations

import json
import os
from typing import Tuple, cast

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer, StandardScaler

from core.config import Settings


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(s: Settings) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load recipe metadata, raw reviews, and gold-labeled reviews.

    Returns:
        recipes_df:  Raw recipe metadata (modeling_recipe.parquet).
        reviews_df:  Raw reviews (modeling_reviews.parquet) — loaded for
                     completeness but not used downstream in preprocess_data().
        labels_df:   Gold labeled reviews with pred_* / sim_* tag columns.
    """
    recipes_df = pd.read_parquet(s.raw_recipes_path)
    reviews_df = pd.read_parquet(s.raw_reviews_path)
    labels_df  = pd.read_parquet(s.gold_reviews_path)

    if recipes_df.empty:
        raise ValueError(f"Recipes DataFrame is empty: {s.raw_recipes_path}")
    if reviews_df.empty:
        raise ValueError(f"Reviews DataFrame is empty: {s.raw_reviews_path}")
    if labels_df.empty:
        raise ValueError(f"Labeled reviews DataFrame is empty: {s.gold_reviews_path}")

    return recipes_df, reviews_df, labels_df


# ---------------------------------------------------------------------------
# Review aggregation
# ---------------------------------------------------------------------------

def bayesian_rating(
    df: pd.DataFrame,
    global_avg_rating: float,
    rating_col: str = "rating",
    review_count_col: str = "review_count",
    m_threshold: float | None = None,
) -> pd.Series:
    """Bayesian smoothed average: pulls low-count recipes toward the global mean."""
    C = global_avg_rating
    if m_threshold is None:
        m_threshold = df[review_count_col].quantile(0.25)
    v = df[review_count_col]
    R = df[rating_col]
    return (v * R + m_threshold * C) / (v + m_threshold)


def review_aggregation(reviews_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate gold-labeled reviews to recipe level.

    For each tag t:
        pred_t      = proportion of reviews where pred_t == 1
        intensity_t = mean sim_t for reviews where pred_t == 1

    Only retains recipes with at least one positive tag signal.
    """
    reviews_df = reviews_df.copy()
    reviews_df["recipe_id"] = reviews_df["recipe_id"].astype(str)

    global_avg_rating = reviews_df["rating"].mean()
    tags = [col.replace("pred_", "") for col in reviews_df.columns if col.startswith("pred_")]

    agg_dict: dict = {
        "rating": ["mean", "count"],
        **{f"pred_{tag}": "mean" for tag in tags},
    }

    recipe_level_df = reviews_df.groupby("recipe_id").agg(agg_dict)

    # Intensity: mean sim_tag where predicted == 1
    for tag in tags:
        intensity = (
            reviews_df[reviews_df[f"pred_{tag}"] == 1]
            .groupby("recipe_id")[f"sim_{tag}"]
            .mean()
        )
        recipe_level_df[f"intensity_{tag}"] = intensity

    recipe_level_df = recipe_level_df.fillna(0)

    # Flatten MultiIndex columns
    recipe_level_df.columns = [
        f"{c[0]}_{c[1]}" if isinstance(c, tuple) and c[1] != "mean" else c[0]
        for c in recipe_level_df.columns
    ]

    recipe_level_df = (
        recipe_level_df
        .rename(columns={"rating": "raw_mean_rating", "rating_count": "review_count"})
        .reset_index()
    )

    recipe_level_df["bayesian_rating"] = bayesian_rating(
        recipe_level_df,
        global_avg_rating=global_avg_rating,
        rating_col="raw_mean_rating",
        review_count_col="review_count",
    )

    # Reorder columns
    all_cols = recipe_level_df.columns.tolist()
    p_cols = sorted(c for c in all_cols if c.startswith("pred_"))
    i_cols = sorted(c for c in all_cols if c.startswith("intensity_"))
    base_cols = ["recipe_id", "raw_mean_rating", "review_count", "bayesian_rating"]
    recipe_level_df = recipe_level_df[base_cols + p_cols + i_cols]

    # Keep only recipes with at least one detected semantic signal
    has_signals = recipe_level_df[p_cols].sum(axis=1) > 0
    return recipe_level_df[has_signals].copy()


# ---------------------------------------------------------------------------
# Merging
# ---------------------------------------------------------------------------

def _normalise_recipe_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce recipe_id to clean integer string to avoid float-artifact mismatches."""
    df = df.copy()
    df["recipe_id"] = (
        pd.to_numeric(df["recipe_id"], errors="coerce")
        .fillna(0)
        .astype(int)
        .astype(str)
    )
    return df


def merge_data(recipe_df: pd.DataFrame, review_agg_df: pd.DataFrame) -> pd.DataFrame:
    """Inner-join recipe metadata with aggregated review features."""
    recipe_df    = _normalise_recipe_ids(recipe_df)
    review_agg_df = _normalise_recipe_ids(review_agg_df)
    return pd.merge(recipe_df, review_agg_df, on="recipe_id", how="inner")


def validate_merge(recipe_df: pd.DataFrame, review_agg_df: pd.DataFrame) -> None:
    """Diagnostic QA — warn if overlap is unexpectedly low."""
    recipe_ids = set(recipe_df["recipe_id"].unique())
    review_ids = set(review_agg_df["recipe_id"].unique())
    intersection = recipe_ids & review_ids

    print("\n--- Merge QA ---")
    print(f"  Recipe IDs:   {len(recipe_ids):,}")
    print(f"  Review IDs:   {len(review_ids):,}")
    print(f"  Intersection: {len(intersection):,}")

    if len(intersection) == 0:
        raise ValueError(
            "Zero overlap between recipe and review IDs. "
            "Check that both files are from the same corpus version."
        )
    if 0 < len(intersection) < 100:
        print("  WARNING: very low overlap — verify corpus files match.")


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def scale_features(
    df: pd.DataFrame,
    standard_cols: list[str] | None,
    minmax_cols: list[str] | None,
) -> pd.DataFrame:
    """StandardScaler for continuous features; MinMaxScaler for intensity tags."""
    if standard_cols:
        df[standard_cols] = StandardScaler().fit_transform(df[standard_cols])
    if minmax_cols:
        df[minmax_cols] = MinMaxScaler().fit_transform(df[minmax_cols])
    elif not standard_cols:
        raise ValueError("Provide at least one of standard_cols or minmax_cols.")
    return df


def encode_multi_label_features(
    df: pd.DataFrame,
    column: str,
    prefix: str,
    top_n: int = 100,
) -> pd.DataFrame:
    """One-hot encode a space-separated multi-label column (top-N items)."""
    item_lists = df[column].str.split()
    all_items  = [item for sublist in item_lists for item in sublist]
    top_items  = pd.Series(all_items).value_counts().head(top_n).index.tolist()

    filtered = item_lists.apply(lambda x: [i for i in x if i in top_items])
    mlb = MultiLabelBinarizer(classes=top_items, sparse_output=False)
    encoded = cast(np.ndarray, mlb.fit_transform(filtered))

    encoded_df = pd.DataFrame(
        encoded.astype(np.int32),
        columns=[f"{prefix}_{item.replace('-', '_').replace(' ', '_')}" for item in top_items],
        index=df.index,
    )
    return pd.concat([df, encoded_df], axis=1)


# ---------------------------------------------------------------------------
# Search / ES formatting
# ---------------------------------------------------------------------------

def format_for_search(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare a recipe DataFrame for Elasticsearch ingestion.

    - Cast pred_* to bool
    - Parse ingredients_clean and tags_clean to lists
    """
    search_df = df.copy()

    pred_cols = [c for c in search_df.columns if c.startswith("pred_")]
    search_df[pred_cols] = search_df[pred_cols].astype(bool)

    search_df["ingredients_clean"] = (
        search_df["ingredients_clean"]
        .fillna("")
        .apply(lambda x: [ing.replace("_", " ") for ing in str(x).split()])
    )
    search_df["tags_clean"] = (
        search_df["tags_clean"]
        .fillna("")
        .apply(lambda x: str(x).split())
    )
    return search_df


# ---------------------------------------------------------------------------
# Static column mapping export
# ---------------------------------------------------------------------------

def export_static_mapping(df: pd.DataFrame, settings: Settings) -> None:
    """
    Save a JSON mapping of {column_name: index} for all model input features.

    Excludes identifier / target columns so the mapping reflects exactly what
    RecipeNet sees as input. Written to settings.column_mapping_path.
    """
    exclude = {"recipe_id", "name", "bayesian_rating", "raw_mean_rating", "review_count"}
    model_features = [col for col in df.columns if col not in exclude]
    mapping = {col: i for i, col in enumerate(model_features)}

    path = settings.column_mapping_path
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(mapping, f, indent=4)
    print(f"Column mapping saved → {path}")


# ---------------------------------------------------------------------------
# Main preprocessing pipeline
# ---------------------------------------------------------------------------

def preprocess_data(settings: Settings, overwrite_processed: bool = False) -> pd.DataFrame:
    """
    Full preprocessing pipeline: load → aggregate → merge → scale → encode → save.

    If settings.processed_recipes_path already exists and overwrite_processed
    is False, the cached parquet is returned immediately.

    Side effects:
        - Writes settings.recipes_path          (ES-ready search parquet)
        - Writes settings.processed_recipes_path (scaled+encoded training parquet)
        - Writes settings.column_mapping_path    (JSON feature→index mapping)
    """
    out_path = settings.processed_recipes_path

    if not overwrite_processed and out_path.exists():
        print(f"Using cached preprocessed data: {out_path}")
        return pd.read_parquet(out_path)

    # Load
    recipe_df, _, label_df = load_data(settings)

    # Aggregate reviews → recipe-level tag features
    review_agg_df = review_aggregation(label_df)
    validate_merge(recipe_df, review_agg_df)
    merged_df = merge_data(recipe_df, review_agg_df)

    # Write ES-ready search parquet (no normalisation, boolean pred_*)
    search_df = format_for_search(merged_df)
    settings.recipes_path.parent.mkdir(parents=True, exist_ok=True)
    search_df.to_parquet(settings.recipes_path, index=False)
    print(f"Search parquet saved → {settings.recipes_path}")

    # Scale
    standard_cols = [
        col for col in merged_df.columns
        if col not in {"recipe_id", "raw_mean_rating", "review_count", "bayesian_rating", "name"}
        and not col.startswith("pred_")
        and not col.startswith("intensity_")
        and merged_df[col].dtype in ("float64", "int64")
    ]
    minmax_cols = [col for col in merged_df.columns if col.startswith("intensity_")]
    scaled_df = scale_features(merged_df, standard_cols, minmax_cols)

    # Encode multi-label features
    encoded_df = encode_multi_label_features(scaled_df, "tags_clean", "cat", top_n=100)
    encoded_df = encode_multi_label_features(encoded_df, "ingredients_clean", "ing", top_n=100)

    # Export column mapping and save training parquet
    export_static_mapping(encoded_df, settings)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    encoded_df.to_parquet(out_path, index=False)
    print(f"Training parquet saved → {out_path}")

    return encoded_df


def preprocess_report(df: pd.DataFrame) -> None:
    """Print a summary of the preprocessed DataFrame."""
    print("\n=== Preprocessing Report ===")
    print(f"  Recipes: {df['recipe_id'].nunique():,}")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"  Numeric features: {len(numeric_cols)}")
    for col in numeric_cols:
        print(
            f"  {col}: mean={df[col].mean():.4f} "
            f"std={df[col].std():.4f} "
            f"[{df[col].min():.4f}, {df[col].max():.4f}]"
        )
