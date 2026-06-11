"""Builds a user taste profile from their review history.

Produces a recency-weighted embedding centroid and a tag affinity vector
used by the hybrid recommender to personalize results.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


GLOBAL_RATING_MEAN = 4.4   # approximate; used for mean-centering star ratings


@dataclass
class ReviewRecord:
    """One review: id, rating, timestamp, and per-tag intensity scores."""
    recipe_id: int
    star_rating: float
    timestamp: pd.Timestamp
    tag_intensities: Dict[str, float]   # {tag_name: intensity_score}


@dataclass
class UserProfile:
    """Aggregated taste signal for one user: embedding centroid + tag affinity vector."""
    user_id: Any
    embedding: Optional[torch.Tensor]
    tag_affinity: np.ndarray
    review_count: int
    reviews: List[ReviewRecord] = field(default_factory=list)
    rated_recipe_ids: "set[int]" = field(default_factory=set)

def _recency_weight(timestamp: pd.Timestamp, now: pd.Timestamp, half_life_days: int) -> float:
    """Exponential decay: weight = 2^(-(days_ago / half_life))."""
    days_ago = max((now - timestamp).days, 0)
    return math.pow(2.0, -(days_ago / half_life_days))


def _rating_weight(star_rating: float) -> float:
    """Mean-centered rating mapped to a positive weight in [0.1, 2.0]."""
    return float(np.clip(1.0 + (star_rating - GLOBAL_RATING_MEAN), 0.1, 2.0))


def _tag_columns(df: pd.DataFrame, culinary_tags: tuple[str, ...]) -> List[str]:
    """Find tag columns in df, checking for both 'intensity_*' and 'sim_*' prefixes."""
    available = set(df.columns)
    cols = []
    for t in culinary_tags:
        if f"intensity_{t}" in available:
            cols.append(f"intensity_{t}")
        elif f"sim_{t}" in available:
            cols.append(f"sim_{t}")
    return cols


# Public API

def load_user_review_history(
    gold_reviews_path: Any,
    user_id: Any,
    culinary_tags: tuple[str, ...],
    min_reviews: int = 1,
) -> Tuple[List[ReviewRecord], bool]:
    """Load one user's reviews from the gold_labeled_reviews parquet.

    Returns (records, True) on success, or ([], False) if the user has fewer
    than min_reviews reviews.
    """
    df = pd.read_parquet(gold_reviews_path)

    # Normalise types — compare as strings so int/str column dtype doesn't matter
    user_id_str = str(user_id)
    user_df = df[df["user_id"].astype(str) == user_id_str].copy()

    if len(user_df) < min_reviews:
        return [], False

    # Parse timestamp — column may be "date" or "submitted"
    date_col = "date" if "date" in user_df.columns else "submitted"
    user_df[date_col] = pd.to_datetime(user_df[date_col], errors="coerce")
    user_df = user_df.dropna(subset=[date_col])

    intensity_cols = _tag_columns(user_df, culinary_tags)

    records: List[ReviewRecord] = []
    for _, row in user_df.iterrows():
        intensities = {}
        for col in intensity_cols:
            if pd.isna(row[col]):
                continue
            tag = col.removeprefix("intensity_").removeprefix("sim_")
            intensities[tag] = float(row[col])
        records.append(ReviewRecord(
            recipe_id=int(row["recipe_id"]),
            star_rating=float(row["rating"]),
            timestamp=row[date_col],
            tag_intensities=intensities,
        ))

    return records, True


def get_eligible_user_ids(
    gold_reviews_path: Any,
    min_reviews: int = 5,
) -> List[Any]:
    """Return users with at least min_reviews reviews, sorted for the UI dropdown."""
    df = pd.read_parquet(gold_reviews_path, columns=["user_id"])
    counts = df["user_id"].value_counts()
    return sorted(counts[counts >= min_reviews].index.tolist())


def build_user_profile(
    records: List[ReviewRecord],
    user_id: Any,
    bundle: Dict[str, Any],
    culinary_tags: tuple[str, ...],
    half_life_days: int = 365,
) -> UserProfile:
    """Build a UserProfile from review history and the Phase 2 embedding bundle."""
    if not records:
        return UserProfile(
            user_id=user_id,
            embedding=None,
            tag_affinity=np.zeros(len(culinary_tags), dtype=np.float32),
            review_count=0,
            reviews=records,
            rated_recipe_ids=set(),
        )

    # Anchor to most recent review — Food.com data ends ~2013 so using today's
    # date would decay all signals to near-zero.
    now = max(r.timestamp for r in records)
    rated_ids = {r.recipe_id for r in records}

    # Fast recipe_id → index lookup into the embedding matrix
    bundle_ids = [int(rid) for rid in bundle["recipe_ids"]]
    id_to_idx = {rid: idx for idx, rid in enumerate(bundle_ids)}
    raw_embeddings: torch.Tensor = bundle["embeddings"]  # (N, 128)

    # Weighted embedding mean
    weighted_vecs: List[torch.Tensor] = []
    embedding_weights: List[float] = []

    for rec in records:
        idx = id_to_idx.get(rec.recipe_id)
        if idx is None:
            continue   # recipe not in bundle — skip
        rw = _rating_weight(rec.star_rating)
        dw = _recency_weight(rec.timestamp, now, half_life_days)
        w  = rw * dw
        weighted_vecs.append(raw_embeddings[idx].float() * w)
        embedding_weights.append(w)

    if weighted_vecs:
        total_w = sum(embedding_weights)
        mean_vec = torch.stack(weighted_vecs).sum(dim=0) / total_w
        user_embedding = F.normalize(mean_vec, p=2, dim=0)   # unit vector
    else:
        user_embedding = None

    # Tag affinity: weighted mean of (rating deviation × tag intensity) per tag
    tag_affinity = np.zeros(len(culinary_tags), dtype=np.float32)
    tag_counts   = np.zeros(len(culinary_tags), dtype=np.float32)

    for rec in records:
        deviation = rec.star_rating - GLOBAL_RATING_MEAN
        rw = _recency_weight(rec.timestamp, now, half_life_days)

        for t_idx, tag in enumerate(culinary_tags):
            intensity = rec.tag_intensities.get(tag, 0.0)
            if intensity > 0.0:
                tag_affinity[t_idx] += rw * deviation * intensity
                tag_counts[t_idx]   += rw

    nonzero = tag_counts > 0
    tag_affinity[nonzero] /= tag_counts[nonzero]

    return UserProfile(
        user_id=user_id,
        embedding=user_embedding,
        tag_affinity=tag_affinity,
        review_count=len(records),
        reviews=records,
        rated_recipe_ids=rated_ids,
    )
