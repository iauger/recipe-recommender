"""User profile builder for history-aware recommendations.

Computes a recency-weighted 128-D taste centroid and a 17-D Phase 1 tag affinity
vector from a user's gold_labeled_reviews history.
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
    """Minimal structured view of a single user review."""
    recipe_id: int
    star_rating: float
    timestamp: pd.Timestamp
    tag_intensities: Dict[str, float]   # {tag_name: intensity_score}


@dataclass
class UserProfile:
    """Aggregated preference signal for one user: 128-D taste centroid + 17-D tag affinity."""
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
    """
    Convert a 1-5 star rating to a positive weight.

    Mean-centering (subtract ~4.4) yields a signed value; we add 1 so that
    even average-rated recipes (deviation ~0) still contribute positively.
    Clamped to [0.1, 2.0] to prevent negative weights from inverting the pull.
    """
    return float(np.clip(1.0 + (star_rating - GLOBAL_RATING_MEAN), 0.1, 2.0))


def _tag_columns(df: pd.DataFrame, culinary_tags: tuple[str, ...]) -> List[str]:
    """Return intensity/sim column names present in the DataFrame.

    Supports both 'intensity_*' (preprocessed recipes) and 'sim_*' (gold reviews
    from Phase 1 Word2Vec pipeline) column naming conventions.
    """
    available = set(df.columns)
    cols = []
    for t in culinary_tags:
        if f"intensity_{t}" in available:
            cols.append(f"intensity_{t}")
        elif f"sim_{t}" in available:
            cols.append(f"sim_{t}")
    return cols


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_user_review_history(
    gold_reviews_path: Any,
    user_id: Any,
    culinary_tags: tuple[str, ...],
    min_reviews: int = 1,
) -> Tuple[List[ReviewRecord], bool]:
    """
    Load and parse a single user's review history from the gold_labeled_reviews parquet.

    Parameters
    ----------
    gold_reviews_path : str | Path
        Path to gold_labeled_reviews_*.parquet.
    user_id : int | str
        The Food.com user_id to look up.
    culinary_tags : tuple[str, ...]
        Ordered list of the 17 tag names (from Settings.culinary_tags).
    min_reviews : int
        Minimum number of reviews required; returns ([], False) if below threshold.

    Returns
    -------
    records : list[ReviewRecord]
    ok : bool
        False if the user has fewer than min_reviews reviews.
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
    """
    Return sorted list of user_ids with >= min_reviews reviews.
    Used to populate the Streamlit dropdown.
    """
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
    """
    Compute a UserProfile from a list of ReviewRecords and the embedding bundle.

    Parameters
    ----------
    records : list[ReviewRecord]
        The user's review history (output of load_user_review_history).
    user_id : int | str
        User identifier (stored on the profile for reference).
    bundle : dict
        The Phase 2 embedding bundle: {recipe_ids, embeddings, predictions}.
    culinary_tags : tuple[str, ...]
        Ordered tag names — defines the axis ordering of tag_affinity.
    half_life_days : int
        Recency decay half-life in days.

    Returns
    -------
    UserProfile
    """
    if not records:
        return UserProfile(
            user_id=user_id,
            embedding=None,
            tag_affinity=np.zeros(len(culinary_tags), dtype=np.float32),
            review_count=0,
            reviews=records,
            rated_recipe_ids=set(),
        )

    # Anchor decay to the user's most recent review rather than today.
    # Food.com data is historical (reviews end ~2013); using today's date
    # reduces all signals to near-zero for older users.
    now = max(r.timestamp for r in records)
    rated_ids = {r.recipe_id for r in records}

    # Build a fast recipe_id → embedding index lookup
    bundle_ids = [int(rid) for rid in bundle["recipe_ids"]]
    id_to_idx = {rid: idx for idx, rid in enumerate(bundle_ids)}
    raw_embeddings: torch.Tensor = bundle["embeddings"]  # (N, 128)

    # ── Weighted embedding mean ──────────────────────────────────────────────
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

    # ── Tag affinity vector ──────────────────────────────────────────────────
    # affinity[t] = mean over reviews of (rating_deviation * intensity_t)
    # Positive tags pull affinity up when rated highly; negative tags pull
    # affinity down (negative deviation) when their intensity is high — both
    # cases correctly signal what the user does/doesn't want.
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
