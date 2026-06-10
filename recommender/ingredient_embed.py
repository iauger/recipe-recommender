"""TF-IDF ingredient embedding for culinary-identity Stage 1 retrieval.

Encodes what a recipe IS (ingredients → cuisine, style, dietary profile) rather
than how it is rated. Drop-in Stage 1 alternative to CandidateGenerator.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


def _to_doc(ingredients) -> str:
    """Join ingredients into a TF-IDF document; multi-word names are underscore-joined."""
    if ingredients is None:
        return ""
    if hasattr(ingredients, "tolist"):
        ingredients = ingredients.tolist()
    return " ".join(str(ing).strip().replace(" ", "_") for ing in ingredients if ing)


class IngredientEmbedder:
    """TF-IDF over ingredients_clean. min_df/max_df filter rare tokens and ubiquitous ones (salt, water)."""

    def __init__(
        self,
        recipes_df: pd.DataFrame,
        bundle_ids: Set[int],
        min_df: int = 5,
        max_df: float = 0.7,
    ) -> None:
        _df = recipes_df.reset_index() if "recipe_id" not in recipes_df.columns \
              else recipes_df.reset_index()

        recipe_ids: List[int] = []
        docs: List[str] = []

        for _, row in _df.iterrows():
            rid = int(row.get("recipe_id", row.name))
            if rid not in bundle_ids:
                continue
            doc = _to_doc(row.get("ingredients_clean", []))
            if not doc:
                continue
            recipe_ids.append(rid)
            docs.append(doc)

        self._recipe_ids: List[int] = recipe_ids

        self._vectorizer = TfidfVectorizer(
            analyzer="word",
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=True,   # log(1 + tf) dampens high-frequency ingredients
        )
        raw_matrix = self._vectorizer.fit_transform(docs)

        # L2-normalise rows so cosine similarity = dot product
        self._matrix: sp.csr_matrix = normalize(raw_matrix, norm="l2")

        self._id_to_idx: Dict[int, int] = {rid: i for i, rid in enumerate(recipe_ids)}

    def get_candidates(
        self,
        recipe_ids: List[int],
        ratings: List[float],
        top_n: int = 1000,
        exclude_ids: Optional[Set[int]] = None,
        timestamps: Optional[List] = None,
        liked_rating_threshold: float = 4.0,
        recency_half_life_days: float = 180.0,
    ) -> List[Tuple[int, float]]:
        """Build a recency-weighted ingredient centroid and return top-n (recipe_id, cosine_sim)."""
        exclude = exclude_ids or set()

        if timestamps is not None and len(timestamps) == len(ratings):
            try:
                ref_date = max(timestamps)
                lam = math.log(2.0) / recency_half_life_days
                recency_weights = [
                    math.exp(-lam * max(0, (ref_date - ts).days))
                    for ts in timestamps
                ]
            except Exception:
                recency_weights = [1.0] * len(ratings)
        else:
            recency_weights = [1.0] * len(ratings)

        centroid: Optional[sp.csr_matrix] = None
        total_weight = 0.0

        for rid, rating, rw in zip(recipe_ids, ratings, recency_weights):
            if rating < liked_rating_threshold:
                continue
            idx = self._id_to_idx.get(rid)
            if idx is None:
                continue
            weight = (rating - 3.0) * rw
            row_vec = self._matrix[idx]   # sparse (1, vocab)
            centroid = row_vec if centroid is None else centroid + weight * row_vec
            total_weight += weight

        if centroid is None or total_weight <= 0:
            return []

        centroid_dense = np.asarray(centroid.todense()).ravel()
        norm = float(np.linalg.norm(centroid_dense))
        if norm == 0.0:
            return []
        centroid_dense /= norm

        sims = self._matrix.dot(centroid_dense)  # (N,) rows already L2-normalised
        ranked_idx = np.argsort(sims)[::-1]
        results: List[Tuple[int, float]] = []
        for i in ranked_idx:
            rid = self._recipe_ids[i]
            if rid in exclude:
                continue
            results.append((rid, float(sims[i])))
            if len(results) >= top_n:
                break

        return results

    @property
    def vocab_size(self) -> int:
        return len(self._vectorizer.vocabulary_)

    @property
    def n_recipes(self) -> int:
        return len(self._recipe_ids)
