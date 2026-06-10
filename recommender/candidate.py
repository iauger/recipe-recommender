"""
Tag-based Stage 1 candidate generation for the recommender pipeline.

Builds an IDF-weighted inverted index over the dataset's taxonomy tags (cuisine,
course, protein, method, dietary, occasion — not the Phase-1 sentiment tags).
Candidate scoring is IDF-weighted tag recall so rare diagnostic tags outweigh
common ones, and tag-rich recipes aren't penalised for having extra tags.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


# Category header labels in the taxonomy (not semantic values — strip these).
_STRUCTURAL_TAGS: FrozenSet[str] = frozenset({
    "time-to-make",
    "course",
    "main-ingredient",
    "cuisine",
    "preparation",
    "occasion",
    "dietary",
    "equipment",
    "technique",
    "for-large-groups",
    "number-of-servings",
    "presentation",
    "served-hot",
    "served-cold",
    "from-scratch",
    "copycat-recipe",
})

_TIME_TAGS: FrozenSet[str] = frozenset({
    "15-minutes-or-less",
    "30-minutes-or-less",
    "60-minutes-or-less",
    "4-hours-or-less",
    "weeknight",          # kept for occasion matching; low IDF weight in practice
})

_MIN_TAG_FREQ = 10  # rare tags below this are excluded from the inverted index


class CandidateGenerator:
    """
    Content-based Stage 1 retrieval via IDF-weighted taxonomy tag overlap.

    bundle_ids restricts candidates to recipes that have a Phase 2 embedding,
    so Stage 2 KNN always has an embedding for every candidate.
    """

    def __init__(
        self,
        recipes_df: pd.DataFrame,
        bundle_ids: Set[int],
    ) -> None:
        self._bundle_ids = bundle_ids

        self._inverted: Dict[str, List[int]] = defaultdict(list)
        self._forward: Dict[int, FrozenSet[str]] = {}
        tag_freq: Dict[str, int] = defaultdict(int)

        _df = recipes_df.reset_index() if "recipe_id" not in recipes_df.columns \
              else recipes_df.reset_index()

        for _, row in _df.iterrows():
            rid = int(row.get("recipe_id", row.name))
            if rid not in bundle_ids:
                continue
            raw_tags = row.get("tags_clean", [])
            semantic = _extract_semantic_tags(raw_tags)
            self._forward[rid] = frozenset(semantic)
            for t in semantic:
                tag_freq[t] += 1

        n_recipes = len(self._forward)

        self._idf: Dict[str, float] = {
            tag: float(np.log(n_recipes / max(freq, 1)))
            for tag, freq in tag_freq.items()
            if freq >= _MIN_TAG_FREQ
        }

        for rid, tags in self._forward.items():
            for t in tags:
                if t in self._idf:
                    self._inverted[t].append(rid)

    def get_candidates(
        self,
        query_tags: Set[str],
        top_n: int = 500,
        exclude_ids: Optional[Set[int]] = None,
    ) -> List[Tuple[int, float]]:
        """Return top-n (recipe_id, score) by IDF-weighted tag recall, excluding exclude_ids."""
        if not query_tags:
            return []

        exclude = exclude_ids or set()

        active_tags = {t for t in query_tags if t in self._idf}
        if not active_tags:
            return []

        scores: Dict[int, float] = defaultdict(float)
        for tag in active_tags:
            idf = self._idf[tag]
            for rid in self._inverted.get(tag, []):
                if rid not in exclude:
                    scores[rid] += idf

        # Normalise by query IDF sum → tag recall. Avoids penalising tag-rich
        # recipes that match all query tags but also carry extras (Jaccard problem).
        query_idf_sum = sum(self._idf.get(t, 0.0) for t in active_tags)
        if query_idf_sum > 0:
            for rid in scores:
                scores[rid] = scores[rid] / query_idf_sum

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_n]

    def tags_for_recipe(self, recipe_id: int) -> FrozenSet[str]:
        """Return the semantic tag set for a single recipe."""
        return self._forward.get(recipe_id, frozenset())

    def tags_for_recipes(self, recipe_ids: List[int]) -> Set[str]:
        """Union of semantic tags across multiple recipes."""
        result: Set[str] = set()
        for rid in recipe_ids:
            result |= self._forward.get(rid, frozenset())
        return result

    def tags_for_user_history(
        self,
        recipe_ids: List[int],
        ratings: List[float],
        top_n_tags: int = 20,
        liked_rating_threshold: float = 4.0,
        timestamps: Optional[List] = None,
        recency_half_life_days: float = 180.0,
        negative_threshold: float = 3.0,
        neg_amplifier: float = 2.0,
    ) -> Set[str]:
        """
        Return tags from review history for Stage 1 retrieval.

        Unions two signals:
        1. Affinity tags — top-N by IDF-weighted deviation from global mean,
           with recency decay and negative amplification for disliked recipes.
        2. Liked-recipe tags — top-N by IDF from recipes >= liked_rating_threshold,
           filtered to net-positive deviation (prevents ambiguous tags from entering
           via the coverage path when their overall signal is negative).
        """
        global_mean = 4.4

        # Compute per-review recency weights
        if timestamps is not None and len(timestamps) == len(ratings):
            try:
                ref_date = max(timestamps)
                lam = math.log(2.0) / recency_half_life_days
                recency_weights = []
                for ts in timestamps:
                    days_ago = max(0, (ref_date - ts).days)
                    recency_weights.append(math.exp(-lam * days_ago))
            except Exception:
                recency_weights = [1.0] * len(ratings)
        else:
            recency_weights = [1.0] * len(ratings)

        tag_scores: Dict[str, float] = defaultdict(float)
        liked_tag_idf: Dict[str, float] = {}

        for rid, rating, rw in zip(recipe_ids, ratings, recency_weights):
            deviation = rating - global_mean
            # Amplify negative signal for explicitly disliked recipes
            weight = rw * (neg_amplifier if rating <= negative_threshold else 1.0)
            recipe_tags = self._forward.get(rid, frozenset())
            for t in recipe_tags:
                if t in self._idf:
                    tag_scores[t] += deviation * weight * self._idf[t]
            if rating >= liked_rating_threshold:
                for t in recipe_tags:
                    if t in self._idf:
                        liked_tag_idf[t] = self._idf[t]

        # Signal 1: affinity-ranked tags (net positive deviation score)
        positive = {t: s for t, s in tag_scores.items() if s > 0}
        ranked_affinity = sorted(positive.items(), key=lambda x: x[1], reverse=True)
        affinity_tags = {t for t, _ in ranked_affinity[:top_n_tags]}

        # Signal 2: IDF-ranked liked-recipe tags, filtered to net-positive only.
        # Prevents tags that are net-negative (appear in too many disliked recipes)
        # from entering the query through the liked-recipe coverage path.
        ranked_liked = sorted(
            [(t, idf) for t, idf in liked_tag_idf.items() if tag_scores.get(t, 0.0) >= 0],
            key=lambda x: x[1],
            reverse=True,
        )
        liked_tags = {t for t, _ in ranked_liked[:top_n_tags]}

        return affinity_tags | liked_tags

    def build_user_tag_freq(
        self,
        recipe_ids: List[int],
        ratings: List[float],
        liked_rating_threshold: float = 4.0,
    ) -> Dict[str, float]:
        """
        Build an IDF-weighted, normalized tag frequency vector from a user's
        liked recipes.

        For each recipe rated >= liked_rating_threshold, accumulate a count
        per semantic tag, then multiply by IDF so that rare diagnostic tags
        (cuisine, protein) outweigh common ones (weeknight).  The result is
        normalised to [0, 1] so it can be used as a direct additive boost in
        Stage 2 scoring.

        Used to implement the taxonomy tag preference term:
            tag_pref(u, i) = Σ_t  user_tag_freq[t] × (t ∈ recipe_tags[i])
        """
        freq: Dict[str, float] = defaultdict(float)
        for rid, rating in zip(recipe_ids, ratings):
            if rating < liked_rating_threshold:
                continue
            for t in self._forward.get(rid, frozenset()):
                if t in self._idf:
                    freq[t] += 1.0

        if not freq:
            return {}

        weighted: Dict[str, float] = {t: count * self._idf[t] for t, count in freq.items()}
        max_val = max(weighted.values())
        if max_val <= 0:
            return {}
        return {t: v / max_val for t, v in weighted.items()}

    def synthesize_query_string(
        self,
        recipe_ids: List[int],
        ratings: List[float],
        top_n_tags: int = 12,
    ) -> str:
        """
        Synthesize a natural-language query from a user's review history.

        The returned string is a space-joined list of the user's top
        affinity-weighted taxonomy tags.  It is designed to be passed
        straight into parse_user_intent (and through to SearchEngine.run)
        as if the user had typed it: the intent parser will pattern-match
        cuisine / protein / course / method / occasion tokens against its
        regex tables, and any unmatched tokens fall through to the BM25
        lexical clause via clean_text.
        caller is responsible for choosing a fallback (e.g. cold-start).
        """
        tags = self.tags_for_user_history(
            recipe_ids=recipe_ids,
            ratings=ratings,
            top_n_tags=top_n_tags,
        )
        if not tags:
            return ""
        return " ".join(sorted(tags))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_semantic_tags(raw_tags: Any) -> List[str]:
    """Extract meaningful semantic tags, filtering structural category headers."""
    if raw_tags is None:
        return []
    if isinstance(raw_tags, str):
        tags = raw_tags.split()
    elif hasattr(raw_tags, "tolist"):
        tags = raw_tags.tolist()
    else:
        tags = list(raw_tags)
    return [
        t for t in tags
        if isinstance(t, str)
        and t not in _STRUCTURAL_TAGS
        and len(t) > 2
    ]
