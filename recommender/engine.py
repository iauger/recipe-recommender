# recommender/engine.py

"""
RecommenderEngine -- two-stage personalized recommendation pipeline.

Stage 1 (Candidate Generation): CandidateGenerator uses IDF-weighted
Jaccard overlap over taxonomy tags (tags_clean) to retrieve a culinarily
coherent pool of ~500 recipes. This replaces raw KNN over the full 173K
corpus, which degenerates because RecipeNet embeddings encode rating
quality rather than culinary identity.

Stage 2 (Personalized Scoring): cosine_knn ranks the candidate pool by
embedding similarity to the query vector (seed centroid for session-based,
taste centroid for history-aware), with an optional tag affinity boost
from the 17 Phase-1 sentiment tags.

Modes
-----
SESSION_BASED
    Seeds: 1-3 recipes the user selects.
    Stage 1 query: union of seed recipe taxonomy tags.
    Stage 2 query: unit-normalised mean of seed embeddings.
    Excludes seed recipes from output.

HISTORY_AWARE (simulated from dataset)
    Seeds: all recipes reviewed by a user (from gold_labeled_reviews).
    Stage 1 query: affinity-weighted top tags from the user's history.
    Stage 2 query: rating*recency weighted embedding centroid.
    Stage 2 boost: 17-D sentiment tag affinity vector.
    Excludes already-reviewed recipes from output.

COLD_START
    No user signal. Returns highest quality_score recipes from the
    full embedding bundle -- the existing IR baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from recommender.candidate import CandidateGenerator
from recommender.ingredient_embed import IngredientEmbedder
from recommender.knn import KNNResult, cosine_knn, session_query_vector
from recommender.profile import (
    UserProfile,
    build_user_profile,
    get_eligible_user_ids,
    load_user_review_history,
)


# ---------------------------------------------------------------------------
# Enums and result types
# ---------------------------------------------------------------------------

class RecommendMode(Enum):
    COLD_START    = "cold_start"
    SESSION_BASED = "session_based"
    HISTORY_AWARE = "history_aware"


@dataclass
class RecommendResult:
    """
    Output bundle for a single recommendation request.

    Attributes
    ----------
    mode : RecommendMode
    results : list[KNNResult]
        Ranked recommendations.
    candidate_pool_size : int
        Number of candidates entering Stage 2 (for transparency/eval).
    user_profile : UserProfile | None
        Populated for HISTORY_AWARE mode.
    seed_recipe_ids : list[int]
        Populated for SESSION_BASED mode.
    query_tags : set[str]
        Semantic tags used for Stage 1 candidate generation.
    """
    mode: RecommendMode
    results: List[KNNResult]
    candidate_pool_size: int = 0
    user_profile: Optional[UserProfile] = None
    seed_recipe_ids: List[int] = field(default_factory=list)
    query_tags: Set[str] = field(default_factory=set)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class RecommenderEngine:
    """
    Two-stage personalized recommender engine.

    Parameters
    ----------
    settings : Settings
        Loaded via core.config.load_settings().
    search_engine : SearchEngine | None
        If provided, the embedding bundle is shared from the already-loaded
        SemanticReranker to avoid double memory usage.
        Pass None to load the bundle independently (no ES required).
    """

    # Stage 1 candidate pool size -- larger pool = better recall, slower Stage 2
    CANDIDATE_POOL_SIZE: int = 1000
    # Minimum tag overlap score to include a recipe in Stage 2
    MIN_OVERLAP_SCORE: float = 0.0
    # Blend weight for the 17-D sentiment tag affinity boost
    TAG_AFFINITY_WEIGHT: float = 0.25
    # Blend weight for the taxonomy tag preference score (Chen & Zhang Stage 2 term)
    TAXONOMY_TAG_WEIGHT: float = 0.15
    # Blend weight for the RecipeNet quality score in Stage 2 ranking
    QUALITY_WEIGHT: float = 0.0
    # Number of top tags to use from user history for Stage 1
    USER_HISTORY_TOP_TAGS: int = 40
    # Number of embedding-NN candidates to inject into Stage 1 from the full bundle.
    # Tested at 300: collapsed TagSim 0.182→0.109 (embedding and tag spaces are orthogonal --
    # cosine-similar recipes do not share culinary tags, dominate Stage 2 pre-filter).
    # Kept at 0; left in place as an ablation handle.
    EMB_POOL_SIZE: int = 0
    # Stage 2 pre-filter: keep top (top_k * STAGE2_PREFILTER_MULT) by cosine similarity
    # before re-scoring.  Default 3 was tuned for a 1000-candidate tag-only pool.
    # Hybrid pools (~1800 candidates) may benefit from a larger multiplier.
    STAGE2_PREFILTER_MULT: int = 3

    def __init__(self, settings: Any, search_engine: Optional[Any] = None):
        self.s = settings
        self.search_engine = search_engine

        # ── Embedding bundle (shared with SemanticReranker if available) ────
        if search_engine is not None:
            self._bundle     = search_engine.reranker.bundle
            self._embeddings = search_engine.reranker.embeddings  # pre-normalised (N, 128)
        else:
            self._bundle     = torch.load(settings.embeddings_path, weights_only=False)
            raw              = self._bundle["embeddings"]
            self._embeddings = F.normalize(raw.float(), p=2, dim=1)

        self._recipe_ids: List[int] = [int(r) for r in self._bundle["recipe_ids"]]
        self._predictions = self._bundle["predictions"]
        self._bundle_id_set: Set[int] = set(self._recipe_ids)

        # ── Recipe metadata DataFrame ────────────────────────────────────────
        self._recipes_df: pd.DataFrame = pd.read_parquet(settings.recipes_path)
        if "recipe_id" in self._recipes_df.columns:
            self._recipes_df["recipe_id"] = pd.to_numeric(
                self._recipes_df["recipe_id"], errors="coerce"
            ).astype(int)
            self._recipes_df = self._recipes_df.set_index("recipe_id")

        # ── Stage 1: CandidateGenerator (tag inverted index) ─────────────────
        # Build at init -- takes ~2-3s, then all candidate lookups are O(|tags|)
        self._candidates = CandidateGenerator(
            recipes_df=self._recipes_df,
            bundle_ids=self._bundle_id_set,
        )

        # ── Stage 1 (alt): TF-IDF ingredient embedder ────────────────────────
        # Culinary-identity embedding: encodes what a recipe IS (ingredients)
        # rather than how it is rated.  Used by the two_stage_ingr eval condition.
        self._ingr_embedder = IngredientEmbedder(
            recipes_df=self._recipes_df,
            bundle_ids=self._bundle_id_set,
        )

        # Precompute prediction lookup
        preds = self._predictions
        if hasattr(preds, "tolist"):
            preds = preds.tolist()
        self._id_to_pred: Dict[int, float] = {
            rid: float(p)
            for rid, p in zip(self._recipe_ids, preds)
        }

        # Lazy-loaded name fallback for recipes absent from the processed parquet.
        # Populated on first miss in get_recipe_by_id; None means not yet loaded.
        self._raw_name_lookup: Optional[Dict[int, str]] = None

    # ── Public interface ────────────────────────────────────────────────────

    def recommend_session(
        self,
        seed_recipe_ids: List[int],
        top_k: int = 10,
    ) -> RecommendResult:
        """
        Session-based recommendations from 1-3 seed recipes.

        Stage 1: Union of seed taxonomy tags -> IDF-weighted Jaccard overlap
                 candidate pool (up to CANDIDATE_POOL_SIZE recipes).
        Stage 2: KNN over 128D embeddings within the candidate pool,
                 using the mean seed embedding as the query vector.
        """
        # Stage 1 -- tag-based candidate generation
        query_tags = self._candidates.tags_for_recipes(seed_recipe_ids)
        exclude    = set(seed_recipe_ids)

        candidates = self._candidates.get_candidates(
            query_tags=query_tags,
            top_n=self.CANDIDATE_POOL_SIZE,
            exclude_ids=exclude,
        )

        if not candidates:
            return RecommendResult(
                mode=RecommendMode.SESSION_BASED,
                results=[],
                seed_recipe_ids=seed_recipe_ids,
                query_tags=query_tags,
            )

        candidate_ids = [rid for rid, _ in candidates]

        # Stage 2 -- embedding KNN within the candidate pool
        query_vec = session_query_vector(
            seed_recipe_ids, self._embeddings, self._recipe_ids
        )
        if query_vec is None:
            # Fall back to tag-overlap ordering if seeds not in bundle
            results = self._overlap_fallback(candidates, top_k)
            return RecommendResult(
                mode=RecommendMode.SESSION_BASED,
                results=results,
                candidate_pool_size=len(candidates),
                seed_recipe_ids=seed_recipe_ids,
                query_tags=query_tags,
            )

        results = self._stage2_knn(
            query_vec=query_vec,
            candidate_ids=candidate_ids,
            top_k=top_k,
            tag_affinity=None,  # no sentiment affinity for session mode
        )

        return RecommendResult(
            mode=RecommendMode.SESSION_BASED,
            results=results,
            candidate_pool_size=len(candidates),
            seed_recipe_ids=seed_recipe_ids,
            query_tags=query_tags,
        )

    def recommend_for_user(
        self,
        user_id: Any,
        top_k: int = 10,
    ) -> RecommendResult:
        """
        History-aware recommendations for a simulated user from the dataset.

        Stage 1: User's affinity-weighted top taxonomy tags -> candidate pool.
        Stage 2: KNN from user's weighted embedding centroid + 17-D tag
                 affinity re-score.
        """
        records, ok = load_user_review_history(
            self.s.gold_reviews_path,
            user_id,
            self.s.culinary_tags,
            min_reviews=self.s.min_reviews_for_cf,
        )

        if not ok or not records:
            return RecommendResult(
                mode=RecommendMode.HISTORY_AWARE,
                results=[],
                user_profile=None,
            )

        profile = build_user_profile(
            records=records,
            user_id=user_id,
            bundle=self._bundle,
            culinary_tags=self.s.culinary_tags,
            half_life_days=self.s.recency_half_life_days,
        )

        # Stage 1 -- derive query tags from user's history
        rated_ids   = [r.recipe_id for r in records]
        ratings     = [r.star_rating for r in records]
        timestamps  = [r.timestamp for r in records]
        query_tags  = self._candidates.tags_for_user_history(
            recipe_ids=rated_ids,
            ratings=ratings,
            timestamps=timestamps,
            top_n_tags=self.USER_HISTORY_TOP_TAGS,
        )

        candidates = self._candidates.get_candidates(
            query_tags=query_tags,
            top_n=self.CANDIDATE_POOL_SIZE,
            exclude_ids=profile.rated_recipe_ids,
        )

        # Stage 1b: inject embedding-NN candidates from the full bundle.
        # These are the globally top-EMB_POOL_SIZE recipes by cosine similarity
        # to the user's taste centroid -- they capture held-out items that share
        # embedding proximity but don't surface via tag overlap alone.
        if self.EMB_POOL_SIZE > 0 and profile.embedding is not None:
            tag_id_set = {rid for rid, _ in candidates}
            emb_ids = self._embedding_nn_candidates(
                profile.embedding,
                n=self.EMB_POOL_SIZE + len(tag_id_set),  # overshoot to account for overlap
                exclude_ids=profile.rated_recipe_ids,
            )
            extra = [rid for rid in emb_ids if rid not in tag_id_set][:self.EMB_POOL_SIZE]
            candidate_ids = [rid for rid, _ in candidates] + extra
        else:
            candidate_ids = [rid for rid, _ in candidates]

        if not candidate_ids or profile.embedding is None:
            return RecommendResult(
                mode=RecommendMode.HISTORY_AWARE,
                results=[],
                user_profile=profile,
                query_tags=query_tags,
            )

        # Build taxonomy tag frequency profile from highly-rated reviews.
        # Normalized frequency over tags_clean of recipes rated >= 4 stars.
        user_tag_freq: Dict[str, float] = {}
        for rec in records:
            if rec.star_rating >= 4.0:
                for tag in self._candidates.tags_for_recipe(rec.recipe_id):
                    user_tag_freq[tag] = user_tag_freq.get(tag, 0.0) + 1.0
        if user_tag_freq:
            _total = sum(user_tag_freq.values())
            user_tag_freq = {t: v / _total for t, v in user_tag_freq.items()}

        # Stage 2 -- KNN from taste centroid + tag affinity boost
        results = self._stage2_knn(
            query_vec=profile.embedding,
            candidate_ids=candidate_ids,
            top_k=top_k,
            tag_affinity=profile.tag_affinity,
            user_tag_freq=user_tag_freq,
        )

        return RecommendResult(
            mode=RecommendMode.HISTORY_AWARE,
            results=results,
            candidate_pool_size=len(candidate_ids),
            user_profile=profile,
            query_tags=query_tags,
        )

    def recommend_cold_start(self, top_k: int = 10) -> RecommendResult:
        """
        Cold-start baseline: highest quality_score recipes in the bundle.
        No user signal. Used as evaluation baseline.
        """
        ranked = sorted(self._id_to_pred.items(), key=lambda x: x[1], reverse=True)
        results = []
        for rid, quality in ranked[:top_k]:
            results.append(KNNResult(
                recipe_id=rid,
                similarity=0.0,
                quality_score=quality,
                source=self._get_source(rid),
            ))
        return RecommendResult(
            mode=RecommendMode.COLD_START,
            results=results,
            candidate_pool_size=len(self._recipe_ids),
        )

    def recommend_knn_only(
        self,
        seed_recipe_ids: Optional[List[int]] = None,
        user_embedding: Optional[torch.Tensor] = None,
        top_k: int = 10,
        exclude_ids: Optional[Set[int]] = None,
    ) -> RecommendResult:
        """
        Raw KNN over the full bundle -- no tag filtering.
        Used as an ablation baseline for evaluation (vs. two-stage).
        Supply either seed_recipe_ids (session) or user_embedding (history).
        """
        if seed_recipe_ids is not None:
            query_vec = session_query_vector(
                seed_recipe_ids, self._embeddings, self._recipe_ids
            )
            exclude = set(seed_recipe_ids) | (exclude_ids or set())
        elif user_embedding is not None:
            query_vec = user_embedding
            exclude   = exclude_ids or set()
        else:
            return RecommendResult(mode=RecommendMode.SESSION_BASED, results=[])

        if query_vec is None:
            return RecommendResult(mode=RecommendMode.SESSION_BASED, results=[])

        hits = cosine_knn(
            query_vec, self._embeddings, self._recipe_ids,
            k=top_k, exclude_ids=exclude,
        )
        results = [
            KNNResult(
                recipe_id=rid,
                similarity=sim,
                quality_score=self._id_to_pred.get(rid, 0.0),
                source=self._get_source(rid),
            )
            for rid, sim in hits
        ]
        return RecommendResult(
            mode=RecommendMode.SESSION_BASED,
            results=results,
            candidate_pool_size=len(self._recipe_ids),
        )

    def build_profile_for_user(
        self,
        user_id: Any,
    ) -> Optional[Tuple[Any, Set[str]]]:
        """
        Build a UserProfile and extract affinity tags for a given user.

        Returns (profile, user_tags) or None if the user is ineligible
        (too few reviews or no recipes in the embedding bundle).

        Used by the Streamlit Personalized Search mode to obtain user signals
        for SearchEngine.search() without going through recommend_for_user.
        The returned values map directly to search() parameters:

            result = search_engine.search(
                query=query,
                user_embedding=profile.embedding,
                user_tag_affinity=profile.tag_affinity,
                user_tags=user_tags,
                ...
            )
        """
        records, ok = load_user_review_history(
            self.s.gold_reviews_path,
            user_id,
            self.s.culinary_tags,
            min_reviews=self.s.min_reviews_for_cf,
        )
        if not ok or not records:
            return None

        profile = build_user_profile(
            records=records,
            user_id=user_id,
            bundle=self._bundle,
            culinary_tags=self.s.culinary_tags,
            half_life_days=self.s.recency_half_life_days,
        )
        if profile.embedding is None:
            return None

        rated_ids  = [r.recipe_id for r in records]
        ratings    = [r.star_rating for r in records]
        timestamps = [r.timestamp for r in records]
        user_tags  = self._candidates.tags_for_user_history(
            recipe_ids=rated_ids,
            ratings=ratings,
            timestamps=timestamps,
            top_n_tags=self.USER_HISTORY_TOP_TAGS,
        )
        return profile, user_tags

    # ── UI helpers ──────────────────────────────────────────────────────────

    def get_eligible_users(self, min_reviews: Optional[int] = None) -> List[Any]:
        n = min_reviews if min_reviews is not None else self.s.min_reviews_for_cf
        return get_eligible_user_ids(self.s.gold_reviews_path, min_reviews=n)

    def get_recipe_by_id(self, recipe_id: int) -> Optional[Dict[str, Any]]:
        try:
            row = self._recipes_df.loc[recipe_id]
            return row.to_dict() if isinstance(row, pd.Series) else None
        except KeyError:
            name = self._raw_name_fallback(recipe_id)
            return {"name": name} if name else None

    def _raw_name_fallback(self, recipe_id: int) -> Optional[str]:
        """
        Lazy-load name from RAW_recipes.csv for recipes not in the processed
        parquet (e.g. filtered during preprocessing).  Reads only the id and
        name columns; result is cached for the lifetime of the engine instance.
        """
        if self._raw_name_lookup is None:
            raw_path = self.s.food_raw_recipes_path
            try:
                raw_df = pd.read_csv(raw_path, usecols=["id", "name"])
                self._raw_name_lookup = {
                    int(rid): str(name)
                    for rid, name in zip(raw_df["id"], raw_df["name"])
                    if pd.notna(name)
                }
            except Exception:
                self._raw_name_lookup = {}
        return self._raw_name_lookup.get(recipe_id)

    def search_recipes_by_name(self, query: str, top_n: int = 20) -> List[Dict[str, Any]]:
        q   = query.lower()
        df  = self._recipes_df.reset_index()
        mask = df["name"].str.lower().str.contains(q, na=False)
        hits = df[mask].head(top_n)
        out  = []
        for _, row in hits.iterrows():
            out.append({
                "recipe_id":      int(row.get("recipe_id", row.name)),
                "name":           str(row.get("name", "")),
                "minutes":        row.get("minutes"),
                "bayesian_rating": row.get("bayesian_rating"),
            })
        return out

    # ── Private helpers ─────────────────────────────────────────────────────

    def _stage2_knn(
        self,
        query_vec: torch.Tensor,
        candidate_ids: List[int],
        top_k: int,
        tag_affinity: Optional[np.ndarray],
        user_tag_freq: Optional[Dict[str, float]] = None,
    ) -> List[KNNResult]:
        """
        Score ALL candidates in the pool with embedding cosine + quality +
        optional affinity signals, then return top_k.

        Scores the full candidate pool via direct matrix multiply rather than
        pre-filtering to top_k*3 by cosine similarity first.  This allows the
        quality and affinity terms to rescue candidates that are slightly below
        the cosine cutoff but have a much higher quality score.
        """
        id_to_bundle_idx = {rid: idx for idx, rid in enumerate(self._recipe_ids)}

        local_indices = [
            id_to_bundle_idx[rid]
            for rid in candidate_ids
            if rid in id_to_bundle_idx
        ]
        valid_ids = [
            candidate_ids[i]
            for i, rid in enumerate(candidate_ids)
            if rid in id_to_bundle_idx
        ]

        if not local_indices:
            return []

        local_embeddings = self._embeddings[local_indices]  # (M, 128)

        # Pre-filter to top_k*3 by cosine similarity, then re-score.
        # The pre-filter concentrates Stage 2 competition among the most
        # embedding-similar candidates, preserving ranking precision for
        # the held-out items that make it into the pool.
        hits = cosine_knn(
            query_vec=query_vec,
            embeddings=local_embeddings,
            recipe_ids=valid_ids,
            k=min(top_k * self.STAGE2_PREFILTER_MULT, len(valid_ids)),
        )

        results: List[KNNResult] = []
        for rid, sim in hits:
            source  = self._get_source(rid)
            quality = self._id_to_pred.get(rid, 0.0)

            final_sim = sim + self.QUALITY_WEIGHT * quality

            if tag_affinity is not None:
                tag_vec = self._get_sentiment_tag_vector(source)
                if tag_vec is not None:
                    boost = float(np.dot(tag_affinity, tag_vec))
                    final_sim += self.TAG_AFFINITY_WEIGHT * boost

            if user_tag_freq:
                recipe_tags = self._candidates.tags_for_recipe(rid)
                if recipe_tags:
                    tp_score = sum(user_tag_freq.get(t, 0.0) for t in recipe_tags) / len(recipe_tags)
                    final_sim += self.TAXONOMY_TAG_WEIGHT * tp_score

            results.append(KNNResult(
                recipe_id=rid,
                similarity=final_sim,
                quality_score=quality,
                source=source,
            ))

        results.sort(key=lambda r: r.similarity, reverse=True)
        return results[:top_k]

    def _embedding_nn_candidates(
        self,
        user_embedding: torch.Tensor,
        n: int,
        exclude_ids: Optional[Set[int]] = None,
    ) -> List[int]:
        """Top-n recipe IDs from the full bundle by cosine similarity to user_embedding."""
        hits = cosine_knn(
            user_embedding, self._embeddings, self._recipe_ids,
            k=n, exclude_ids=exclude_ids or set(),
        )
        return [rid for rid, _ in hits]

    def _overlap_fallback(
        self,
        candidates: List[Tuple[int, float]],
        top_k: int,
    ) -> List[KNNResult]:
        """Return top-k by tag overlap score when embedding lookup fails."""
        results = []
        for rid, overlap in candidates[:top_k]:
            results.append(KNNResult(
                recipe_id=rid,
                similarity=overlap,
                quality_score=self._id_to_pred.get(rid, 0.0),
                source=self._get_source(rid),
            ))
        return results

    def _get_source(self, recipe_id: int) -> Dict[str, Any]:
        try:
            row = self._recipes_df.loc[recipe_id]
            if isinstance(row, pd.Series):
                return row.to_dict()
        except KeyError:
            pass
        return {"recipe_id": recipe_id}

    def _get_sentiment_tag_vector(
        self,
        source: Dict[str, Any],
    ) -> Optional[np.ndarray]:
        """Build 17-D intensity vector from recipe source metadata."""
        tags = self.s.culinary_tags
        vec  = np.array(
            [float(source.get("intensity_{}_".format(t), source.get("intensity_{}".format(t), 0.0)))
             for t in tags],
            dtype=np.float32,
        )
        return vec if vec.sum() > 0 else None
