# recommender/evaluate.py

"""
Offline evaluation for the personalized recommender (Phase 4).

Protocol: Leave-N-Out (LNO)
-----------------------------
For each sampled user with >= min_reviews reviews:
  1. Sort reviews by timestamp ascending.
  2. Hold out the last n_holdout reviews that are in the embedding bundle
     as the test set.
  3. Train on all remaining reviews.
  4. Generate top-k recommendations from training history.
  5. Score with Recall@k and Tag Similarity@k.

Switching from single-item LOO to multi-held-out is the primary fix for
the metric-floor problem: with 1 held-out item in a 173K corpus, even a
near-perfect recommender lands near 0 HR. Holding out 5 items gives 5x
signal density, and Recall@k (fraction of held-outs recovered) is a
continuous metric instead of binary hit/miss.

Tag Similarity@k (supplementary):
  For each recommendation in top-k, compute Jaccard similarity of its
  taxonomy tags against the union of the held-out recipes' tags. Mean
  over top-k. This is a continuous relevance signal that works even when
  the exact held-out items are never recovered.

Conditions compared
-------------------
  cold_start   Quality-ranked baseline (no user signal).
  knn_only     Raw KNN over the full bundle -- no tag filtering (Stage 1 ablation).
  two_stage    Tag-filtered candidate pool + embedding KNN + tag affinity boost.
  unified      Personalized search: tag-direct ES retrieval + user embedding +
               affinity blended into the existing SemanticReranker. Requires ES.

Usage
-----
    python -m recommender.evaluate [--n-holdout 5] [--no-unified] [--n-users 200]
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from core.config import load_settings, get_es_client
from recommender.engine import RecommenderEngine
from recommender.profile import build_user_profile
from search.engine import SearchEngine, SearchMode


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

UNIFIED_DEFAULT_ALPHA: float = 0.5
UNIFIED_DEFAULT_AFFINITY_WEIGHT: float = 0.25
UNIFIED_CANDIDATE_POOL: int = 500
DEFAULT_K_VALUES: Tuple[int, ...] = (5, 10, 20, 50)
EVAL_WINDOW: int = 50
DEFAULT_N_HOLDOUT: int = 5    # number of reviews held out per user


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TestCase:
    user_id: Any
    train_records: list          # list[ReviewRecord]
    held_out_ids: List[int]      # recipe IDs held out (all in bundle)


@dataclass
class PerCaseResult:
    """Per-test-case diagnostic record."""
    user_id: Any
    held_out_ids: List[int]
    recommended_ids: List[int]                    # ranked, length = eval_window
    # single-item metrics (first hit from held_out_ids -- backward compat)
    first_hit_rank: Optional[int]                 # 1-based rank of first hit, else None
    # multi-item metrics
    recall_at_k: Dict[int, float]                 # k -> fraction of held-outs in top-k
    tag_sim_at_k: Dict[int, float]                # k -> mean Jaccard vs held-out tags
    dcg_tag_sim_at_k: Dict[int, float]            # k -> rank-discounted TagSim (NDCG-style)
    tag_coverage_at_k: Dict[int, float]           # k -> fraction of held-out tags covered by union of top-k tags
    # pool diagnostics
    pool_size: int
    held_out_in_pool: Optional[bool]              # True if ANY held-out in Stage 1 pool
    held_out_pool_count: int                      # number of held-outs in pool
    n_train_reviews: int


@dataclass
class ConditionResult:
    """Aggregate metrics for one condition across all test cases."""
    condition: str
    n_users: int
    coverage: float
    # HR / NDCG / MRR use first-hit semantics (matches prior LOO framing)
    hr_at_k: Dict[int, float]
    hr_ci_at_k: Dict[int, Tuple[float, float]]
    ndcg_at_k: Dict[int, float]
    ndcg_ci_at_k: Dict[int, Tuple[float, float]]
    mrr: float
    mrr_ci: Tuple[float, float]
    # new multi-holdout metrics
    recall_at_k: Dict[int, float]
    recall_ci_at_k: Dict[int, Tuple[float, float]]
    tag_sim_at_k: Dict[int, float]
    tag_sim_ci_at_k: Dict[int, Tuple[float, float]]
    dcg_tag_sim_at_k: Dict[int, float]
    dcg_tag_sim_ci_at_k: Dict[int, Tuple[float, float]]
    tag_coverage_at_k: Dict[int, float]
    tag_coverage_ci_at_k: Dict[int, Tuple[float, float]]
    # pool diagnostics
    pool_recall: Optional[float]
    pool_recall_mean_count: Optional[float]       # mean held-outs-in-pool count
    rank_in_pool_when_present: Optional[float]
    per_case: List[PerCaseResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def leave_n_out_sample(
    gold_reviews_path: Any,
    bundle_ids: Set[int],
    min_reviews: int = 15,
    n_holdout: int = DEFAULT_N_HOLDOUT,
    n_users: int = 200,
    seed: int = 42,
) -> List[TestCase]:
    """
    Sample up to n_users eligible test cases for Leave-N-Out evaluation.

    Eligibility:
      - User has >= min_reviews reviews.
      - At least 1 of the last n_holdout reviews is in the bundle.
      - At least (min_reviews - n_holdout) training reviews remain.
    """
    rng = random.Random(seed)

    df = pd.read_parquet(gold_reviews_path)
    date_col = "date" if "date" in df.columns else "submitted"
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df["recipe_id"] = pd.to_numeric(df["recipe_id"], errors="coerce").astype(int)

    intensity_cols = [c for c in df.columns
                      if c.startswith("intensity_") or c.startswith("sim_")]

    counts = df.groupby("user_id").size()
    eligible_users = counts[counts >= min_reviews].index.tolist()
    rng.shuffle(eligible_users)

    cases: List[TestCase] = []
    from recommender.profile import ReviewRecord

    for user_id in eligible_users:
        if len(cases) >= n_users:
            break

        user_df = df[df["user_id"] == user_id].sort_values(date_col)

        # Hold out last n_holdout reviews that are in the bundle
        candidates_for_holdout = [
            int(r["recipe_id"])
            for _, r in user_df.tail(n_holdout + 5).iterrows()  # over-fetch, then filter
            if int(r["recipe_id"]) in bundle_ids
        ]
        held_out_ids = candidates_for_holdout[-n_holdout:]  # keep the most recent n

        if not held_out_ids:
            continue

        held_out_set = set(held_out_ids)
        train_df = user_df[~user_df["recipe_id"].isin(held_out_set)]

        if len(train_df) < max(1, min_reviews - n_holdout):
            continue

        train_records = []
        for _, row in train_df.iterrows():
            intensities = {}
            for col in intensity_cols:
                val = row.get(col)
                if pd.notna(val):
                    tag = col.removeprefix("intensity_").removeprefix("sim_").rstrip("_")
                    intensities[tag] = float(val)
            train_records.append(ReviewRecord(
                recipe_id=int(row["recipe_id"]),
                star_rating=float(row["rating"]),
                timestamp=row[date_col],
                tag_intensities=intensities,
            ))

        cases.append(TestCase(
            user_id=user_id,
            train_records=train_records,
            held_out_ids=held_out_ids,
        ))

    return cases


# ---------------------------------------------------------------------------
# Tag similarity helper
# ---------------------------------------------------------------------------

def _held_out_tag_union(
    held_out_ids: List[int],
    forward: Dict[int, FrozenSet[str]],
) -> FrozenSet[str]:
    """Union of taxonomy tags across all held-out recipes."""
    result: Set[str] = set()
    for rid in held_out_ids:
        result |= forward.get(rid, frozenset())
    return frozenset(result)


def _tag_sim(
    rec_tags: FrozenSet[str],
    held_out_tags: FrozenSet[str],
) -> float:
    """Jaccard similarity between recommendation tags and held-out tag union."""
    if not rec_tags or not held_out_tags:
        return 0.0
    intersection = len(rec_tags & held_out_tags)
    union = len(rec_tags | held_out_tags)
    return intersection / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def find_first_hit_rank(
    recommended_ids: List[int],
    held_out_ids: List[int],
) -> Optional[int]:
    """1-based rank of the first item from held_out_ids in recommended_ids."""
    held_out_set = set(held_out_ids)
    for rank, rid in enumerate(recommended_ids, start=1):
        if rid in held_out_set:
            return rank
    return None


def recall_at_k(
    recommended_ids: List[int],
    held_out_ids: List[int],
    k: int,
) -> float:
    """Fraction of held-out items recovered in top-k recommendations."""
    if not held_out_ids:
        return 0.0
    recs_at_k = set(recommended_ids[:k])
    hits = len(recs_at_k & set(held_out_ids))
    return hits / len(held_out_ids)


def compute_tag_sim_at_k(
    recommended_ids: List[int],
    held_out_tags: FrozenSet[str],
    forward: Dict[int, FrozenSet[str]],
    k: int,
) -> float:
    """Mean Jaccard tag similarity of top-k recommendations vs. held-out tag union."""
    recs_at_k = recommended_ids[:k]
    if not recs_at_k:
        return 0.0
    sims = [_tag_sim(forward.get(rid, frozenset()), held_out_tags) for rid in recs_at_k]
    return float(np.mean(sims))


def compute_dcg_tag_sim_at_k(
    recommended_ids: List[int],
    held_out_tags: FrozenSet[str],
    forward: Dict[int, FrozenSet[str]],
    k: int,
) -> float:
    """
    Rank-discounted tag similarity (NDCG-style, continuous relevance).

    Uses TagSim as a continuous relevance score and discounts by log2(rank+1).
    Normalized by the ideal DCG (all positions have TagSim=1.0) so the result
    is in [0, 1].  Higher = tag-similar items appear earlier in the list.
    """
    recs_at_k = recommended_ids[:k]
    if not recs_at_k:
        return 0.0
    dcg = sum(
        _tag_sim(forward.get(rid, frozenset()), held_out_tags) / math.log2(r + 2)
        for r, rid in enumerate(recs_at_k)
    )
    idcg = sum(1.0 / math.log2(r + 2) for r in range(len(recs_at_k)))
    return float(dcg / idcg) if idcg > 0 else 0.0


def compute_tag_coverage_at_k(
    recommended_ids: List[int],
    held_out_tags: FrozenSet[str],
    forward: Dict[int, FrozenSet[str]],
    k: int,
) -> float:
    """
    Fraction of held-out tags covered by the union of top-k recommendation tags.

    Measures collective tag recall: whether the recommendation set as a whole
    covers the user's future interests, not just whether individual items are
    tag-similar.
    """
    if not held_out_tags:
        return 0.0
    rec_tag_union: Set[str] = set()
    for rid in recommended_ids[:k]:
        rec_tag_union |= forward.get(rid, frozenset())
    return len(rec_tag_union & held_out_tags) / len(held_out_tags)


def hit_from_rank(rank: Optional[int], k: int) -> float:
    return 1.0 if rank is not None and rank <= k else 0.0


def ndcg_from_rank(rank: Optional[int], k: int) -> float:
    if rank is not None and rank <= k:
        return 1.0 / math.log2(rank + 1)
    return 0.0


def reciprocal_rank(rank: Optional[int]) -> float:
    return 1.0 / rank if rank is not None else 0.0


def bootstrap_ci(
    values: List[float],
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    means = np.array([rng.choice(arr, size=n, replace=True).mean() for _ in range(n_bootstrap)])
    return (float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))))


# ---------------------------------------------------------------------------
# Condition runner
# ---------------------------------------------------------------------------

def _parse_unified_condition(condition: str) -> Optional[float]:
    """Parse alpha from 'unified_a{float}' condition name."""
    if condition == "unified":
        return None
    prefix = "unified_a"
    if condition.startswith(prefix) and not condition.startswith("unified_aw"):
        try:
            return float(condition[len(prefix):])
        except ValueError:
            return None
    return None


def _parse_unified_affinity_condition(condition: str) -> Optional[float]:
    """Parse affinity_weight from 'unified_aw{float}' condition name."""
    prefix = "unified_aw"
    if condition.startswith(prefix):
        try:
            return float(condition[len(prefix):])
        except ValueError:
            return None
    return None


def _run_condition_for_case(
    engine: RecommenderEngine,
    case: TestCase,
    condition: str,
    eval_window: int,
    *,
    search_engine: Optional[SearchEngine] = None,
    unified_alpha: float = UNIFIED_DEFAULT_ALPHA,
    unified_affinity_weight: float = UNIFIED_DEFAULT_AFFINITY_WEIGHT,
) -> Tuple[List[int], int, Optional[bool], int]:
    """
    Returns (recommended_ids, pool_size, any_held_out_in_pool, held_out_pool_count).
    """
    held_out_set = set(case.held_out_ids)

    if condition == "cold_start":
        result = engine.recommend_cold_start(top_k=eval_window)
        return [r.recipe_id for r in result.results], 0, None, 0

    is_unified = condition == "unified" or condition.startswith("unified_a")
    if is_unified:
        aw_override = _parse_unified_affinity_condition(condition)
        if aw_override is not None:
            unified_affinity_weight = aw_override
        else:
            alpha_override = _parse_unified_condition(condition)
            if alpha_override is not None:
                unified_alpha = alpha_override

    profile = build_user_profile(
        records=case.train_records,
        user_id=case.user_id,
        bundle=engine._bundle,
        culinary_tags=engine.s.culinary_tags,
        half_life_days=engine.s.recency_half_life_days,
    )
    if profile.embedding is None:
        return [], 0, None, 0

    if condition == "knn_only":
        result = engine.recommend_knn_only(
            user_embedding=profile.embedding,
            top_k=eval_window,
            exclude_ids=profile.rated_recipe_ids,
        )
        rec_ids = [r.recipe_id for r in result.results]
        pool_count = sum(1 for rid in case.held_out_ids if rid in set(engine._recipe_ids))
        return rec_ids, len(engine._recipe_ids), True, pool_count

    if condition == "two_stage":
        rated_ids  = [r.recipe_id for r in case.train_records]
        ratings    = [r.star_rating for r in case.train_records]
        timestamps = [r.timestamp for r in case.train_records]
        query_tags = engine._candidates.tags_for_user_history(
            recipe_ids=rated_ids, ratings=ratings, timestamps=timestamps,
            top_n_tags=engine.USER_HISTORY_TOP_TAGS,
        )
        candidates = engine._candidates.get_candidates(
            query_tags=query_tags, top_n=engine.CANDIDATE_POOL_SIZE,
            exclude_ids=profile.rated_recipe_ids,
        )
        if not candidates:
            return [], 0, False, 0

        candidate_ids = [rid for rid, _ in candidates]

        # Stage 1b: inject embedding-NN candidates from the full bundle
        if engine.EMB_POOL_SIZE > 0 and profile.embedding is not None:
            tag_id_set = set(candidate_ids)
            emb_ids = engine._embedding_nn_candidates(
                profile.embedding,
                n=engine.EMB_POOL_SIZE + len(tag_id_set),
                exclude_ids=profile.rated_recipe_ids,
            )
            extra = [rid for rid in emb_ids if rid not in tag_id_set][:engine.EMB_POOL_SIZE]
            candidate_ids = candidate_ids + extra

        pool_set = set(candidate_ids)
        pool_count = sum(1 for rid in case.held_out_ids if rid in pool_set)
        any_in_pool = pool_count > 0

        results = engine._stage2_knn(
            query_vec=profile.embedding,
            candidate_ids=candidate_ids,
            top_k=eval_window,
            tag_affinity=profile.tag_affinity,
        )
        return [r.recipe_id for r in results], len(candidate_ids), any_in_pool, pool_count

    if condition == "two_stage_tp":
        rated_ids  = [r.recipe_id for r in case.train_records]
        ratings    = [r.star_rating for r in case.train_records]
        timestamps = [r.timestamp for r in case.train_records]
        query_tags = engine._candidates.tags_for_user_history(
            recipe_ids=rated_ids, ratings=ratings, timestamps=timestamps,
            top_n_tags=engine.USER_HISTORY_TOP_TAGS,
        )
        candidates = engine._candidates.get_candidates(
            query_tags=query_tags, top_n=engine.CANDIDATE_POOL_SIZE,
            exclude_ids=profile.rated_recipe_ids,
        )
        if not candidates:
            return [], 0, False, 0

        candidate_ids = [rid for rid, _ in candidates]
        pool_set = set(candidate_ids)
        pool_count = sum(1 for rid in case.held_out_ids if rid in pool_set)
        any_in_pool = pool_count > 0

        user_tag_freq = engine._candidates.build_user_tag_freq(
            recipe_ids=rated_ids,
            ratings=ratings,
        )
        results = engine._stage2_knn(
            query_vec=profile.embedding,
            candidate_ids=candidate_ids,
            top_k=eval_window,
            tag_affinity=profile.tag_affinity,
            user_tag_freq=user_tag_freq,
        )
        return [r.recipe_id for r in results], len(candidate_ids), any_in_pool, pool_count

    if condition == "two_stage_ingr":
        rated_ids  = [r.recipe_id for r in case.train_records]
        ratings    = [r.star_rating for r in case.train_records]
        timestamps = [r.timestamp for r in case.train_records]

        candidates = engine._ingr_embedder.get_candidates(
            recipe_ids=rated_ids,
            ratings=ratings,
            timestamps=timestamps,
            top_n=engine.CANDIDATE_POOL_SIZE,
            exclude_ids=profile.rated_recipe_ids,
            recency_half_life_days=float(engine.s.recency_half_life_days),
        )
        if not candidates:
            return [], 0, False, 0

        candidate_ids = [rid for rid, _ in candidates]
        pool_set = set(candidate_ids)
        pool_count = sum(1 for rid in case.held_out_ids if rid in pool_set)
        any_in_pool = pool_count > 0

        results = engine._stage2_knn(
            query_vec=profile.embedding,
            candidate_ids=candidate_ids,
            top_k=eval_window,
            tag_affinity=profile.tag_affinity,
        )
        return [r.recipe_id for r in results], len(candidate_ids), any_in_pool, pool_count

    if condition == "two_stage_hybrid":
        rated_ids  = [r.recipe_id for r in case.train_records]
        ratings    = [r.star_rating for r in case.train_records]
        timestamps = [r.timestamp for r in case.train_records]

        # Stage 1a: tag recall
        tag_cands = engine._candidates.get_candidates(
            query_tags=engine._candidates.tags_for_user_history(
                recipe_ids=rated_ids, ratings=ratings, timestamps=timestamps,
                top_n_tags=engine.USER_HISTORY_TOP_TAGS,
            ),
            top_n=engine.CANDIDATE_POOL_SIZE,
            exclude_ids=profile.rated_recipe_ids,
        )

        # Stage 1b: ingredient TF-IDF
        ingr_cands = engine._ingr_embedder.get_candidates(
            recipe_ids=rated_ids,
            ratings=ratings,
            timestamps=timestamps,
            top_n=engine.CANDIDATE_POOL_SIZE,
            exclude_ids=profile.rated_recipe_ids,
            recency_half_life_days=float(engine.s.recency_half_life_days),
        )

        if not tag_cands and not ingr_cands:
            return [], 0, False, 0

        # Union: tag pool first, then ingredient-only additions
        tag_id_set = {rid for rid, _ in tag_cands}
        candidate_ids = [rid for rid, _ in tag_cands] + [
            rid for rid, _ in ingr_cands if rid not in tag_id_set
        ]

        pool_set = set(candidate_ids)
        pool_count = sum(1 for rid in case.held_out_ids if rid in pool_set)
        any_in_pool = pool_count > 0

        results = engine._stage2_knn(
            query_vec=profile.embedding,
            candidate_ids=candidate_ids,
            top_k=eval_window,
            tag_affinity=profile.tag_affinity,
        )
        return [r.recipe_id for r in results], len(candidate_ids), any_in_pool, pool_count

    if is_unified:
        if search_engine is None:
            raise RuntimeError("unified condition requires a SearchEngine")

        rated_ids  = [r.recipe_id for r in case.train_records]
        ratings    = [r.star_rating for r in case.train_records]
        timestamps = [r.timestamp for r in case.train_records]

        # Use tags_for_user_history directly -- bypasses the
        # synthesize_query_string -> parse_user_intent round-trip that
        # drops ~40% of canonical tags (e.g. 'main-dish', 'soups-stews')
        # because TAG_MAPPINGS has no reverse patterns for those forms.
        # search() detects no free-text query and routes to
        # retrieve_candidates_from_tags, keeping all user signal intact.
        user_tags = engine._candidates.tags_for_user_history(
            recipe_ids=rated_ids,
            ratings=ratings,
            timestamps=timestamps,
            top_n_tags=engine.USER_HISTORY_TOP_TAGS,
        )
        if not user_tags:
            return [], 0, False, 0

        result = search_engine.search(
            user_tags=user_tags,
            user_embedding=profile.embedding,
            user_tag_affinity=profile.tag_affinity,
            alpha=unified_alpha,
            affinity_weight=unified_affinity_weight,
            mode=SearchMode.HYBRID,
            top_k=eval_window,
            candidate_pool=UNIFIED_CANDIDATE_POOL,
            exclude_ids=profile.rated_recipe_ids,
        )
        rec_ids = [int(r.recipe_id) for r in result.results]
        candidate_id_set = {int(c["_id"]) for c in result.candidates}
        pool_count = sum(1 for rid in case.held_out_ids if rid in candidate_id_set)
        any_in_pool = pool_count > 0
        return rec_ids, len(result.candidates), any_in_pool, pool_count

    raise ValueError(f"Unknown condition: {condition}")


# ---------------------------------------------------------------------------
# evaluate_condition
# ---------------------------------------------------------------------------

def evaluate_condition(
    engine: RecommenderEngine,
    test_cases: List[TestCase],
    condition: str,
    ks: Tuple[int, ...] = DEFAULT_K_VALUES,
    eval_window: int = EVAL_WINDOW,
    *,
    search_engine: Optional[SearchEngine] = None,
    unified_alpha: float = UNIFIED_DEFAULT_ALPHA,
    unified_affinity_weight: float = UNIFIED_DEFAULT_AFFINITY_WEIGHT,
) -> ConditionResult:
    forward = engine._candidates._forward   # recipe_id -> frozenset of taxonomy tags
    per_case: List[PerCaseResult] = []

    for case in test_cases:
        try:
            rec_ids, pool_size, any_in_pool, pool_count = _run_condition_for_case(
                engine, case, condition, eval_window,
                search_engine=search_engine,
                unified_alpha=unified_alpha,
                unified_affinity_weight=unified_affinity_weight,
            )
        except Exception as e:
            print("  [WARN] {} user {}: {}".format(condition, case.user_id, e))
            rec_ids, pool_size, any_in_pool, pool_count = [], 0, None, 0

        # First-hit rank (backward compat for HR/NDCG/MRR)
        first_rank = find_first_hit_rank(rec_ids, case.held_out_ids) if rec_ids else None

        # Multi-holdout recall per k
        recall_k = {k: recall_at_k(rec_ids, case.held_out_ids, k) for k in ks}

        # Tag similarity per k
        held_tags = _held_out_tag_union(case.held_out_ids, forward)
        tag_sim_k     = {k: compute_tag_sim_at_k(rec_ids, held_tags, forward, k) for k in ks}
        dcg_tag_sim_k = {k: compute_dcg_tag_sim_at_k(rec_ids, held_tags, forward, k) for k in ks}
        tag_coverage_k = {k: compute_tag_coverage_at_k(rec_ids, held_tags, forward, k) for k in ks}

        per_case.append(PerCaseResult(
            user_id=case.user_id,
            held_out_ids=case.held_out_ids,
            recommended_ids=rec_ids,
            first_hit_rank=first_rank,
            recall_at_k=recall_k,
            tag_sim_at_k=tag_sim_k,
            dcg_tag_sim_at_k=dcg_tag_sim_k,
            tag_coverage_at_k=tag_coverage_k,
            pool_size=pool_size,
            held_out_in_pool=any_in_pool,
            held_out_pool_count=pool_count,
            n_train_reviews=len(case.train_records),
        ))

    n = len(per_case)
    if n == 0:
        empty = {k: 0.0 for k in ks}
        empty_ci = {k: (0.0, 0.0) for k in ks}
        return ConditionResult(
            condition=condition, n_users=0, coverage=0.0,
            hr_at_k=empty, hr_ci_at_k=empty_ci,
            ndcg_at_k=empty, ndcg_ci_at_k=empty_ci,
            mrr=0.0, mrr_ci=(0.0, 0.0),
            recall_at_k=empty, recall_ci_at_k=empty_ci,
            tag_sim_at_k=empty, tag_sim_ci_at_k=empty_ci,
            dcg_tag_sim_at_k=empty, dcg_tag_sim_ci_at_k=empty_ci,
            tag_coverage_at_k=empty, tag_coverage_ci_at_k=empty_ci,
            pool_recall=None, pool_recall_mean_count=None,
            rank_in_pool_when_present=None,
            per_case=per_case,
        )

    coverage = float(np.mean([1.0 if pc.recommended_ids else 0.0 for pc in per_case]))

    # HR / NDCG / MRR (first-hit semantics)
    hr_at_k: Dict[int, float] = {}
    hr_ci_at_k: Dict[int, Tuple[float, float]] = {}
    ndcg_at_k_: Dict[int, float] = {}
    ndcg_ci_at_k_: Dict[int, Tuple[float, float]] = {}
    for k in ks:
        hits  = [hit_from_rank(pc.first_hit_rank, k) for pc in per_case]
        ndcgs = [ndcg_from_rank(pc.first_hit_rank, k) for pc in per_case]
        hr_at_k[k]       = float(np.mean(hits))
        ndcg_at_k_[k]    = float(np.mean(ndcgs))
        hr_ci_at_k[k]    = bootstrap_ci(hits)
        ndcg_ci_at_k_[k] = bootstrap_ci(ndcgs)
    rr_values = [reciprocal_rank(pc.first_hit_rank) for pc in per_case]
    mrr    = float(np.mean(rr_values))
    mrr_ci = bootstrap_ci(rr_values)

    # Recall@k (multi-holdout)
    recall_at_k_: Dict[int, float] = {}
    recall_ci_at_k_: Dict[int, Tuple[float, float]] = {}
    for k in ks:
        vals = [pc.recall_at_k[k] for pc in per_case]
        recall_at_k_[k]    = float(np.mean(vals))
        recall_ci_at_k_[k] = bootstrap_ci(vals)

    # Tag sim@k
    tag_sim_at_k_: Dict[int, float] = {}
    tag_sim_ci_at_k_: Dict[int, Tuple[float, float]] = {}
    for k in ks:
        vals = [pc.tag_sim_at_k[k] for pc in per_case]
        tag_sim_at_k_[k]    = float(np.mean(vals))
        tag_sim_ci_at_k_[k] = bootstrap_ci(vals)

    # DCG-TagSim@k (rank-discounted)
    dcg_tag_sim_at_k_: Dict[int, float] = {}
    dcg_tag_sim_ci_at_k_: Dict[int, Tuple[float, float]] = {}
    for k in ks:
        vals = [pc.dcg_tag_sim_at_k[k] for pc in per_case]
        dcg_tag_sim_at_k_[k]    = float(np.mean(vals))
        dcg_tag_sim_ci_at_k_[k] = bootstrap_ci(vals)

    # Tag coverage@k
    tag_coverage_at_k_: Dict[int, float] = {}
    tag_coverage_ci_at_k_: Dict[int, Tuple[float, float]] = {}
    for k in ks:
        vals = [pc.tag_coverage_at_k[k] for pc in per_case]
        tag_coverage_at_k_[k]    = float(np.mean(vals))
        tag_coverage_ci_at_k_[k] = bootstrap_ci(vals)

    # Pool diagnostics
    pool_cases = [pc for pc in per_case if pc.held_out_in_pool is not None]
    if pool_cases:
        pool_recall_ = float(np.mean([1.0 if pc.held_out_in_pool else 0.0 for pc in pool_cases]))
        pool_count_mean = float(np.mean([pc.held_out_pool_count for pc in pool_cases]))
        in_pool_ranks = [pc.first_hit_rank for pc in pool_cases
                         if pc.held_out_in_pool and pc.first_hit_rank is not None]
        rank_in_pool = float(np.mean(in_pool_ranks)) if in_pool_ranks else None
    else:
        pool_recall_ = pool_count_mean = rank_in_pool = None

    return ConditionResult(
        condition=condition, n_users=n, coverage=coverage,
        hr_at_k=hr_at_k, hr_ci_at_k=hr_ci_at_k,
        ndcg_at_k=ndcg_at_k_, ndcg_ci_at_k=ndcg_ci_at_k_,
        mrr=mrr, mrr_ci=mrr_ci,
        recall_at_k=recall_at_k_, recall_ci_at_k=recall_ci_at_k_,
        tag_sim_at_k=tag_sim_at_k_, tag_sim_ci_at_k=tag_sim_ci_at_k_,
        dcg_tag_sim_at_k=dcg_tag_sim_at_k_, dcg_tag_sim_ci_at_k=dcg_tag_sim_ci_at_k_,
        tag_coverage_at_k=tag_coverage_at_k_, tag_coverage_ci_at_k=tag_coverage_ci_at_k_,
        pool_recall=pool_recall_, pool_recall_mean_count=pool_count_mean,
        rank_in_pool_when_present=rank_in_pool,
        per_case=per_case,
    )


# ---------------------------------------------------------------------------
# Full evaluation suite
# ---------------------------------------------------------------------------

def run_evaluation(
    settings=None,
    n_users: int = 200,
    min_reviews: int = 15,
    n_holdout: int = DEFAULT_N_HOLDOUT,
    ks: Tuple[int, ...] = DEFAULT_K_VALUES,
    eval_window: int = EVAL_WINDOW,
    seed: int = 42,
    unified_alphas: Tuple[float, ...] = (UNIFIED_DEFAULT_ALPHA,),
    unified_affinity_weight: float = UNIFIED_DEFAULT_AFFINITY_WEIGHT,
    affinity_weights: Optional[Tuple[float, ...]] = None,
    include_unified: bool = True,
) -> List[ConditionResult]:
    if settings is None:
        settings = load_settings()

    search_engine: Optional[SearchEngine] = None
    if include_unified:
        print("Connecting to Elasticsearch...")
        es_client = get_es_client(settings)
        search_engine = SearchEngine(settings, es_client)

    print("Loading recommender engine...")
    engine = RecommenderEngine(settings, search_engine=search_engine)

    print("Sampling {} LNO test cases (min_reviews={}, n_holdout={})...".format(
        n_users, min_reviews, n_holdout))
    t0 = time.time()
    test_cases = leave_n_out_sample(
        gold_reviews_path=settings.gold_reviews_path,
        bundle_ids=engine._bundle_id_set,
        min_reviews=min_reviews,
        n_holdout=n_holdout,
        n_users=n_users,
        seed=seed,
    )
    print("  Sampled {} valid test cases in {:.1f}s".format(len(test_cases), time.time() - t0))
    if not test_cases:
        print("No valid test cases found.")
        return []

    mean_held = float(np.mean([len(tc.held_out_ids) for tc in test_cases]))
    print("  Mean held-out set size: {:.1f} recipes".format(mean_held))

    conditions: List[str] = ["cold_start", "knn_only", "two_stage", "two_stage_ingr", "two_stage_hybrid"]
    if include_unified:
        if affinity_weights is not None:
            for aw in affinity_weights:
                conditions.append("unified_aw{:.2f}".format(aw))
        elif len(unified_alphas) == 1 and unified_alphas[0] == UNIFIED_DEFAULT_ALPHA:
            conditions.append("unified")
        else:
            for a in unified_alphas:
                conditions.append("unified_a{:.2f}".format(a))

    results: List[ConditionResult] = []
    for cond in conditions:
        print("Evaluating {}...".format(cond), flush=True)
        t0 = time.time()
        cr = evaluate_condition(
            engine, test_cases, cond,
            ks=ks, eval_window=eval_window,
            search_engine=search_engine,
            unified_alpha=UNIFIED_DEFAULT_ALPHA,
            unified_affinity_weight=unified_affinity_weight,
        )
        results.append(cr)
        print("  done in {:.1f}s".format(time.time() - t0))

    _print_summary(results, ks)
    _print_diagnostics(results)
    return results


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def _fmt(mean: float, ci: Tuple[float, float]) -> str:
    return "{:.4f} [{:.3f},{:.3f}]".format(mean, ci[0], ci[1])


def _print_summary(results: List[ConditionResult], ks: Tuple[int, ...]) -> None:
    W = 24
    print()
    print("=" * 80)
    print("Recall@k  (multi-holdout -- primary metric)  95% bootstrap CI")
    print("=" * 80)
    header = "{:<16}".format("Condition")
    for k in ks:
        header += " {:>{}}".format("Recall@{}".format(k), W)
    print(header)
    print("-" * 80)
    for r in results:
        best_k = max(ks, key=lambda k: r.recall_at_k[k])
        row = "{:<16}".format(r.condition)
        for k in ks:
            cell = _fmt(r.recall_at_k[k], r.recall_ci_at_k[k])
            if k == best_k and r.recall_at_k[k] > 0:
                cell = cell + " *"
            row += " {:>{}}".format(cell, W)
        print(row)

    print()
    print("=" * 80)
    print("Tag Similarity@k  (Jaccard vs held-out tags -- continuous quality signal)")
    print("=" * 80)
    header = "{:<16}".format("Condition")
    for k in ks:
        header += " {:>{}}".format("TagSim@{}".format(k), W)
    print(header)
    print("-" * 80)
    for r in results:
        row = "{:<16}".format(r.condition)
        for k in ks:
            row += " {:>{}}".format(_fmt(r.tag_sim_at_k[k], r.tag_sim_ci_at_k[k]), W)
        print(row)

    print()
    print("=" * 80)
    print("DCG-TagSim@k  (rank-discounted -- rewards tag-similar items ranked early)")
    print("=" * 80)
    header = "{:<16}".format("Condition")
    for k in ks:
        header += " {:>{}}".format("DCG-TS@{}".format(k), W)
    print(header)
    print("-" * 80)
    for r in results:
        row = "{:<16}".format(r.condition)
        for k in ks:
            row += " {:>{}}".format(_fmt(r.dcg_tag_sim_at_k[k], r.dcg_tag_sim_ci_at_k[k]), W)
        print(row)

    print()
    print("=" * 80)
    print("Tag Coverage@k  (fraction of held-out tags covered by top-k union)")
    print("=" * 80)
    header = "{:<16}".format("Condition")
    for k in ks:
        header += " {:>{}}".format("TagCov@{}".format(k), W)
    print(header)
    print("-" * 80)
    for r in results:
        row = "{:<16}".format(r.condition)
        for k in ks:
            row += " {:>{}}".format(_fmt(r.tag_coverage_at_k[k], r.tag_coverage_ci_at_k[k]), W)
        print(row)

    print()
    print("=" * 80)
    print("HR@k / NDCG@k / MRR  (first-hit semantics -- backward compat)")
    print("=" * 80)
    header = "{:<16}".format("Condition")
    for k in (5, 10, 20):
        header += " {:>{}}".format("HR@{}".format(k), W)
    header += " {:>{}}".format("MRR", W)
    print(header)
    print("-" * 80)
    for r in results:
        row = "{:<16}".format(r.condition)
        for k in (5, 10, 20):
            row += " {:>{}}".format(_fmt(r.hr_at_k[k], r.hr_ci_at_k[k]), W)
        row += " {:>{}}".format(_fmt(r.mrr, r.mrr_ci), W)
        print(row)


def _print_diagnostics(results: List[ConditionResult]) -> None:
    print()
    print("=" * 80)
    print("Stage 1 pool diagnostics")
    print("=" * 80)
    print("{:<16} {:>10} {:>14} {:>16} {:>18}".format(
        "Condition", "Coverage", "Pool recall", "Mean in-pool #", "Mean rank (in-pool)"
    ))
    print("-" * 80)
    for r in results:
        pr   = "{:.3f}".format(r.pool_recall) if r.pool_recall is not None else "n/a"
        pc   = "{:.2f}".format(r.pool_recall_mean_count) if r.pool_recall_mean_count is not None else "n/a"
        rk   = "{:.1f}".format(r.rank_in_pool_when_present) if r.rank_in_pool_when_present is not None else "n/a"
        print("{:<16} {:>10.3f} {:>14} {:>16} {:>18}".format(r.condition, r.coverage, pr, pc, rk))
    print()

    unified_conds = [r for r in results if r.condition.startswith("unified")]
    if unified_conds:
        print("Note: unified conditions use effective_alpha=0.0 (Stage 1 is tag-based, no free-text")
        print("query; alpha has no effect). affinity_weight is the active hyperparameter — use")
        print("--affinity-weights to sweep it (e.g. --affinity-weights 0.0,0.1,0.25,0.5,0.75,1.0).")
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Recommender Leave-N-Out evaluation")
    parser.add_argument("--n-users",     type=int, default=200)
    parser.add_argument("--min-reviews", type=int, default=15)
    parser.add_argument("--n-holdout",   type=int, default=DEFAULT_N_HOLDOUT,
                        help="Reviews held out per user (default {})".format(DEFAULT_N_HOLDOUT))
    parser.add_argument("--ks",          type=str, default="5,10,20,50")
    parser.add_argument("--eval-window", type=int, default=EVAL_WINDOW)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--unified-alphas", type=str,
                        default="{:.2f}".format(UNIFIED_DEFAULT_ALPHA))
    parser.add_argument("--unified-affinity-weight", type=float,
                        default=UNIFIED_DEFAULT_AFFINITY_WEIGHT)
    parser.add_argument("--affinity-weights", type=str, default=None,
                        help="Sweep affinity_weight for unified (e.g. 0.0,0.1,0.25,0.5,0.75,1.0). "
                             "Generates unified_aw* conditions. Takes precedence over --unified-alphas.")
    parser.add_argument("--no-unified", action="store_true")
    args = parser.parse_args()

    affinity_weights = None
    if args.affinity_weights:
        affinity_weights = tuple(float(x) for x in args.affinity_weights.split(",") if x.strip())

    run_evaluation(
        n_users=args.n_users,
        min_reviews=args.min_reviews,
        n_holdout=args.n_holdout,
        ks=tuple(int(x) for x in args.ks.split(",") if x.strip()),
        eval_window=args.eval_window,
        seed=args.seed,
        unified_alphas=tuple(float(x) for x in args.unified_alphas.split(",") if x.strip()),
        unified_affinity_weight=args.unified_affinity_weight,
        affinity_weights=affinity_weights,
        include_unified=not args.no_unified,
    )
