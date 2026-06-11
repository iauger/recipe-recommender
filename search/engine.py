"""Search engine orchestration: routes queries through ES retrieval and SemanticReranker."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

import copy
import sys
import io
if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import torch

from search.query_encoding import QueryFeatureProjector
from search.reranker import RerankedResult, SemanticReranker
from search.search import (
    retrieve_candidates,
    retrieve_candidates_from_tags,
    retrieve_candidates_personalized,
)


class SearchMode(Enum):
    LEXICAL         = "lexical"
    SEMANTIC        = "semantic"
    QUALITY         = "quality"
    ABLATION_NO_SEM = "ablation_no_sem"
    HYBRID          = "hybrid"


_FIXED_WEIGHTS: Dict[SearchMode, Dict[str, Any]] = {
    SearchMode.LEXICAL: {
        "tier": "lexical",
        "lex": 1.0,
        "alignment": 0.0,
        "semantic": 0.0,
        "quality": 0.0,
    },
    SearchMode.SEMANTIC: {
        "tier": "semantic",
        "lex": 0.0,
        "alignment": 0.0,
        "semantic": 1.0,
        "quality": 0.5,
    },
    SearchMode.QUALITY: {
        "tier": "quality",
        "lex": 0.0,
        "alignment": 0.0,
        "semantic": 0.0,
        "quality": 1.0,
    },
}

_STRATIFIED_MODES       = {SearchMode.HYBRID, SearchMode.ABLATION_NO_SEM}
_ABLATION_SEMANTIC_ZERO = {SearchMode.ABLATION_NO_SEM}  # stratified weights, semantic zeroed


@dataclass
class SearchResult:
    """Result bundle for a single query/mode execution. Personalization fields default to None."""
    query: str
    mode: SearchMode
    tier: str                           # intent tier label (or mode name for fixed modes)
    weights: Dict[str, Any]             # weights actually applied
    intent: Dict[str, Any]              # parsed query intent
    candidates: List[Dict[str, Any]]    # raw Stage 1 ES hits
    results: List[RerankedResult]       # reranked / ordered results
    query_embedding: Optional[Any] = field(default=None, repr=False)

    # Personalization metadata -- None unless run_personalized() set them.
    is_personalized: bool = False
    alpha: Optional[float] = None
    affinity_weight: Optional[float] = None
    synthesized_query: Optional[str] = None


class SearchEngine:
    """Unified search interface over all five search modes."""

    def __init__(self, settings: Any, es_client: Any, debug: bool = False):
        self.s = settings
        self.es_client = es_client
        self.projector = QueryFeatureProjector(settings)
        self.reranker  = SemanticReranker(settings, debug=debug)

    def run(
        self,
        query: str,
        mode: SearchMode = SearchMode.HYBRID,
        top_k: int = 10,
        return_query_embedding: bool = False,
    ) -> SearchResult:
        """
        Execute a non-personalized search query.

        return_query_embedding attaches the 128-D embedding for visualisation;
        skip it in evaluation loops to avoid the extra forward pass.
        """
        candidates, intent = retrieve_candidates(self.es_client, self.s.es_index, query, top_k=top_k)
        projected_query = self.projector.project(query, intent)
        weights, tier = self._resolve_weights(projected_query, mode)

        if mode == SearchMode.LEXICAL:
            results = self._wrap_lexical_results(candidates)
        else:
            results = self.reranker.rerank(
                projected_query=projected_query,
                candidates=candidates,
                mode_weights=weights,
            )

        query_embedding = None
        if return_query_embedding and mode != SearchMode.LEXICAL:
            query_embedding = self.reranker.encode_query(projected_query)

        return SearchResult(
            query=query,
            mode=mode,
            tier=tier,
            weights=weights,
            intent=intent,
            candidates=candidates,
            results=results,
            query_embedding=query_embedding,
        )

    def run_all_modes(
        self,
        query: str,
        top_k: int = 10,
    ) -> Dict[SearchMode, SearchResult]:
        """Run all five modes on one query, sharing Stage 1 candidates for a fair comparison."""
        # Retrieve once; deep-copy per mode to prevent in-place mutation.
        candidates, intent = retrieve_candidates(self.es_client, self.s.es_index, query, top_k=top_k)
        projected_query = self.projector.project(query, intent)

        results: Dict[SearchMode, SearchResult] = {}

        for mode in SearchMode:
            weights, tier = self._resolve_weights(projected_query, mode)
            mode_candidates = copy.deepcopy(candidates)

            if mode == SearchMode.LEXICAL:
                mode_results = self._wrap_lexical_results(mode_candidates)
            else:
                mode_results = self.reranker.rerank(
                    projected_query=projected_query,
                    candidates=mode_candidates,
                    mode_weights=weights,
                )

            results[mode] = SearchResult(
                query=query,
                mode=mode,
                tier=tier,
                weights=weights,
                intent=intent,
                candidates=candidates,
                results=mode_results,
            )

        return results

    def run_personalized(
        self,
        query: Optional[str] = None,
        user_embedding: Optional[torch.Tensor] = None,
        user_tag_affinity: Optional[np.ndarray] = None,
        user_tags: Optional[Set[str]] = None,
        alpha: float = 0.5,
        affinity_weight: float = 0.25,
        mode: SearchMode = SearchMode.HYBRID,
        top_k: int = 10,
        candidate_pool: int = 100,
        exclude_ids: Optional[Set[int]] = None,
    ) -> SearchResult:
        """
        Personalized search over all four (query × user) input combinations.

        alpha=1.0 → pure query; alpha=0.0 → pure user taste centroid.
        exclude_ids filters out already-rated recipes from the final list.
        """
        effective_query = query if query is not None else ""
        synthesized = None if effective_query else ""

        # Tag-direct path when no free-text query: bypasses parse_user_intent
        # (which only maps human phrases → canonical tags, not the reverse).
        # Without a query to blend, alpha is forced to 0.0.
        if user_tags is not None and not effective_query:
            candidates, intent = retrieve_candidates_from_tags(
                self.es_client, self.s.es_index, user_tags, top_k=candidate_pool
            )
            effective_alpha = 0.0  # no query → no query embedding to blend
        else:
            candidates, intent = retrieve_candidates_personalized(
                self.es_client, self.s.es_index, effective_query, top_k=candidate_pool
            )
            effective_alpha = alpha

        projected_query = self.projector.project(effective_query, intent)
        weights, tier = self._resolve_weights(projected_query, mode)

        # Scale query-dependent terms by alpha so the slider is a true blend:
        # alpha=0 → lex=align=0 (pure user centroid); alpha=1 → pure query.
        # Without this, BM25 dominates even at alpha=0.
        if user_embedding is not None and effective_alpha < 1.0 and mode != SearchMode.LEXICAL:
            weights = dict(weights)
            weights["lex"]       = weights.get("lex", 0.0)       * effective_alpha
            weights["alignment"] = weights.get("alignment", 0.0) * effective_alpha

        if mode == SearchMode.LEXICAL:
            results = self._wrap_lexical_results(candidates)
        else:
            results = self.reranker.rerank(
                projected_query=projected_query,
                candidates=candidates,
                mode_weights=weights,
                user_embedding=user_embedding,
                user_tag_affinity=user_tag_affinity,
                alpha=effective_alpha,
                affinity_weight=affinity_weight,
            )

        if exclude_ids:
            excluded_str = {str(x) for x in exclude_ids}
            results = [r for r in results if r.recipe_id not in excluded_str]

        results = results[:top_k]

        return SearchResult(
            query=effective_query,
            mode=mode,
            tier=tier,
            weights=weights,
            intent=intent,
            candidates=candidates,
            results=results,
            is_personalized=(user_embedding is not None or user_tag_affinity is not None),
            alpha=alpha,
            affinity_weight=affinity_weight,
            synthesized_query=synthesized,
        )

    def search(
        self,
        query: Optional[str] = None,
        user_embedding: Optional[torch.Tensor] = None,
        user_tag_affinity: Optional[np.ndarray] = None,
        user_tags: Optional[Set[str]] = None,
        alpha: float = 0.5,
        affinity_weight: float = 0.25,
        mode: SearchMode = SearchMode.HYBRID,
        top_k: int = 10,
        candidate_pool: int = 500,
        exclude_ids: Optional[Set[int]] = None,
    ) -> SearchResult:
        """
        Primary entry point. Routes all four (query × user) combinations:

          query + user_tags  → personalized search
          query only         → standard search (alpha forced to 1.0)
          user_tags only     → tag retrieval, user embedding ranking
          neither            → cold-start quality ranking
        """
        has_query = bool(query and query.strip())
        has_user  = bool(user_tags)

        if not has_query and not has_user:
            return self.run("", mode=SearchMode.QUALITY, top_k=top_k)  # cold start

        return self.run_personalized(
            query=query if has_query else None,
            user_embedding=user_embedding,
            user_tag_affinity=user_tag_affinity,
            user_tags=user_tags if has_user else None,
            alpha=alpha,
            affinity_weight=affinity_weight,
            mode=mode,
            top_k=top_k,
            candidate_pool=candidate_pool,
            exclude_ids=exclude_ids,
        )

    def _resolve_weights(
        self,
        projected_query: Any,
        mode: SearchMode,
    ) -> tuple[Dict[str, Any], str]:
        if mode in _STRATIFIED_MODES:
            weights = dict(self.reranker.get_weight_profile(projected_query))
            tier = weights["tier"]
            if mode in _ABLATION_SEMANTIC_ZERO:
                weights["semantic"] = 0.0
                weights["tier"] = f"{tier}_no_sem"
            return weights, weights["tier"]
        weights = dict(_FIXED_WEIGHTS[mode])
        return weights, weights["tier"]

    @staticmethod
    def _wrap_lexical_results(
        candidates: List[Dict[str, Any]],
    ) -> List[RerankedResult]:
        return [
            RerankedResult(
                recipe_id=str(hit["_id"]),
                base_score=float(hit.get("_score", 0.0)),
                alignment_score=0.0,
                semantic_sim=0.0,
                quality_score=0.0,
                final_score=float(hit.get("_score", 0.0)),
                source=hit["_source"],
            )
            for hit in candidates
        ]


