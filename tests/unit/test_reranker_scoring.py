# tests/unit/test_reranker_scoring.py
"""
Tests for SemanticReranker scoring logic: weight profiles, score_alignment,
and combine_scores.  Uses lightweight mock objects — no model weights or ES.
"""

import pytest
from search.reranker import SemanticReranker
from search.query_encoding import ProjectedQuery
import numpy as np


def _make_projected_query(
    proteins=None, dietary_tags=None, cuisines=None, courses=None,
    methods=None, occasions=None, taste=None, dish_type=None,
    target_minutes=None, max_minutes=None, clean_text="",
):
    """Helper to build a minimal ProjectedQuery for scoring tests."""
    return ProjectedQuery(
        raw_query=clean_text,
        lexical_query=clean_text,
        clean_text=clean_text,
        active_meta_features=[],
        active_tag_features=[],
        active_intensity_features=[],
        meta_vector=np.zeros(10, dtype=np.float32),
        tag_vector=np.zeros(17, dtype=np.float32),
        target_minutes=target_minutes,
        max_minutes=max_minutes,
        proteins=proteins or [],
        dietary_tags=dietary_tags or [],
        cuisines=cuisines or [],
        methods=methods or [],
        occasions=occasions or [],
        courses=courses or [],
        taste=taste or [],
        dish_type=dish_type or [],
    )


class TestWeightProfiles:
    def test_high_intent_three_signals(self):
        pq = _make_projected_query(
            proteins=["chicken"], cuisines=["italian"], courses=["main-dish"]
        )
        profile = SemanticReranker.get_weight_profile(pq)
        assert profile["tier"] == "high"
        assert profile["alignment"] > profile["semantic"]

    def test_medium_intent_one_signal(self):
        pq = _make_projected_query(proteins=["beef"])
        profile = SemanticReranker.get_weight_profile(pq)
        assert profile["tier"] == "medium"

    def test_low_intent_no_signals(self):
        pq = _make_projected_query()
        profile = SemanticReranker.get_weight_profile(pq)
        assert profile["tier"] == "low"
        assert profile["semantic"] > profile["alignment"]

    def test_all_weights_non_negative(self):
        for signals in [0, 1, 3]:
            pq = _make_projected_query(
                proteins=["chicken"] * min(signals, 1),
                cuisines=["italian"] * min(signals - 1, 1) if signals > 1 else [],
                courses=["main-dish"] * min(signals - 2, 1) if signals > 2 else [],
            )
            profile = SemanticReranker.get_weight_profile(pq)
            for key in ("lex", "alignment", "semantic", "quality"):
                assert profile[key] >= 0.0, f"Negative weight for {key} at {signals} signals"


class TestScoreAlignment:
    """
    score_alignment is a static-ish method that only needs a ProjectedQuery
    and a candidate source dict — no model required.
    """

    def _score(self, pq, source):
        reranker = object.__new__(SemanticReranker)
        return reranker.score_alignment(pq, source)

    def test_cuisine_match_in_tags(self):
        pq = _make_projected_query(cuisines=["italian"])
        source = {"tags_clean": ["italian", "dinner"], "name": "pasta", "ingredients_clean": "", "minutes": None}
        score = self._score(pq, source)
        assert score > 0.0

    def test_cuisine_match_in_name(self):
        pq = _make_projected_query(cuisines=["mexican"])
        source = {"tags_clean": [], "name": "mexican chicken bowl", "ingredients_clean": "", "minutes": None}
        score = self._score(pq, source)
        assert score > 0.0

    def test_protein_match_in_ingredients(self):
        pq = _make_projected_query(proteins=["salmon"], clean_text="salmon")
        source = {"tags_clean": [], "name": "baked fish", "ingredients_clean": "salmon lemon butter", "minutes": None}
        score = self._score(pq, source)
        assert score > 0.0

    def test_time_proximity_exact(self):
        pq = _make_projected_query(target_minutes=30)
        source = {"tags_clean": [], "name": "dish", "ingredients_clean": "", "minutes": 30}
        score = self._score(pq, source)
        assert score > 0.0

    def test_leftover_token_penalty_when_no_match(self):
        pq = _make_projected_query(clean_text="quinoa")
        source = {"tags_clean": [], "name": "pasta carbonara", "ingredients_clean": "pasta eggs bacon", "minutes": None}
        score = self._score(pq, source)
        assert score <= 0.0

    def test_no_crash_on_none_tags(self):
        pq = _make_projected_query(cuisines=["italian"])
        source = {"tags_clean": None, "name": "pasta", "ingredients_clean": "", "minutes": None}
        score = self._score(pq, source)
        assert isinstance(score, float)


class TestCombineScores:
    def _combine(self, base=1.0, align=1.0, sem=0.5, quality=4.0, weights=None):
        reranker = object.__new__(SemanticReranker)
        w = weights or {"lex": 1.0, "alignment": 1.0, "semantic": 1.0, "quality": 0.25}
        return reranker.combine_scores(base, align, sem, quality, w)

    def test_returns_float(self):
        assert isinstance(self._combine(), float)

    def test_zero_weights_returns_zero(self):
        result = self._combine(
            base=5.0, align=5.0, sem=5.0, quality=5.0,
            weights={"lex": 0.0, "alignment": 0.0, "semantic": 0.0, "quality": 0.0},
        )
        assert result == 0.0

    def test_negative_semantic_clamped_to_zero(self):
        w = {"lex": 0.0, "alignment": 0.0, "semantic": 1.0, "quality": 0.0}
        result = self._combine(sem=-0.5, weights=w)
        assert result == 0.0

    def test_higher_quality_increases_score(self):
        w = {"lex": 0.0, "alignment": 0.0, "semantic": 0.0, "quality": 1.0}
        low = self._combine(quality=2.0, weights=w)
        high = self._combine(quality=5.0, weights=w)
        assert high > low
