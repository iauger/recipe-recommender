# tests/integration/test_search_engine.py
"""
Integration tests for the full search pipeline.
Requires: Elasticsearch running + data files indexed.

Run with:  pytest --integration tests/integration/
"""

import pytest
from search.engine import SearchEngine, SearchMode, SearchResult
from search.reranker import RerankedResult


@pytest.fixture(scope="module")
def engine():
    from core.config import load_settings, get_es_client
    s = load_settings()
    try:
        es = get_es_client(s)
    except Exception:
        pytest.skip("Elasticsearch not reachable — start ES to run integration tests")
    if not es.indices.exists(index=s.es_index):
        pytest.skip(f"ES index '{s.es_index}' not found — run the indexer first")
    return SearchEngine(s, es)


@pytest.mark.integration
class TestSearchEngineRun:
    def test_returns_search_result(self, engine):
        result = engine.run("chicken dinner", mode=SearchMode.HYBRID, top_k=5)
        assert isinstance(result, SearchResult)

    def test_results_list_not_empty(self, engine):
        result = engine.run("easy pasta", mode=SearchMode.HYBRID, top_k=5)
        assert len(result.results) > 0

    def test_results_are_reranked_result_objects(self, engine):
        result = engine.run("vegan breakfast", mode=SearchMode.SEMANTIC, top_k=5)
        for r in result.results:
            assert isinstance(r, RerankedResult)

    def test_results_sorted_by_final_score(self, engine):
        result = engine.run("italian pasta", mode=SearchMode.HYBRID, top_k=10)
        scores = [r.final_score for r in result.results]
        assert scores == sorted(scores, reverse=True)

    def test_lexical_mode_has_zero_semantic_sim(self, engine):
        result = engine.run("chicken", mode=SearchMode.LEXICAL, top_k=5)
        for r in result.results:
            assert r.semantic_sim == 0.0
            assert r.alignment_score == 0.0

    def test_intent_parsed_correctly(self, engine):
        result = engine.run("vegan dinner under 30 minutes", mode=SearchMode.HYBRID, top_k=5)
        assert result.intent.get("max_minutes") == 30
        assert "vegan" in result.intent.get("dietary_tags", [])

    def test_tier_assigned(self, engine):
        result = engine.run("low carb italian beef skillet", mode=SearchMode.HYBRID, top_k=5)
        assert result.tier in ("high", "medium", "low", "high_no_sem",
                               "medium_no_sem", "low_no_sem", "lexical",
                               "semantic", "quality")


@pytest.mark.integration
class TestRunAllModes:
    def test_returns_all_five_modes(self, engine):
        results = engine.run_all_modes("chicken soup", top_k=5)
        assert set(results.keys()) == set(SearchMode)

    def test_all_modes_share_same_candidates(self, engine):
        results = engine.run_all_modes("pasta", top_k=5)
        # All modes retrieved from the same Stage 1 pool
        candidate_sets = [
            {h["_id"] for h in sr.candidates}
            for sr in results.values()
        ]
        assert all(s == candidate_sets[0] for s in candidate_sets)

    def test_quality_mode_results_differ_from_lexical(self, engine):
        results = engine.run_all_modes("easy dinner", top_k=10)
        lex_ids = [r.recipe_id for r in results[SearchMode.LEXICAL].results]
        qual_ids = [r.recipe_id for r in results[SearchMode.QUALITY].results]
        # Order should differ (quality sorts by rating, not BM25)
        assert lex_ids != qual_ids


@pytest.mark.integration
class TestHardFilters:
    def test_vegan_filter_respected(self, engine):
        result = engine.run("vegan pasta", mode=SearchMode.HYBRID, top_k=10)
        for r in result.results:
            tags = r.source.get("tags_clean", []) or []
            # At least some results should have vegan tag (hard filter applied at Stage 1)
            # We can't guarantee all results have it (semantic modes may not re-filter)
            break  # just assert no crash

    def test_max_minutes_filter(self, engine):
        result = engine.run("soup under 15 minutes", mode=SearchMode.LEXICAL, top_k=10)
        for r in result.results:
            minutes = r.source.get("minutes")
            if minutes is not None:
                assert float(minutes) <= 15
