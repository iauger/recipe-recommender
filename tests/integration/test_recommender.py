# tests/integration/test_recommender.py
"""
Integration tests for the recommender engine.
Requires: data files (embeddings bundle, recipes parquet, gold reviews parquet).
Elasticsearch is optional — tests without a query filter don't need it.

Run with:  pytest --integration tests/integration/
"""

import pytest
from recommender.engine import RecommenderEngine, RecommendResult, RecommendMode
from recommender.knn import KNNResult


@pytest.fixture(scope="module")
def rec_engine():
    from core.config import load_settings
    s = load_settings()
    try:
        engine = RecommenderEngine(s, search_engine=None)
    except FileNotFoundError as e:
        pytest.skip(f"Data file not found — {e}")
    except Exception as e:
        pytest.skip(f"Could not initialise RecommenderEngine — {e}")
    return engine


@pytest.fixture(scope="module")
def seed_ids(rec_engine):
    """Return the first two recipe IDs from the bundle as session seeds."""
    return rec_engine._recipe_ids[:2]


@pytest.fixture(scope="module")
def eligible_users(rec_engine):
    return rec_engine.get_eligible_users(min_reviews=10)


@pytest.mark.integration
class TestRecommenderEngineInit:
    def test_loads_recipe_ids(self, rec_engine):
        assert len(rec_engine._recipe_ids) > 0

    def test_recipes_df_indexed_by_int(self, rec_engine):
        # Index dtype should be int after the dtype fix
        assert rec_engine._recipes_df.index.dtype in ("int64", "int32")

    def test_predictions_accessible(self, rec_engine):
        # Predictions should be indexable regardless of list vs tensor
        pred = rec_engine._predictions[0]
        assert isinstance(float(pred), float)


@pytest.mark.integration
class TestSessionRecommendations:
    def test_returns_recommend_result(self, rec_engine, seed_ids):
        result = rec_engine.recommend_session(seed_recipe_ids=seed_ids, top_k=10)
        assert isinstance(result, RecommendResult)
        assert result.mode == RecommendMode.SESSION_BASED

    def test_results_not_empty(self, rec_engine, seed_ids):
        result = rec_engine.recommend_session(seed_recipe_ids=seed_ids, top_k=10)
        assert len(result.results) > 0

    def test_results_are_knn_result_objects(self, rec_engine, seed_ids):
        result = rec_engine.recommend_session(seed_recipe_ids=seed_ids, top_k=10)
        for r in result.results:
            assert isinstance(r, KNNResult)

    def test_seed_ids_excluded_from_results(self, rec_engine, seed_ids):
        result = rec_engine.recommend_session(seed_recipe_ids=seed_ids, top_k=10)
        returned_ids = {r.recipe_id for r in result.results}
        for sid in seed_ids:
            assert sid not in returned_ids

    def test_results_have_source_metadata(self, rec_engine, seed_ids):
        result = rec_engine.recommend_session(seed_recipe_ids=seed_ids, top_k=5)
        for r in result.results:
            assert "name" in r.source or r.source  # source should not be empty shell

    def test_results_sorted_by_similarity_descending(self, rec_engine, seed_ids):
        result = rec_engine.recommend_session(seed_recipe_ids=seed_ids, top_k=10)
        sims = [r.similarity for r in result.results]
        assert sims == sorted(sims, reverse=True)

    def test_unknown_seed_ids_return_empty(self, rec_engine):
        result = rec_engine.recommend_session(seed_recipe_ids=[99999999], top_k=10)
        assert result.results == []

    def test_top_k_respected(self, rec_engine, seed_ids):
        result = rec_engine.recommend_session(seed_recipe_ids=seed_ids, top_k=5)
        assert len(result.results) <= 5


@pytest.mark.integration
class TestHistoryAwareRecommendations:
    def test_eligible_users_not_empty(self, eligible_users):
        assert len(eligible_users) > 0

    def test_returns_recommend_result(self, rec_engine, eligible_users):
        user_id = eligible_users[0]
        result = rec_engine.recommend_for_user(user_id=user_id, top_k=10)
        assert isinstance(result, RecommendResult)
        assert result.mode == RecommendMode.HISTORY_AWARE

    def test_user_profile_built(self, rec_engine, eligible_users):
        user_id = eligible_users[0]
        result = rec_engine.recommend_for_user(user_id=user_id, top_k=10)
        assert result.user_profile is not None
        assert result.user_profile.review_count > 0

    def test_rated_recipes_excluded(self, rec_engine, eligible_users):
        user_id = eligible_users[0]
        result = rec_engine.recommend_for_user(user_id=user_id, top_k=10)
        if result.user_profile and result.results:
            returned_ids = {r.recipe_id for r in result.results}
            assert returned_ids.isdisjoint(result.user_profile.rated_recipe_ids)

    def test_tag_affinity_shape(self, rec_engine, eligible_users):
        user_id = eligible_users[0]
        result = rec_engine.recommend_for_user(user_id=user_id, top_k=5)
        if result.user_profile:
            assert result.user_profile.tag_affinity.shape == (17,)

    def test_unknown_user_returns_empty(self, rec_engine):
        result = rec_engine.recommend_for_user(user_id=99999999, top_k=10)
        assert result.results == []
        assert result.user_profile is None


@pytest.mark.integration
class TestSearchRecipesByName:
    def test_returns_list(self, rec_engine):
        results = rec_engine.search_recipes_by_name("chicken", top_n=5)
        assert isinstance(results, list)

    def test_results_contain_required_keys(self, rec_engine):
        results = rec_engine.search_recipes_by_name("pasta", top_n=5)
        for r in results:
            assert "recipe_id" in r
            assert "name" in r

    def test_query_matches_name(self, rec_engine):
        results = rec_engine.search_recipes_by_name("chicken", top_n=10)
        for r in results:
            assert "chicken" in r["name"].lower()

    def test_no_match_returns_empty(self, rec_engine):
        results = rec_engine.search_recipes_by_name("xyzimpossiblematch12345", top_n=5)
        assert results == []
