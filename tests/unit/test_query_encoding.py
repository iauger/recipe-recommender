# tests/unit/test_query_encoding.py
"""
Tests for search/query_encoding.py: QueryFeatureProjector.
Uses the col_map_file fixture from conftest — no ES or data files required.
"""

import numpy as np
import pytest
from search.query_encoding import QueryFeatureProjector
from search.search import parse_user_intent


@pytest.fixture
def projector(mock_settings):
    return QueryFeatureProjector(mock_settings)


class TestProjectorInit:
    def test_loads_col_map(self, projector):
        assert len(projector.col_map) > 0

    def test_meta_features_excludes_pred(self, projector):
        for f in projector.meta_features:
            assert not f.startswith("pred_")

    def test_tag_features_only_pred(self, projector):
        for f in projector.tag_features:
            assert f.startswith("pred_")


class TestProjection:
    def test_returns_projected_query(self, projector):
        from search.query_encoding import ProjectedQuery
        intent = parse_user_intent("chicken dinner")
        result = projector.project("chicken dinner", intent)
        assert isinstance(result, ProjectedQuery)

    def test_meta_vector_shape(self, projector):
        intent = parse_user_intent("chicken dinner")
        result = projector.project("chicken dinner", intent)
        assert result.meta_vector.shape == (len(projector.meta_features),)

    def test_tag_vector_shape(self, projector):
        intent = parse_user_intent("chicken dinner")
        result = projector.project("chicken dinner", intent)
        assert result.tag_vector.shape == (len(projector.tag_features),)

    def test_known_ingredient_activated(self, projector):
        intent = parse_user_intent("chicken onion")
        result = projector.project("chicken onion", intent)
        ing_chicken_idx = projector.meta_index.get("ing_chicken")
        if ing_chicken_idx is not None:
            assert result.meta_vector[ing_chicken_idx] == 1.0

    def test_unknown_token_does_not_crash(self, projector):
        intent = parse_user_intent("xyzunknowntoken")
        result = projector.project("xyzunknowntoken", intent)
        assert result.meta_vector.sum() == 0.0

    def test_intent_fields_populated(self, projector):
        intent = parse_user_intent("vegan italian dinner under 30 minutes")
        result = projector.project("vegan italian dinner under 30 minutes", intent)
        assert "vegan" in result.dietary_tags
        assert "italian" in result.cuisines
        assert result.max_minutes == 30

    def test_proteins_passed_through(self, projector):
        intent = parse_user_intent("grilled chicken with garlic")
        result = projector.project("grilled chicken with garlic", intent)
        assert "chicken" in result.proteins
