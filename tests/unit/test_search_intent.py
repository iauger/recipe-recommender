# tests/unit/test_search_intent.py
"""
Tests for search/search.py: query intent parsing, filter building, boost application.
No Elasticsearch connection required.
"""

import pytest
from search.search import (
    parse_user_intent,
    build_candidate_query,
    IntentKey,
)


class TestTimeConstraints:
    def test_under_minutes(self):
        intent = parse_user_intent("chicken dinner under 30 minutes")
        assert intent[IntentKey.MAX_MINUTES.value] == 30
        assert intent[IntentKey.TARGET_MINUTES.value] is None

    def test_less_than_minutes(self):
        intent = parse_user_intent("pasta less than 45 mins")
        assert intent[IntentKey.MAX_MINUTES.value] == 45

    def test_max_minutes(self):
        intent = parse_user_intent("soup max 20 min")
        assert intent[IntentKey.MAX_MINUTES.value] == 20

    def test_around_minutes(self):
        intent = parse_user_intent("something around 30 minutes")
        assert intent[IntentKey.TARGET_MINUTES.value] == 30
        assert intent[IntentKey.MAX_MINUTES.value] is None

    def test_no_time_constraint(self):
        intent = parse_user_intent("easy chicken soup")
        assert intent[IntentKey.MAX_MINUTES.value] is None
        assert intent[IntentKey.TARGET_MINUTES.value] is None

    def test_time_phrase_removed_from_clean_text(self):
        intent = parse_user_intent("chicken under 30 minutes")
        assert "30" not in intent[IntentKey.CLEAN_TEXT.value]
        assert "under" not in intent[IntentKey.CLEAN_TEXT.value]


class TestProteinExtraction:
    def test_single_protein(self):
        intent = parse_user_intent("grilled chicken with vegetables")
        assert "chicken" in intent[IntentKey.PROTEINS.value]

    def test_multiple_proteins(self):
        intent = parse_user_intent("beef and shrimp stir fry")
        assert "beef" in intent[IntentKey.PROTEINS.value]
        assert "shrimp" in intent[IntentKey.PROTEINS.value]

    def test_no_protein(self):
        intent = parse_user_intent("chocolate cake with frosting")
        assert intent[IntentKey.PROTEINS.value] == []

    def test_protein_case_insensitive(self):
        intent = parse_user_intent("CHICKEN tikka masala")
        assert "chicken" in intent[IntentKey.PROTEINS.value]


class TestTagExtraction:
    def test_dietary_vegan(self):
        intent = parse_user_intent("vegan pasta salad")
        assert "vegan" in intent[IntentKey.DIETARY_TAGS.value]

    def test_dietary_gluten_free(self):
        intent = parse_user_intent("gluten-free bread recipe")
        assert "gluten-free" in intent[IntentKey.DIETARY_TAGS.value]

    def test_keto_maps_to_low_carb(self):
        intent = parse_user_intent("keto dinner ideas")
        assert "low-carb" in intent[IntentKey.DIETARY_TAGS.value]

    def test_course_dinner(self):
        intent = parse_user_intent("easy dinner for weeknights")
        assert "main-dish" in intent[IntentKey.COURSES.value]

    def test_cuisine_italian(self):
        intent = parse_user_intent("classic italian pasta")
        assert "italian" in intent[IntentKey.CUISINES.value]

    def test_method_slow_cooker(self):
        intent = parse_user_intent("slow cooker beef stew")
        assert "crock-pot-slow-cooker" in intent[IntentKey.METHODS.value]

    def test_occasion_thanksgiving(self):
        intent = parse_user_intent("thanksgiving turkey recipe")
        assert "thanksgiving" in intent[IntentKey.OCCASIONS.value]

    def test_no_tags_on_plain_query(self):
        intent = parse_user_intent("something delicious")
        assert intent[IntentKey.DIETARY_TAGS.value] == []
        assert intent[IntentKey.COURSES.value] == []
        assert intent[IntentKey.CUISINES.value] == []


class TestQueryBuilder:
    def test_must_clause_always_present(self):
        intent = parse_user_intent("chicken soup")
        query = build_candidate_query(intent)
        top = query.get("bool") or query.get("function_score", {}).get("query", {}).get("bool")
        assert top is not None
        assert len(top["must"]) >= 1

    def test_hard_filter_applied_for_max_minutes(self):
        intent = parse_user_intent("soup under 20 minutes")
        query = build_candidate_query(intent)
        bool_q = query.get("bool") or query["function_score"]["query"]["bool"]
        filter_keys = [list(f.keys())[0] for f in bool_q["filter"]]
        assert "range" in filter_keys

    def test_function_score_wraps_query_when_target_minutes_set(self):
        intent = parse_user_intent("pasta around 30 minutes")
        query = build_candidate_query(intent)
        assert "function_score" in query

    def test_no_function_score_without_target_minutes(self):
        intent = parse_user_intent("easy pasta")
        query = build_candidate_query(intent)
        assert "bool" in query
        assert "function_score" not in query

    def test_protein_boost_in_should(self):
        intent = parse_user_intent("grilled salmon")
        query = build_candidate_query(intent)
        bool_q = query.get("bool") or query["function_score"]["query"]["bool"]
        should_queries = [list(s.keys())[0] for s in bool_q["should"]]
        assert "match" in should_queries
