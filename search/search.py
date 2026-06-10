"""
Query parsing and Elasticsearch query builders for recipe retrieval.

Heavily reliant on the dataset's taxonomy tags for filtering and boosting — this is
intentional; the tags are comprehensive and more reliable than free-text NER for this
domain, though it's not a generalizable approach.
"""

import re
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Set, Tuple


class SearchField(str, Enum):
    NAME = "name^2"
    INGREDIENTS = "ingredients_clean"
    DESCRIPTION = "description_clean"
    TAGS = "tags_clean"
    MINUTES = "minutes"


class BoostValue(float, Enum):
    MULTI_MATCH = 0.5
    PROTEIN_MATCH = 2.0
    TAG_MATCH = 1.5
    TASTE_MATCH = 0.5


class RetrievalConfig(int, Enum):
    DEFAULT_TOP_K = 10
    TARGET_MINUTES_OFFSET = 5
    TARGET_MINUTES_SCALE = 15


class IntentKey(str, Enum):
    RAW_QUERY = "raw_query"
    LEXICAL_QUERY = "lexical_query"
    CLEAN_TEXT = "clean_text"
    MAX_MINUTES = "max_minutes"
    TARGET_MINUTES = "target_minutes"
    PROTEINS = "proteins"
    DIETARY_TAGS = "dietary_tags"
    COURSES = "courses"
    CUISINES = "cuisines"
    METHODS = "methods"
    OCCASIONS = "occasions"
    TASTE = "taste"
    DISH_TYPE = "dish_type"


PROTEIN_KEYWORDS = [
    "chicken",
    "beef",
    "pork",
    "tofu",
    "fish",
    "turkey",
    "lamb",
    "duck",
    "seafood",
    "shrimp",
    "salmon",
    "tuna",
    "eggs",
]

# Maps common user phrases to canonical dataset tags. Unmapped phrases stay in the lexical query.
TAG_MAPPINGS: Dict[str, Dict[str, str]] = {
    IntentKey.DIETARY_TAGS.value: {
        r"\bvegan\b": "vegan",
        r"\bvegetarian\b": "vegetarian",
        r"\bmeatless\b": "vegetarian",
        r"\bplant[-\s]?based\b": "vegan",
        r"\bgluten[-\s]?free\b": "gluten-free",
        r"\bdairy[-\s]?free\b": "dairy-free",
        r"\blactose[-\s]?free\b": "dairy-free",
        r"\begg[-\s]?free\b": "egg-free",
        r"\bnut[-\s]?free\b": "nut-free",
        r"\blow[-\s]?carb\b": "low-carb",
        r"\bvery[-\s]?low[-\s]?carb\b": "very-low-carbs",
        r"\blow[-\s]?calorie\b": "low-calorie",
        r"\blow[-\s]?cal\b": "low-calorie",
        r"\blow[-\s]?fat\b": "low-fat",
        r"\blow[-\s]?saturated[-\s]?fat\b": "low-saturated-fat",
        r"\blow[-\s]?sodium\b": "low-sodium",
        r"\blow[-\s]?cholesterol\b": "low-cholesterol",
        r"\blow[-\s]?protein\b": "low-protein",
        r"\bhigh[-\s]?protein\b": "high-protein",
        r"\bhigh[-\s]?calcium\b": "high-calcium",
        r"\bhigh[-\s]?fiber\b": "high-fiber",
        r"\bdiabetic\b": "diabetic",
        r"\bketo\b": "low-carb",
        r"\bhealthy\b": "healthy",
        r"\bkosher\b": "kosher",
    },
    IntentKey.COURSES.value: {
        r"\bbreakfast\b": "breakfast",
        r"\bbrunch\b": "brunch",
        r"\blunch\b": "lunch",
        r"\bdinner\b": "main-dish",
        r"\bappetizer[s]?\b": "appetizers",
        r"\bsnack[s]?\b": "snacks",
        r"\bdessert[s]?\b": "desserts",
        r"\b(side[-\s]?dish(?:es)?|side(?:s)?|accompaniment(?:s)?)\b": "side-dishes",
        r"\bsoup[s]?\b": "soups-stews",
        r"\bsalad[s]?\b": "salads",
        r"\bbeverage[s]?\b": "beverages",
        r"\bdrink[s]?\b": "beverages",
    },
    IntentKey.CUISINES.value: {
        r"\bmexican\b": "mexican",
        r"\bitalian\b": "italian",
        r"\basian\b": "asian",
        r"\bchinese\b": "chinese",
        r"\bindian\b": "indian",
        r"\bthai\b": "thai",
        r"\bgreek\b": "greek",
        r"\bfrench\b": "french",
        r"\bspanish\b": "spanish",
        r"\bmediterranean\b": "european",
        r"\bcajun\b": "cajun",
        r"\bsouthern\b": "southern-united-states",
    },
    IntentKey.METHODS.value: {
        r"\bcrock[-\s]?pot\b": "crock-pot-slow-cooker",
        r"\bslow[-\s]?cooker\b": "crock-pot-slow-cooker",
        r"\binstant[-\s]?pot\b": "pressure-cooker",
        r"\bmicrowave\b": "microwave",
        r"\bgrill(?:ing|ed)?\b": "grilling",
        r"\bbbq\b": "barbecue",
        r"\bno[-\s]?cook\b": "no-cook",
        r"\b(?:one|1)[-\s]?pot\b": "one-dish-meal",
        r"\b(?:one|1)[-\s]?pan\b": "one-dish-meal",
    },
    IntentKey.OCCASIONS.value: {
        r"\bthanksgiving\b": "thanksgiving",
        r"\bchristmas\b": "christmas",
        r"\bsuper[-\s]?bowl\b": "superbowl",
        r"\bpotluck\b": "potluck",
        r"\bcamping\b": "camping",
        r"\bkid[-\s]?friendly\b": "kid-friendly",
        r"\btoddler[-\s]?friendly\b": "toddler-friendly",
    },
    IntentKey.TASTE.value: {
        r"\bsweet\b": "sweet",
        r"\bspicy\b": "spicy",
        r"\bsavory\b": "savory",
        r"\bsalty\b": "salty",
        r"\bsour\b": "sour",
        r"\btangy\b": "tangy",
        r"\bsmoky\b": "smoky",
        r"\bcreamy\b": "creamy",
        r"\bgarlicky\b": "garlicky",
        r"\bcheesy\b": "cheesy",
    },
    IntentKey.DISH_TYPE.value: {
        r"\btaco[s]?\b": "tacos",
        r"\bburrito[s]?\b": "burritos",
        r"\bwrap[s]?\b": "wraps",
        r"\bsandwich(?:es)?\b": "sandwiches",
        r"\bburger[s]?\b": "burgers",
        r"\bpizza\b": "pizza",
        r"\bpasta\b": "pasta",
        r"\bskillet\b": "skillet",
        r"\bsoup[s]?\b": "soups",
        r"\bstew[s]?\b": "stews",
        r"\bsalad[s]?\b": "salads",
    }
}


def initialize_intent(raw_query: str) -> Dict[str, Any]:
    return {
        IntentKey.RAW_QUERY.value: raw_query,
        IntentKey.LEXICAL_QUERY.value: raw_query.strip(),
        IntentKey.CLEAN_TEXT.value: raw_query.strip(),
        IntentKey.MAX_MINUTES.value: None,
        IntentKey.TARGET_MINUTES.value: None,
        IntentKey.PROTEINS.value: [],
        IntentKey.DIETARY_TAGS.value: [],
        IntentKey.COURSES.value: [],
        IntentKey.CUISINES.value: [],
        IntentKey.METHODS.value: [],
        IntentKey.OCCASIONS.value: [],
        IntentKey.TASTE.value: [],
        IntentKey.DISH_TYPE.value: [],
    }


def extract_time_constraints(intent: Dict[str, Any], raw_query: str) -> None:
    """Extract max_minutes (hard filter) and target_minutes (soft Gaussian) from time phrases."""
    under_match = re.search(
        r"(?:under|less than|max)\s*(\d+)\s*(?:min|minute)s?",
        raw_query,
        re.IGNORECASE,
    )
    if under_match:
        intent[IntentKey.MAX_MINUTES.value] = int(under_match.group(1))
        intent[IntentKey.CLEAN_TEXT.value] = (
            raw_query[: under_match.start()] + raw_query[under_match.end() :]
        ).strip()

    around_match = re.search(
        r"(?:around|about|approx)\s*(\d+)\s*(?:min|minute)s?",
        raw_query,
        re.IGNORECASE,
    )
    if around_match:
        intent[IntentKey.TARGET_MINUTES.value] = int(around_match.group(1))
        intent[IntentKey.CLEAN_TEXT.value] = (
            intent[IntentKey.CLEAN_TEXT.value][: around_match.start()]
            + intent[IntentKey.CLEAN_TEXT.value][around_match.end() :]
        ).strip()


def extract_proteins(intent: Dict[str, Any], raw_query: str) -> None:
    """Detect protein keywords for soft ingredient boosting (not hard filtering)."""
    for protein in PROTEIN_KEYWORDS:
        if re.search(rf"\b{re.escape(protein)}\b", raw_query, re.IGNORECASE):
            intent[IntentKey.PROTEINS.value].append(protein)


def extract_tag_intent(intent: Dict[str, Any], raw_query: str) -> None:
    """Map query phrases to canonical tags via TAG_MAPPINGS; strip matched phrases from clean_text."""
    clean_text = intent[IntentKey.CLEAN_TEXT.value]

    for category, mapping in TAG_MAPPINGS.items():
        for pattern, tag in mapping.items():
            if re.search(pattern, raw_query, re.IGNORECASE):
                if tag not in intent[category]:
                    intent[category].append(tag)
                clean_text = re.sub(pattern, "", clean_text, flags=re.IGNORECASE).strip()

    intent[IntentKey.CLEAN_TEXT.value] = re.sub(r"\s+", " ", clean_text).strip()


def parse_user_intent(raw_query: str) -> Dict[str, Any]:
    """Parse a raw query into structured intent (hard constraints + soft preferences)."""
    intent = initialize_intent(raw_query)
    extract_time_constraints(intent, raw_query)
    extract_proteins(intent, raw_query)
    extract_tag_intent(intent, raw_query)
    return intent


def build_base_bool_query(intent: Dict[str, Any]) -> Dict[str, Any]:
    """Build the base ES bool query with a must:multi_match and empty filter/should stubs."""
    lexical_query = intent[IntentKey.LEXICAL_QUERY.value] or "recipe"

    return {
        "bool": {
            "must": [
                {
                    "multi_match": {
                        "query": lexical_query,
                        "fields": [
                            SearchField.NAME.value,
                            SearchField.INGREDIENTS.value,
                            SearchField.DESCRIPTION.value,
                        ],
                        "boost": float(BoostValue.MULTI_MATCH.value),
                    }
                }
            ],
            "filter": [],
            "should": [],
        }
    }


def apply_hard_filters(base_query: Dict[str, Any], intent: Dict[str, Any]) -> None:
    """Apply max_minutes range filter and dietary tag term filters."""
    max_minutes = intent[IntentKey.MAX_MINUTES.value]
    if max_minutes is not None:
        base_query["bool"]["filter"].append(
            {
                "range": {
                    SearchField.MINUTES.value: {
                        "lte": max_minutes,
                    }
                }
            }
        )

    for tag in intent[IntentKey.DIETARY_TAGS.value]:
        base_query["bool"]["filter"].append(
            {"term": {SearchField.TAGS.value: tag}}
        )


def apply_soft_boosts(base_query: Dict[str, Any], intent: Dict[str, Any]) -> None:
    """Add should clauses for protein, taxonomy tag, and taste matches."""
    for protein in intent[IntentKey.PROTEINS.value]:
        base_query["bool"]["should"].append(
            {
                "match": {
                    SearchField.INGREDIENTS.value: {
                        "query": protein,
                        "boost": float(BoostValue.PROTEIN_MATCH.value),
                    }
                }
            }
        )

    soft_tags = (
        intent[IntentKey.COURSES.value]
        + intent[IntentKey.CUISINES.value]
        + intent[IntentKey.METHODS.value]
        + intent[IntentKey.OCCASIONS.value]
        + intent[IntentKey.DISH_TYPE.value]
    )

    for tag in soft_tags:
        base_query["bool"]["should"].append(
            {
                "term": {
                    SearchField.TAGS.value: {
                        "value": tag,
                        "boost": float(BoostValue.TAG_MATCH.value),
                    }
                }
            }
        )

    for taste in intent[IntentKey.TASTE.value]:
        base_query["bool"]["should"].append(
            {
                "term": {
                    SearchField.TAGS.value: {
                        "value": taste,
                        "boost": float(BoostValue.TASTE_MATCH.value),
                    }
                }
            }
        )


def apply_time_proximity_scoring(
    base_query: Dict[str, Any],
    intent: Dict[str, Any],
) -> Dict[str, Any]:
    """Wrap in a function_score Gaussian if target_minutes is set."""
    target_minutes = intent[IntentKey.TARGET_MINUTES.value]
    if target_minutes is None:
        return base_query

    return {
        "function_score": {
            "query": base_query,
            "functions": [
                {
                    "gauss": {
                        SearchField.MINUTES.value: {
                            "origin": str(target_minutes),
                            "offset": str(RetrievalConfig.TARGET_MINUTES_OFFSET.value),
                            "scale": str(RetrievalConfig.TARGET_MINUTES_SCALE.value),
                        }
                    }
                }
            ],
            "boost_mode": "multiply",
        }
    }


def build_candidate_query(intent: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build first-stage lexical retrieval query only.
    No dense vectors, no query encoder, no reranking logic.
    """
    base_query = build_base_bool_query(intent)
    apply_hard_filters(base_query, intent)
    apply_soft_boosts(base_query, intent)
    return apply_time_proximity_scoring(base_query, intent)


def retrieve_candidates(
    es_client: Any,
    index_name: str,
    raw_query: str,
    top_k: int = RetrievalConfig.DEFAULT_TOP_K.value,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Parse intent, build ES query, execute, return (hits, intent)."""
    intent = parse_user_intent(raw_query)
    final_query = build_candidate_query(intent)

    search_payload = {
        "size": top_k,
        "query": final_query,
    }

    response = es_client.search(index=index_name, **search_payload)
    return response["hits"]["hits"], intent


# When a query is synthesized from user history (not typed), strict must:multi_match
# drops candidates whose tags match but whose text doesn't contain the synthesized tokens.
# Diagnostics showed pool recall falling from 2% (tag Jaccard) to 0.5% (BM25-on-text).
# Fix: tags drive retrieval (minimum_should_match=1); text match becomes a weak boost.

def build_personalized_bool_query(intent: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tag-permissive query for personalized retrieval.

    No must:multi_match — text is a weak should boost (0.3) so tag-matched
    recipes aren't gated out by text absence. minimum_should_match=1 ensures
    at least one should clause qualifies each candidate.
    """
    lexical_query = intent[IntentKey.LEXICAL_QUERY.value] or ""

    base = {
        "bool": {
            "must": [],
            "filter": [],
            "should": [],
            "minimum_should_match": 1,
        }
    }

    if lexical_query:
        # Text contributes to score but doesn't gate; tags drive retrieval.
        base["bool"]["should"].append({
            "multi_match": {
                "query": lexical_query,
                "fields": [
                    SearchField.NAME.value,
                    SearchField.INGREDIENTS.value,
                    SearchField.DESCRIPTION.value,
                ],
                "boost": 0.3,
            }
        })

    return base


def build_personalized_candidate_query(intent: Dict[str, Any]) -> Dict[str, Any]:
    """Assemble the personalized query using the tag-permissive base."""
    base_query = build_personalized_bool_query(intent)
    apply_hard_filters(base_query, intent)
    apply_soft_boosts(base_query, intent)
    return apply_time_proximity_scoring(base_query, intent)


def retrieve_candidates_personalized(
    es_client: Any,
    index_name: str,
    raw_query: str,
    top_k: int = 500,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Tag-permissive retrieval for synthesized queries; returns (hits, intent)."""
    intent = parse_user_intent(raw_query)
    final_query = build_personalized_candidate_query(intent)

    response = es_client.search(
        index=index_name,
        size=top_k,
        query=final_query,
    )
    return response["hits"]["hits"], intent


# synthesize_query_string emits canonical tags ('main-dish', 'soups-stews') that have
# no reverse pattern in TAG_MAPPINGS — passing them through parse_user_intent drops
# ~40% of the signal. These functions bypass the parser and inject tags directly as
# ES term queries on tags_clean.

_PROTEIN_KEYWORD_SET: FrozenSet[str] = frozenset(PROTEIN_KEYWORDS)


def build_candidate_query_from_tags(tags: Set[str]) -> Dict[str, Any]:
    """
    Build an ES bool query directly from canonical taxonomy tags.

    Proteins match on ingredients_clean (boost 2.0); other tags use term on
    tags_clean (boost 1.5). minimum_should_match=1 keeps the pool open without
    requiring every tag to match.
    """
    protein_tags = {t for t in tags if t in _PROTEIN_KEYWORD_SET}
    other_tags   = tags - protein_tags

    should: List[Dict[str, Any]] = []

    for protein in protein_tags:
        should.append({
            "match": {
                SearchField.INGREDIENTS.value: {
                    "query": protein,
                    "boost": float(BoostValue.PROTEIN_MATCH),
                }
            }
        })

    for tag in other_tags:
        should.append({
            "term": {
                SearchField.TAGS.value: {
                    "value": tag,
                    "boost": float(BoostValue.TAG_MATCH),
                }
            }
        })

    if not should:
        should.append({"match_all": {}})  # no tags — retrieve broadly

    return {
        "bool": {
            "should": should,
            "minimum_should_match": 1,
        }
    }


def retrieve_candidates_from_tags(
    es_client: Any,
    index_name: str,
    tags: Set[str],
    top_k: int = 500,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Retrieve candidates from canonical tags, no text parsing.

    Returns (hits, empty_intent). Alignment score will be 0 for all candidates
    (correct — alignment only makes sense for typed queries). Cosine sim and
    tag affinity carry all personalization signal.
    """
    query    = build_candidate_query_from_tags(tags)
    response = es_client.search(index=index_name, size=top_k, query=query)
    return response["hits"]["hits"], initialize_intent("")


