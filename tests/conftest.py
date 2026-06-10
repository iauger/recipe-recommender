# tests/conftest.py
"""
Shared pytest fixtures used across unit and integration test modules.

Fixtures that require real data files (parquet, .pt bundle) are marked
with the 'integration' marker and skipped unless --integration is passed.
"""

import json
import numpy as np
import pandas as pd
import pytest
import torch

# ---------------------------------------------------------------------------
# Pytest marker registration
# ---------------------------------------------------------------------------

def pytest_addoption(parser):
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="Run integration tests that require Elasticsearch and data files.",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: mark test as requiring Elasticsearch and data files",
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--integration"):
        skip = pytest.mark.skip(reason="Pass --integration to run")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip)


# ---------------------------------------------------------------------------
# Lightweight synthetic fixtures (no data files required)
# ---------------------------------------------------------------------------

NUM_RECIPES = 50
EMBED_DIM = 128
N_CULINARY_TAGS = 17

CULINARY_TAGS = (
    "family_hit", "delicious_tasty", "would_make_again", "easy_quick",
    "substitution_modification", "ingredient_issue", "crispy_crunchy",
    "moist_tender", "would_not_make_again", "too_spicy",
    "time_consuming_complex", "bland_lacks_flavor", "dry",
    "too_sweet", "mushy_soggy", "too_acidic", "too_salty",
)


@pytest.fixture
def culinary_tags():
    return CULINARY_TAGS


@pytest.fixture
def tiny_embeddings():
    """Pre-normalised (N, 128) float tensor — unit vectors."""
    raw = torch.randn(NUM_RECIPES, EMBED_DIM)
    return torch.nn.functional.normalize(raw, p=2, dim=1)


@pytest.fixture
def recipe_ids():
    """Parallel list of int recipe IDs matching tiny_embeddings rows."""
    return list(range(1000, 1000 + NUM_RECIPES))


@pytest.fixture
def tiny_bundle(tiny_embeddings, recipe_ids):
    """
    Minimal embedding bundle mirroring the structure saved by pipeline/inference.py.
    predictions is a plain Python list (as inference.py produces).
    """
    return {
        "recipe_ids":   [str(r) for r in recipe_ids],   # stored as strings by inference
        "recipe_names": [f"Recipe {r}" for r in recipe_ids],
        "targets":      [4.0] * NUM_RECIPES,
        "predictions":  [3.5 + 0.1 * i % 5 for i in range(NUM_RECIPES)],  # plain list
        "embeddings":   tiny_embeddings,
    }


@pytest.fixture
def tiny_recipes_df(recipe_ids):
    """
    Minimal recipes DataFrame matching PROCESSED_search_recipes.parquet schema.
    recipe_id stored as string (matching preprocessing output).
    """
    rng = np.random.default_rng(42)
    rows = []
    for rid in recipe_ids:
        rows.append({
            "recipe_id":       str(rid),          # string, as preprocessing stores it
            "name":            f"recipe {rid}",
            "minutes":         rng.integers(10, 120),
            "bayesian_rating": float(rng.uniform(3.0, 5.0)),
            "tags_clean":      "dinner main-dish",
            "ingredients_clean": "chicken onion garlic",
            **{f"pred_{t}": rng.random() > 0.7 for t in CULINARY_TAGS},
            **{f"intensity_{t}": float(rng.uniform(0, 1)) for t in CULINARY_TAGS},
        })
    return pd.DataFrame(rows)


@pytest.fixture
def col_map_dict():
    """
    Minimal column_mapping.json matching the feature schema used by
    QueryFeatureProjector and SemanticReranker.
    Includes numeric, cat_*, ing_*, pred_*, intensity_* columns.
    """
    cols = [
        "recipe_id", "name", "tags_clean", "ingredients_clean",   # text (indices 0-3)
        "minutes", "n_steps", "n_ingredients",                    # numeric
        "cat_chicken", "cat_italian", "cat_dinner",               # categorical tags
        "ing_chicken", "ing_onion", "ing_garlic",                 # ingredient OHE
    ]
    cols += [f"pred_{t}" for t in CULINARY_TAGS]
    cols += [f"intensity_{t}" for t in CULINARY_TAGS]
    return {col: i for i, col in enumerate(cols)}


@pytest.fixture
def col_map_file(tmp_path, col_map_dict):
    """Write col_map_dict to a temp JSON file and return the path."""
    p = tmp_path / "column_mapping.json"
    p.write_text(json.dumps(col_map_dict))
    return p


@pytest.fixture
def mock_settings(tmp_path, col_map_file, tiny_bundle, tiny_recipes_df):
    """
    Minimal Settings-like object with all paths pointing to tmp_path fixtures.
    Does NOT require Elasticsearch — es_host is set to a dummy value.
    """
    import types
    import torch as _torch

    bundle_path = tmp_path / "embeddings.pt"
    _torch.save(tiny_bundle, bundle_path)

    recipes_path = tmp_path / "recipes.parquet"
    tiny_recipes_df.to_parquet(recipes_path, index=False)

    return types.SimpleNamespace(
        es_host="http://localhost:9200",
        es_index="recipes",
        es_timeout=30,
        embeddings_path=bundle_path,
        column_mapping_path=col_map_file,
        recipes_path=recipes_path,
        candidate_pool_size=20,
        min_reviews_for_cf=2,
        recency_half_life_days=365,
        culinary_tags=CULINARY_TAGS,
    )
