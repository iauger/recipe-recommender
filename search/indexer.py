"""Elasticsearch index creation and bulk ingestion for the recipe corpus."""

import pandas as pd
import numpy as np
import torch
from elasticsearch import helpers

if not hasattr(np, 'float_'):
    np.float_ = np.float64  # type: ignore  # ES 8.12 requires np.float_


def get_tags(df: pd.DataFrame) -> list[str]:
    """Extract base tag names from pred_* columns."""
    return [col.removeprefix("pred_") for col in df.columns if col.startswith("pred_")]


def create_index(settings, es_client):
    """Drop and recreate the ES index with explicit field mappings."""
    print("Testing connection to Elasticsearch...")
    try:
        info = es_client.info()
        print(f"Successfully connected to Elasticsearch v{info['version']['number']}")
    except Exception as e:
        print(f"Connection failed: {e}")
        print("Check if a VPN or proxy is blocking the connection.")
        return

    print(f"Resetting index: {settings.es_index}")
    es_client.indices.delete(index=settings.es_index, ignore_unavailable=True)

    properties = {
        "name": {"type": "text", "analyzer": "english"},
        "description_clean": {"type": "text", "analyzer": "english"},
        "steps_clean": {"type": "text", "analyzer": "english"},
        "ingredients_clean": {"type": "text", "analyzer": "standard"},
        "tags_clean": {"type": "keyword"},
        "minutes": {"type": "float"},
        "n_steps": {"type": "integer"},
        "n_ingredients": {"type": "integer"},
        "bayesian_rating": {"type": "float"},
        "review_count": {"type": "integer"}
    }

    tags = get_tags(pd.read_parquet(settings.recipes_path))

    for tag in tags:
        properties[f"pred_{tag}"] = {"type": "boolean"}
        properties[f"intensity_{tag}"] = {"type": "float"}

    es_client.indices.create(index=settings.es_index, mappings={"properties": properties})
    print(f"Created index '{settings.es_index}' with hybrid search mappings.")


def generate_documents(df, bundle, index_name):
    """Yield bulk-API documents from the recipe DataFrame."""

    for idx, row in df.iterrows():
        r_id = str(row['recipe_id'])

        doc = row.to_dict()

        yield {
            "_index": index_name,
            "_id": r_id,
            "_source": doc
        }


def run_ingestion(settings, es_client):
    print("Loading search data into memory.")
    df = pd.read_parquet(settings.recipes_path)
    bundle = torch.load(settings.embeddings_path, weights_only=False)

    create_index(settings, es_client)

    print(f"Starting bulk ingestion of {len(df)} recipes.")
    success, failed = helpers.bulk(
        es_client,
        generate_documents(df, bundle, settings.es_index),
        chunk_size=500,
        stats_only=True
    )
    print(f"Successfully indexed {success} documents.")
    if failed:
        print(f"Warning: {failed} documents failed to index.")


if __name__ == "__main__":
    from core.config import load_settings, get_es_client

    s = load_settings()
    es_client = get_es_client(s)
    run_ingestion(s, es_client)
