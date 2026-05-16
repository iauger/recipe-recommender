# Recipe Recommender System

Personalized recipe recommender built on top of a three-phase Food.com corpus pipeline (DSCI 632 → CS 615 → INFO 624). Extends the Phase 3 two-stage IR system (Elasticsearch BM25 + semantic reranker) with user preference modeling.

## Project context

| Phase | Repo | Output used here |
|-------|------|-----------------|
| 1 — Tagging pipeline (DSCI 632) | `recipe-review-tags` | `gold_labeled_reviews` — static input, not a runtime dependency |
| 2 — Embedding model (CS 615) | `RecipeFeedback-ResNet` | `final_residual_v2_embeddings.pt`, `best_model_residual_v2_all_features_mse.pth`, `column_mapping.json`, `PROCESSED_search_recipes.parquet` |
| 3 — IR system (INFO 624) | `recipe-search-engine` | Elasticsearch index, reranker, Streamlit app — extended here |

## Architecture

Three recommendation modes, activated based on available user context:

- **Cold-start** — quality-weighted BM25 + semantic reranker, no user signal (`user_affinity_score = 0.0`)
- **Session-based** — mean of seed recipe embeddings → `sklearn.NearestNeighbors` cosine NN over the 128D embedding bundle
- **History-aware** — 17D tag affinity vector derived from the user's review history (mean-centered, IDF-weighted, recency-decayed); injected as a 5th signal into `combine_scores()`

Item-item collaborative filtering (LensKit) is a stretch goal for users with ≥5 reviews.

## Repository layout

```
recipe-recommender/
├── core/           # Shared: config, models, layers, schema
├── pipeline/       # Phase 1/2 wrappers (offline use only)
├── search/         # Phase 3 IR system — extended with user context
├── recommender/    # User profiles, affinity reranker, session NN, CF
├── app/            # Streamlit frontend
├── docker/         # docker-compose for Elasticsearch
├── notebooks/      # Exploratory analysis
└── main.py         # Unified CLI entry point
```

## Setup

### Prerequisites

- Python 3.10+
- Docker Desktop (for Elasticsearch)

### Install

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Start Elasticsearch

```bash
docker-compose -f docker/docker-compose.yml up -d
```

### Configure

Copy `.env.example` to `.env` and set paths to Phase 2 artifacts:

```bash
cp .env.example .env
# Edit .env — set EMBEDDINGS_PATH, MODEL_PATH, RECIPES_PATH, COLUMN_MAPPING_PATH
```

### Run

```bash
# Launch Streamlit app
streamlit run app/main.py

# Or via CLI
python main.py --help
```

## Evaluation

Temporal train/test split on review timestamps.

| Metric | What it measures |
|--------|-----------------|
| NDCG@5 | Ranking quality |
| Precision@1 | Top hit accuracy |
| Tag Diversity H@5 | Normalized entropy over culinary tags in top-5 results |
| Coverage | Fraction of recipe catalog surfaced |

Key risk: tag diversity collapse toward `family_hit` / `delicious_tasty` — mitigated by IDF weighting in the affinity vector.

## Data

Data files are excluded from version control (see `.gitignore`). Place Phase 2 artifacts in `data/` or point `.env` to their existing location.

## License

Academic project — Drexel University DSCI 641.
