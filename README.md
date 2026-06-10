# Recipe Search & Recommender

End-to-end personalized recipe recommendation system built on the Food.com corpus. Unifies four course projects into a single pipeline: a PySpark tagging pipeline (Phase 1), a RecipeNet dual-encoder MLP (Phase 2), a BM25 + SemanticReranker IR system (Phase 3), and a two-stage personalized recommender (Phase 4).

**Course:** DSCI 641 — Drexel University MSDS

---

## Reviewer quick start

**Requires only [Docker Desktop](https://www.docker.com/products/docker-desktop/) — no Python, no data files, no configuration.**  
Works on Windows, macOS, and Linux.

### 1 — Start the app

```bash
docker compose -f docker/docker-compose.prebuilt.yml up
```

- **First run:** Docker pulls ~3–4 GB of images from Docker Hub, then Elasticsearch starts with the recipe index pre-loaded (~1 minute total). No indexing step.
- **Subsequent runs:** start in ~30 seconds.

Watch for this line in the logs before opening the browser:
```
[startup] Launching Streamlit on port 8501
```

### 2 — Open the app

**http://localhost:8501**

### 3 — Two modes to explore

| Mode | How to use |
|------|-----------|
| **Search** | Type any recipe query (e.g. "easy chicken dinner under 45 mins"). Switch search modes (Hybrid / Lexical / Semantic / Quality) via the sidebar. Add results to a session to see personalized "Based on your session" recommendations appear alongside results. |
| **User History** | Select one of 10 pre-loaded Food.com reviewer archetypes (Global Foodie, Baker, Critic, etc.). The system builds a full recommendation list from their real review history. Optionally add a query and use the alpha slider to blend query relevance with personal taste. |

### 4 — Stop

`Ctrl-C`, then:
```bash
docker compose -f docker/docker-compose.prebuilt.yml down
```

---

## Project lineage

| Phase | Course | What it produced |
|-------|--------|-----------------|
| 1 — Tagging pipeline | DSCI 632 | `gold_labeled_reviews` — 17 review-sentiment tags via PySpark Word2Vec |
| 2 — Embedding model | CS 615 | RecipeNet dual-encoder MLP; 128D recipe embeddings; Bayesian rating predictions |
| 3 — IR system | INFO 624 | BM25 + SemanticReranker; 5-mode ablation; stratified intent-based weights |
| 4 — Personalization | DSCI 641 | Two-stage personalized recommender: tag-filtered candidate generation + embedding KNN |

---

## Phase 4 — New files (DSCI 641 submission)

The following files were written for this course. Everything under `core/`, `pipeline/train*.py`, `search/`, and the base search mode in `app/main.py` is carried forward from prior phases.

| File | What it does |
|------|-------------|
| `recommender/candidate.py` | `CandidateGenerator` — inverted index over Food.com taxonomy tags; IDF-weighted recall scoring for Stage 1 candidate retrieval |
| `recommender/ingredient_embed.py` | `IngredientEmbedder` — TF-IDF matrix over `ingredients_clean` (169K recipes, 7,007-token vocab); user centroid for ingredient-based retrieval |
| `recommender/knn.py` | `cosine_knn()`, `session_query_vector()`, `KNNResult` — cosine KNN restricted to Stage 1 candidate pool |
| `recommender/profile.py` | `UserProfile`, `build_user_profile()` — rating×recency-weighted embedding centroid; 17D Phase-1 sentiment tag affinity vector |
| `recommender/engine.py` | `RecommenderEngine` — unified two-stage pipeline; cold-start, session-based, and history-aware modes |
| `recommender/evaluate.py` | Leave-N-out evaluation harness — TagSim@k, TagCov@k, HR@k, MRR, pool recall, mean rank in-pool |
| `pipeline/grid_search.py` | Hyperparameter sweep — `CANDIDATE_POOL_SIZE`, `USER_HISTORY_TOP_TAGS`, `STAGE2_PREFILTER_MULT`, Stage 1 scoring variants |
| `pipeline/identify_personas.py` | Offline archetype clustering — maps real Food.com reviewer histories to 10 curated personas for the UI |
| `app/main.py` | Streamlit UI — Search mode (Phase 3, extended) + User History mode (Phase 4 new): persona selector, taste-profile chips, `alpha` blend slider |
| `main.py` *(extended)* | Added `recommend` and `compare` CLI commands on top of Phase 3's `search` / `index` / `app` |

---

## Architecture

### Stage 1 — Lexical retrieval (Elasticsearch BM25)

Query intent is parsed into structured slots (dietary constraints, proteins, cuisines, courses, methods, occasions, taste, time bounds). Hard filters and soft boosts applied. Returns a candidate pool (default 1,000).

For personalized queries with no free-text input, Stage 1 uses a tag-direct ES query over the user's top-40 history tags (`retrieve_candidates_from_tags`), bypassing intent parsing entirely to preserve all user-history signal.

### Stage 2 — Reranking (SemanticReranker)

Queries are projected into the 128D RecipeNet latent space via `QueryFeatureProjector`. Four signals are combined with stratified weights based on query intent richness:

| Signal | Role |
|--------|------|
| BM25 base score | Lexical relevance from Stage 1 |
| Rule-based alignment | Structured slot matching score |
| Cosine similarity | Embedding distance in RecipeNet latent space |
| Bayesian quality score | Predicted smoothed rating from Phase 2 |

**Personalized blend:** When a user profile is active, Stage 2 uses `alpha * query_embedding + (1 - alpha) * user_embedding` for cosine similarity. The `lex` and `alignment` weights are also scaled by `alpha` so that at `alpha=0` only the user taste centroid and quality signal drive ranking.

### Five search modes

| Mode | Description |
|------|-------------|
| `HYBRID` | Full pipeline — all four signals with stratified weights |
| `LEXICAL` | BM25 only; no reranking |
| `SEMANTIC` | Embedding similarity + quality only |
| `QUALITY` | Predicted rating only; query-agnostic |
| `ABLATION_NO_SEM` | Hybrid minus embedding similarity |

### Phase 4 — Two-stage recommender (complete)

**Key design decision:** RecipeNet embeddings were trained to predict ratings, not encode culinary identity. Raw KNN over the full 173K corpus degenerates — all high-quality recipe embeddings cluster together in cosine space, producing incoherent recommendations. The fix is a two-stage pipeline that uses Food.com taxonomy tags to constrain the embedding KNN to a culinarily coherent candidate pool.

**Stage 1a — Tag candidate generation (`recommender/candidate.py`)**  
Inverted index over `tags_clean` taxonomy. For a user history query, retrieves candidates by IDF-weighted recall scoring (intersection / query_idf_sum) over the user's top-40 tags. Structural noise tags are filtered; rare/common tags are balanced by IDF. Pool size: 1,000.

**Stage 1b — Ingredient TF-IDF (`recommender/ingredient_embed.py`)**  
IngredientEmbedder builds a TF-IDF matrix over `ingredients_clean` (169K recipes, 7,007-token vocab). Used as an ablation signal and in the hybrid retrieval condition.

**Stage 2 — Personalized scoring**  
Cosine KNN within the candidate pool using the user's rating×recency-weighted embedding centroid (half-life 180 days). Reviews ≤ 3★ have their deviation amplified 2× as a negative signal. An optional 17-D sentiment tag affinity boost is applied on top.

**Final optimized configuration (grid-searched)**

| Parameter | Value |
|-----------|-------|
| `CANDIDATE_POOL_SIZE` | 1,000 |
| `USER_HISTORY_TOP_TAGS` | 40 |
| `recency_half_life_days` | 180 |
| Stage 1 scoring | Recall-only (intersection / query_idf_sum) |
| Negative signal amplification | 2× for ≤ 3★ reviews |
| `QUALITY_WEIGHT` | 0.0 |
| `STAGE2_PREFILTER_MULT` | 3 |

---

## Evaluation results

### Phase 3 — IR system (5-mode ablation, 14 stratified queries)

HYBRID dominates on high- and medium-intent queries. ABLATION_NO_SEM confirms the semantic embedding layer contributes meaningfully to ranking quality. Full results in `search/evaluate.py`.

### Phase 4 — Recommender (leave-N-out, n=200 users, n_holdout=5)

Primary metric: **TagSim@10** — average cosine similarity between recommended recipe tags and held-out review tags.

| Condition | TagSim@10 | Pool recall | HR@10 |
|-----------|-----------|-------------|-------|
| `cold_start` | 0.095 | — | 0.000 |
| `knn_only` | 0.104 | 1.000 | 0.005 |
| `two_stage` | 0.186 | 0.120 | 0.005 |
| `two_stage_ingr` | 0.113 | 0.090 | 0.015 |
| `two_stage_hybrid` | 0.155 | 0.195 | 0.010 |
| `unified` (ceiling) | 0.206 | 0.070 | 0.005 |

**Key findings:**
1. RecipeNet embeddings and taxonomy tags occupy orthogonal spaces — embeddings encode quality, tags encode culinary identity. Two-stage architecture is required to bridge them.
2. Ingredient TF-IDF and taxonomy tags are complementary: ingredients improve HR/MRR, tags drive TagSim.
3. Injecting embedding NN results into Stage 1 destroyed TagSim (−40%) — confirmed orthogonality empirically.
4. Recency decay + negative signal boosted pool recall from 0.060 → 0.095 (+58%); MRR doubled.
5. The `unified` TagSim ceiling (~0.206) comes from ES BM25 full-text retrieval — not reachable by tuning within the two-stage architecture alone.

---

## Repository layout

```
recipe-recommender/
├── core/
│   ├── config.py               # Settings dataclass; load_settings(); get_es_client()
│   ├── layers.py               # FullyConnectedBlock, ResidualBlock, ResidualLinearBlock
│   └── models.py               # RecipeNet; PRODUCTION_HEAD = HeadType.RESIDUAL_V2
│
├── pipeline/
│   ├── train_config.py
│   ├── preprocessing.py
│   ├── dataset.py
│   ├── trainer.py
│   ├── inference.py
│   ├── train.py
│   ├── grid_search.py          # Phase 4 hyperparameter sweeps (PREFILTER_MULT, TOP_TAGS)
│   └── identify_personas.py    # Offline archetype identification from gold_labeled_reviews
│
├── search/
│   ├── search.py               # Intent parsing; ES query builder; retrieve_candidates()
│   ├── query_encoding.py       # QueryFeatureProjector → ProjectedQuery
│   ├── reranker.py             # SemanticReranker; 4-signal scoring; 5 weight profiles
│   ├── engine.py               # SearchEngine; SearchMode enum; SearchResult; unified search()
│   ├── indexer.py              # ES index creation and bulk ingestion
│   └── evaluate.py             # NDCG@k, P@1; 5-mode ablation evaluation suite
│
├── recommender/
│   ├── candidate.py            # CandidateGenerator: inverted tag index; IDF-weighted recall
│   ├── ingredient_embed.py     # IngredientEmbedder: TF-IDF over ingredients_clean
│   ├── knn.py                  # cosine_knn(); session_query_vector(); KNNResult
│   ├── profile.py              # ReviewRecord, UserProfile; build_user_profile()
│   ├── engine.py               # RecommenderEngine: two-stage pipeline; all modes
│   └── evaluate.py             # LOO evaluation; TagSim@k, TagCov@k, HR@k, MRR
│
├── app/
│   └── main.py                 # Streamlit UI — Search + User History modes
│
├── docker/
│   └── docker-compose.yml      # Single-node Elasticsearch 8
│
├── data/                       # Excluded from git — see Data section
├── main.py                     # Unified CLI entry point
├── CAPSTONE_OVERVIEW.md        # Full project narrative for presentation/report
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup

### Build the pre-indexed images (submitter only)

Run once before submission to build and push the images the reviewer will pull.  
Requires: Docker Desktop, Python + venv active, data files in `data/`, `docker login` done.

```powershell
# From the project root with your venv active:
docker\build-and-push.ps1 -DockerUser yourdockerhubusername
```

This takes 15–30 minutes (ES indexing + image builds + upload). When done it writes `docker/docker-compose.prebuilt.yml` — commit and include that file in your submission.

---

### Local development setup

### Prerequisites

- Docker Desktop (required — runs both Elasticsearch and the app)

### 1 — Place data files

Create a `data/` directory at the project root and add the following files:

| File | Source |
|------|--------|
| `PROCESSED_search_recipes.parquet` | Phase 3 ES-ready recipe corpus |
| `final_residual_v2_embeddings.pt` | Phase 2 embedding bundle (recipe IDs + 128D vectors + quality scores) |
| `best_model_residual_v2_all_features_mse.pth` | Phase 2 model checkpoint |
| `column_mapping.json` | Phase 2 feature index map |
| `gold_labeled_reviews.parquet` | Phase 1 sentiment-tagged reviews — rename your timestamped file to this name, or see the note below |

> **Gold reviews filename:** If your Phase 1 output has a timestamp (`gold_labeled_reviews_<timestamp>.parquet`), either rename it to `gold_labeled_reviews.parquet` **or** uncomment and set the `GOLD_REVIEWS_PATH` override in `docker/docker-compose.yml`.

### 2 — Build and run

```bash
docker compose -f docker/docker-compose.yml up --build
```

On first run this will:
1. Pull the Elasticsearch 8 image and the Python 3.11 base image
2. Install all Python dependencies (including CPU-only PyTorch — ~500 MB total)
3. Wait for Elasticsearch to become healthy
4. Index all recipes into Elasticsearch (~2–5 minutes depending on hardware)
5. Launch Streamlit at **http://localhost:8501**

Subsequent runs skip steps 1–4 and start in seconds.

To stop: `Ctrl-C`, then `docker compose -f docker/docker-compose.yml down`.  
The ES index persists in a named Docker volume (`esdata`) so re-indexing is not needed on restart.

---

### Manual setup (alternative — requires Python 3.10+)

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # then edit .env with your data file paths
```

Start Elasticsearch:

```bash
docker compose -f docker/docker-compose.yml up elasticsearch -d
```

Index and launch:

```bash
python main.py index
python main.py app
```

---

## CLI

```bash
# Run a search query
python main.py search "easy chicken dinner under 45 mins" --mode hybrid --top-k 10

# Personalized recommendations for a user
python main.py recommend --user-id 12345 --top-k 10

# Compare personalized results across multiple users
python main.py compare "quick pasta" --users 100526 1001924 1177249 --alpha 0.0 --top-k 7

# Ingest recipes into Elasticsearch
python main.py index

# Launch Streamlit app
python main.py app
```

---

## Streamlit app

Two input modes:

**Search** — BM25/hybrid IR with optional session seeding. Results can be added to a session (up to 5 seeds); subsequent searches show personalized "Based on your session" recommendations alongside results in a two-column layout. Supports all five search modes and a full mode-comparison view with NDCG@5 per mode.

**User History** — Choose from 10 curated Food.com reviewer archetypes identified offline by `pipeline/identify_personas.py` (Global Foodie, Critic, Baker, Comfort Food Devotee, Weeknight Cook, Specialist, Lapsed Reviewer, Health Enthusiast, Casual, Holiday Cook). A persona card shows each archetype's tagline and trait chips before recommendations load. Optionally add a free-text query and use the `Query ↔ User blend` slider (alpha) to control how much text relevance vs. user taste profile drives ranking. Displays a polarity-aware taste profile (Likes/Dislikes chips derived from 17-D sentiment tag affinity) and the user's recent review history.

> **Note on alpha with a query:** `alpha=0` applies to Stage 2 reranking weights only — Stage 1 still retrieves candidates using the query text. For purely taste-driven recommendations, leave the query blank; the no-query path uses tag-direct ES retrieval from the user's history, which produces stronger archetype differentiation.

---

## Run evaluations

```bash
# Phase 3 — 5-mode IR ablation
python -m search.evaluate

# Phase 4 — Leave-N-out recommender evaluation
python -m recommender.evaluate --n-users 200 --min-reviews 15 --k 10

# Phase 4 — Hyperparameter grid search
python -m pipeline.grid_search
```

---

## Key design decisions

**Two-stage recommender — why not pure embedding KNN:** RecipeNet was trained for rating prediction. In cosine space all high-quality embeddings cluster together, making full-corpus KNN degenerate. Stage 1 tag filtering constrains the search space to a culinarily coherent pool before embedding KNN runs.

**Orthogonal embedding spaces:** RecipeNet embeddings encode quality; Food.com taxonomy tags encode culinary identity. They are empirically orthogonal — injecting embedding NN neighbors into Stage 1 destroyed TagSim by 40%. The two-stage architecture is the correct way to bridge them.

**Alpha weight scaling:** At `alpha=0`, BM25 would otherwise still dominate the final score through fixed `w_lex`. The fix scales `weights["lex"]` and `weights["alignment"]` by `effective_alpha` so the slider is a true blend: at 0, only the user taste centroid and quality signal rank results.

**History-aware via simulation:** No production user storage or auth layer is required. Recommendation modes use real user histories from the Food.com dataset with proper temporal splits for offline evaluation.

**Frozen RecipeNet:** Phase 2 model is used in eval mode as a feature encoder only. `PRODUCTION_HEAD = HeadType.RESIDUAL_V2` is a permanent constant.

**Stratified reranking weights:** SemanticReranker adapts to query intent richness. High-intent queries (≥3 structured slots) weight alignment heavily; low-intent queries lean on semantic similarity.

---

## Retrain Phase 2 model

```bash
python -m pipeline.train
```

Uses `TrainConfig` defaults: lr=1e-4, batch=256, epochs=300, patience=20, RESIDUAL_V2 head.

---

## Data

Data files are excluded from version control. Phase 2 artifacts are produced by the CS 615 project. Phase 1 output (`gold_labeled_reviews`) is a static input from the DSCI 632 project. Raw Food.com corpus available on Kaggle.

---

## License

Academic project — Drexel University DSCI 641.
