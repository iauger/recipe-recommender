"""
Targeted grid search for the two-stage personalized recommender.

Sweep 1: STAGE2_PREFILTER_MULT (two_stage_hybrid) — the hybrid Stage 1 widened
the pool (0.105 → 0.180 recall) but TagSim dropped 0.182 → 0.152; hypothesis is
the pre-filter needs resizing for ~1800 candidates instead of ~1000.

Sweep 2: USER_HISTORY_TOP_TAGS (two_stage) — tests whether a wider tag query
can push TagSim@10 past 0.20 from the current 0.182 baseline.

    python -m pipeline.grid_search
"""

import time
from typing import List

from core.config import load_settings, get_es_client
from search.engine import SearchEngine
from recommender.engine import RecommenderEngine
from recommender.evaluate import (
    DEFAULT_K_VALUES,
    EVAL_WINDOW,
    leave_n_out_sample,
    evaluate_condition,
    ConditionResult,
)

STAGE2_MULT_GRID: List[int] = [3, 6, 10, 20]   # pre-filter keeps top_k * mult candidates
TOP_TAGS_GRID:    List[int] = [20, 30, 40, 50]

N_USERS  = 100
SEED     = 42
TARGET_K = 10

def _header(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def _row(label: str, result: ConditionResult) -> str:
    ts  = result.tag_sim_at_k.get(TARGET_K, 0.0)
    ts_ci = result.tag_sim_ci_at_k.get(TARGET_K, (0.0, 0.0))
    rec = result.recall_at_k.get(TARGET_K, 0.0)
    pr  = result.pool_recall or 0.0
    return (
        f"  {label:<35}  TagSim@{TARGET_K}: {ts:.4f} [{ts_ci[0]:.3f},{ts_ci[1]:.3f}]"
        f"  Recall@{TARGET_K}: {rec:.4f}  PoolRecall: {pr:.3f}"
    )


def main() -> None:
    settings = load_settings()

    print("Connecting to Elasticsearch...")
    es_client = get_es_client(settings)
    search_engine = SearchEngine(settings, es_client)

    print("Loading recommender engine...")
    t0 = time.time()
    engine = RecommenderEngine(settings, search_engine=search_engine)
    print(f"Engine loaded in {time.time() - t0:.1f}s")

    print(f"\nSampling {N_USERS} LNO test cases (seed={SEED})...")
    t0 = time.time()
    test_cases = leave_n_out_sample(
        gold_reviews_path=settings.gold_reviews_path,
        bundle_ids=engine._bundle_id_set,
        min_reviews=15,
        n_holdout=5,
        n_users=N_USERS,
        seed=SEED,
    )
    print(f"Sampled {len(test_cases)} valid test cases in {time.time() - t0:.1f}s")
    if not test_cases:
        print("No test cases found. Exiting.")
        return

    _header(
        f"Sweep 1: Stage 2 pre-filter multiplier (two_stage_hybrid)"
        f"  --  optimising TagSim@{TARGET_K}"
    )
    print(f"  Grid: {STAGE2_MULT_GRID}  (pre-filter keeps top_k * mult candidates)")
    print()

    best_ts1, best_mult = -1.0, STAGE2_MULT_GRID[0]
    sweep1_results = []

    for mult in STAGE2_MULT_GRID:
        engine.STAGE2_PREFILTER_MULT = mult
        label = f"STAGE2_PREFILTER_MULT={mult}"
        print(f"  Running {label}...", end="", flush=True)
        t0 = time.time()
        result = evaluate_condition(
            engine=engine,
            test_cases=test_cases,
            condition="two_stage_hybrid",
            ks=DEFAULT_K_VALUES,
            eval_window=EVAL_WINDOW,
            search_engine=search_engine,
        )
        elapsed = time.time() - t0
        ts = result.tag_sim_at_k.get(TARGET_K, 0.0)
        print(f" done in {elapsed:.1f}s")
        print(_row(label, result))
        sweep1_results.append((mult, result))
        if ts > best_ts1:
            best_ts1 = ts
            best_mult = mult

    engine.STAGE2_PREFILTER_MULT = 3  # restore default

    print()
    print(f"  >> Best STAGE2_PREFILTER_MULT = {best_mult}  (TagSim@{TARGET_K} = {best_ts1:.4f})")

    _header(
        f"Sweep 2: Tag query width (two_stage)"
        f"  --  optimising TagSim@{TARGET_K}"
    )
    print(f"  Grid: {TOP_TAGS_GRID}  (affinity + liked tags each capped at N)")
    print()

    best_ts2, best_top_tags = -1.0, TOP_TAGS_GRID[0]
    sweep2_results = []

    for n_tags in TOP_TAGS_GRID:
        engine.USER_HISTORY_TOP_TAGS = n_tags
        label = f"USER_HISTORY_TOP_TAGS={n_tags}"
        print(f"  Running {label}...", end="", flush=True)
        t0 = time.time()
        result = evaluate_condition(
            engine=engine,
            test_cases=test_cases,
            condition="two_stage",
            ks=DEFAULT_K_VALUES,
            eval_window=EVAL_WINDOW,
            search_engine=search_engine,
        )
        elapsed = time.time() - t0
        ts = result.tag_sim_at_k.get(TARGET_K, 0.0)
        print(f" done in {elapsed:.1f}s")
        print(_row(label, result))
        sweep2_results.append((n_tags, result))
        if ts > best_ts2:
            best_ts2 = ts
            best_top_tags = n_tags

    engine.USER_HISTORY_TOP_TAGS = 30  # restore default

    print()
    print(f"  >> Best USER_HISTORY_TOP_TAGS = {best_top_tags}  (TagSim@{TARGET_K} = {best_ts2:.4f})")

    _header("Grid Search Summary")
    print(f"  Sweep 1 (two_stage_hybrid -- STAGE2_PREFILTER_MULT)")
    for mult, r in sweep1_results:
        marker = " <-- best" if mult == best_mult else ""
        print(_row(f"  mult={mult}", r) + marker)

    print()
    print(f"  Sweep 2 (two_stage -- USER_HISTORY_TOP_TAGS)")
    for n_tags, r in sweep2_results:
        marker = " <-- best" if n_tags == best_top_tags else ""
        print(_row(f"  top_tags={n_tags}", r) + marker)

    print()
    print(f"  Recommended settings:")
    print(f"    two_stage_hybrid: STAGE2_PREFILTER_MULT = {best_mult}")
    print(f"    two_stage:        USER_HISTORY_TOP_TAGS = {best_top_tags}")


if __name__ == "__main__":
    main()
