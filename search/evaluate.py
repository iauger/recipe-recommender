"""Ablation evaluation for all five search modes across a stratified query set."""

import sys
import io
if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding='utf-8')

from typing import Any, Dict, List

import numpy as np

from search.engine import SearchEngine, SearchMode


QUERIES_BY_TIER: Dict[str, List[str]] = {
    "high": [
        "easy chicken dinner under 45 mins",     # protein + time + course
        "vegan thanksgiving sides",              # dietary + occasion + course
        "low carb italian beef skillet",         # dietary + cuisine + protein + dish_type
        "spicy pork tacos",                      # taste + protein + dish_type
    ],
    "medium": [
        "quick healthy breakfast under 15 mins", # course + time (healthy maps to tag)
        "quick chicken dish",                    # protein only
        "italian pasta dinner",                  # cuisine + course
    ],
    "low": [
        "something cozy and warming",            # pure vibe
        "weeknight comfort food",                # exploratory
        "impressive but not too hard",           # exploratory
    ],
}

ALL_QUERIES: List[str] = [
    q for queries in QUERIES_BY_TIER.values() for q in queries
]

QUERY_TIER: Dict[str, str] = {
    q: tier
    for tier, queries in QUERIES_BY_TIER.items()
    for q in queries
}


def score_result(result: Any, intent: Dict[str, Any]) -> float:
    """
    Rule-based relevance proxy. Accepts both raw ES hits and RerankedResult objects.

    Returns 0.0 immediately if hard constraints (max_minutes, dietary) are violated.
    Awards bonuses for structured matches; penalises unmatched residual tokens.
    """
    source = result["_source"] if isinstance(result, dict) else result.source

    score = 0.0

    tags = source.get("tags_clean", [])
    if tags is None:
        tags = []
    if isinstance(tags, str):
        tags = [tags]

    tags_norm = {str(tag).strip().lower() for tag in tags}
    name_text = str(source.get("name", "")).lower()
    ingredients_text = str(source.get("ingredients_clean", "")).lower()
    minutes = source.get("minutes")

    max_minutes = intent.get("max_minutes")
    if max_minutes is not None and minutes is not None:
        try:
            if float(minutes) > float(max_minutes):
                return 0.0
        except (TypeError, ValueError):
            return 0.0

    for diet in intent.get("dietary_tags", []):
        if str(diet).strip().lower() not in tags_norm:
            return 0.0

    for group in ["cuisines", "methods", "occasions", "courses"]:
        for value in intent.get(group, []):
            if str(value).strip().lower() in tags_norm:
                score += 1.0

    for protein in intent.get("proteins", []):
        protein_norm = str(protein).strip().lower()
        if protein_norm in name_text:
            score += 1.0
        elif protein_norm in ingredients_text:
            score += 0.75

    target_minutes = intent.get("target_minutes")
    if target_minutes is not None and minutes is not None:
        try:
            distance = abs(float(minutes) - float(target_minutes))
            if distance <= 5:
                score += 0.75
            elif distance <= 15:
                score += 0.4
        except (TypeError, ValueError):
            pass

    clean_text = str(intent.get("clean_text", "")).lower()
    structured_tokens = set(
        [str(x).strip().lower() for x in intent.get("proteins", [])]
        + [str(x).strip().lower() for x in intent.get("dietary_tags", [])]
        + [str(x).strip().lower() for x in intent.get("cuisines", [])]
        + [str(x).strip().lower() for x in intent.get("methods", [])]
        + [str(x).strip().lower() for x in intent.get("occasions", [])]
        + [str(x).strip().lower() for x in intent.get("courses", [])]
    )
    leftover_tokens = [
        t.strip()
        for t in clean_text.split()
        if t.strip() and t.strip() not in structured_tokens
    ]

    leftover_match = False
    for token in leftover_tokens:
        if token in name_text:
            score += 0.75
            leftover_match = True
        elif token in ingredients_text:
            score += 0.25
            leftover_match = True

    if leftover_tokens and not leftover_match:
        score -= 0.75

    return max(score, 0.0)


def ndcg_at_k(results: list, intent: Dict[str, Any], k: int = 5) -> float:
    """NDCG@k using score_result as the gain function."""
    gains = [score_result(r, intent) for r in results[:k]]
    dcg   = sum(gain / np.log2(i + 2) for i, gain in enumerate(gains))
    idcg  = sum(gain / np.log2(i + 2) for i, gain in enumerate(sorted(gains, reverse=True)))
    return dcg / idcg if idcg > 0 else 0.0


def precision_at_1(results: list, intent: Dict[str, Any]) -> float:
    """1.0 if the top result has score_result > 0, else 0.0."""
    if not results:
        return 0.0
    return 1.0 if score_result(results[0], intent) > 0.0 else 0.0


def evaluate_query(
    query: str,
    engine: SearchEngine,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Run one query through all modes; return NDCG and P@1 per mode."""
    all_mode_results = engine.run_all_modes(query, top_k=top_k)

    intent = all_mode_results[SearchMode.LEXICAL].intent
    tier   = QUERY_TIER.get(query, "unknown")

    mode_scores: Dict[str, List[float]] = {}
    mode_means:  Dict[str, float] = {}
    mode_p1:     Dict[str, float] = {}

    for mode, search_result in all_mode_results.items():
        scores = [score_result(r, intent) for r in search_result.results[:top_k]]
        mode_scores[mode.value] = scores
        mode_means[mode.value]  = ndcg_at_k(search_result.results, intent, k=top_k)
        mode_p1[mode.value]     = precision_at_1(search_result.results, intent)

    return {
        "query": query,
        "tier": tier,
        "intent": intent,
        "mode_scores": mode_scores,
        "mode_means": mode_means,
        "mode_p1": mode_p1,
        "all_mode_results": all_mode_results,
    }


def evaluate_engine(top_k: int = 5) -> Dict[str, Any]:
    """Run the full ablation evaluation; return aggregated metrics dict."""
    from core.config import load_settings, get_es_client

    s = load_settings()
    es_client = get_es_client(s)
    engine = SearchEngine(s, es_client)

    print(f"Running evaluation: {len(ALL_QUERIES)} queries x {len(SearchMode)} modes\n")

    query_reports = []
    for query in ALL_QUERIES:
        print(f"  Evaluating: {query}")
        report = evaluate_query(query, engine, top_k=top_k)
        query_reports.append(report)

    overall_means: Dict[str, float] = {
        mode.value: float(np.mean([r["mode_means"][mode.value] for r in query_reports]))
        for mode in SearchMode
    }
    overall_p1: Dict[str, float] = {
        mode.value: float(np.mean([r["mode_p1"][mode.value] for r in query_reports]))
        for mode in SearchMode
    }

    tier_means: Dict[str, Dict[str, float]] = {}
    tier_p1:    Dict[str, Dict[str, float]] = {}

    for tier in QUERIES_BY_TIER:
        tier_reports = [r for r in query_reports if r["tier"] == tier]
        tier_means[tier] = {
            mode.value: float(np.mean([r["mode_means"][mode.value] for r in tier_reports]))
            for mode in SearchMode
        }
        tier_p1[tier] = {
            mode.value: float(np.mean([r["mode_p1"][mode.value] for r in tier_reports]))
            for mode in SearchMode
        }

    hybrid_overall = overall_means[SearchMode.HYBRID.value]
    deltas_vs_hybrid: Dict[str, float] = {
        mode.value: hybrid_overall - overall_means[mode.value]
        for mode in SearchMode
        if mode != SearchMode.HYBRID
    }

    return {
        "query_reports": query_reports,
        "overall_means": overall_means,
        "overall_p1": overall_p1,
        "tier_means": tier_means,
        "tier_p1": tier_p1,
        "deltas_vs_hybrid": deltas_vs_hybrid,
        "top_k": top_k,
    }


def print_summary(summary: Dict[str, Any]) -> None:
    """Print evaluation results as five numbered tables."""

    top_k = summary["top_k"]
    modes = [m.value for m in SearchMode]
    col_w = 16
    lbl_w = 46
    sep   = "=" * (lbl_w + col_w * len(modes))
    dash  = "-" * (lbl_w + col_w * len(modes))

    def _query_header() -> str:
        return f"{'Query':<38} {'Tier':<8}" + "".join(f"{m:>{col_w}}" for m in modes)

    def _tier_header() -> str:
        return f"{'':38} {'':8}" + "".join(f"{m:>{col_w}}" for m in modes)

    def _query_rows(key: str) -> None:
        current_tier = None
        for report in summary["query_reports"]:
            tier = report["tier"]
            if tier != current_tier:
                if current_tier is not None:
                    print()
                current_tier = tier
            row = f"{report['query'][:36]:<38} {tier:<8}"
            row += "".join(f"{report[key][m]:>{col_w}.3f}" for m in modes)
            print(row)

    print(f"\n{sep}")
    print(f"TABLE 1: NDCG@{top_k} PER QUERY")
    print(sep)
    print(_query_header())
    print(dash)
    _query_rows("mode_means")

    print(f"\n{sep}")
    print(f"TABLE 2: PRECISION@1 PER QUERY")
    print(sep)
    print(_query_header())
    print(dash)
    _query_rows("mode_p1")

    print(f"\n{sep}")
    print(f"TABLE 3: TIER SUMMARY  (mean NDCG@{top_k} / mean P@1)")
    print(sep)
    print(_tier_header())
    print(dash)

    for tier in QUERIES_BY_TIER:
        ndcg_row = f"{tier + ' (NDCG)':<38} {'':8}"
        ndcg_row += "".join(f"{summary['tier_means'][tier][m]:>{col_w}.3f}" for m in modes)
        p1_row = f"{tier + ' (P@1) ':<38} {'':8}"
        p1_row += "".join(f"{summary['tier_p1'][tier][m]:>{col_w}.3f}" for m in modes)
        print(ndcg_row)
        print(p1_row)
        print()

    print(f"\n{sep}")
    print(f"TABLE 4: OVERALL SUMMARY  (mean across all {len(ALL_QUERIES)} queries)")
    print(sep)
    print(_tier_header())
    print(dash)

    ndcg_row = f"{'MEAN NDCG@' + str(top_k):<38} {'ALL':<8}"
    ndcg_row += "".join(f"{summary['overall_means'][m]:>{col_w}.3f}" for m in modes)
    p1_row = f"{'MEAN P@1':<38} {'ALL':<8}"
    p1_row += "".join(f"{summary['overall_p1'][m]:>{col_w}.3f}" for m in modes)
    print(ndcg_row)
    print(p1_row)

    print(f"\n{sep}")
    print("TABLE 5: HYBRID IMPROVEMENT  (HYBRID - other, positive = HYBRID wins)")
    print(sep)
    print(f"  {'Comparison':<30} {'NDCG delta':>14}  {'P@1 delta':>12}")
    print(f"  {'-'*60}")

    for mode, ndcg_delta in summary["deltas_vs_hybrid"].items():
        p1_delta = (summary["overall_p1"][SearchMode.HYBRID.value]
                    - summary["overall_p1"][mode])
        nd = "+" if ndcg_delta >= 0 else ""
        pd = "+" if p1_delta   >= 0 else ""
        print(f"  HYBRID vs {mode:<22} {nd}{ndcg_delta:>12.3f}  {pd}{p1_delta:>10.3f}")

    print(f"\n{sep}")
    print("KEY FINDING: HYBRID vs ABLATION_NO_SEM delta by tier")
    print("(isolates the marginal contribution of the embedding layer)")
    print(sep)
    print(f"  {'Tier':<10}  {'HYBRID NDCG':>14}  {'NO_SEM NDCG':>14}  "
          f"{'NDCG delta':>12}  {'P@1 delta':>12}")
    print(f"  {'-'*68}")

    for tier in QUERIES_BY_TIER:
        h_ndcg = summary["tier_means"][tier][SearchMode.HYBRID.value]
        a_ndcg = summary["tier_means"][tier][SearchMode.ABLATION_NO_SEM.value]
        h_p1   = summary["tier_p1"][tier][SearchMode.HYBRID.value]
        a_p1   = summary["tier_p1"][tier][SearchMode.ABLATION_NO_SEM.value]
        nd = "+" if (h_ndcg - a_ndcg) >= 0 else ""
        pd = "+" if (h_p1   - a_p1)   >= 0 else ""
        print(f"  {tier:<10}  {h_ndcg:>14.3f}  {a_ndcg:>14.3f}  "
              f"{nd}{h_ndcg - a_ndcg:>10.3f}  {pd}{h_p1 - a_p1:>10.3f}")


def print_query_detail(report: Dict[str, Any], top_k: int = 5) -> None:
    """Print per-mode result lists for a single query (qualitative inspection)."""
    print(f"\n{'='*70}")
    print(f"QUERY : {report['query']}")
    print(f"TIER  : {report['tier']}")
    print(f"INTENT: {report['intent']}")
    print(f"{'='*70}")

    for mode in SearchMode:
        mode_result = report["all_mode_results"][mode]
        scores      = report["mode_scores"][mode.value]
        ndcg        = report["mode_means"][mode.value]
        p1          = report["mode_p1"][mode.value]
        w           = mode_result.weights

        print(f"\n  [{mode.value.upper()}]  tier={mode_result.tier}  "
              f"lex={w['lex']}  align={w['alignment']}  "
              f"sem={w['semantic']}  quality={w['quality']}  "
              f"NDCG={ndcg:.3f}  P@1={p1:.1f}")

        for i, (r, s) in enumerate(
            zip(mode_result.results[:top_k], scores), start=1
        ):
            name = r.source.get("name", "").title()
            mins = r.source.get("minutes")
            print(
                f"    {i}. [{r.final_score:.3f}] {name} ({mins}m)"
                f"  eval={s:.2f}"
                f"  align={r.alignment_score:.2f}"
                f"  sim={r.semantic_sim:.3f}"
                f"  quality={r.quality_score:.2f}"
            )


if __name__ == "__main__":
    summary = evaluate_engine(top_k=5)
    print_summary(summary)

