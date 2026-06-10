"""
CLI entry point for the recipe recommender system.

    python main.py search "chicken tikka masala" --mode hybrid --top-k 10
    python main.py recommend --user-id 12345 --top-k 10
    python main.py compare "easy chicken dinner" --users 100526 1001924 1177249 --alpha 0.0 --top-k 7
    python main.py index
    python main.py app
"""

import argparse
import sys


def cmd_search(args):
    from core.config import load_settings, get_es_client
    from search.engine import SearchEngine, SearchMode

    settings = load_settings()
    es = get_es_client(settings)
    engine = SearchEngine(settings, es)

    mode = SearchMode[args.mode.upper()]
    result = engine.run(query=args.query, mode=mode, top_k=args.top_k)

    for i, r in enumerate(result.results, 1):
        print(f"{i:>3}. [{r.final_score:.4f}] {r.source.get('name', 'Unknown')}")


def cmd_recommend(args):
    from core.config import load_settings, get_es_client
    from search.engine import SearchEngine
    from recommender.engine import RecommenderEngine

    settings = load_settings()
    es = get_es_client(settings)
    search_engine = SearchEngine(settings, es)
    rec = RecommenderEngine(settings, search_engine=search_engine)
    result = rec.recommend_for_user(user_id=args.user_id, top_k=args.top_k)

    for i, r in enumerate(result.results, 1):
        print(f"{i:>3}. {r.source.get('name', 'Unknown')}")


def cmd_compare(args):
    from collections import defaultdict
    from core.config import load_settings, get_es_client
    from search.engine import SearchEngine, SearchMode
    from recommender.engine import RecommenderEngine

    settings = load_settings()
    es       = get_es_client(settings)
    search_engine = SearchEngine(settings, es)
    rec_engine    = RecommenderEngine(settings, search_engine=search_engine)

    query = args.query
    alpha = args.alpha
    k     = args.top_k
    mode  = SearchMode[args.mode.upper()]

    W = 72  # column width for the divider
    print()
    print("=" * W)
    print(f'  Query : "{query}"')
    print(f"  Alpha : {alpha:.2f}  |  Mode: {mode.value}  |  k={k}")
    print("=" * W)

    appearances: dict = defaultdict(list)  # recipe_id → [(user_id, rank)]
    user_results: dict = {}

    for uid in args.users:
        profile_data = rec_engine.build_profile_for_user(uid)
        if profile_data is None:
            print(f"\n  USER {uid}: could not build profile (too few reviews or no embeddings)")
            continue

        profile, user_tags = profile_data
        result = search_engine.search(
            query=query if query.strip() else None,
            user_embedding=profile.embedding,
            user_tag_affinity=profile.tag_affinity,
            user_tags=user_tags,
            alpha=alpha,
            affinity_weight=0.25,
            mode=mode,
            top_k=k,
            candidate_pool=500,
            exclude_ids=profile.rated_recipe_ids,
        )

        user_results[uid] = result.results

        print(f"\n  USER {uid}  ({profile.review_count} reviews)")

        # Split by polarity: high affinity for a negative tag ≠ liking that attribute.
        culinary_tags  = settings.culinary_tags
        negative_tags  = settings.negative_tags
        tagged = sorted(zip(culinary_tags, profile.tag_affinity.tolist()),
                        key=lambda x: x[1], reverse=True)
        pos = [(t.replace("_", " "), v) for t, v in tagged
               if t not in negative_tags][:4]
        neg = [(t.replace("_", " "), v) for t, v in reversed(tagged)
               if v < -0.01 and t in negative_tags][:3]
        if pos:
            print("  Likes:    " + "  ".join(f"{t} +{v:.2f}" for t, v in pos))
        if neg:
            print("  Dislikes: " + "  ".join(f"{t} {v:.2f}" for t, v in neg))

        if user_tags:
            tag_sample = sorted(user_tags)[:10]
            print("  Tags:     " + ", ".join(tag_sample))

        recent_highs = sorted(
            [r for r in profile.reviews if r.star_rating >= 4],
            key=lambda r: r.timestamp, reverse=True,
        )[:4]
        if recent_highs:
            parts = []
            for rev in recent_highs:
                meta = rec_engine.get_recipe_by_id(rev.recipe_id)
                name = (meta.get("name", f"#{rev.recipe_id}") if meta else f"#{rev.recipe_id}").title()[:28]
                parts.append(f"{name} ★{int(rev.star_rating)}")
            print("  Recent ★★★★+: " + " · ".join(parts))

        print(f"  {'#':<4} {'Recipe':<42} {'Sim':>6}  {'Qual':>5}  {'Score':>7}")
        print(f"  {'-'*4} {'-'*42} {'-'*6}  {'-'*5}  {'-'*7}")

        for rank, r in enumerate(result.results, 1):
            name  = r.source.get("name", "Unknown").title()[:42]
            score = getattr(r, "final_score", getattr(r, "similarity", 0.0))
            sim   = getattr(r, "semantic_sim", getattr(r, "similarity", 0.0))
            qual  = getattr(r, "quality_score", 0.0)
            rid   = str(r.recipe_id)
            print(f"  {rank:<4} {name:<42} {sim:>6.3f}  {qual:>5.2f}  {score:>7.3f}")
            appearances[rid].append((uid, rank))

    print()
    print("-" * W)
    print("  OVERLAP ANALYSIS")
    print("-" * W)

    n_users = len(user_results)
    in_all  = [(rid, apps) for rid, apps in appearances.items() if len(apps) == n_users]
    in_some = [(rid, apps) for rid, apps in appearances.items() if 1 < len(apps) < n_users]
    unique  = [(rid, apps) for rid, apps in appearances.items() if len(apps) == 1]

    def _name(rid):
        for uid, results in user_results.items():
            for r in results:
                if str(r.recipe_id) == rid:
                    return r.source.get("name", "Recipe #" + rid).title()[:45]
        return "Recipe #" + rid

    if in_all:
        print(f"\n  In all {n_users} users:")
        for rid, apps in sorted(in_all, key=lambda x: min(r for _, r in x[1])):
            ranks = ", ".join(f"#{r} for {u}" for u, r in apps)
            print(f"    • {_name(rid):<45}  [{ranks}]")
    else:
        print(f"\n  In all {n_users} users:  (none)")

    if in_some:
        print(f"\n  Shared by some users:")
        for rid, apps in sorted(in_some, key=lambda x: -len(x[1])):
            uids  = ", ".join(str(u) for u, _ in apps)
            ranks = ", ".join(f"#{r}" for _, r in apps)
            print(f"    • {_name(rid):<45}  users [{uids}]  ranks [{ranks}]")

    print(f"\n  Unique to one user:")
    by_user: dict = defaultdict(list)
    for rid, apps in unique:
        by_user[apps[0][0]].append((apps[0][1], _name(rid)))
    for uid in args.users:
        items = sorted(by_user.get(uid, []))
        if items:
            names = ", ".join(f"#{r} {n}" for r, n in items)
            print(f"    User {uid}: {names}")

    print()


def cmd_index(args):
    from core.config import load_settings, get_es_client
    from search.indexer import run_ingestion

    settings = load_settings()
    es = get_es_client(settings)
    run_ingestion(settings, es)


def cmd_app(args):
    import subprocess
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app/main.py"], check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recipe Recommender System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_search = sub.add_parser("search", help="Run a search query")
    p_search.add_argument("query", type=str)
    p_search.add_argument("--mode", default="hybrid",
                          choices=["lexical", "semantic", "quality", "hybrid", "ablation_no_sem"])
    p_search.add_argument("--top-k", type=int, default=10)
    p_search.set_defaults(func=cmd_search)

    p_rec = sub.add_parser("recommend", help="Get personalized recommendations")
    p_rec.add_argument("--user-id", type=str, required=True)
    p_rec.add_argument("--top-k", type=int, default=10)
    p_rec.set_defaults(func=cmd_recommend)

    p_cmp = sub.add_parser("compare", help="Compare personalized search results across users")
    p_cmp.add_argument("query", type=str, help="Search query (use '' for pure recommendation)")
    p_cmp.add_argument("--users", nargs="+", required=True, metavar="USER_ID",
                       help="Two or more user IDs to compare")
    p_cmp.add_argument("--alpha", type=float, default=0.0,
                       help="Query↔User blend (0=pure user, 1=pure query). Default: 0.0")
    p_cmp.add_argument("--top-k", type=int, default=7)
    p_cmp.add_argument("--mode", default="hybrid",
                       choices=["lexical", "semantic", "quality", "hybrid", "ablation_no_sem"])
    p_cmp.set_defaults(func=cmd_compare)

    p_idx = sub.add_parser("index", help="Ingest recipes into Elasticsearch")
    p_idx.set_defaults(func=cmd_index)

    p_app = sub.add_parser("app", help="Launch Streamlit frontend")
    p_app.set_defaults(func=cmd_app)

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
