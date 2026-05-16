"""
main.py
-------
Unified CLI entry point for the recipe recommender system.

Usage:
    python main.py search "chicken tikka masala" --mode hybrid --top-k 10
    python main.py recommend --user-id 12345 --top-k 10
    python main.py index          # ingest recipes into Elasticsearch
    python main.py app            # launch Streamlit frontend
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
    results = engine.run(query=args.query, mode=mode, top_k=args.top_k)

    for i, r in enumerate(results, 1):
        print(f"{i:>3}. [{r.get('score', 0):.4f}] {r.get('name', 'Unknown')}")


def cmd_recommend(args):
    from core.config import load_settings
    from recommender.session import SessionRecommender

    settings = load_settings()
    rec = SessionRecommender(settings)
    results = rec.recommend(user_id=args.user_id, top_k=args.top_k)

    for i, r in enumerate(results, 1):
        print(f"{i:>3}. {r.get('name', 'Unknown')}")


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

    # search
    p_search = sub.add_parser("search", help="Run a search query")
    p_search.add_argument("query", type=str)
    p_search.add_argument("--mode", default="hybrid",
                          choices=["lexical", "semantic", "quality", "hybrid", "ablation_no_sem"])
    p_search.add_argument("--top-k", type=int, default=10)
    p_search.set_defaults(func=cmd_search)

    # recommend
    p_rec = sub.add_parser("recommend", help="Get personalized recommendations")
    p_rec.add_argument("--user-id", type=str, required=True)
    p_rec.add_argument("--top-k", type=int, default=10)
    p_rec.set_defaults(func=cmd_recommend)

    # index
    p_idx = sub.add_parser("index", help="Ingest recipes into Elasticsearch")
    p_idx.set_defaults(func=cmd_index)

    # app
    p_app = sub.add_parser("app", help="Launch Streamlit frontend")
    p_app.set_defaults(func=cmd_app)

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
