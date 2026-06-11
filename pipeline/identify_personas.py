"""
Offline script to identify one representative user per persona archetype via greedy
exclusive assignment (highest-confidence archetype first, so no user appears twice).

    python -m pipeline.identify_personas
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, FrozenSet, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.config import load_settings


CUISINE_GROUPS: Dict[str, List[str]] = {
    "mexican":       ["mexican", "tex-mex", "southwest-american"],
    "italian":       ["italian", "pasta", "pizza"],
    "asian":         ["asian", "chinese", "japanese", "thai", "korean",
                      "vietnamese", "indian", "pakistani"],
    "american":      ["american", "southern-united-states", "midwestern-united-states"],
    "french":        ["french"],
    "greek":         ["greek"],
    "middle_eastern":["middle-eastern"],
}

INTERNATIONAL_CUISINE_TAGS: FrozenSet[str] = frozenset(
    t for tags in CUISINE_GROUPS.values() for t in tags
)

BAKING_TAGS: FrozenSet[str] = frozenset([
    "desserts", "cookies-and-brownies", "cakes", "breads", "pies-and-tarts",
    "candy", "muffins", "brownies", "baking", "chocolate",
])

HEALTH_TAGS: FrozenSet[str] = frozenset([
    "healthy", "low-calorie", "low-in-something", "low-fat", "low-carb",
    "low-sodium", "low-protein", "diabetic", "vegetarian", "vegan",
    "low-cholesterol",
])

COMFORT_TAGS: FrozenSet[str] = frozenset([
    "soups-stews", "comfort-food", "casseroles", "stews", "chowders",
    "pot-roast", "one-dish-meal",
])

QUICK_TAGS: FrozenSet[str] = frozenset([
    "30-minutes-or-less", "15-minutes-or-less", "60-minutes-or-less",
])

HOLIDAY_TAGS: FrozenSet[str] = frozenset([
    "holiday-event", "christmas", "thanksgiving", "halloween", "hanukkah",
    "easter", "valentines-day", "new-years", "independence-day",
    "st-patricks-day", "labor-day",
])



def _parse_tags(raw) -> FrozenSet[str]:
    """Normalise tags_clean regardless of whether it was stored as list, string, or array."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return frozenset()
    if isinstance(raw, (list, np.ndarray)):
        return frozenset(str(t).strip() for t in raw if t)
    return frozenset(str(raw).split())


def load_data(s) -> Tuple[pd.DataFrame, Dict[int, FrozenSet[str]], Dict[int, float]]:
    """Returns (reviews_df, recipe_tags dict, recipe_minutes dict)."""
    print("  Loading gold reviews...")
    reviews = pd.read_parquet(s.gold_reviews_path,
                              columns=["user_id", "recipe_id", "rating", "date"])
    date_col = "date" if "date" in reviews.columns else "submitted"
    reviews = pd.read_parquet(s.gold_reviews_path)
    if "date" not in reviews.columns and "submitted" in reviews.columns:
        reviews = reviews.rename(columns={"submitted": "date"})
    reviews["date"] = pd.to_datetime(reviews["date"], errors="coerce")
    reviews = reviews.dropna(subset=["date"])
    reviews["user_id"]   = reviews["user_id"].astype(str)
    reviews["recipe_id"] = reviews["recipe_id"].astype(int)
    reviews = reviews[["user_id", "recipe_id", "rating", "date"]].copy()

    print("  Loading recipe tags + minutes...")
    rdf = pd.read_parquet(s.recipes_path)
    if "recipe_id" not in rdf.columns:  # may have been stored as the DataFrame index
        rdf = rdf.reset_index().rename(columns={"index": "recipe_id"})
    rdf["recipe_id"] = rdf["recipe_id"].astype(int)

    recipe_tags: Dict[int, FrozenSet[str]] = {}
    recipe_minutes: Dict[int, float] = {}
    for _, row in rdf.iterrows():
        rid = int(row["recipe_id"])
        recipe_tags[rid] = _parse_tags(row.get("tags_clean"))
        mins = row.get("minutes")
        recipe_minutes[rid] = float(mins) if pd.notna(mins) and float(mins) > 0 else np.nan

    return reviews, recipe_tags, recipe_minutes

def compute_user_features(
    reviews: pd.DataFrame,
    recipe_tags: Dict[int, FrozenSet[str]],
    recipe_minutes: Dict[int, float],
    min_reviews: int = 15,
) -> pd.DataFrame:
    """Build one row per eligible user with all features needed for archetype scoring."""
    print("  Computing per-user features...")

    rows = []
    for uid, grp in reviews.groupby("user_id"):
        n = len(grp)
        if n < min_reviews:
            continue

        ratings   = grp["rating"].values
        dates     = grp["date"]
        recipe_ids = grp["recipe_id"].values

        mean_rating    = float(np.mean(ratings))
        frac_negative  = float(np.mean(ratings <= 3))
        median_date    = dates.median()
        max_date       = dates.max()

        tag_group_hits     = defaultdict(int)   # reviews touching each tag group
        cuisine_group_hits = defaultdict(int)
        n_with_tags = 0

        minutes_list = []

        for rid, rating in zip(recipe_ids, ratings):
            rtags = recipe_tags.get(rid, frozenset())
            if rtags:
                n_with_tags += 1

            if rtags & BAKING_TAGS:
                tag_group_hits["baking"] += 1
            if rtags & HEALTH_TAGS:
                tag_group_hits["health"] += 1
            if rtags & COMFORT_TAGS:
                tag_group_hits["comfort"] += 1
            if rtags & QUICK_TAGS:
                tag_group_hits["quick"] += 1
            if rtags & HOLIDAY_TAGS:
                tag_group_hits["holiday"] += 1

            for cname, ctags in CUISINE_GROUPS.items():
                if rtags & frozenset(ctags):
                    cuisine_group_hits[cname] += 1

            m = recipe_minutes.get(rid, np.nan)
            if not np.isnan(m) and m < 600:  # cap outliers
                minutes_list.append(m)

        denom = max(n, 1)

        baking_frac   = tag_group_hits["baking"]  / denom
        health_frac   = tag_group_hits["health"]  / denom
        comfort_frac  = tag_group_hits["comfort"] / denom
        quick_frac    = tag_group_hits["quick"]   / denom
        holiday_frac  = tag_group_hits["holiday"] / denom
        mean_minutes  = float(np.mean(minutes_list)) if minutes_list else np.nan

        cuisine_fracs = {c: v / denom for c, v in cuisine_group_hits.items()}
        dominant_cuisine = max(cuisine_fracs, key=cuisine_fracs.get) if cuisine_fracs else ""
        dominant_cuisine_frac = cuisine_fracs.get(dominant_cuisine, 0.0)
        cuisine_diversity = sum(1 for v in cuisine_fracs.values() if v > 0.05)

        rows.append({
            "user_id":              uid,
            "review_count":         n,
            "mean_rating":          mean_rating,
            "frac_negative":        frac_negative,
            "median_date":          median_date,
            "max_date":             max_date,
            "baking_frac":          baking_frac,
            "health_frac":          health_frac,
            "comfort_frac":         comfort_frac,
            "quick_frac":           quick_frac,
            "holiday_frac":         holiday_frac,
            "mean_minutes":         mean_minutes,
            "dominant_cuisine":     dominant_cuisine,
            "dominant_cuisine_frac":dominant_cuisine_frac,
            "cuisine_diversity":    cuisine_diversity,
        })

    return pd.DataFrame(rows)


EPOCH = pd.Timestamp("2000-01-01")
NOW   = pd.Timestamp.now(tz=None)

def _score_specialist(row) -> float:
    if not (20 <= row.review_count <= 70):
        return np.nan
    if row.dominant_cuisine_frac < 0.40:
        return np.nan
    return row.dominant_cuisine_frac

def _score_baker(row) -> float:
    if row.baking_frac < 0.40:
        return np.nan
    return row.baking_frac

def _score_health(row) -> float:
    if row.health_frac < 0.30:
        return np.nan
    return row.health_frac

def _score_comfort(row) -> float:
    if row.comfort_frac < 0.30:
        return np.nan
    # Comfort food users tend not to be health-focused
    return row.comfort_frac - 0.5 * row.health_frac

def _score_weeknight(row) -> float:
    if row.quick_frac < 0.35:
        return np.nan
    if pd.isna(row.mean_minutes) or row.mean_minutes > 55:
        return np.nan
    # Combine quick_frac with low mean_minutes (normalised to [0,1])
    minutes_score = max(0.0, 1.0 - row.mean_minutes / 60.0)
    return 0.6 * row.quick_frac + 0.4 * minutes_score

def _score_global_foodie(row) -> float:
    if row.cuisine_diversity < 3:
        return np.nan
    if row.review_count < 35:
        return np.nan
    # diversity × log(reviews) — rewards breadth at volume
    return row.cuisine_diversity * np.log1p(row.review_count)

def _score_critic(row) -> float:
    if row.frac_negative < 0.18:
        return np.nan
    if row.review_count < 20:
        return np.nan
    return row.frac_negative

def _score_lapsed(row) -> float:
    if row.review_count < 20:
        return np.nan
    median_dt = pd.Timestamp(row.median_date)
    if pd.isna(median_dt) or median_dt >= pd.Timestamp("2010-01-01"):
        return np.nan
    # Older median date → higher score
    days_old = (NOW - median_dt).days
    return days_old / 10_000.0

def _score_casual(row) -> float:
    if not (15 <= row.review_count <= 25):
        return np.nan
    return 1.0 / row.review_count  # closest to 15 scores highest

def _score_holiday(row) -> float:
    if row.holiday_frac < 0.22:
        return np.nan
    return row.holiday_frac


ARCHETYPES: List[Tuple[str, callable, str]] = [
    ("The Specialist",       _score_specialist,  "dominant_cuisine_frac"),
    ("The Baker",            _score_baker,        "baking_frac"),
    ("The Health Enthusiast",_score_health,       "health_frac"),
    ("The Comfort Food Devotee", _score_comfort,  "comfort_frac"),
    ("The Weeknight Cook",   _score_weeknight,    "quick_frac"),
    ("The Global Foodie",    _score_global_foodie, "cuisine_diversity"),
    ("The Critic",           _score_critic,       "frac_negative"),
    ("The Lapsed Reviewer",  _score_lapsed,       "median_date"),
    ("The Casual",           _score_casual,       "review_count"),
    ("The Holiday Cook",     _score_holiday,      "holiday_frac"),
]

def assign_archetypes(
    features: pd.DataFrame,
    top_n: int = 3,
) -> Dict[str, pd.DataFrame]:
    """
    Greedily assign each user to their best-fit archetype.

    Order of assignment is by confidence margin (gap between rank-1 and rank-2
    scores), so the most clear-cut picks go first and aren't stolen by weaker ones.
    Returns {archetype_name: top_n candidates DataFrame}.
    """
    features = features.copy()
    excluded: set = set()
    results: Dict[str, pd.DataFrame] = {}

    scored: Dict[str, pd.Series] = {}
    for name, fn, _ in ARCHETYPES:
        scored[name] = features.apply(fn, axis=1)

    def _margin(name: str) -> float:
        s = scored[name].dropna().sort_values(ascending=False)
        if len(s) < 2:
            return float(s.iloc[0]) if len(s) == 1 else 0.0
        return float(s.iloc[0] - s.iloc[1])

    order = sorted([a[0] for a in ARCHETYPES], key=_margin, reverse=True)

    for name in order:
        fn       = next(fn for n, fn, _ in ARCHETYPES if n == name)
        stat_col = next(s for n, _, s in ARCHETYPES if n == name)

        eligible = features[~features["user_id"].isin(excluded)].copy()
        eligible["_score"] = eligible.apply(fn, axis=1)
        eligible = eligible.dropna(subset=["_score"])
        eligible = eligible.sort_values("_score", ascending=False).head(top_n)

        if eligible.empty:
            results[name] = eligible
            continue

        top_user = eligible.iloc[0]["user_id"]
        excluded.add(top_user)

        results[name] = eligible[
            ["user_id", "review_count", "mean_rating", "frac_negative",
             "dominant_cuisine", "dominant_cuisine_frac",
             "baking_frac", "health_frac", "comfort_frac", "quick_frac",
             "holiday_frac", "cuisine_diversity", "mean_minutes",
             "median_date", "max_date", "_score"]
        ].copy()

    return results



def print_results(results: Dict[str, pd.DataFrame]) -> None:
    divider = "=" * 80
    for name, df in results.items():
        print(f"\n{divider}")
        print(f"  {name.upper()}")
        print(divider)
        if df.empty:
            print("  No eligible users found — consider relaxing filters.")
            continue
        for rank, (_, row) in enumerate(df.iterrows(), 1):
            marker = "  >>> TOP PICK <<<" if rank == 1 else ""
            print(f"\n  Rank {rank}{marker}")
            print(f"    user_id        : {row['user_id']}")
            print(f"    review_count   : {int(row['review_count'])}")
            print(f"    mean_rating    : {row['mean_rating']:.2f}")
            print(f"    frac_negative  : {row['frac_negative']:.1%}  (<=3 stars)")
            print(f"    score          : {row['_score']:.4f}")
            print(f"    dominant_cuisine: {row['dominant_cuisine']} "
                  f"({row['dominant_cuisine_frac']:.1%})")
            print(f"    baking_frac    : {row['baking_frac']:.1%}")
            print(f"    health_frac    : {row['health_frac']:.1%}")
            print(f"    comfort_frac   : {row['comfort_frac']:.1%}")
            print(f"    quick_frac     : {row['quick_frac']:.1%}")
            print(f"    holiday_frac   : {row['holiday_frac']:.1%}")
            print(f"    cuisine_diversity: {int(row['cuisine_diversity'])} groups")
            mins_str = "N/A" if pd.isna(row["mean_minutes"]) else f"{row['mean_minutes']:.0f} min"
            print(f"    mean_minutes   : {mins_str}")
            if pd.notna(row["median_date"]):
                print(f"    median_date    : {pd.Timestamp(row['median_date']).date()}")
            if pd.notna(row["max_date"]):
                print(f"    last_review    : {pd.Timestamp(row['max_date']).date()}")

    print(f"\n{divider}")
    print("  ASSIGNMENT SUMMARY (copy these IDs into app/main.py PERSONAS dict)")
    print(divider)
    for name, df in results.items():
        if df.empty:
            print(f"  {name:<28}  NO CANDIDATE FOUND")
        else:
            row = df.iloc[0]
            print(f"  {name:<28}  {row['user_id']}  "
                  f"({int(row['review_count'])} reviews, "
                  f"{row['mean_rating']:.2f} stars, score={row['_score']:.4f})")

def main() -> None:
    print("Persona Identification Script")
    print("=" * 80)

    s = load_settings()

    print("\n[1/3] Loading data...")
    reviews, recipe_tags, recipe_minutes = load_data(s)

    print("\n[2/3] Computing per-user features (min 15 reviews)...")
    features = compute_user_features(reviews, recipe_tags, recipe_minutes, min_reviews=15)
    print(f"      {len(features):,} eligible users")

    print("\n[3/3] Scoring archetypes and assigning...")
    results = assign_archetypes(features, top_n=3)

    print_results(results)


if __name__ == "__main__":
    main()
