# app/main.py
# Run with: python -m streamlit run app/main.py

"""
Recipe Search & Recommender -- unified Streamlit UI.

Two input modes, one result interface:
  Search       -- BM25/hybrid IR with optional session seeding.
                  Each result can be added to a session; subsequent searches
                  also show a personalised "Based on your session" section.
  User History -- Personalized recommendations for a real Food.com user.
                  Optional free-text query blends BM25 with the user's
                  embedding centroid (alpha controls the blend).
"""

import sys
import io
if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding='utf-8')

import streamlit as st

from core.config import load_settings, get_es_client
from search.engine import SearchEngine, SearchMode
from search.evaluate import ndcg_at_k
from recommender.engine import RecommenderEngine


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Recipe Search & Recommender",
    page_icon="🍳",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=DM+Sans:wght@300;400;500&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    h1, h2, h3 { font-family: 'Playfair Display', serif; }
    [data-testid="stSidebar"] { background-color: #3d4a35; }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] p { color: #f5f0e8 !important; }
    [data-testid="stSidebar"] .stMarkdown { color: #f5f0e8; }
    .app-header { padding: 8px 0 24px 0; border-bottom: 2px solid #d8d0c0; margin-bottom: 28px; }
    .app-title { font-family: 'Playfair Display', serif; font-size: 36px; font-weight: 700; color: #3d4a35; margin: 0; line-height: 1.1; }
    .app-subtitle { font-size: 14px; color: #8a7a6a; margin-top: 6px; }
    .compare-table { width: 100%; border-collapse: collapse; font-size: 13px; margin-bottom: 24px; }
    .compare-table th { background: #3d4a35; color: #f5f0e8; padding: 10px 14px; text-align: left; font-weight: 500; }
    .compare-table td { padding: 9px 14px; border-bottom: 1px solid #e8e0d4; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading search engine...")
def get_engine():
    s = load_settings()
    es_client = get_es_client(s)
    return SearchEngine(s, es_client)


@st.cache_resource(show_spinner="Loading recommender engine...")
def get_recommender():
    s = load_settings()
    engine = get_engine()
    return RecommenderEngine(s, search_engine=engine)


@st.cache_data(show_spinner="Loading eligible users...")
def get_eligible_users(min_reviews: int):
    return get_recommender().get_eligible_users(min_reviews=min_reviews)


# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

CARD_STYLE   = "background:#faf8f4;border:1px solid #ddd5c5;border-radius:12px;padding:18px 22px;margin-bottom:6px;box-shadow:0 2px 8px rgba(0,0,0,0.05);"
RANK_STYLE   = "font-size:11px;font-weight:500;letter-spacing:0.1em;color:#8a7a62;text-transform:uppercase;margin-bottom:4px;"
NAME_STYLE   = "font-size:18px;font-weight:700;color:#3d4a35;margin-bottom:6px;line-height:1.3;"
META_STYLE   = "font-size:13px;color:#6a6258;margin-bottom:10px;"
SROW_STYLE   = "display:flex;gap:10px;flex-wrap:wrap;margin-bottom:10px;"
PILL_STYLE   = "font-size:11px;font-weight:500;padding:3px 10px;border-radius:20px;background:#ede8dc;color:#5a4e3a;"
PILLHI_STYLE = "font-size:11px;font-weight:500;padding:3px 10px;border-radius:20px;background:#3d4a35;color:#f5f0e8;"
TAGROW_STYLE = "display:flex;gap:6px;flex-wrap:wrap;margin-top:8px;"
CHIP_STYLE   = "font-size:11px;padding:2px 8px;border-radius:4px;background:#f0ebe0;color:#3d4a35;border:1px solid #d0c8b8;"
ING_STYLE    = "font-size:12px;color:#7a7060;margin-top:6px;line-height:1.6;"
INFO_BOX     = "background:#f5f0e8;border-radius:8px;padding:8px 14px;margin-bottom:16px;font-size:12px;color:#6a6258;"
SEED_BTN_STYLE = "margin-bottom:14px;"

# Food.com taxonomy parent-node tags — category names, not descriptive.
# They appear on nearly every recipe and carry no recipe-specific signal.
_STRUCTURAL_TAGS = frozenset([
    "time-to-make", "course", "main-ingredient", "cuisine",
    "preparation", "occasion", "dietary", "technique", "equipment",
])

MODE_LABELS = {
    SearchMode.HYBRID:          "Hybrid (full pipeline)",
    SearchMode.LEXICAL:         "Lexical (BM25 only)",
    SearchMode.SEMANTIC:        "Semantic (embeddings only)",
    SearchMode.QUALITY:         "Quality (rating signal only)",
    SearchMode.ABLATION_NO_SEM: "Ablation (no semantic)",
}

MODE_DESCRIPTIONS = {
    SearchMode.HYBRID:
        "Full pipeline: BM25 + alignment + embedding similarity + quality.",
    SearchMode.LEXICAL:
        "Elasticsearch BM25 only. Baseline -- no reranking.",
    SearchMode.SEMANTIC:
        "Embedding cosine similarity + quality. No structured rules.",
    SearchMode.QUALITY:
        "Quality-ranked by Bayesian rating. Query-agnostic.",
    SearchMode.ABLATION_NO_SEM:
        "Hybrid minus embedding similarity. Isolates embedding contribution.",
}

# ---------------------------------------------------------------------------
# Persona definitions (hardcoded Food.com user IDs, identified offline)
# ---------------------------------------------------------------------------

PERSONAS = {
    "The Global Foodie": {
        "user_id": 192581,
        "tagline": "Explores cuisines from every corner of the world, rarely repeating the same style twice.",
        "traits": ["World cuisines", "High diversity", "Frequent reviewer"],
    },
    "The Critic": {
        "user_id": 716215,
        "tagline": "Holds recipes to a high standard — over a quarter of reviews are 3 stars or below.",
        "traits": ["High standards", "Critical eye", "Discerning palate"],
    },
    "The Baker": {
        "user_id": 776846,
        "tagline": "Almost exclusively bakes — desserts, breads, and pastries dominate the review history.",
        "traits": ["Baking focused", "Sweets & breads", "Consistent rater"],
    },
    "The Comfort Food Devotee": {
        "user_id": 361621,
        "tagline": "Gravitates toward soups, stews, casseroles, and hearty home-cooked meals.",
        "traits": ["Comfort classics", "Hearty dishes", "Home cooking"],
    },
    "The Weeknight Cook": {
        "user_id": 342784,
        "tagline": "Keeps it practical — nearly every recipe reviewed can be made in 30 minutes or less.",
        "traits": ["Quick recipes", "30 min or less", "Efficient cook"],
    },
    "The Specialist": {
        "user_id": 121553,
        "tagline": "Deep expertise in one cuisine — focused on a single culinary tradition above all others.",
        "traits": ["Cuisine specialist", "Deep focus", "Expert palate"],
    },
    "The Lapsed Reviewer": {
        "user_id": 15609,
        "tagline": "Most active in the early 2000s, with reviews that reflect Food.com's founding era.",
        "traits": ["Early adopter", "Pre-2010 activity", "Rare recent reviews"],
    },
    "The Health Enthusiast": {
        "user_id": 1272254,
        "tagline": "Seeks out healthy, low-calorie, and vegetarian recipes with minimal overlap with comfort food.",
        "traits": ["Health focused", "Low-calorie", "Vegetarian leaning"],
    },
    "The Casual": {
        "user_id": 1026723,
        "tagline": "Reviews occasionally across a wide range of recipes with no strong specialty or pattern.",
        "traits": ["Infrequent reviewer", "Broad tastes", "No strong niche"],
    },
    "The Holiday Cook": {
        "user_id": 643883,
        "tagline": "Comes alive around the holidays — a large share of reviews are for seasonal and festive recipes.",
        "traits": ["Holiday recipes", "Seasonal cook", "Festive focus"],
    },
}


def _persona_name_for_user(user_id: int) -> "str | None":
    for name, p in PERSONAS.items():
        if p["user_id"] == user_id:
            return name
    return None


# ---------------------------------------------------------------------------
# Session state initialisation (must run before sidebar widgets)
# ---------------------------------------------------------------------------

_STATE_DEFAULTS = {
    "session_seeds":        [],     # list of {recipe_id, name}
    "search_display":       None,   # "search" | "compare" | None
    "last_search_result":   None,   # SearchResult
    "last_search_top_k":    10,
    "last_compare_results": None,   # Dict[SearchMode, SearchResult]
    "last_compare_top_k":   10,
    "last_user_result":     None,   # (user_id, SearchResult, profile, user_tags)
    "cached_user_profile":  None,   # (user_id, profile, user_tags)
}
for _k, _v in _STATE_DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ---------------------------------------------------------------------------
# Shared render helpers
# ---------------------------------------------------------------------------

def _recipe_id_from_result(result) -> "int | None":
    """Return integer recipe_id from either a RerankedResult or KNNResult."""
    try:
        return int(result.recipe_id)
    except (ValueError, TypeError, AttributeError):
        return None


def tier_badge(tier: str) -> str:
    colours = {
        "high":   "background:#fce8e8;color:#b03030;",
        "medium": "background:#f5edd8;color:#8a7535;",
        "low":    "background:#e4ede0;color:#3d4a35;",
    }
    base   = "display:inline-block;font-size:12px;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;padding:4px 14px;border-radius:20px;margin-bottom:16px;"
    colour = colours.get(tier.split("_")[0], "background:#f0ebe0;color:#6a6258;")
    return '<span style="{}{}">{} INTENT</span>'.format(base, colour, tier.upper())


def render_intent_summary(intent: dict) -> None:
    parts = []
    if intent.get("dietary_tags"):
        parts.append("<b>Dietary:</b> " + ", ".join(intent["dietary_tags"]))
    if intent.get("proteins"):
        parts.append("<b>Protein:</b> " + ", ".join(intent["proteins"]))
    if intent.get("cuisines"):
        parts.append("<b>Cuisine:</b> " + ", ".join(intent["cuisines"]))
    if intent.get("courses"):
        parts.append("<b>Course:</b> " + ", ".join(intent["courses"]))
    if intent.get("methods"):
        parts.append("<b>Method:</b> " + ", ".join(intent["methods"]))
    if intent.get("max_minutes"):
        parts.append("<b>Max time:</b> {} mins".format(intent["max_minutes"]))
    if intent.get("taste"):
        parts.append("<b>Taste:</b> " + ", ".join(intent["taste"]))
    if parts:
        st.markdown(
            '<div style="{}">Parsed intent &nbsp;|&nbsp; '.format(INFO_BOX)
            + " &nbsp;&middot;&nbsp; ".join(parts) + "</div>",
            unsafe_allow_html=True,
        )


def _build_source_card(rank, name, mins_str, rating_str, score_html, tag_html, ing_html=""):
    return (
        '<div style="{}">'
        '<div style="{}">{}</div>'
        '<div style="{}">{}</div>'
        '<div style="{}">{} &nbsp;&middot;&nbsp; {}</div>'
        '{}{}{}'
        '</div>'
    ).format(CARD_STYLE,
             RANK_STYLE, "#{:d}".format(rank),
             NAME_STYLE, name,
             META_STYLE, mins_str, rating_str,
             score_html, tag_html, ing_html)


def render_search_card(rank: int, result, show_scores: bool = True) -> None:
    source  = result.source
    name    = source.get("name", "Unknown").title()
    minutes = source.get("minutes")
    rating  = source.get("bayesian_rating")
    mins_str   = "{}m".format(int(float(minutes))) if minutes and float(minutes) > 0 else "?"
    rating_str = "&#9733; {:.2f}".format(float(rating)) if rating else ""

    tags = source.get("tags_clean")
    if tags is None:
        tags = []
    elif isinstance(tags, str):
        tags = tags.split()
    else:
        tags = list(tags)
    tags = [t for t in tags if t not in _STRUCTURAL_TAGS]
    ingredients = source.get("ingredients_clean")
    if ingredients is None:
        ingredients = []
    elif isinstance(ingredients, str):
        ingredients = ingredients.split()
    else:
        ingredients = list(ingredients)

    score_html = ""
    if show_scores and hasattr(result, "final_score"):
        pills = ['<span style="{}">{}</span>'.format(PILLHI_STYLE, "Score {:.3f}".format(result.final_score))]
        if result.alignment_score:
            pills.append('<span style="{}">{}</span>'.format(PILL_STYLE, "Align {:.2f}".format(result.alignment_score)))
        if result.semantic_sim:
            pills.append('<span style="{}">{}</span>'.format(PILL_STYLE, "Sim {:.3f}".format(result.semantic_sim)))
        if result.quality_score:
            pills.append('<span style="{}">{}</span>'.format(PILL_STYLE, "Quality {:.2f}".format(result.quality_score)))
        score_html = '<div style="{}">{}</div>'.format(SROW_STYLE, "".join(pills))

    tag_html = ""
    if tags:
        chips = "".join('<span style="{}">{}</span>'.format(CHIP_STYLE, t.replace("-", " ").replace("_", " ")) for t in tags[:8])
        tag_html = '<div style="{}">{}</div>'.format(TAGROW_STYLE, chips)

    ing_html = ""
    if ingredients:
        ing_clean = [i.replace("_", " ") for i in ingredients[:10]]
        ing_html = '<div style="{}"><b>Ingredients:</b> {}</div>'.format(ING_STYLE, ", ".join(ing_clean))

    st.markdown(_build_source_card(rank, name, mins_str, rating_str, score_html, tag_html, ing_html), unsafe_allow_html=True)


def render_rec_card(rank: int, result) -> None:
    source   = result.source
    name     = source.get("name", "Unknown").title()
    minutes  = source.get("minutes")
    rating   = source.get("bayesian_rating")
    mins_str   = "{}m".format(int(float(minutes))) if minutes and float(minutes) > 0 else "?"
    rating_str = "&#9733; {:.2f}".format(float(rating)) if rating else ""

    tags = source.get("tags_clean")
    if tags is None:
        tags = []
    elif isinstance(tags, str):
        tags = tags.split()
    else:
        tags = list(tags)
    tags = [t for t in tags if t not in _STRUCTURAL_TAGS]
    tag_html = ""
    if tags:
        chips = "".join('<span style="{}">{}</span>'.format(CHIP_STYLE, t.replace("-", " ").replace("_", " ")) for t in tags[:8])
        tag_html = '<div style="{}">{}</div>'.format(TAGROW_STYLE, chips)

    pills = [
        '<span style="{}">{}</span>'.format(PILLHI_STYLE, "Sim {:.3f}".format(result.similarity)),
        '<span style="{}">{}</span>'.format(PILL_STYLE, "Quality {:.2f}".format(result.quality_score)),
    ]
    score_html = '<div style="{}">{}</div>'.format(SROW_STYLE, "".join(pills))

    st.markdown(_build_source_card(rank, name, mins_str, rating_str, score_html, tag_html), unsafe_allow_html=True)


def render_stage1_box(query_tags: set, pool_size: int) -> None:
    if not query_tags:
        return
    tag_str = ", ".join(sorted(query_tags)[:14])
    st.markdown(
        '<div style="{}"><b>Stage 1</b> &nbsp;|&nbsp; {} candidate recipes &nbsp;|&nbsp; tags: {}</div>'.format(
            INFO_BOX, pool_size, tag_str),
        unsafe_allow_html=True,
    )


def _render_seed_button(result, rank: int) -> None:
    """Render an '+ Add to session' button below a search result card."""
    rid = _recipe_id_from_result(result)
    if rid is None:
        return
    seeds_set = {s["recipe_id"] for s in st.session_state.session_seeds}
    if rid in seeds_set:
        st.markdown(
            '<div style="{}"><span style="font-size:12px;color:#2e7d32;">✓ In session</span></div>'.format(SEED_BTN_STYLE),
            unsafe_allow_html=True,
        )
    elif len(st.session_state.session_seeds) < 5:
        btn_col, _ = st.columns([2, 8])
        with btn_col:
            if st.button("＋ Add to session", key="seed_{}_r{}".format(rid, rank), use_container_width=True):
                name = result.source.get("name", "Recipe #{}".format(rid)).title()
                st.session_state.session_seeds.append({"recipe_id": rid, "name": name})
                st.rerun()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 🍳 Recipe Search & Recommender")
    st.markdown("---")

    input_mode = st.radio(
        "Input mode",
        options=["Search", "User History"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    # ── Search controls ──────────────────────────────────────────────────────
    if input_mode == "Search":
        query = st.text_input(
            "Search query",
            placeholder="e.g. easy chicken dinner under 45 mins",
            label_visibility="collapsed",
        )
        st.markdown("#### Search mode")
        selected_mode = st.radio(
            "mode",
            options=list(SearchMode),
            format_func=lambda m: MODE_LABELS[m],
            label_visibility="collapsed",
        )
        top_k      = st.slider("Results", min_value=5, max_value=20, value=10, step=5)
        run_search = st.button("Search", use_container_width=True, type="primary")
        run_compare = st.button("Compare all modes", use_container_width=True)

        # Session seeds management
        seeds = st.session_state.session_seeds
        if seeds:
            st.markdown("---")
            st.markdown("**Session** ({}/5)".format(len(seeds)))
            for s in list(seeds):
                c1, c2 = st.columns([4, 1])
                with c1:
                    st.caption("🌱 {}".format(s["name"][:26]))
                with c2:
                    if st.button("×", key="rm_{}".format(s["recipe_id"])):
                        st.session_state.session_seeds = [
                            x for x in seeds if x["recipe_id"] != s["recipe_id"]
                        ]
                        st.rerun()
            if st.button("Clear session", use_container_width=True):
                st.session_state.session_seeds = []
                st.rerun()

        st.markdown("---")
        st.markdown(
            "<small style='color:#aaa'>{}</small>".format(MODE_DESCRIPTIONS[selected_mode]),
            unsafe_allow_html=True,
        )

    # ── User History controls ────────────────────────────────────────────────
    else:
        st.markdown(
            "Choose a user archetype to explore personalised recommendations "
            "based on a real Food.com reviewer profile."
        )

        persona_names = list(PERSONAS.keys())
        selected_persona_name = st.selectbox(
            "Select archetype",
            options=[None] + persona_names,
            format_func=lambda n: "— Choose an archetype —" if n is None else n,
            label_visibility="collapsed",
        )

        selected_user = PERSONAS[selected_persona_name]["user_id"] if selected_persona_name else None

        query = st.text_input(
            "Optional query",
            placeholder="e.g. quick pasta (leave blank for pure recommendations)",
            label_visibility="collapsed",
        )

        if query.strip():
            alpha = st.slider(
                "Query ↔ User blend",
                min_value=0.0, max_value=1.0, value=0.5, step=0.1,
                help="1.0 = pure text search  ·  0.0 = pure user taste",
            )
        else:
            alpha = 0.0

        top_k   = st.slider("Results", min_value=5, max_value=20, value=10, step=5, key="hist_top_k")
        run_rec = st.button(
            "Get recommendations",
            use_container_width=True,
            type="primary",
            disabled=(selected_user is None),
        )


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown("""
<div class="app-header">
    <p class="app-title">Recipe Search &amp; Recommender</p>
    <p class="app-subtitle">
        DSCI 641 &nbsp;&middot;&nbsp; BM25 + RecipeNet Embeddings + Two-Stage Personalization
    </p>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Results area
# ---------------------------------------------------------------------------

# ── SEARCH ───────────────────────────────────────────────────────────────────

if input_mode == "Search":

    # Handle new search submission
    if run_search and query.strip():
        with st.spinner("Searching..."):
            _result = get_engine().run(query.strip(), mode=selected_mode, top_k=top_k)
        st.session_state.last_search_result = _result
        st.session_state.last_search_top_k  = top_k
        st.session_state.search_display     = "search"

    elif run_compare and query.strip():
        with st.spinner("Running all five modes..."):
            _all = get_engine().run_all_modes(query.strip(), top_k=top_k)
        st.session_state.last_compare_results = _all
        st.session_state.last_compare_top_k   = top_k
        st.session_state.search_display       = "compare"

    display = st.session_state.search_display

    # ── Search results ────────────────────────────────────────────────────────
    if display == "search":
        result = st.session_state.last_search_result
        seeds  = st.session_state.session_seeds

        st.markdown(tier_badge(result.tier), unsafe_allow_html=True)
        w = result.weights
        st.caption("mode: {} | lex={} align={} sem={} quality={}".format(
            result.mode.value, w["lex"], w["alignment"], w["semantic"], w["quality"]))
        render_intent_summary(result.intent)
        st.markdown("---")

        if seeds:
            # Two-column layout: search results left, session picks right
            left, right = st.columns([3, 2])

            with left:
                st.markdown("**Search Results**")
                for i, r in enumerate(result.results, start=1):
                    render_search_card(i, r, show_scores=(result.mode != SearchMode.LEXICAL))
                    _render_seed_button(r, rank=i)

            with right:
                st.markdown("**🌱 Based on your session**")
                seed_names = " · ".join(s["name"][:18] for s in seeds)
                st.caption(seed_names)

                with st.spinner(""):
                    rec_result = get_recommender().recommend_session(
                        seed_recipe_ids=[s["recipe_id"] for s in seeds],
                        top_k=st.session_state.last_search_top_k,
                    )
                render_stage1_box(rec_result.query_tags, rec_result.candidate_pool_size)

                if rec_result.results:
                    for i, r in enumerate(rec_result.results, start=1):
                        render_rec_card(i, r)
                else:
                    st.info("No candidates found. Try adding different recipes.")

        else:
            # No session seeds — full-width search results
            for i, r in enumerate(result.results, start=1):
                render_search_card(i, r, show_scores=(result.mode != SearchMode.LEXICAL))
                _render_seed_button(r, rank=i)

    # ── Compare results ───────────────────────────────────────────────────────
    elif display == "compare":
        all_results = st.session_state.last_compare_results
        top_k_used  = st.session_state.last_compare_top_k

        tier   = all_results[SearchMode.HYBRID].tier
        intent = all_results[SearchMode.HYBRID].intent
        st.markdown(tier_badge(tier), unsafe_allow_html=True)
        render_intent_summary(intent)
        st.markdown("### Mode comparison")

        mode_ndcg = {
            mode: ndcg_at_k(sr.results, intent, k=min(5, top_k_used))
            for mode, sr in all_results.items()
        }
        best_mode = max(mode_ndcg, key=lambda m: mode_ndcg[m])

        rows = ""
        for mode, sr in all_results.items():
            ndcg = mode_ndcg[mode]
            w    = sr.weights
            win  = mode == best_mode
            bg   = "background:#fffbf0;" if win else ""
            fw   = "font-weight:700;" if win else ""
            td   = "padding:9px 14px;border-bottom:1px solid #eee;{}".format(bg)
            rows += (
                "<tr>"
                "<td style='{}{}'>{}</td>"
                "<td style='{}'>{}</td>"
                "<td style='{}'>{}</td>"
                "<td style='{}'>{}</td>"
                "<td style='{}'>{}</td>"
                "<td style='{}'>{}</td>"
                "<td style='{}{}'>{:.4f}{}</td>"
                "</tr>"
            ).format(
                td, fw, MODE_LABELS[mode],
                td, sr.tier,
                td, w["lex"],
                td, w["alignment"],
                td, w["semantic"],
                td, w["quality"],
                td, fw, ndcg, " ★" if win else "",
            )

        st.markdown(
            '<table class="compare-table"><thead><tr>'
            '<th>Mode</th><th>Tier</th><th>w_lex</th>'
            '<th>w_align</th><th>w_sem</th><th>w_quality</th><th>NDCG@5</th>'
            '</tr></thead><tbody>{}</tbody></table>'.format(rows),
            unsafe_allow_html=True,
        )

        st.markdown("### Results by mode")
        for mode, sr in all_results.items():
            with st.expander(
                "{} — NDCG@5: {:.4f}".format(MODE_LABELS[mode], mode_ndcg[mode]),
                expanded=(mode == SearchMode.HYBRID),
            ):
                for i, r in enumerate(sr.results, start=1):
                    render_search_card(i, r, show_scores=(mode != SearchMode.LEXICAL))

    # ── Landing page ──────────────────────────────────────────────────────────
    else:
        st.markdown("""
        <div style="text-align:center;padding:80px 0;color:#aaa;">
            <div style="font-size:48px;margin-bottom:16px;">&#127859;</div>
            <div style="font-size:22px;color:#555;margin-bottom:8px;">What are you hungry for?</div>
            <div style="font-size:14px;">
                Try <em>easy chicken dinner under 45 mins</em>,
                <em>vegan thanksgiving sides</em>, or
                <em>something cozy and warming</em><br><br>
                Add results to your <b>session</b> to get personalised recommendations alongside your next search.<br>
                Switch to <b>User History</b> to recommend from a real Food.com user profile.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ── USER HISTORY ──────────────────────────────────────────────────────────────

else:

    if run_rec and selected_user is not None:

        _spinner_label = selected_persona_name or "User {}".format(selected_user)

        # Load or reuse cached user profile
        cached = st.session_state.cached_user_profile
        if cached is None or cached[0] != selected_user:
            with st.spinner("Building profile for {}...".format(_spinner_label)):
                profile_data = get_recommender().build_profile_for_user(selected_user)
            if profile_data is not None:
                st.session_state.cached_user_profile = (selected_user, profile_data[0], profile_data[1])
            else:
                st.session_state.cached_user_profile = None

        cached = st.session_state.cached_user_profile

        if cached is None:
            st.warning(
                "Could not build a profile for User {}. "
                "They may have too few recipes in the embedding bundle.".format(selected_user)
            )
            st.session_state.last_user_result = None
        else:
            _, profile, user_tags = cached

            with st.spinner("Finding recommendations for {}...".format(_spinner_label)):
                result = get_engine().search(
                    query=query.strip() if query.strip() else None,
                    user_embedding=profile.embedding,
                    user_tag_affinity=profile.tag_affinity,
                    user_tags=user_tags,
                    alpha=alpha,
                    affinity_weight=0.25,
                    mode=SearchMode.HYBRID,
                    top_k=top_k,
                    candidate_pool=500,
                    exclude_ids=profile.rated_recipe_ids,
                )
            st.session_state.last_user_result = (selected_user, result, profile, user_tags)

    cached_result = st.session_state.last_user_result

    if cached_result is not None:
        display_user, result, profile, user_tags = cached_result

        # Header
        _pname = _persona_name_for_user(display_user) or "User {}".format(display_user)
        if result.is_personalized and result.alpha is not None and query.strip():
            blend_pct = int((1.0 - result.alpha) * 100)
            st.markdown(
                "**{}** &nbsp;·&nbsp; {:,} reviews &nbsp;·&nbsp; "
                "{}% user taste / {}% query".format(
                    _pname, profile.review_count, blend_pct, 100 - blend_pct
                )
            )
            render_intent_summary(result.intent)
        else:
            st.markdown(
                "**{}** &nbsp;·&nbsp; {:,} reviews &nbsp;·&nbsp; "
                "pure taste profile".format(_pname, profile.review_count)
            )

        # Profile expanders
        rec_engine = get_recommender()

        if profile.tag_affinity is not None and profile.tag_affinity.any():
            culinary_tags = rec_engine.s.culinary_tags
            negative_tags = rec_engine.s.negative_tags
            tagged = list(zip(culinary_tags, profile.tag_affinity.tolist()))
            # Always show top 4 positive-polarity tags — no threshold so users
            # with recency-discounted histories (e.g. only old reviews) still
            # see their best positive signals rather than an empty Likes row.
            likes = sorted(
                [(t, v) for t, v in tagged if t not in negative_tags],
                key=lambda x: x[1], reverse=True,
            )[:4]
            dislikes = sorted(
                [(t, v) for t, v in tagged if v < -0.01 and t in negative_tags],
                key=lambda x: x[1],
            )
            if likes or dislikes:
                with st.expander("Taste profile"):
                    LIKE_CHIP    = "font-size:12px;padding:3px 10px;border-radius:4px;background:#e4ede0;color:#3d4a35;border:1px solid #bfd0b8;margin:2px;"
                    DISLIKE_CHIP = "font-size:12px;padding:3px 10px;border-radius:4px;background:#fce8e8;color:#b03030;border:1px solid #f5c6c6;margin:2px;"
                    if likes:
                        chips = "".join(
                            '<span style="{}">{}</span>'.format(LIKE_CHIP, t.replace("_", " "))
                            for t, _ in likes
                        )
                        st.markdown(
                            "**Likes** &nbsp; " + chips,
                            unsafe_allow_html=True,
                        )
                    if dislikes:
                        chips = "".join(
                            '<span style="{}">{}</span>'.format(DISLIKE_CHIP, t.replace("_", " "))
                            for t, _ in dislikes
                        )
                        st.markdown(
                            "**Dislikes** &nbsp; " + chips,
                            unsafe_allow_html=True,
                        )

        with st.expander("Review history ({} reviews)".format(profile.review_count)):
            for rec in sorted(profile.reviews, key=lambda r: r.timestamp, reverse=True)[:20]:
                meta  = rec_engine.get_recipe_by_id(rec.recipe_id)
                rname = (
                    meta.get("name", "Recipe #{}".format(rec.recipe_id)).title()
                    if meta else "Recipe #{}".format(rec.recipe_id)
                )
                stars = "&#9733;" * int(rec.star_rating) + "&#9734;" * (5 - int(rec.star_rating))
                st.markdown(
                    "- **{}** &nbsp; {} &nbsp; {}".format(rname, stars, rec.timestamp.date()),
                    unsafe_allow_html=True,
                )

        st.markdown("---")
        for i, r in enumerate(result.results, start=1):
            render_search_card(i, r, show_scores=True)

    elif selected_persona_name is not None:
        # Persona selected but no results yet — show persona card
        persona = PERSONAS[selected_persona_name]
        TRAIT_CHIP = (
            "font-size:12px;padding:3px 12px;border-radius:4px;"
            "background:#f0ebe0;color:#3d4a35;border:1px solid #d0c8b8;margin:2px;"
        )
        trait_chips = "".join(
            '<span style="{}">{}</span>'.format(TRAIT_CHIP, t)
            for t in persona["traits"]
        )
        st.markdown(
            '<div style="{}">'.format(CARD_STYLE)
            + '<div style="font-size:20px;font-weight:700;color:#3d4a35;margin-bottom:8px;">{}</div>'.format(selected_persona_name)
            + '<div style="font-size:14px;color:#6a6258;margin-bottom:14px;">{}</div>'.format(persona["tagline"])
            + '<div style="display:flex;gap:6px;flex-wrap:wrap;">{}</div>'.format(trait_chips)
            + '</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div style="text-align:center;padding:24px 0;font-size:13px;color:#aaa;">'
            'Click <b>Get recommendations</b> to load this profile.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown("""
        <div style="text-align:center;padding:60px 0;color:#aaa;">
            <div style="font-size:36px;margin-bottom:12px;">&#128100;</div>
            <div style="font-size:18px;color:#555;margin-bottom:8px;">Choose an archetype in the sidebar</div>
            <div style="font-size:13px;">
                Select one of 10 curated Food.com reviewer profiles, then click
                <b>Get recommendations</b>.<br><br>
                Add an optional text query to blend search relevance with the
                user's taste profile. Use the <b>Query ↔ User blend</b> slider
                to control how much each signal contributes.
            </div>
        </div>
        """, unsafe_allow_html=True)
