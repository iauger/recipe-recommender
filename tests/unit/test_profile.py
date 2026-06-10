# tests/unit/test_profile.py
"""
Tests for recommender/profile.py: recency/rating weights, build_user_profile.
Pure logic — no data files required.
"""

import math
import numpy as np
import pandas as pd
import pytest
import torch

from recommender.profile import (
    ReviewRecord,
    UserProfile,
    build_user_profile,
    _recency_weight,
    _rating_weight,
    GLOBAL_RATING_MEAN,
)


# ---------------------------------------------------------------------------
# Weight helpers
# ---------------------------------------------------------------------------

class TestRecencyWeight:
    def test_recent_review_weight_near_one(self):
        now = pd.Timestamp.now()
        recent = now - pd.Timedelta(days=1)
        w = _recency_weight(recent, now, half_life_days=365)
        assert w > 0.99

    def test_half_life_gives_half_weight(self):
        now = pd.Timestamp("2025-01-01")
        past = now - pd.Timedelta(days=365)
        w = _recency_weight(past, now, half_life_days=365)
        assert abs(w - 0.5) < 1e-6

    def test_older_review_lower_weight(self):
        now = pd.Timestamp.now()
        old = now - pd.Timedelta(days=730)
        recent = now - pd.Timedelta(days=30)
        assert _recency_weight(old, now, 365) < _recency_weight(recent, now, 365)

    def test_future_timestamp_clamped_to_one(self):
        now = pd.Timestamp.now()
        future = now + pd.Timedelta(days=10)
        w = _recency_weight(future, now, half_life_days=365)
        assert w == 1.0


class TestRatingWeight:
    def test_average_rating_gives_weight_near_one(self):
        w = _rating_weight(GLOBAL_RATING_MEAN)
        assert abs(w - 1.0) < 1e-5

    def test_high_rating_above_one(self):
        assert _rating_weight(5.0) > 1.0

    def test_low_rating_positive_but_small(self):
        w = _rating_weight(1.0)
        assert 0.0 < w <= 1.0

    def test_clamp_lower_bound(self):
        assert _rating_weight(0.0) >= 0.1

    def test_clamp_upper_bound(self):
        assert _rating_weight(10.0) <= 2.0


# ---------------------------------------------------------------------------
# build_user_profile
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_records(culinary_tags):
    now = pd.Timestamp.now()
    return [
        ReviewRecord(
            recipe_id=1000,
            star_rating=5.0,
            timestamp=now - pd.Timedelta(days=10),
            tag_intensities={"family_hit": 0.8, "delicious_tasty": 0.6},
        ),
        ReviewRecord(
            recipe_id=1001,
            star_rating=2.0,
            timestamp=now - pd.Timedelta(days=400),
            tag_intensities={"bland_lacks_flavor": 0.7},
        ),
    ]


@pytest.fixture
def sample_bundle(culinary_tags):
    raw = torch.randn(50, 128)
    embeddings = torch.nn.functional.normalize(raw, p=2, dim=1)
    return {
        "recipe_ids": list(range(1000, 1050)),
        "embeddings": embeddings,
        "predictions": [4.0] * 50,
    }


class TestBuildUserProfile:
    def test_returns_user_profile(self, sample_records, sample_bundle, culinary_tags):
        profile = build_user_profile(sample_records, user_id=42, bundle=sample_bundle, culinary_tags=culinary_tags)
        assert isinstance(profile, UserProfile)

    def test_embedding_is_unit_vector(self, sample_records, sample_bundle, culinary_tags):
        profile = build_user_profile(sample_records, user_id=42, bundle=sample_bundle, culinary_tags=culinary_tags)
        assert profile.embedding is not None
        norm = profile.embedding.norm().item()
        assert abs(norm - 1.0) < 1e-5

    def test_tag_affinity_shape(self, sample_records, sample_bundle, culinary_tags):
        profile = build_user_profile(sample_records, user_id=42, bundle=sample_bundle, culinary_tags=culinary_tags)
        assert profile.tag_affinity.shape == (len(culinary_tags),)

    def test_positive_tag_gets_positive_affinity(self, sample_records, sample_bundle, culinary_tags):
        profile = build_user_profile(sample_records, user_id=42, bundle=sample_bundle, culinary_tags=culinary_tags)
        family_hit_idx = list(culinary_tags).index("family_hit")
        # Recipe 1000 rated 5 stars (above mean) with family_hit intensity 0.8 → positive affinity
        assert profile.tag_affinity[family_hit_idx] > 0.0

    def test_negative_tag_gets_negative_affinity(self, sample_records, sample_bundle, culinary_tags):
        profile = build_user_profile(sample_records, user_id=42, bundle=sample_bundle, culinary_tags=culinary_tags)
        bland_idx = list(culinary_tags).index("bland_lacks_flavor")
        # Recipe 1001 rated 2 stars (below mean) with bland_lacks_flavor → negative affinity
        assert profile.tag_affinity[bland_idx] < 0.0

    def test_rated_recipe_ids_populated(self, sample_records, sample_bundle, culinary_tags):
        profile = build_user_profile(sample_records, user_id=42, bundle=sample_bundle, culinary_tags=culinary_tags)
        assert 1000 in profile.rated_recipe_ids
        assert 1001 in profile.rated_recipe_ids

    def test_review_count_matches_records(self, sample_records, sample_bundle, culinary_tags):
        profile = build_user_profile(sample_records, user_id=42, bundle=sample_bundle, culinary_tags=culinary_tags)
        assert profile.review_count == len(sample_records)

    def test_empty_records_returns_none_embedding(self, sample_bundle, culinary_tags):
        profile = build_user_profile([], user_id=99, bundle=sample_bundle, culinary_tags=culinary_tags)
        assert profile.embedding is None
        assert profile.review_count == 0

    def test_records_with_no_bundle_match(self, culinary_tags):
        records = [
            ReviewRecord(
                recipe_id=99999,
                star_rating=5.0,
                timestamp=pd.Timestamp.now(),
                tag_intensities={},
            )
        ]
        bundle = {
            "recipe_ids": [1, 2, 3],
            "embeddings": torch.randn(3, 128),
            "predictions": [4.0, 4.0, 4.0],
        }
        profile = build_user_profile(records, user_id=42, bundle=bundle, culinary_tags=culinary_tags)
        assert profile.embedding is None
