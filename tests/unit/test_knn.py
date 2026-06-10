# tests/unit/test_knn.py
"""
Tests for recommender/knn.py: cosine_knn and session_query_vector.
Pure tensor math — no data files or ES required.
"""

import pytest
import torch
import torch.nn.functional as F

from recommender.knn import cosine_knn, session_query_vector


@pytest.fixture
def unit_embeddings():
    """20 unit-normalised 128-D embeddings."""
    raw = torch.randn(20, 128)
    return F.normalize(raw, p=2, dim=1)


@pytest.fixture
def ids():
    return list(range(100, 120))


class TestCosineKnn:
    def test_returns_k_results(self, unit_embeddings, ids):
        query = F.normalize(torch.randn(128), p=2, dim=0)
        results = cosine_knn(query, unit_embeddings, ids, k=5)
        assert len(results) == 5

    def test_results_sorted_descending(self, unit_embeddings, ids):
        query = F.normalize(torch.randn(128), p=2, dim=0)
        results = cosine_knn(query, unit_embeddings, ids, k=10)
        sims = [sim for _, sim in results]
        assert sims == sorted(sims, reverse=True)

    def test_exact_match_is_top_result(self, unit_embeddings, ids):
        # Query is identical to the first embedding — should be rank 1
        query = unit_embeddings[0].clone()
        results = cosine_knn(query, unit_embeddings, ids, k=5)
        assert results[0][0] == ids[0]

    def test_similarities_in_valid_range(self, unit_embeddings, ids):
        query = F.normalize(torch.randn(128), p=2, dim=0)
        results = cosine_knn(query, unit_embeddings, ids, k=10)
        for _, sim in results:
            assert -1.0 <= sim <= 1.0 + 1e-5

    def test_exclude_ids_removes_from_results(self, unit_embeddings, ids):
        query = unit_embeddings[0].clone()
        exclude = {ids[0], ids[1]}
        results = cosine_knn(query, unit_embeddings, ids, k=5, exclude_ids=exclude)
        returned_ids = {rid for rid, _ in results}
        assert returned_ids.isdisjoint(exclude)

    def test_k_larger_than_corpus_returns_all(self, unit_embeddings, ids):
        query = F.normalize(torch.randn(128), p=2, dim=0)
        results = cosine_knn(query, unit_embeddings, ids, k=1000)
        assert len(results) == len(ids)

    def test_2d_query_squeezed(self, unit_embeddings, ids):
        query = F.normalize(torch.randn(1, 128), p=2, dim=1)
        results = cosine_knn(query, unit_embeddings, ids, k=3)
        assert len(results) == 3

    def test_invalid_query_shape_raises(self, unit_embeddings, ids):
        bad_query = torch.randn(2, 128)
        with pytest.raises(ValueError):
            cosine_knn(bad_query, unit_embeddings, ids, k=3)


class TestSessionQueryVector:
    def test_single_seed(self, unit_embeddings, ids):
        vec = session_query_vector([ids[0]], unit_embeddings, ids)
        assert vec is not None
        assert vec.shape == (128,)

    def test_result_is_unit_vector(self, unit_embeddings, ids):
        vec = session_query_vector([ids[0], ids[1]], unit_embeddings, ids)
        assert vec is not None
        norm = vec.norm().item()
        assert abs(norm - 1.0) < 1e-5

    def test_mean_of_two_identical_seeds(self, unit_embeddings, ids):
        # Mean of same vector with itself == that vector
        vec = session_query_vector([ids[0], ids[0]], unit_embeddings, ids)
        expected = unit_embeddings[0]
        assert torch.allclose(vec, expected, atol=1e-5)

    def test_unknown_seed_ids_ignored(self, unit_embeddings, ids):
        vec = session_query_vector([ids[0], 99999], unit_embeddings, ids)
        assert vec is not None  # only the valid seed is used

    def test_all_unknown_returns_none(self, unit_embeddings, ids):
        vec = session_query_vector([99999, 88888], unit_embeddings, ids)
        assert vec is None

    def test_empty_seeds_returns_none(self, unit_embeddings, ids):
        vec = session_query_vector([], unit_embeddings, ids)
        assert vec is None
