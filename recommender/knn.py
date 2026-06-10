"""KNN retrieval over the Phase 2 RecipeNet embedding bundle via matrix-vector multiply.

Stateless: caller owns the pre-normalised embedding matrix so it can be shared
with SemanticReranker without loading the bundle twice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

import torch
import torch.nn.functional as F


@dataclass
class KNNResult:
    """Single nearest-neighbor hit from the embedding bundle."""
    recipe_id: int
    similarity: float
    quality_score: float
    source: Dict[str, Any]


def cosine_knn(
    query_vec: torch.Tensor,
    embeddings: torch.Tensor,
    recipe_ids: List[int],
    k: int = 20,
    exclude_ids: Optional[Set[int]] = None,
) -> List[tuple[int, float]]:
    """Return top-k (recipe_id, similarity) by cosine sim; query_vec is normalised internally."""
    qv = query_vec.float().squeeze()
    if qv.dim() != 1:
        raise ValueError(f"query_vec must be 1-D, got shape {qv.shape}")

    qv = F.normalize(qv, p=2, dim=0)
    sims: torch.Tensor = embeddings.float() @ qv

    if exclude_ids:
        for idx, rid in enumerate(recipe_ids):
            if rid in exclude_ids:
                sims[idx] = -1.0   # push to bottom

    # Over-fetch to have headroom after exclude filtering
    fetch_k = min(k + len(exclude_ids or set()) + 50, len(recipe_ids))
    top_vals, top_idxs = torch.topk(sims, fetch_k)

    results: List[tuple[int, float]] = []
    for val, idx in zip(top_vals.tolist(), top_idxs.tolist()):
        rid = recipe_ids[idx]
        if exclude_ids and rid in exclude_ids:
            continue
        results.append((rid, float(val)))
        if len(results) >= k:
            break

    return results


def session_query_vector(
    seed_recipe_ids: List[int],
    embeddings: torch.Tensor,
    recipe_ids: List[int],
) -> Optional[torch.Tensor]:
    """Unit-normalised mean embedding of seed recipes; None if none are in the bundle."""
    id_to_idx = {rid: idx for idx, rid in enumerate(recipe_ids)}
    vecs = []
    for rid in seed_recipe_ids:
        idx = id_to_idx.get(rid)
        if idx is not None:
            vecs.append(embeddings[idx].float())

    if not vecs:
        return None

    mean_vec = torch.stack(vecs).mean(dim=0)
    return F.normalize(mean_vec, p=2, dim=0)
