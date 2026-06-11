"""Two-stage reranker combining BM25, alignment, embedding cosine similarity, and quality score."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from core.models import RecipeNet, HeadType, AblationType

_TEXT_COL_INDICES = {0, 1, 2, 3}   # columns skipped when building input tensors

META_DIM   = 210  # meta features (numeric + OHE categorical) fed to the meta encoder
TAG_DIM    =  34  # review tag features including pred_* and intensity_* fed to the tag encoder
NUM_META   =  10  # continuous numeric features
CAT_META   = 200  # categorical features (cat_* + ing_*)
HIDDEN_DIM = 128  # dim of the shared latent space where query and recipe embeddings live


@dataclass
class RerankedResult:
    """
    Single reranked result. user_affinity_score is 0.0 unless a user_tag_affinity
    vector was passed, so unpersonalized callers see an unchanged result shape.
    """
    recipe_id: str
    base_score: float
    alignment_score: float
    semantic_sim: float
    quality_score: float
    final_score: float
    source: Dict[str, Any]
    user_affinity_score: float = 0.0


class SemanticReranker:
    """
    Scores candidates on four signals: BM25, intent alignment, RecipeNet embedding cosine sim,
    and quality (predicted rating). Weights are stratified by query intent richness.
    """

    def __init__(self, s: Any, debug: bool = False):
        self.s = s
        self.debug = debug
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.bundle = self._load_bundle(s.embeddings_path)
        self.recipe_ids   = self.bundle["recipe_ids"]
        self.predictions  = self.bundle["predictions"]

        # Normalise once at init so cosine sim is just a dot product at query time.
        raw_embeddings  = self.bundle["embeddings"]
        self.embeddings = F.normalize(raw_embeddings.float(), p=2, dim=1)

        self.id_to_index = {str(r_id): idx for idx, r_id in enumerate(self.recipe_ids)}

        with open(s.column_mapping_path, "r") as f:
            self.col_map: Dict[str, int] = json.load(f)
        self._meta_cols, self._tag_cols = self._split_feature_columns()

        self.model = self._load_model(s.model_path)

    def _load_bundle(self, path: str) -> Dict[str, Any]:
        return torch.load(path, map_location=torch.device("cpu"), weights_only=False)

    def _load_model(self, weights_path: str) -> RecipeNet:
        """Load the RESIDUAL_V2 checkpoint frozen for query encoding."""
        model = RecipeNet(
            meta_in=META_DIM,
            tag_in=TAG_DIM,
            hidden_dim=HIDDEN_DIM,
            head_type=HeadType.RESIDUAL_V2,
            num_meta=NUM_META,
            cat_meta=CAT_META,
        ).to(self.device)

        state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)

        # Phase 2 checkpoints used "legacy_meta_encoder"; remap for compatibility.
        remapped_state_dict = {
            k.replace('legacy_meta_encoder', 'default_meta_encoder'): v
            for k, v in state_dict.items()
        }
        model.load_state_dict(remapped_state_dict)
        model.eval()

        for param in model.parameters():
            param.requires_grad = False

        print(f"RecipeNet loaded from {weights_path} (frozen, eval mode).")
        return model

    def _split_feature_columns(self):
        meta_entries = []
        tag_entries  = []

        for col, raw_idx in self.col_map.items():
            if raw_idx in _TEXT_COL_INDICES:
                continue  # text columns have no tensor slot

            if col.startswith("pred_") or col.startswith("intensity_"):
                tag_entries.append((col, raw_idx))
            else:
                meta_entries.append((col, raw_idx))

        meta_entries.sort(key=lambda x: x[1])   # preserve training column order
        tag_entries.sort(key=lambda x: x[1])

        meta_cols = [(col, i) for i, (col, _) in enumerate(meta_entries)]
        tag_cols  = [(col, i) for i, (col, _) in enumerate(tag_entries)]

        return meta_cols, tag_cols

    def _build_query_tensors(
        self,
        projected_query: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert a ProjectedQuery into (meta_tensor, tag_tensor) for RecipeNet's dual encoders."""
        meta_tensor = torch.zeros(1, META_DIM, dtype=torch.float32)
        tag_tensor  = torch.zeros(1, TAG_DIM,  dtype=torch.float32)

        for col, tensor_pos in self._meta_cols:
            raw_idx = self.col_map.get(col)
            if raw_idx is None:
                continue
            val = projected_query.meta_vector[tensor_pos] if tensor_pos < len(projected_query.meta_vector) else 0.0
            meta_tensor[0, tensor_pos] = float(val)

        for col, tensor_pos in self._tag_cols:
            raw_idx = self.col_map.get(col)
            if raw_idx is None:
                continue
            val = projected_query.tag_vector[tensor_pos] if tensor_pos < len(projected_query.tag_vector) else 0.0
            tag_tensor[0, tensor_pos] = float(val)

        return meta_tensor.to(self.device), tag_tensor.to(self.device)

    def encode_query(self, projected_query: Any) -> torch.Tensor:
        """Encode a projected query into the 128-D RecipeNet latent space."""
        meta_tensor, tag_tensor = self._build_query_tensors(projected_query)

        with torch.no_grad():
            _, embedding = self.model(
                meta_tensor,
                tag_tensor,
                return_embeddings=True,
                ablation=AblationType.ALL_FEATURES,
            )

        query_embedding = F.normalize(embedding.cpu().float(), p=2, dim=1).squeeze(0)
        return query_embedding

    def blend_embeddings(
        self,
        query_embedding: torch.Tensor,
        user_embedding: torch.Tensor,
        alpha: float = 0.5,
    ) -> torch.Tensor:
        """Linearly interpolate between query and user embeddings."""
        blended = (alpha * query_embedding) + ((1 - alpha) * user_embedding)
        return F.normalize(blended, p=2, dim=0)

    def compute_affinity_score(
        self,
        candidate_source: Dict[str, Any],
        user_tag_affinity: Optional[np.ndarray] = None,
    ) -> float:
        """dot(user_tag_affinity, recipe_pred_vector) or 0.0 if no affinity provided."""
        if user_tag_affinity is None:
            return 0.0

        tag_keys = [f"pred_{tag}" for tag in self.s.culinary_tags]
        recipe_vector = np.array(
            [candidate_source.get(key, 0.0) for key in tag_keys],
            dtype=np.float32
        )
        return float(np.dot(user_tag_affinity, recipe_vector))

    def compute_semantic_similarity(
        self,
        query_embedding: torch.Tensor,
        recipe_id: str,
    ) -> float:
        """Dot product of unit-normalised query and recipe embeddings (== cosine sim)."""
        idx = self.id_to_index.get(str(recipe_id))
        if idx is None:
            return 0.0

        recipe_embedding = self.embeddings[idx]
        sim = torch.dot(query_embedding, recipe_embedding).item()
        return float(sim)

    def get_quality_score(self, recipe_id: str) -> float:
        """Scalar rating prediction from the Phase 2 bundle (range ~1-5)."""
        idx = self.id_to_index.get(str(recipe_id))
        if idx is None:
            return 0.0

        pred = self.predictions[idx]
        if isinstance(pred, torch.Tensor):
            pred = pred.item()
        return float(pred)

    def combine_scores(
        self,
        base_score: float,
        alignment_score: float,
        semantic_sim: float,
        quality_score: float,
        user_affinity_score: float,
        weights: Dict[str, Any],
        affinity_weight: Optional[float] = None,
    ) -> float:
        score = (
            (base_score * weights.get("lex", 0.0)) +
            (alignment_score * weights.get("alignment", 0.0)) +
            (semantic_sim * weights.get("semantic", 0.0)) +
            (quality_score * weights.get("quality", 0.0))
        )
        if affinity_weight is not None and user_affinity_score > 0:
            score += user_affinity_score * affinity_weight
        return score

    def score_alignment(
        self,
        projected_query: Any,
        candidate_source: Dict[str, Any],
    ) -> float:
        """Rule-based score for how well the candidate matches the structured query intent."""
        score = 0.0

        tags = candidate_source.get("tags_clean", [])
        if tags is None:
            tags = []
        if isinstance(tags, str):
            tags = [tags]
        tags_norm = {str(t).strip().lower() for t in tags}

        ingredients_text = str(candidate_source.get("ingredients_clean", "")).lower()
        name_text = str(candidate_source.get("name", "")).lower()
        minutes = candidate_source.get("minutes")

        # Weights are hand-tuned; a future version could learn them from interaction data.
        high_value_groups = [
            projected_query.cuisines,
            projected_query.methods,
            projected_query.occasions,
            projected_query.courses,
            projected_query.dish_type,
            projected_query.dietary_tags,
        ]
        for group in high_value_groups:
            for value in group:
                value_norm = str(value).strip().lower()
                if value_norm in tags_norm:
                    score += 1.0
                elif value_norm in name_text:
                    score += 0.75

        for value in projected_query.taste:
            value_norm = str(value).strip().lower()
            if value_norm in tags_norm:
                score += 0.25
            elif value_norm in name_text:
                score += 0.25
            elif value_norm in ingredients_text:
                score += 0.10

        for protein in projected_query.proteins:
            protein_norm = str(protein).strip().lower()
            if protein_norm in ingredients_text:
                score += 0.75
            elif protein_norm in name_text:
                score += 0.50

        if projected_query.target_minutes is not None and minutes is not None:
            try:
                distance = abs(float(minutes) - projected_query.target_minutes)
                if distance <= 5:
                    score += 0.75
                elif distance <= 15:
                    score += 0.4
            except (TypeError, ValueError):
                pass

        # Penalise if residual query text (after structured extraction) finds no match.
        structured_tokens = set(
            projected_query.proteins
            + projected_query.dietary_tags
            + projected_query.cuisines
            + projected_query.methods
            + projected_query.occasions
            + projected_query.courses
        )
        leftover_tokens = [
            t.strip()
            for t in projected_query.clean_text.lower().split()
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

        return score

    @staticmethod
    def get_weight_profile(projected_query: Any) -> Dict[str, Any]:
        """Return a weight dict tiered by structured intent richness (high/medium/low)."""
        intent_signals = (
            len(projected_query.dietary_tags)
            + len(projected_query.cuisines)
            + len(projected_query.courses)
            + len(projected_query.methods)
            + len(projected_query.occasions)
            + len(projected_query.proteins)
        )

        if intent_signals >= 3:
            return {"tier": "high",   "lex": 1.0, "alignment": 1.5, "semantic": 0.1,  "quality": 0.25}
        elif intent_signals >= 1:
            return {"tier": "medium", "lex": 1.0, "alignment": 1.0, "semantic": 0.5,  "quality": 0.25}
        else:
            return {"tier": "low",    "lex": 0.75, "alignment": 0.25, "semantic": 1.5, "quality": 0.5}

    def _recipe_tag_vector(self, source: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Build an intensity vector in culinary_tags order for dot product with user affinity.

        Accepts both `intensity_<tag>` and `intensity_<tag>_` (trailing underscore)
        from the parquet for forward compatibility. Returns None if all values are zero.
        """
        tags = self.s.culinary_tags
        values = []
        for t in tags:
            v = source.get("intensity_{}_".format(t),
                           source.get("intensity_{}".format(t), 0.0))
            try:
                values.append(float(v))
            except (TypeError, ValueError):
                values.append(0.0)
        vec = np.asarray(values, dtype=np.float32)
        return vec if float(vec.sum()) > 0 else None

    def rerank(
        self,
        projected_query: Any,
        candidates: List[Dict[str, Any]],
        mode_weights: Dict[str, Any],
        *,
        user_embedding: Optional[torch.Tensor] = None,
        user_tag_affinity: Optional[np.ndarray] = None,
        alpha: float = 0.5,
        affinity_weight: Optional[float] = None,
    ) -> List[RerankedResult]:
        """Score all candidates on four signals, blend user embedding if provided, sort descending."""
        if not candidates:
            return []

        query_embedding = self.encode_query(projected_query)

        if user_embedding is not None:
            effective_embedding = self.blend_embeddings(
                query_embedding, user_embedding, alpha=alpha
            )
        else:
            effective_embedding = query_embedding

        results = []
        for cand in candidates:
            recipe_id = str(cand["_id"])
            source = cand["_source"]

            base_score = cand["_score"] if cand["_score"] is not None else 0.0
            alignment_score = self.score_alignment(projected_query, source)
            semantic_sim = self.compute_semantic_similarity(effective_embedding, recipe_id)
            quality_score = self.get_quality_score(recipe_id)
            user_affinity_score = self.compute_affinity_score(source, user_tag_affinity)

            final_score = self.combine_scores(
                base_score,
                alignment_score,
                semantic_sim,
                quality_score,
                user_affinity_score,
                mode_weights,
                affinity_weight,
            )

            results.append(
                RerankedResult(
                    recipe_id=recipe_id,
                    base_score=base_score,
                    alignment_score=alignment_score,
                    semantic_sim=semantic_sim,
                    quality_score=quality_score,
                    user_affinity_score=user_affinity_score,
                    final_score=final_score,
                    source=source,
                )
            )

        sorted_results = sorted(results, key=lambda x: x.final_score, reverse=True)

        if self.debug:
            print("\n--- Reranker Scores (Top 10) ---")
            for i, res in enumerate(sorted_results[:10]):
                print(f"--- Candidate: {res.recipe_id} (Rank: {i+1}) ---")
                print(f"  Lexical:   {res.base_score:.4f} (w: {mode_weights.get('lex', 0.0):.2f})")
                print(f"  Alignment: {res.alignment_score:.4f} (w: {mode_weights.get('alignment', 0.0):.2f})")
                print(f"  Semantic:  {res.semantic_sim:.4f} (w: {mode_weights.get('semantic', 0.0):.2f})")
                print(f"  Quality:   {res.quality_score:.4f} (w: {mode_weights.get('quality', 0.0):.2f})")
                if res.user_affinity_score > 0:
                    affinity_w = affinity_weight if affinity_weight is not None else 0.0
                    print(f"  Affinity:  {res.user_affinity_score:.4f} (w: {affinity_w:.2f})")
                print("  -----------------------------")
                print(f"  Combined Score: {res.final_score:.4f}")

        return sorted_results
