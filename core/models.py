"""
core/models.py
--------------
RecipeNet — dual-encoder neural network for recipe quality prediction
and 128D embedding generation.

Ported from Phase 2 (CS 615 / RecipeFeedback-ResNet).

Production model: HeadType.RESIDUAL_V2 (selected after full sweep in Phase 2).
All head variants are retained so that any Phase 2 checkpoint can be loaded
without architecture mismatch. Use PRODUCTION_HEAD as the default when
instantiating for inference or retraining.

Architecture overview:
    Metadata encoder  (default_meta_encoder)  ──┐
                                                  ├─ concat → head → 128D embedding
    Tag encoder       (tag_encoder)           ──┘                         │
                                                                    regressor → [1,5]

Heads:
    SHALLOW       — single FC block from fused dim (baseline)
    DEEP          — 10-layer MLP stack (overfit risk)
    RESIDUAL      — 2× ResidualBlock (same-dim skip)
    RESIDUAL_V2   — 6× ResidualLinearBlock (production)
    RESIDUAL_V3   — RESIDUAL_V2 head + PLQP numeric encoder
    TWO_TOWER     — asymmetric late-fusion (separate meta/tag compression)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from enum import Enum

from core.layers import FullyConnectedBlock, ResidualBlock, ResidualLinearBlock, PLQPLayer


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class HeadType(Enum):
    SHALLOW    = "shallow"
    DEEP       = "deep"
    RESIDUAL   = "residual"
    RESIDUAL_V2 = "residual_v2"
    RESIDUAL_V3 = "residual_v3"
    TWO_TOWER  = "two_tower"


class AblationType(Enum):
    META_ONLY    = "meta_only"
    TAG_ONLY     = "tag_only"
    ALL_FEATURES = "all_features"


# Production model — RESIDUAL_V2 selected after Phase 2 sweep.
# Use this constant everywhere instead of hardcoding the enum value.
PRODUCTION_HEAD = HeadType.RESIDUAL_V2


# ---------------------------------------------------------------------------
# RecipeNet
# ---------------------------------------------------------------------------

class RecipeNet(nn.Module):
    """
    Dual-encoder architecture for recipe rating prediction and embedding.

    Args:
        meta_in:    Total metadata feature dimension (num + cat features).
        tag_in:     Tag feature dimension (pred_* + intensity_* columns).
        hidden_dim: Shared hidden dimension — must match the trained checkpoint
                    (default 128, set in core/config.py).
        head_type:  Which prediction head to build. Defaults to RESIDUAL_V2
                    (production). Must match the checkpoint being loaded.
        num_meta:   Number of continuous numeric features (10 in Phase 2).
        cat_meta:   Number of one-hot categorical features.

    Returns (forward):
        prediction: Tensor of shape (B, 1), bounded to [1, 5] via sigmoid.
        embeddings: Tensor of shape (B, hidden_dim) — only when
                    return_embeddings=True.
    """

    def __init__(
        self,
        meta_in: int,
        tag_in: int,
        hidden_dim: int = 128,
        head_type: HeadType = PRODUCTION_HEAD,
        num_meta: int = 10,
        cat_meta: int = 200,
    ):
        super().__init__()

        self.meta_in    = meta_in
        self.tag_in     = tag_in
        self.hidden_dim = hidden_dim
        self.head_type  = head_type
        self.num_meta   = num_meta
        self.cat_meta   = cat_meta

        # ── Metadata encoders ────────────────────────────────────────────────
        # Default path: used by all heads except RESIDUAL_V3 and TWO_TOWER
        self.default_meta_encoder = nn.Sequential(
            FullyConnectedBlock(meta_in, hidden_dim),
            FullyConnectedBlock(hidden_dim, hidden_dim),
        )

        # TWO_TOWER path: compresses to hidden_dim // 4 (32D when hidden_dim=128)
        self.two_tower_meta_encoder = nn.Sequential(
            FullyConnectedBlock(meta_in, hidden_dim // 2),
            FullyConnectedBlock(hidden_dim // 2, hidden_dim // 4),
        )

        # RESIDUAL_V3 heterogeneous path: PLQP for numerics + FC for categoricals
        self.plqp     = PLQPLayer(num_features=num_meta, num_bins=15, embeddings_dim=16)
        self.num_proj = FullyConnectedBlock(num_meta * 16, hidden_dim // 2)
        self.cat_proj = FullyConnectedBlock(cat_meta, hidden_dim // 2)

        # ── Tag encoder ──────────────────────────────────────────────────────
        self.tag_encoder = nn.Sequential(
            FullyConnectedBlock(tag_in, hidden_dim),
            FullyConnectedBlock(hidden_dim, hidden_dim),
        )

        # ── Prediction head ──────────────────────────────────────────────────
        fusion_dim = hidden_dim * 2

        if head_type == HeadType.SHALLOW:
            self.head = self._build_shallow_head(fusion_dim, hidden_dim)
        elif head_type == HeadType.DEEP:
            self.head = self._build_deep_head(fusion_dim, hidden_dim)
        elif head_type == HeadType.RESIDUAL:
            self.head = self._build_residual_head(fusion_dim, hidden_dim)
        elif head_type in (HeadType.RESIDUAL_V2, HeadType.RESIDUAL_V3):
            self.head = self._build_residual_head_v2(fusion_dim, hidden_dim)
        elif head_type == HeadType.TWO_TOWER:
            # Tag-only residual head; meta stream stays narrow and joins at output
            self.head = self._build_residual_head_v2(hidden_dim, hidden_dim)
        else:
            raise ValueError(f"Unsupported head type: {head_type}")

        # ── Regressor / output head ──────────────────────────────────────────
        if head_type == HeadType.TWO_TOWER:
            self.meta_norm = nn.LayerNorm(hidden_dim // 4)  # 32D
            self.tag_norm  = nn.LayerNorm(hidden_dim)       # 128D
            two_tower_fusion_dim = (hidden_dim // 4) + hidden_dim
            self.regressor = nn.Sequential(
                nn.Linear(two_tower_fusion_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )
        else:
            regressor = nn.Linear(hidden_dim, 1)
            nn.init.xavier_uniform_(regressor.weight)
            if regressor.bias is not None:
                nn.init.zeros_(regressor.bias)
            self.regressor = regressor

    # ── Head builders ────────────────────────────────────────────────────────

    def _build_shallow_head(self, fusion_dim: int, hidden_dim: int) -> nn.Sequential:
        return nn.Sequential(
            FullyConnectedBlock(fusion_dim, hidden_dim),
        )

    def _build_deep_head(self, fusion_dim: int, hidden_dim: int) -> nn.Sequential:
        layers = [FullyConnectedBlock(fusion_dim, fusion_dim)]
        for _ in range(8):
            layers.append(FullyConnectedBlock(fusion_dim, fusion_dim))
        layers.append(FullyConnectedBlock(fusion_dim, hidden_dim))
        return nn.Sequential(*layers)

    def _build_residual_head(self, fusion_dim: int, hidden_dim: int) -> nn.Sequential:
        return nn.Sequential(
            FullyConnectedBlock(fusion_dim, fusion_dim),
            ResidualBlock(fusion_dim),
            ResidualBlock(fusion_dim),
            FullyConnectedBlock(fusion_dim, hidden_dim),
        )

    def _build_residual_head_v2(self, fusion_dim: int, hidden_dim: int) -> nn.Sequential:
        """
        6× ResidualLinearBlock with internal 2× expansion — the production head.
        Fused dim → (expand → contract) × 6 → hidden_dim.
        """
        return nn.Sequential(
            FullyConnectedBlock(fusion_dim, fusion_dim),
            *[ResidualLinearBlock(fusion_dim, fusion_dim, expansion=2) for _ in range(6)],
            FullyConnectedBlock(fusion_dim, hidden_dim),
        )

    # ── Forward pass ─────────────────────────────────────────────────────────

    def forward(
        self,
        meta_x: torch.Tensor,
        tag_x: torch.Tensor,
        return_embeddings: bool = False,
        ablation: AblationType = AblationType.ALL_FEATURES,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            meta_x:            (B, meta_in) — structured recipe features.
            tag_x:             (B, tag_in)  — review tag features.
            return_embeddings: If True, returns (prediction, embedding).
            ablation:          Zero out one input stream for ablation studies.

        Returns:
            prediction: (B, 1) bounded to [1, 5].
            embeddings: (B, hidden_dim) — only when return_embeddings=True.
        """
        # Optional ablation
        if ablation == AblationType.META_ONLY:
            tag_x = torch.zeros_like(tag_x)
        elif ablation == AblationType.TAG_ONLY:
            meta_x = torch.zeros_like(meta_x)

        # Encode metadata
        if self.head_type == HeadType.RESIDUAL_V3:
            num_x = meta_x[:, : self.num_meta]
            cat_x = meta_x[:, self.num_meta :]

            num_out = (
                self.num_proj(self.plqp(num_x))
                if self.num_meta > 0
                else torch.zeros(meta_x.shape[0], self.hidden_dim // 2, device=meta_x.device)
            )
            cat_out = (
                self.cat_proj(cat_x)
                if self.cat_meta > 0
                else torch.zeros(meta_x.shape[0], self.hidden_dim // 2, device=meta_x.device)
            )
            meta_out = torch.cat((num_out, cat_out), dim=1)

        elif self.head_type == HeadType.TWO_TOWER:
            meta_out = self.two_tower_meta_encoder(meta_x)
        else:
            meta_out = self.default_meta_encoder(meta_x)

        # Encode tags
        tag_out = self.tag_encoder(tag_x)

        # Fuse and predict
        if self.head_type == HeadType.TWO_TOWER:
            deep_tags  = self.head(tag_out)
            fused      = torch.cat((self.meta_norm(meta_out), self.tag_norm(deep_tags)), dim=1)
            embeddings = fused
            raw_pred   = self.regressor(fused)
        else:
            fused      = torch.cat((meta_out, tag_out), dim=1)
            embeddings = self.head(fused)
            raw_pred   = self.regressor(embeddings)

        # Bound to [1, 5]
        prediction = 1.0 + 4.0 * torch.sigmoid(raw_pred)

        if return_embeddings:
            return prediction, embeddings
        return prediction
