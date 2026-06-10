"""
RecipeNet — dual-encoder for recipe quality prediction and 128D embedding generation.

All Phase 2 head variants are retained so any checkpoint loads without architecture
mismatch. PRODUCTION_HEAD == RESIDUAL_V2, selected after the Phase 2 sweep.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from enum import Enum

from core.layers import FullyConnectedBlock, ResidualBlock, ResidualLinearBlock, PLQPLayer


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


# Use this constant rather than hardcoding the enum — RESIDUAL_V2 selected after Phase 2 sweep.
PRODUCTION_HEAD = HeadType.RESIDUAL_V2


class RecipeNet(nn.Module):
    """
    Dual-encoder for recipe rating prediction and embedding extraction.

    hidden_dim must match the checkpoint (default 128). head_type must also
    match — use PRODUCTION_HEAD unless loading a non-production checkpoint.
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

        # --- Metadata encoders
        # Default path used by all heads except RESIDUAL_V3 and TWO_TOWER.
        self.default_meta_encoder = nn.Sequential(
            FullyConnectedBlock(meta_in, hidden_dim),
            FullyConnectedBlock(hidden_dim, hidden_dim),
        )

        # TWO_TOWER compresses meta to hidden_dim // 4 so the tag stream dominates.
        self.two_tower_meta_encoder = nn.Sequential(
            FullyConnectedBlock(meta_in, hidden_dim // 2),
            FullyConnectedBlock(hidden_dim // 2, hidden_dim // 4),
        )

        self.plqp     = PLQPLayer(num_features=num_meta, num_bins=15, embeddings_dim=16)
        self.num_proj = FullyConnectedBlock(num_meta * 16, hidden_dim // 2)
        self.cat_proj = FullyConnectedBlock(cat_meta, hidden_dim // 2)

        self.tag_encoder = nn.Sequential(
            FullyConnectedBlock(tag_in, hidden_dim),
            FullyConnectedBlock(hidden_dim, hidden_dim),
        )

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
            # Meta stream stays narrow; tag stream gets the full residual head.
            self.head = self._build_residual_head_v2(hidden_dim, hidden_dim)
        else:
            raise ValueError(f"Unsupported head type: {head_type}")

        if head_type == HeadType.TWO_TOWER:
            self.meta_norm = nn.LayerNorm(hidden_dim // 4)
            self.tag_norm  = nn.LayerNorm(hidden_dim)
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
        """6× ResidualLinearBlock with 2× internal expansion — the production head."""
        return nn.Sequential(
            FullyConnectedBlock(fusion_dim, fusion_dim),
            *[ResidualLinearBlock(fusion_dim, fusion_dim, expansion=2) for _ in range(6)],
            FullyConnectedBlock(fusion_dim, hidden_dim),
        )

    def forward(
        self,
        meta_x: torch.Tensor,
        tag_x: torch.Tensor,
        return_embeddings: bool = False,
        ablation: AblationType = AblationType.ALL_FEATURES,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Returns (prediction, embeddings) if return_embeddings else prediction."""
        if ablation == AblationType.META_ONLY:
            tag_x = torch.zeros_like(tag_x)
        elif ablation == AblationType.TAG_ONLY:
            meta_x = torch.zeros_like(meta_x)

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

        tag_out = self.tag_encoder(tag_x)

        if self.head_type == HeadType.TWO_TOWER:
            deep_tags  = self.head(tag_out)
            fused      = torch.cat((self.meta_norm(meta_out), self.tag_norm(deep_tags)), dim=1)
            embeddings = fused
            raw_pred   = self.regressor(fused)
        else:
            fused      = torch.cat((meta_out, tag_out), dim=1)
            embeddings = self.head(fused)
            raw_pred   = self.regressor(embeddings)

        prediction = 1.0 + 4.0 * torch.sigmoid(raw_pred)  # bounded to [1, 5]

        if return_embeddings:
            return prediction, embeddings
        return prediction
