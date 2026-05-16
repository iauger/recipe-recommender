"""
core/layers.py
--------------
Reusable neural network building blocks for RecipeNet.

Ported from Phase 2 (CS 615 / RecipeFeedback-ResNet).
All components are self-contained and import-safe — no Elasticsearch or
data dependencies.

Classes:
    FullyConnectedBlock     — Linear + BatchNorm + ReLU + Dropout
    ResidualBlock           — Same-dimension skip connection (baseline)
    ResidualLinearBlock     — Dimension-expanding skip with shortcut Linear
                              (used in RESIDUAL_V2 head — production model)
    PLQPLayer               — Piecewise Linear Quantile Projection for
                              continuous numeric features (used in RESIDUAL_V3)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import cast


class FullyConnectedBlock(nn.Module):
    """
    Standard fully connected block: Linear → BatchNorm1d → ReLU → Dropout.

    Kaiming (He) initialization on the linear weight for stable training
    with ReLU activations.
    """

    def __init__(self, in_size: int, out_size: int, dropout: float = 0.2):
        super().__init__()
        self.linear = nn.Linear(in_size, out_size)
        self.batchnorm = nn.BatchNorm1d(out_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.kaiming_normal_(self.linear.weight, nonlinearity="relu")
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class ResidualBlock(nn.Module):
    """
    Baseline residual block: two FC layers with a same-dimension skip connection.

    identity → [Linear → BN → ReLU → Dropout → Linear → BN] → add → ReLU

    Requires in_size == out_size. Used in the RESIDUAL head (not production).
    """

    def __init__(self, size: int, dropout: float = 0.2):
        super().__init__()
        self.path = nn.Sequential(
            nn.Linear(size, size),
            nn.BatchNorm1d(size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(size, size),
            nn.BatchNorm1d(size),
        )
        self.final_relu = nn.ReLU()
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.path(x)
        out = out + identity
        return self.final_relu(out)


class ResidualLinearBlock(nn.Module):
    """
    Upgraded residual block with internal dimension expansion and a learned
    shortcut projection to handle in/out dimension mismatches.

    Architecture (with expansion=2):
        identity → shortcut Linear → BN  ──────────────────────┐
        x → fc1(in → mid) → BN → ReLU → fc2(mid → out) → BN  ──┤
                                                          Dropout ┘ → add → ReLU

    Used in the RESIDUAL_V2 head — the production model.

    Reference: pattern from Weights & Biases sample implementations.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        expansion: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        mid_features = out_features * expansion

        self.fc1 = nn.Linear(in_features, mid_features)
        self.bn1 = nn.BatchNorm1d(mid_features)
        self.relu = nn.ReLU(inplace=True)

        self.fc2 = nn.Linear(mid_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(dropout)

        # Shortcut: identity if dims match, otherwise learned projection
        if in_features == out_features:
            self.shortcut: nn.Module = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features),
            )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in [self.fc1, self.fc2]:
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.fc2(out)
        out = self.bn2(out)
        out = self.dropout(out)

        out = out + identity
        return self.relu(out)


class PLQPLayer(nn.Module):
    """
    Piecewise Linear Quantile Projection (PLQP) layer.

    Embeds each continuous feature into a richer learned representation by
    computing soft bin weights against fixed bin centres, then projecting
    those weights through learnable per-feature embeddings.

    Output shape: (batch_size, num_features * embeddings_dim)

    Used in the RESIDUAL_V3 heterogeneous metadata encoder.

    Reference: Gorishniy et al., "On Embeddings for Numerical Features in
    Tabular Deep Learning" (NeurIPS 2022).
    """

    def __init__(
        self,
        num_features: int,
        num_bins: int = 15,
        embeddings_dim: int = 16,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.num_bins = num_bins
        self.embeddings_dim = embeddings_dim

        # Fixed bin centres on the pre-standardised range [-3, 3]
        bin_centers = torch.linspace(-3, 3, num_bins)
        self.register_buffer("bin_centers", bin_centers)
        self.delta = bin_centers[1] - bin_centers[0]

        # Learnable per-(feature, bin) embeddings
        self.embeddings = nn.Parameter(
            torch.empty(num_features, num_bins, embeddings_dim)
        )
        nn.init.normal_(self.embeddings, mean=0.0, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, num_features)
        batch_size = x.size(0)

        x_expanded = x.unsqueeze(-1)                          # (B, F, 1)
        bin_centers = cast(torch.Tensor, self.bin_centers)    # (num_bins,)

        distances = torch.abs(x_expanded - bin_centers)       # (B, F, num_bins)
        weights = torch.relu(1.0 - (distances / self.delta))  # triangle kernel

        out = torch.einsum("bfn,fnm->bfm", weights, self.embeddings)  # (B, F, emb_dim)
        return out.reshape(batch_size, -1)                     # (B, F * emb_dim)
