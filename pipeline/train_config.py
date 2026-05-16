"""
pipeline/train_config.py
------------------------
Training hyperparameters for RecipeNet retraining.

Kept separate from core/config.py so that runtime search/recommender
components have no dependency on training concerns.

Winning configuration from Phase 2 sweep:
    head_type = RESIDUAL_V2
    ablation  = ALL_FEATURES
    loss_fn   = MSE
    lr        = 1e-4  (head params get lr * lr_mult = 1e-3)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from core.models import AblationType, HeadType, PRODUCTION_HEAD
from pipeline.trainer import LossFunc


@dataclass
class TrainConfig:
    # ── Optimiser ────────────────────────────────────────────────────────────
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    lr_mult: float = 10.0        # head / regressor params get lr * lr_mult

    # ── Training loop ────────────────────────────────────────────────────────
    batch_size: int = 256
    epochs: int = 300
    early_stopping_patience: int = 20

    # ── Model ────────────────────────────────────────────────────────────────
    hidden_dim: int = 128
    head_type: HeadType = PRODUCTION_HEAD          # RESIDUAL_V2
    ablation: AblationType = AblationType.ALL_FEATURES
    loss_fn: LossFunc = LossFunc.MSE               # winner from Phase 2 sweep

    # ── Output ───────────────────────────────────────────────────────────────
    checkpoint_dir: Path = Path("data/checkpoints")
    results_dir: Path = Path("data/training_results")

    # ── Reproducibility ──────────────────────────────────────────────────────
    seed: int = 42

    # ── DataLoader ───────────────────────────────────────────────────────────
    num_workers: int = 0         # keep 0 for Windows compatibility

    def __post_init__(self) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
