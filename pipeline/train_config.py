"""
Training hyperparameters for RecipeNet retraining.

Kept separate from core/config.py so runtime components have no training
dependency. Defaults reflect the Phase 2 winning config (RESIDUAL_V2 / MSE / lr=1e-4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from core.models import AblationType, HeadType, PRODUCTION_HEAD
from pipeline.trainer import LossFunc


@dataclass
class TrainConfig:
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    lr_mult: float = 10.0        # head / regressor params get lr * lr_mult

    batch_size: int = 256
    epochs: int = 300
    early_stopping_patience: int = 20

    hidden_dim: int = 128
    head_type: HeadType = PRODUCTION_HEAD
    ablation: AblationType = AblationType.ALL_FEATURES
    loss_fn: LossFunc = LossFunc.MSE

    checkpoint_dir: Path = Path("data/checkpoints")
    results_dir: Path = Path("data/training_results")

    seed: int = 42
    num_workers: int = 0         # 0 for Windows compatibility

    def __post_init__(self) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
