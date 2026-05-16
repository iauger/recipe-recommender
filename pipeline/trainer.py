"""
pipeline/trainer.py
-------------------
Training loop, validation, checkpointing, and evaluation for RecipeNet.

Ported from Phase 2 (CS 615 / RecipeFeedback-ResNet src/trainer.py).

Key change from Phase 2:
    fit() no longer calls load_settings() internally to resolve the checkpoint
    path. It now accepts checkpoint_dir: Path explicitly, keeping the trainer
    decoupled from the filesystem layout.
"""

from __future__ import annotations

import os
import time
from enum import Enum
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from core.models import AblationType, HeadType


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

class LossFunc(Enum):
    MSE      = "mse"
    HUBER    = "huber"
    LOG_COSH = "log_cosh"


class LogCoshLoss(nn.Module):
    """Smooth approximation to MAE — less sensitive to outliers than MSE."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        x = y_pred - y_true
        return torch.mean(torch.log(torch.cosh(x + 1e-12)))


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    Encapsulates training, validation, checkpointing, and evaluation.

    Args:
        model:        RecipeNet instance (moved to device internally).
        train_loader: DataLoader for the training split.
        val_loader:   DataLoader for the validation split.
        config:       TrainConfig (or any object with the expected attributes).
    """

    def __init__(self, model: nn.Module, train_loader, val_loader, config) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.config = config

        self.criterion = self._build_criterion(getattr(config, "loss_fn", LossFunc.HUBER))
        self.optimizer = self._setup_optimizer()

        self.history: dict = {
            "model_type":    None,
            "ablation_type": None,
            "loss_type":     None,
            "train_loss":    [],
            "val_loss":      [],
            "grad_norm":     [],
        }

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            patience=5,
            factor=0.5,
        )

        self.current_ablation = AblationType.ALL_FEATURES
        self.best_model_path: Path | None = None

    # ── Setup helpers ────────────────────────────────────────────────────────

    def _build_criterion(self, loss_setting) -> nn.Module:
        val = loss_setting.value if isinstance(loss_setting, LossFunc) else loss_setting
        if val == LossFunc.MSE.value:
            return nn.MSELoss()
        if val == LossFunc.LOG_COSH.value:
            return LogCoshLoss()
        return nn.HuberLoss()

    def _setup_optimizer(self) -> torch.optim.Optimizer:
        base_lr       = self.config.learning_rate
        head_keywords = ("head", "regressor")

        base_params = [p for n, p in self.model.named_parameters()
                       if not any(k in n for k in head_keywords)]
        head_params = [p for n, p in self.model.named_parameters()
                       if any(k in n for k in head_keywords)]

        return torch.optim.AdamW(
            [
                {"params": base_params, "lr": base_lr},
                {"params": head_params, "lr": base_lr * self.config.lr_mult},
            ],
            weight_decay=self.config.weight_decay,
        )

    def _compute_grad_norm(self) -> float:
        total = sum(
            p.grad.data.norm(2).item() ** 2
            for p in self.model.parameters()
            if p.grad is not None
        )
        return total ** 0.5

    # ── Epoch-level methods ──────────────────────────────────────────────────

    def train_epoch(self) -> float:
        self.model.train()
        total_loss, grad_norms = 0.0, []

        for meta_x, tag_x, targets, *_ in self.train_loader:
            meta_x  = meta_x.to(self.device)
            tag_x   = tag_x.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(meta_x, tag_x, ablation=self.current_ablation)
            loss    = self.criterion(outputs, targets)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            grad_norms.append(self._compute_grad_norm())
            self.optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(self.train_loader)
        self.history["train_loss"].append(avg_loss)
        self.history["grad_norm"].append(
            sum(grad_norms) / len(grad_norms) if grad_norms else 0.0
        )
        return avg_loss

    def validate(self) -> float:
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for meta_x, tag_x, targets, *_ in self.val_loader:
                meta_x  = meta_x.to(self.device)
                tag_x   = tag_x.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(meta_x, tag_x, ablation=self.current_ablation)
                total_loss += self.criterion(outputs, targets).item()

        avg_loss = total_loss / len(self.val_loader)
        self.history["val_loss"].append(avg_loss)
        return avg_loss

    # ── Main training entry point ────────────────────────────────────────────

    def fit(
        self,
        epochs: int,
        head_type: HeadType,
        ablation: AblationType,
        loss_fn: LossFunc,
        checkpoint_dir: Path,
    ) -> dict:
        """
        Train for up to `epochs` epochs with early stopping.

        Args:
            epochs:         Maximum number of training epochs.
            head_type:      HeadType of the model being trained (for naming).
            ablation:       AblationType controlling which input streams are active.
            loss_fn:        LossFunc to use during training.
            checkpoint_dir: Directory where the best checkpoint is saved.
                            Replaces the Phase 2 internal load_settings() call.

        Returns:
            history dict with train_loss, val_loss, grad_norm, and metadata.
        """
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.criterion        = self._build_criterion(loss_fn)
        self.current_ablation = ablation

        self.history["loss_type"]     = loss_fn.value
        self.history["ablation_type"] = ablation.value
        self.history["model_type"]    = head_type.value

        checkpoint_path = checkpoint_dir / f"best_{head_type.value}_{ablation.value}_{loss_fn.value}.pth"
        self.best_model_path = checkpoint_path

        best_val_loss  = float("inf")
        patience       = getattr(self.config, "early_stopping_patience", 20)
        trigger_times  = 0

        pbar = tqdm(range(1, epochs + 1), desc="Training", unit="epoch")
        for epoch in pbar:
            t0         = time.time()
            train_loss = self.train_epoch()
            val_loss   = self.validate()
            epoch_sec  = time.time() - t0

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), checkpoint_path)
                trigger_times = 0
            else:
                trigger_times += 1

            self.scheduler.step(val_loss)
            pbar.set_postfix({
                "T-Loss":     f"{train_loss:.4f}",
                "V-Loss":     f"{val_loss:.4f}",
                "Sec/Epoch":  f"{epoch_sec:.1f}",
            })

            if trigger_times >= patience:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs).")
                break

        # Reload best weights before returning
        if checkpoint_path.exists():
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))

        return self.history

    # ── Evaluation ───────────────────────────────────────────────────────────

    def evaluate(
        self,
        loader,
        head_type: HeadType,
        ablation: AblationType,
        return_embeddings: bool = False,
    ) -> tuple[dict, dict | None]:
        """
        Evaluate on a held-out loader. Reports MSE, RMSE, and MAE.

        Returns:
            metrics: dict with test_mse, test_rmse, test_mae.
            bundle:  embedding bundle dict (or None if return_embeddings=False).
        """
        self.model.eval()
        total_mse = total_mae = 0.0
        all_embeddings, all_targets, all_preds, all_ids, all_names = [], [], [], [], []

        print(f"Evaluating {head_type.value} on {len(loader.dataset):,} samples …")

        with torch.no_grad():
            for meta_x, tag_x, targets, ids, names in loader:
                meta_x  = meta_x.to(self.device)
                tag_x   = tag_x.to(self.device)
                targets = targets.to(self.device)

                if return_embeddings:
                    outputs, embeddings = self.model(
                        meta_x, tag_x, return_embeddings=True, ablation=ablation
                    )
                    all_embeddings.append(embeddings.cpu())
                    all_targets.append(targets.cpu())
                    all_preds.append(outputs.cpu())
                    all_ids.extend(ids)
                    all_names.extend(names)
                else:
                    outputs = self.model(meta_x, tag_x, return_embeddings=False, ablation=ablation)

                total_mse += nn.functional.mse_loss(outputs, targets).item()
                total_mae += nn.functional.l1_loss(outputs, targets).item()

        n = len(loader)
        avg_mse = total_mse / n
        metrics = {"test_mse": avg_mse, "test_rmse": avg_mse ** 0.5, "test_mae": total_mae / n}

        bundle = None
        if return_embeddings:
            bundle = {
                "embeddings":   torch.cat(all_embeddings),
                "targets":      torch.cat(all_targets),
                "predictions":  torch.cat(all_preds),
                "recipe_ids":   all_ids,
                "recipe_names": all_names,
            }

        return metrics, bundle
