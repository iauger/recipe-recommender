"""
pipeline/train.py
-----------------
Clean entry point for retraining RecipeNet (RESIDUAL_V2 / ALL_FEATURES / MSE).

Replaces the Phase 2 experiment-matrix main.py. The winning configuration
is the fixed default — no sweep, no flag juggling.

Usage:
    python -m pipeline.train                         # default config
    python -m pipeline.train --epochs 150 --lr 5e-5
    python -m pipeline.train --overwrite             # re-run preprocessing
    python -m pipeline.train --skip-inference        # skip embedding generation

Steps:
    1. Preprocess raw data → PROCESSED_recipes.parquet + column_mapping.json
    2. Split 70 / 15 / 15 (train / val / test) with fixed seed
    3. Train RESIDUAL_V2 with early stopping
    4. Evaluate on test set → print MSE / RMSE / MAE
    5. Run full-corpus inference → overwrite embeddings_path bundle
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from core.config import load_settings
from core.models import AblationType, PRODUCTION_HEAD, RecipeNet
from pipeline.dataset import RecipeDataset
from pipeline.inference import run_inference
from pipeline.preprocessing import preprocess_data, preprocess_report
from pipeline.train_config import TrainConfig
from pipeline.trainer import Trainer


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Retrain RecipeNet (RESIDUAL_V2)")
    p.add_argument("--epochs",          type=int,   default=None,  help="Override max epochs")
    p.add_argument("--lr",              type=float, default=None,  help="Override learning rate")
    p.add_argument("--batch-size",      type=int,   default=None,  help="Override batch size")
    p.add_argument("--overwrite",       action="store_true",       help="Re-run preprocessing even if cache exists")
    p.add_argument("--skip-inference",  action="store_true",       help="Skip embedding generation after training")
    p.add_argument("--report",          action="store_true",       help="Print preprocessing feature report")
    return p


def main() -> None:
    args    = build_parser().parse_args()
    cfg     = TrainConfig()
    s       = load_settings()

    # Apply CLI overrides
    if args.epochs     is not None: cfg.epochs        = args.epochs
    if args.lr         is not None: cfg.learning_rate = args.lr
    if args.batch_size is not None: cfg.batch_size    = args.batch_size

    print(f"=== RecipeNet Retraining ===")
    print(f"  Head      : {cfg.head_type.value}")
    print(f"  Ablation  : {cfg.ablation.value}")
    print(f"  Loss      : {cfg.loss_fn.value}")
    print(f"  Epochs    : {cfg.epochs} (patience={cfg.early_stopping_patience})")
    print(f"  LR        : {cfg.learning_rate} (head ×{cfg.lr_mult})")
    print(f"  Batch     : {cfg.batch_size}")

    # ── 1. Preprocess ────────────────────────────────────────────────────────
    df = preprocess_data(s, overwrite_processed=args.overwrite)
    if args.report:
        preprocess_report(df)

    # ── 2. Dataset and splits ────────────────────────────────────────────────
    full_dataset = RecipeDataset(df)
    total        = len(full_dataset)
    train_size   = int(0.70 * total)
    val_size     = int(0.15 * total)
    test_size    = total - train_size - val_size

    generator = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set, test_set = random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )

    loader_kw = dict(batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    train_loader = DataLoader(train_set, shuffle=True,  **loader_kw)
    val_loader   = DataLoader(val_set,   shuffle=False, **loader_kw)
    test_loader  = DataLoader(test_set,  shuffle=False, **loader_kw)

    print(f"\n  Dataset  : {total:,} recipes")
    print(f"  Train    : {train_size:,} | Val: {val_size:,} | Test: {test_size:,}")
    print(f"  Meta dim : {full_dataset.meta_dim} | Tag dim: {full_dataset.tag_dim}")

    # ── 3. Instantiate model ─────────────────────────────────────────────────
    model = RecipeNet(
        meta_in    = full_dataset.meta_dim,
        tag_in     = full_dataset.tag_dim,
        hidden_dim = cfg.hidden_dim,
        head_type  = cfg.head_type,
        num_meta   = full_dataset.num_dim,
        cat_meta   = full_dataset.cat_dim,
    )

    # ── 4. Train ─────────────────────────────────────────────────────────────
    trainer = Trainer(model, train_loader, val_loader, cfg)
    history = trainer.fit(
        epochs         = cfg.epochs,
        head_type      = cfg.head_type,
        ablation       = cfg.ablation,
        loss_fn        = cfg.loss_fn,
        checkpoint_dir = cfg.checkpoint_dir,
    )

    # ── 5. Evaluate ──────────────────────────────────────────────────────────
    metrics, _ = trainer.evaluate(
        test_loader,
        head_type         = cfg.head_type,
        ablation          = cfg.ablation,
        return_embeddings = False,
    )

    history.update(metrics)
    print(f"\n  Test MSE : {metrics['test_mse']:.4f}")
    print(f"  Test RMSE: {metrics['test_rmse']:.4f}")
    print(f"  Test MAE : {metrics['test_mae']:.4f}")

    # Save training history
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = cfg.results_dir / f"results_{cfg.head_type.value}_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(history, f, indent=4)
    print(f"\n  Results saved → {results_path}")

    # Copy best checkpoint to the canonical model_path so search/reranker can load it
    if trainer.best_model_path and trainer.best_model_path.exists():
        import shutil
        s.model_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(trainer.best_model_path, s.model_path)
        print(f"  Checkpoint promoted → {s.model_path}")

    # ── 6. Full-corpus inference ─────────────────────────────────────────────
    if not args.skip_inference:
        print("\n--- Generating embedding bundle ---")
        run_inference(
            settings          = s,
            model_path        = s.model_path,
            head_type         = PRODUCTION_HEAD,
            output_path       = s.embeddings_path,
            overwrite_processed = False,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
