"""
pipeline/inference.py
---------------------
Full-corpus inference: loads the best RecipeNet checkpoint and produces
the 128D embedding bundle used by the search and recommender components.

Ported from Phase 2 (CS 615 / RecipeFeedback-ResNet src/inference.py).

Changes from Phase 2:
    - Imports updated to core.models / pipeline.dataset / core.config.
    - Output path taken from settings.embeddings_path (unified config)
      rather than os.path.join(s.best_model_dir, output_name).
    - head_type defaults to PRODUCTION_HEAD (RESIDUAL_V2).
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.config import Settings, load_settings
from core.models import AblationType, HeadType, RecipeNet, PRODUCTION_HEAD
from pipeline.dataset import RecipeDataset
from pipeline.preprocessing import preprocess_data


def run_inference(
    settings: Settings | None = None,
    model_path: Path | None = None,
    head_type: HeadType = PRODUCTION_HEAD,
    output_path: Path | None = None,
    batch_size: int = 1024,
    overwrite_processed: bool = False,
) -> dict:
    """
    Run full-corpus inference and save the embedding bundle.

    Args:
        settings:           Loaded Settings instance. Calls load_settings() if None.
        model_path:         Path to the .pth checkpoint. Defaults to
                            settings.model_path if not provided.
        head_type:          Head architecture of the checkpoint. Must match
                            how the model was trained. Defaults to PRODUCTION_HEAD.
        output_path:        Where to save the embedding bundle (.pt).
                            Defaults to settings.embeddings_path.
        batch_size:         Inference batch size.
        overwrite_processed: Re-run preprocessing even if cached parquet exists.

    Returns:
        embedding_bundle dict with keys:
            recipe_ids, recipe_names, targets, predictions, embeddings (Tensor N×128).
    """
    s           = settings or load_settings()
    model_path  = Path(model_path) if model_path else s.model_path
    output_path = Path(output_path) if output_path else s.embeddings_path
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n--- Full-Corpus Inference ---")
    print(f"  Checkpoint : {model_path}")
    print(f"  Head       : {head_type.value}")
    print(f"  Device     : {device}")

    # Load and prepare data
    df           = preprocess_data(s, overwrite_processed=overwrite_processed)
    full_dataset = RecipeDataset(df)
    loader       = DataLoader(full_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Instantiate model
    model = RecipeNet(
        meta_in    = full_dataset.meta_dim,
        tag_in     = full_dataset.tag_dim,
        hidden_dim = s.hidden_dim,
        head_type  = head_type,
        num_meta   = full_dataset.num_dim,
        cat_meta   = full_dataset.cat_dim,
    ).to(device)

    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    state_dict = torch.load(model_path, map_location=device)

    # Handle Phase 2 legacy key name (legacy_meta_encoder → default_meta_encoder)
    state_dict = {
        k.replace("legacy_meta_encoder", "default_meta_encoder"): v
        for k, v in state_dict.items()
    }
    model.load_state_dict(state_dict)
    model.eval()

    # Run inference
    all_ids, all_names, all_targets, all_preds, all_embeddings = [], [], [], [], []

    print(f"  Processing {len(full_dataset):,} recipes …")
    with torch.no_grad():
        for meta_x, tag_x, targets, ids, names in tqdm(loader, desc="Inference"):
            meta_x = meta_x.to(device)
            tag_x  = tag_x.to(device)

            preds, embeddings = model(
                meta_x, tag_x,
                return_embeddings=True,
                ablation=AblationType.ALL_FEATURES,
            )

            all_preds.extend(preds.cpu().view(-1).tolist())
            all_embeddings.append(embeddings.cpu())
            all_targets.extend(targets.view(-1).tolist())
            all_ids.extend(ids)
            all_names.extend(names)

    final_embeddings = torch.cat(all_embeddings, dim=0)

    print(f"  Prediction range: [{min(all_preds):.4f}, {max(all_preds):.4f}]")
    print(f"  Embedding shape : {final_embeddings.shape}")

    bundle = {
        "recipe_ids":    all_ids,
        "recipe_names":  all_names,
        "targets":       all_targets,
        "predictions":   all_preds,
        "embeddings":    final_embeddings,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, output_path)
    print(f"  Bundle saved → {output_path}")

    return bundle
