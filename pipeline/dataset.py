"""
pipeline/dataset.py
-------------------
PyTorch Dataset for RecipeNet training.

Ported from Phase 2 (CS 615 / RecipeFeedback-ResNet src/dataset.py).
No changes to logic — only updated module docstring.

Feature groups (must match the column ordering produced by preprocessing.py):
    num_cols  — 10 continuous features (minutes, n_steps, n_ingredients,
                calories, fat, sugar, sodium, protein, saturated_fat, carbs)
    cat_cols  — one-hot encoded tags and ingredients (cat_*, ing_* prefixes)
    tag_cols  — review sentiment features (pred_*, intensity_* prefixes)
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset
import pandas as pd


class RecipeDataset(Dataset):
    """
    Maps a preprocessed recipe DataFrame to (meta_features, tag_features, target)
    tensors suitable for RecipeNet.

    Args:
        df: Output of pipeline.preprocessing.preprocess_data().
            Must contain the standard num_cols, cat_*, ing_*, pred_*, intensity_*
            columns plus 'bayesian_rating', 'recipe_id', and 'name'.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        # Continuous numeric features
        self.num_cols = [
            "minutes", "n_steps", "n_ingredients",
            "calories", "fat", "sugar", "sodium",
            "protein", "saturated_fat", "carbs",
        ]

        # One-hot categorical features (tags_clean and ingredients_clean encoded)
        self.cat_cols = [c for c in df.columns if c.startswith(("cat_", "ing_"))]

        # Combined metadata fed to the metadata encoder
        self.meta_cols = self.num_cols + self.cat_cols

        # Review sentiment tag features fed to the tag encoder
        self.tag_cols = [c for c in df.columns if c.startswith(("pred_", "intensity_"))]

        # Targets and features → tensors
        self.targets       = torch.tensor(df["bayesian_rating"].values, dtype=torch.float32).view(-1, 1)
        self.meta_features = torch.tensor(df[self.meta_cols].values, dtype=torch.float32)
        self.tag_features  = torch.tensor(df[self.tag_cols].values,  dtype=torch.float32)

        # Identifiers (kept as numpy arrays for the embedding bundle)
        self.recipe_ids  = df["recipe_id"].values
        self.recipe_name = df["name"].values

    # ── Dimension properties used by RecipeNet constructor ───────────────────

    @property
    def meta_dim(self) -> int:
        """Total metadata feature dimension (num + cat)."""
        return len(self.meta_cols)

    @property
    def num_dim(self) -> int:
        """Number of continuous numeric features."""
        return len(self.num_cols)

    @property
    def cat_dim(self) -> int:
        """Number of one-hot categorical features."""
        return len(self.cat_cols)

    @property
    def tag_dim(self) -> int:
        """Number of tag features (pred_* + intensity_*)."""
        return len(self.tag_cols)

    # ── Dataset interface ────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int):
        return (
            self.meta_features[idx],
            self.tag_features[idx],
            self.targets[idx],
            self.recipe_ids[idx],
            self.recipe_name[idx],
        )
