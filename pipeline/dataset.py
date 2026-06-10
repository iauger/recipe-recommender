"""PyTorch Dataset wrapping the preprocessed recipe parquet for RecipeNet training."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset
import pandas as pd


class RecipeDataset(Dataset):
    """Maps preprocess_data() output to (meta_features, tag_features, target) tensors."""

    def __init__(self, df: pd.DataFrame) -> None:
        self.num_cols = [
            "minutes", "n_steps", "n_ingredients",
            "calories", "fat", "sugar", "sodium",
            "protein", "saturated_fat", "carbs",
        ]

        self.cat_cols  = [c for c in df.columns if c.startswith(("cat_", "ing_"))]
        self.meta_cols = self.num_cols + self.cat_cols
        self.tag_cols  = [c for c in df.columns if c.startswith(("pred_", "intensity_"))]

        self.targets       = torch.tensor(df["bayesian_rating"].values, dtype=torch.float32).view(-1, 1)
        self.meta_features = torch.tensor(df[self.meta_cols].values, dtype=torch.float32)
        self.tag_features  = torch.tensor(df[self.tag_cols].values,  dtype=torch.float32)

        self.recipe_ids  = df["recipe_id"].values
        self.recipe_name = df["name"].values

    @property
    def meta_dim(self) -> int:
        return len(self.meta_cols)

    @property
    def num_dim(self) -> int:
        return len(self.num_cols)

    @property
    def cat_dim(self) -> int:
        return len(self.cat_cols)

    @property
    def tag_dim(self) -> int:
        return len(self.tag_cols)

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
