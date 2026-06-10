"""Projects parsed query intent into sparse feature vectors aligned with the RecipeNet column schema."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np


@dataclass
class ProjectedQuery:
    """A query represented as sparse meta/tag vectors in the model's feature space."""
    raw_query: str
    lexical_query: str
    clean_text: str
    active_meta_features: List[str]
    active_tag_features: List[str]
    active_intensity_features: List[str]
    meta_vector: np.ndarray
    tag_vector: np.ndarray
    target_minutes: int | None
    max_minutes: int | None
    proteins: List[str]
    dietary_tags: List[str]
    cuisines: List[str]
    methods: List[str]
    occasions: List[str]
    courses: List[str]
    taste: List[str]
    dish_type: List[str]


class QueryFeatureProjector:
    """Deterministic query-to-feature projector backed by the column_mapping.json schema."""

    def __init__(self, s: Any):
        self.s = s
        with open(s.column_mapping_path, "r") as f:
            self.col_map: Dict[str, int] = json.load(f)

        self.meta_features = [k for k in self.col_map.keys() if not k.startswith("pred_")]
        self.tag_features = [k for k in self.col_map.keys() if k.startswith("pred_")]

        self.meta_index = {name: i for i, name in enumerate(self.meta_features)}
        self.tag_index = {name: i for i, name in enumerate(self.tag_features)}

    @staticmethod
    def _normalize_token(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[a-zA-Z0-9\-]+", text.lower())

    def _activate_feature(
        self,
        feature_name: str,
        active_features: List[str],
        vector: np.ndarray,
        feature_index: Dict[str, int],
    ) -> None:
        idx = feature_index.get(feature_name)
        if idx is None:
            return
        vector[idx] = 1.0
        if feature_name not in active_features:
            active_features.append(feature_name)

    def _project_tokens_to_meta(
        self,
        tokens: List[str],
        active_meta_features: List[str],
        meta_vector: np.ndarray,
    ) -> None:
        for token in tokens:
            normalized = self._normalize_token(token)

            for prefix in ("cat_", "ing_"):
                feature_name = f"{prefix}{normalized}"
                self._activate_feature(
                    feature_name=feature_name,
                    active_features=active_meta_features,
                    vector=meta_vector,
                    feature_index=self.meta_index,
                )

    def _project_structured_tags(
        self,
        intent: Dict[str, Any],
        active_meta_features: List[str],
        active_tag_features: List[str],
        active_intensity_features: List[str],
        meta_vector: np.ndarray,
        tag_vector: np.ndarray,
    ) -> None:
        structured_groups = {
            "dietary_tags": intent.get("dietary_tags", []),
            "courses": intent.get("courses", []),
            "cuisines": intent.get("cuisines", []),
            "methods": intent.get("methods", []),
            "occasions": intent.get("occasions", []),
            "proteins": intent.get("proteins", []),
            "taste": intent.get("taste", []),
            "dish_type": intent.get("dish_type", []),
        }

        for values in structured_groups.values():
            for value in values:
                normalized = self._normalize_token(value)

                cat_feature = f"cat_{normalized}"
                self._activate_feature(
                    feature_name=cat_feature,
                    active_features=active_meta_features,
                    vector=meta_vector,
                    feature_index=self.meta_index,
                )

                ing_feature = f"ing_{normalized}"
                self._activate_feature(
                    feature_name=ing_feature,
                    active_features=active_meta_features,
                    vector=meta_vector,
                    feature_index=self.meta_index,
                )

                pred_feature = f"pred_{normalized}"
                self._activate_feature(
                    feature_name=pred_feature,
                    active_features=active_tag_features,
                    vector=tag_vector,
                    feature_index=self.tag_index,
                )

                intensity_feature = f"intensity_{normalized}"
                if intensity_feature in self.col_map and intensity_feature not in active_intensity_features:
                    active_intensity_features.append(intensity_feature)

    def project(self, raw_query: str, intent: Dict[str, Any]) -> ProjectedQuery:
        """Return the query as activated feature vectors aligned to column_mapping.json."""
        lexical_query = intent.get("lexical_query", raw_query).strip()
        clean_text = intent.get("clean_text", raw_query).strip()
        tokens = self._tokenize(lexical_query)

        meta_vector = np.zeros(len(self.meta_features), dtype=np.float32)
        tag_vector = np.zeros(len(self.tag_features), dtype=np.float32)

        active_meta_features: List[str] = []
        active_tag_features: List[str] = []
        active_intensity_features: List[str] = []

        self._project_tokens_to_meta(
            tokens=tokens,
            active_meta_features=active_meta_features,
            meta_vector=meta_vector,
        )

        self._project_structured_tags(
            intent=intent,
            active_meta_features=active_meta_features,
            active_tag_features=active_tag_features,
            active_intensity_features=active_intensity_features,
            meta_vector=meta_vector,
            tag_vector=tag_vector,
        )

        return ProjectedQuery(
            raw_query=raw_query,
            lexical_query=lexical_query,
            clean_text=clean_text,
            active_meta_features=sorted(active_meta_features),
            active_tag_features=sorted(active_tag_features),
            active_intensity_features=sorted(active_intensity_features),
            meta_vector=meta_vector,
            tag_vector=tag_vector,
            target_minutes=intent.get("target_minutes"),
            max_minutes=intent.get("max_minutes"),
            proteins=list(intent.get("proteins", [])),
            dietary_tags=list(intent.get("dietary_tags", [])),
            cuisines=list(intent.get("cuisines", [])),
            methods=list(intent.get("methods", [])),
            occasions=list(intent.get("occasions", [])),
            courses=list(intent.get("courses", [])),
            taste=list(intent.get("taste", [])),
            dish_type=list(intent.get("dish_type", [])),
        )
