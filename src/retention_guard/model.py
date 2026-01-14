"""Model training and scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from retention_guard.data import REQUIRED_COLUMNS


TARGET_COLUMN = "exit_flag"


@dataclass
class ModelResult:
    model: Pipeline
    auc: float
    feature_importance: Dict[str, float]


def _build_pipeline(cat_features: list[str], num_features: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ]
    )
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=42,
        class_weight="balanced",
    )
    return Pipeline([("prep", preprocessor), ("model", model)])


def train_model(df: pd.DataFrame) -> ModelResult:
    """Train a baseline model and return performance + top drivers.

    Forking tip: swap RandomForest with XGBoost/LightGBM and add validation.
    """
    if TARGET_COLUMN not in df.columns:
        raise ValueError("Training data must include exit_flag column.")

    features = [col for col in REQUIRED_COLUMNS if col in df.columns]
    cat_features = ["dept", "salary_band"]
    num_features = [col for col in features if col not in cat_features and col != "employee_id"]

    x = df[features].copy()
    y = df[TARGET_COLUMN].astype(int).values

    pipeline = _build_pipeline(cat_features=cat_features, num_features=num_features)
    pipeline.fit(x, y)

    proba = pipeline.predict_proba(x)[:, 1]
    auc = roc_auc_score(y, proba)

    feature_importance = _extract_feature_importance(pipeline, cat_features, num_features)
    return ModelResult(model=pipeline, auc=auc, feature_importance=feature_importance)


def score(model: Pipeline, df: pd.DataFrame) -> np.ndarray:
    """Score risk probabilities for each employee.

    Forking tip: keep the output as probabilities to preserve ranking quality.
    """
    features = [col for col in REQUIRED_COLUMNS if col in df.columns]
    return model.predict_proba(df[features])[:, 1]


def _extract_feature_importance(
    pipeline: Pipeline, cat_features: list[str], num_features: list[str]
) -> Dict[str, float]:
    model = pipeline.named_steps["model"]
    prep = pipeline.named_steps["prep"]

    cat_names = list(prep.named_transformers_["cat"].get_feature_names_out(cat_features))
    feature_names = num_features + cat_names

    importances = model.feature_importances_
    pairs = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    return {name: float(weight) for name, weight in pairs[:8]}
