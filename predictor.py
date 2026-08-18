# -*- coding: utf-8 -*-
"""Prediction engine for the prostate cancer Shiny web application."""

from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "Stacking_Top13_Meta_XGBoost_model_bundle.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

BUNDLE = joblib.load(MODEL_PATH)

MODEL_NAME = str(BUNDLE["model_name"])
CUTOFF = float(BUNDLE["cutoff"])
RAW_FEATURES = list(BUNDLE["raw_feature_columns"])
BASE_MODEL_ORDER = list(BUNDLE["base_model_order"])
META_FEATURE_COLUMNS = list(BUNDLE["meta_feature_columns"])
BASE_MODELS = dict(BUNDLE["base_models"])
META_MODEL = BUNDLE["meta_model"]


def _base_model_positive_probability(pipe, x_raw: pd.DataFrame) -> np.ndarray:
    """Return P(class=1) from one fitted base-model pipeline."""
    required_steps = ("preprocess", "scale", "variance", "model")
    missing = [name for name in required_steps if name not in pipe.named_steps]
    if missing:
        raise RuntimeError(f"Saved base-model pipeline is missing steps: {missing}")

    x = pipe.named_steps["preprocess"].transform(x_raw)
    x = pipe.named_steps["scale"].transform(x)
    x = pipe.named_steps["variance"].transform(x)
    prob = np.asarray(pipe.named_steps["model"].predict_proba(x))
    if prob.ndim != 2 or prob.shape[1] < 2:
        raise RuntimeError("A base model returned an invalid predict_proba result.")
    return prob[:, 1].astype(float)


def align_raw_dataframe(x_raw: pd.DataFrame) -> pd.DataFrame:
    """Return a copy containing the model's raw features in the fitted order."""
    if not isinstance(x_raw, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    x = x_raw.copy()
    for feature in RAW_FEATURES:
        if feature not in x.columns:
            x[feature] = np.nan
    return x[RAW_FEATURES]


def predict_raw_dataframe(x_raw: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
    """Predict PCa probability from raw clinical values."""
    x = align_raw_dataframe(x_raw)

    base_probabilities: Dict[str, np.ndarray] = {}
    for model_name in BASE_MODEL_ORDER:
        base_probabilities[model_name] = _base_model_positive_probability(
            BASE_MODELS[model_name], x
        )

    meta_df = pd.DataFrame(
        {
            f"P_{model_name}": base_probabilities[model_name]
            for model_name in BASE_MODEL_ORDER
        },
        index=x.index,
    )
    meta_df = meta_df[META_FEATURE_COLUMNS]

    final_prob = np.asarray(META_MODEL.predict_proba(meta_df))
    if final_prob.ndim != 2 or final_prob.shape[1] < 2:
        raise RuntimeError("Meta XGBoost returned an invalid predict_proba result.")
    return final_prob[:, 1].astype(float), meta_df


def fitted_imputation_values() -> Dict[str, float]:
    """Extract the fitted raw-feature imputation values from a saved base model."""
    reference_model = BASE_MODELS[BASE_MODEL_ORDER[0]]
    preprocess = reference_model.named_steps["preprocess"]
    values: Dict[str, float] = {}

    for name, transformer, columns in preprocess.transformers_:
        if name == "remainder" or not hasattr(transformer, "named_steps"):
            continue
        imputer = transformer.named_steps.get("imputer")
        if imputer is None:
            continue
        for column, statistic in zip(columns, imputer.statistics_):
            values[str(column)] = float(statistic)

    if set(values) != set(RAW_FEATURES):
        missing = [f for f in RAW_FEATURES if f not in values]
        raise RuntimeError(f"Unable to recover fitted imputation values for: {missing}")
    return values


IMPUTATION_VALUES = fitted_imputation_values()


def impute_raw_for_explanation(x_raw: pd.DataFrame) -> pd.DataFrame:
    """Fill missing raw inputs with the fitted training-set imputation values."""
    x = align_raw_dataframe(x_raw)
    for feature in RAW_FEATURES:
        x[feature] = pd.to_numeric(x[feature], errors="coerce").fillna(
            IMPUTATION_VALUES[feature]
        )
    return x


def classify(probability: float) -> int:
    """Apply the fixed training-set Youden cutoff."""
    return int(float(probability) >= CUTOFF)
