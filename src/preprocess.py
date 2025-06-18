"""
src/preprocess.py
-----------------
Transformador sklearn (imputación + One-Hot) y helper de carga.
"""
from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .config import CATEGORICAL_COLS, NUMERIC_COLS, TARGET_COL

logger = logging.getLogger(__name__)

def build_preprocessor() -> ColumnTransformer:
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )
    num_pipe = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    return ColumnTransformer(
        transformers=[
            ("cat", cat_pipe, CATEGORICAL_COLS),
            ("num", num_pipe, NUMERIC_COLS),
        ]
    )

def load_and_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Devuelve X, y binaria.
    Ignora filas cuya etiqueta no sea 'SI' ni 'NO'.
    """
    etiquetas = (
        df[TARGET_COL]
        .astype(str)          # convierte nulos a 'nan'
        .str.strip()          # quita espacios
        .str.upper()          # mayúsculas uniformes
    )

    mask = etiquetas.isin({"SI", "NO"})
    if mask.sum() == 0:
        raise ValueError("No hay filas con etiqueta válida 'SI' o 'NO'.")

    df_clean = df.loc[mask].copy()
    y = etiquetas.loc[mask].map({"SI": 1, "NO": 0}).astype(int).values
    X = df_clean.drop(columns=[TARGET_COL])

    return X, y