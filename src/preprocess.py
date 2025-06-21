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

# Configurar logging si se desea usar en otros scripts
logger = logging.getLogger(__name__)

def build_preprocessor() -> ColumnTransformer:
    """
    Crea el transformador de preprocesamiento con imputación y codificación.

    Returns:
        ColumnTransformer: transformador con dos pipelines para columnas categóricas y numéricas.
    """
    # Pipeline para columnas categóricas: imputar con la moda + OneHotEncoder
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )

    # Pipeline para columnas numéricas: imputar con la mediana
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median"))
        ]
    )

    # Combinar en un ColumnTransformer
    return ColumnTransformer(
        transformers=[
            ("cat", cat_pipe, CATEGORICAL_COLS),
            ("num", num_pipe, NUMERIC_COLS),
        ]
    )

def load_and_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Filtra y divide un DataFrame en características (X) y objetivo binario (y).
    Sólo conserva registros con 'SI' o 'NO' en la columna objetivo.

    Args:
        df (pd.DataFrame): dataset original con la columna de objetivo.

    Returns:
        Tuple[pd.DataFrame, np.ndarray]: X limpio (sin la columna objetivo), y binario (0 o 1).

    Raises:
        ValueError: si no hay ninguna fila válida con 'SI' o 'NO'.
    """

    # Normalizar etiquetas: quitar espacios, pasar a mayúsculas
    etiquetas = (
        df[TARGET_COL]
        .astype(str)        # nulos se convierten a "nan"
        .str.strip()        # eliminar espacios
        .str.upper()        # todo en mayúscula para estandarizar
    )

    # Solo considerar etiquetas válidas
    mask = etiquetas.isin({"SI", "NO"})
    if mask.sum() == 0:
        raise ValueError("No hay filas con etiqueta válida 'SI' o 'NO'.")

    # Filtrar DataFrame
    df_clean = df.loc[mask].copy()

    # Mapear a binario
    y = etiquetas.loc[mask].map({"SI": 1, "NO": 0}).astype(int).values

    # Separar características
    X = df_clean.drop(columns=[TARGET_COL])

    return X, y