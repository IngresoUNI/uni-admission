"""
src/predict.py
--------------
Predice la probabilidad de ingreso para uno o varios postulantes.

Ejemplos:
    python -m src.predict --json "{\"MODALIDAD\":\"Ordinario\",...}"
    python -m src.predict --csv nuevos_postulantes.csv
"""
from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd

from .config import MODEL_OUT, CATEGORICAL_COLS, NUMERIC_COLS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def _ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    for col in CATEGORICAL_COLS + NUMERIC_COLS:
        if col not in df.columns:
            df[col] = np.nan
    return df

def predict_df(df: pd.DataFrame, model_path: Path) -> pd.Series:
    model = joblib.load(model_path)
    df = _ensure_cols(df)
    return pd.Series(model.predict_proba(df)[:, 1], name="prob_ingreso")

def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    src = parser.add_mutually_exclusive_group(required=True)

    # opción original
    src.add_argument("--json", type=str, help="Registro único en formato JSON")

    # ➜ NUEVA opción
    src.add_argument("--json-file", type=Path,
                     help="Ruta a un archivo .json con un registro")

    src.add_argument("--csv", type=Path, help="Archivo CSV con varios registros")
    parser.add_argument("--model", type=Path, default=MODEL_OUT)
    args = parser.parse_args(argv)

    # --- leer JSON ---
    if args.json_file:
        args.json = Path(args.json_file).read_text(encoding="utf-8")

    if args.json:
        df = pd.DataFrame([json.loads(args.json)])
    else:
        df = pd.read_csv(args.csv)

    probs = predict_df(df, args.model)
    for p in probs:
        risk = "Low" if p > 0.7 else "Medium" if p > 0.4 else "High"
        print(f"Probabilidad={p:.2%} | Riesgo={risk}")

if __name__ == "__main__":
    main()
