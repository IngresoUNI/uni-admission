from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import MODEL_OUT, CATEGORICAL_COLS, NUMERIC_COLS

# Configuración de logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Umbral de decisión para considerar ingreso
THRESHOLD = 0.485

# Función auxiliar que garantiza que todas las columnas requeridas estén presentes
def _ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    for col in CATEGORICAL_COLS + NUMERIC_COLS:
        if col not in df.columns:
            df[col] = np.nan
    return df

# Función principal de predicción
def predict_df(df: pd.DataFrame, model_path: Path) -> pd.DataFrame:
    preprocessor, model = joblib.load(model_path)
    df = _ensure_cols(df)
    X_transformed = preprocessor.transform(df)
    probs = model.predict_proba(X_transformed)[:, 1]
    preds = (probs >= THRESHOLD).astype(int)

    df_result = df.copy()
    df_result["probabilidad_ingreso"] = probs
    df_result["prediccion_ingreso"] = preds
    return df_result

# Función de entrada por línea de comandos
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Predicción de ingreso a la UNI")

    # Sólo una fuente de datos permitida
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--json", type=str, help="Registro único en formato JSON")
    src.add_argument("--json-file", type=Path, help="Ruta a un archivo .json con un registro")
    src.add_argument("--csv", type=Path, help="Archivo CSV con varios registros")
    src.add_argument("--xlsx", type=Path, help="Archivo Excel con varios registros")

    # Modelo y salida
    parser.add_argument("--model", type=Path, default=MODEL_OUT, help="Ruta al modelo entrenado")
    parser.add_argument("--output", type=Path, default=None, help="Ruta de salida opcional (.xlsx)")

    args = parser.parse_args(argv)

    # Cargar JSON desde archivo si se indica
    if args.json_file:
        args.json = args.json_file.read_text(encoding="utf-8")

    # Preparar el DataFrame según el tipo de entrada
    if args.json:
        df = pd.DataFrame([json.loads(args.json)])
    elif args.csv:
        df = pd.read_csv(args.csv)
    elif args.xlsx:
        df = pd.read_excel(args.xlsx)
    else:
        raise ValueError("No se proporcionó una fuente de datos válida.")

    # Ejecutar la predicción
    result_df = predict_df(df, args.model)

    # Si se especifica archivo de salida, guardar en Excel
    if args.output:
        result_df.to_excel(args.output, index=False)
        logger.info("Resultados guardados en %s", args.output)
    else:
        # Mostrar por consola
        for _, row in result_df.iterrows():
            prob = row["probabilidad_ingreso"]
            pred = row["prediccion_ingreso"]
            riesgo = "Low" if prob > 0.7 else "Medium" if prob > 0.4 else "High"
            print(f"Probabilidad={prob:.2%} | Predicción={pred} | Riesgo={riesgo}")

# Punto de entrada si se ejecuta como script
if __name__ == "__main__":
    main()
