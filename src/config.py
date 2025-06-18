"""
Config central (PORTABLE)
-------------------------
1.  Si pasas --data / --model desde CLI, esos mandan.
2.  Si defines variables de entorno (UNI_DATA_PATH, UNI_ARTEFACT_DIR),
    tienen prioridad.
3.  Si no, usa rutas relativas dentro del proyecto.
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import List

# ------------------------------------------------------------------ #
#  RUTAS                                                             #
# ------------------------------------------------------------------ #
BASE_DIR = Path(__file__).resolve().parents[1]        # carpeta raíz

# Excel con los históricos
DATA_PATH = Path(os.getenv("UNI_DATA_PATH", BASE_DIR / "Ingresantes_UNI.xlsx"))

# Carpeta donde se guardan modelos/gráficos
ARTEFACT_DIR = Path(os.getenv("UNI_ARTEFACT_DIR", BASE_DIR / "artefacts"))
ARTEFACT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_OUT = ARTEFACT_DIR / "xgb_uni_pipeline.joblib"
PLOT_OUT  = ARTEFACT_DIR / "feature_importances.png"

# ------------------------------------------------------------------ #
#  PARÁMETROS                                                       #
# ------------------------------------------------------------------ #
RANDOM_STATE: int = 42

CATEGORICAL_COLS: List[str] = [
    "COLEGIO_DEPA", "COLEGIO_PAIS", "DOMICILIO_DEPA",
    "NACIMIENTO_PAIS", "NACIMIENTO_DEPA", "SEXO", "MODALIDAD",
]

NUMERIC_COLS: List[str] = [
    "COLEGIO_ANIO_EGRESO", "ANIO_NACIMIENTO",
    # columnas de fuga comentadas
]

TARGET_COL: str = "INGRESO"
