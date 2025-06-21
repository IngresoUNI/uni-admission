from __future__ import annotations
import os
from pathlib import Path
from typing import List

# ==============================================================================
# CONFIGURACIÓN GENERAL DE RUTAS Y PARÁMETROS DEL PROYECTO
# ==============================================================================

# Ruta base del proyecto (nivel superior del paquete)
BASE_DIR = Path(__file__).resolve().parents[1]

# Ruta al archivo Excel con los datos de entrenamiento
# Se puede definir externamente con la variable de entorno UNI_DATA_PATH
DATA_PATH = Path(os.getenv("UNI_DATA_PATH", BASE_DIR / "Ingresantes_UNI.xlsx"))

# Ruta al directorio donde se guardarán los artefactos generados
# Se puede redefinir con la variable de entorno UNI_ARTEFACT_DIR
ARTEFACT_DIR = Path(os.getenv("UNI_ARTEFACT_DIR", BASE_DIR / "artefacts"))
ARTEFACT_DIR.mkdir(parents=True, exist_ok=True)  # Crear si no existe

# Ruta donde se guardará el modelo entrenado (pipeline completo)
MODEL_OUT = ARTEFACT_DIR / "xgb_uni_pipeline.joblib"

# Ruta del gráfico de importancia de variables
PLOT_OUT = ARTEFACT_DIR / "feature_importances.png"

# Semilla aleatoria para reproducibilidad
RANDOM_STATE: int = 42

# ==============================================================================
# DEFINICIÓN DE VARIABLES
# ==============================================================================

# Variables categóricas del dataset
CATEGORICAL_COLS: List[str] = [
    "COLEGIO",
    "COLEGIO_DIST",
    "COLEGIO_PROV",
    "COLEGIO_DEPA",
    "ESPECIALIDAD",
    "MODALIDAD",
    "SEXO"
]

# Variables numéricas del dataset
NUMERIC_COLS: List[str] = [
    "COLEGIO_ANIO_EGRESO",
    "ANIO_POSTULA",           # <- corregido (antes: ANIO_POSTULACION)
    "ANIO_NACIMIENTO",
    "h_e_Matemática",         # <- corregido (antes: h_e_Matematicas)
    "h_e_fisica_quimica",     # <- corregido (antes: h_e_Fisica_Quimica)
    "h_e_Aptitud",
    "h_total_semana"
]

# Variable objetivo
TARGET_COL: str = "INGRESO"
