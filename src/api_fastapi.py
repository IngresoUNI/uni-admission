# ==============================
# API con FastAPI para predicción de ingreso a la UNI
# ==============================

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

# Importar constantes desde config (rutas relativas dentro del proyecto)
from src.config import MODEL_OUT, CATEGORICAL_COLS, NUMERIC_COLS

# Crear instancia de FastAPI
app = FastAPI()

# Cargar las plantillas desde la carpeta "templates" ubicada en la raíz del proyecto
templates = Jinja2Templates(directory="templates")

# Cargar el modelo entrenado y el preprocesador
# Este archivo debe existir en artefacts/xgb_uni_pipeline.joblib después de entrenar
preprocessor, model = joblib.load(MODEL_OUT)

# Umbral de clasificación binaria (ajustado previamente)
THRESHOLD = 0.485

# Definición del esquema de datos de entrada con valores por defecto
class EstudianteInput(BaseModel):
    MODALIDAD: str
    COLEGIO_DEPA: str
    COLEGIO_ANIO_EGRESO: int
    SEXO: str
    ANIO_NACIMIENTO: int
    COLEGIO_PAIS: str = "PERÚ"
    DOMICILIO_DEPA: str = "LIMA"
    NACIMIENTO_PAIS: str = "PERÚ"
    NACIMIENTO_DEPA: str = "LIMA"

# Ruta GET que devuelve el formulario HTML (frontend.html)
@app.get("/", response_class=HTMLResponse)
def mostrar_formulario(request: Request):
    return templates.TemplateResponse("frontend.html", {"request": request})

# Ruta POST para recibir datos del estudiante y hacer predicción
@app.post("/predecir")
def predecir_estudiante(data: EstudianteInput):
    # Convertir datos en un DataFrame
    df = pd.DataFrame([data.dict()])

    # Asegurar que todas las columnas requeridas estén presentes
    for col in CATEGORICAL_COLS + NUMERIC_COLS:
        if col not in df.columns:
            df[col] = np.nan

    # Aplicar preprocesamiento y predicción
    X = preprocessor.transform(df)
    prob = model.predict_proba(X)[:, 1][0]
    pred = int(prob >= THRESHOLD)

    # Asignar nivel de riesgo según la probabilidad
    riesgo = "Low" if prob > 0.7 else "Medium" if prob > 0.4 else "High"

    # Devolver respuesta como JSON
    return {
        "probabilidad_ingreso": round(float(prob) * 100, 2),
        "prediccion": pred,
        "riesgo": riesgo
    }
