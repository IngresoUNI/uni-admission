from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

from src.config import MODEL_OUT, CATEGORICAL_COLS, NUMERIC_COLS, DATA_PATH

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Para producción, cambia esto a una lista específica de dominios permitidos
    allow_credentials=True,
    allow_methods=["*"],  # Permitir todos los métodos (GET, POST, etc.)
    allow_headers=["*"],  # Permitir todos los encabezados
)

templates = Jinja2Templates(directory="templates")

preprocessor, model = joblib.load(MODEL_OUT)
THRESHOLD = 0.485

class EstudianteInput(BaseModel):
    COLEGIO: str
    COLEGIO_DIST: str
    COLEGIO_PROV: str
    COLEGIO_DEPA: str
    COLEGIO_ANIO_EGRESO: int
    ESPECIALIDAD: str
    MODALIDAD: str
    SEXO: str
    ANIO_POSTULA: int
    ANIO_NACIMIENTO: int
    h_e_Matemática: float
    h_e_fisica_quimica: float
    h_e_Aptitud: float
    h_total_semana: float

@app.get("/", response_class=HTMLResponse)
def mostrar_formulario(request: Request):
    return templates.TemplateResponse("frontend.html", {"request": request})

@app.post("/predecir")
def predecir_estudiante(data: EstudianteInput):
    df = pd.DataFrame([data.dict()])

    for col in CATEGORICAL_COLS + NUMERIC_COLS:
        if col not in df.columns:
            df[col] = np.nan

    X = preprocessor.transform(df)
    prob = model.predict_proba(X)[:, 1][0]
    pred = int(prob >= THRESHOLD)
    riesgo = "Low" if prob > 0.7 else "Medium" if prob > 0.4 else "High"

    return {
        "probabilidad_ingreso": round(float(prob) * 100, 2),
        "prediccion": pred,
        "riesgo": riesgo
    }

@app.get("/opciones")
def obtener_opciones():
    df = pd.read_excel(DATA_PATH)

    opciones = {
        "COLEGIO_DIST": sorted(df["COLEGIO_DIST"].dropna().unique().tolist()),
        "COLEGIO_PROV": sorted(df["COLEGIO_PROV"].dropna().unique().tolist()),
        "COLEGIO_DEPA": sorted(df["COLEGIO_DEPA"].dropna().unique().tolist()),
        "ESPECIALIDAD": sorted(df["ESPECIALIDAD"].dropna().unique().tolist()),
        "MODALIDAD": sorted(df["MODALIDAD"].dropna().unique().tolist()),
        "SEXO": sorted(df["SEXO"].dropna().unique().tolist())
    }

    return opciones
