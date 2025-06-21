# 🎓 Sistema de Predicción de Ingreso a la UNI

Este proyecto implementa un modelo de Machine Learning (XGBoost + Pipeline de preprocesamiento) para predecir la probabilidad de que un postulante sea admitido a la Universidad Nacional de Ingeniería (UNI), Perú.

Incluye:
- Entrenamiento con archivo Excel real de postulantes.
- CLI para predicción individual.
- Frontend web interactivo con FastAPI.

---

## 📦 Instalación

1. **Clona el repositorio:**

```bash
git clone https://github.com/<TU-USUARIO>/uni-admission.git
cd uni-admission
```

2. **(Opcional) Crea y activa un entorno virtual:**

```bash
# Crear entorno virtual
python -m venv .venv

# Activar entorno
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# Windows CMD
.\.venv\Scriptsctivate.bat
# macOS / Linux
source .venv/bin/activate
```

3. **Instala las dependencias:**

```bash
pip install -r requirements.txt
```

---

## 🧪 Entrenamiento del Modelo

Asegúrate de tener el archivo `Ingresantes_UNI.xlsx` en la raíz del proyecto.

### Entrenar con el Excel por defecto:

```bash
python -m src.train
```

### Entrenar con un archivo ubicado en otra ruta:

```bash
python -m src.train --data "C:/ruta/Ingresantes_UNI.xlsx"
```

O también:

```bash
# Windows
set UNI_DATA_PATH=C:\ruta\Ingresantes_UNI.xlsx
python -m src.train

# macOS / Linux
export UNI_DATA_PATH=/ruta/Ingresantes_UNI.xlsx
python -m src.train
```

### Archivos generados:

- `artefacts/xgb_uni_pipeline.joblib` → Modelo entrenado + Pipeline de preprocesamiento.
- `artefacts/feature_importances.png` → Gráfico de importancia de variables.

---

## 🔍 Predicción por Línea de Comandos (CLI)

### 1. Predicción directa con JSON embebido:

```bash
python -m src.predict --json "{
  \"MODALIDAD\":\"Ordinario\",
  \"SEXO\":\"M\",
  \"COLEGIO_ANIO_EGRESO\":2023,
  \"ANIO_NACIMIENTO\":2005,
  \"COLEGIO_DEPA\":\"LIMA\",
  \"DOMICILIO_DEPA\":\"LIMA\",
  \"NACIMIENTO_PAIS\":\"PERÚ\",
  \"NACIMIENTO_DEPA\":\"LIMA\"
}"
```

### 2. Predicción desde archivo `.json`:

```bash
python -m src.predict --json-file postulante.json
```

---

## 🌐 Interfaz Web (FastAPI)

### 1. Asegúrate de que el modelo esté entrenado:

El archivo `artefacts/xgb_uni_pipeline.joblib` debe existir. Si no lo tienes, primero entrena el modelo.

### 2. Ejecuta el servidor FastAPI:

```bash
uvicorn src.api_fastapi:app --reload
```

El servidor se iniciará en:

```
http://127.0.0.1:8000
```

### 3. Usa el formulario web:

Abre tu navegador y accede a [http://127.0.0.1:8000](http://127.0.0.1:8000) para llenar el formulario y obtener una predicción visual.

---

## 📁 Estructura del Proyecto

```
uni-admission/
├── artefacts/
│   ├── xgb_uni_pipeline.joblib
│   └── feature_importances.png
├── data/
│   └── (opcional) archivos .xlsx
├── src/
│   ├── train.py
│   ├── predict.py
│   ├── api_fastapi.py
│   └── config.py
├── templates/
│   └── frontend.html
├── postulante.json
├── requirements.txt
└── README.md
```

---

## 🛠 Tecnologías Utilizadas

- Python 3.10+
- FastAPI
- XGBoost
- Pandas, NumPy, scikit-learn
- Uvicorn

---

## 💡 Autor

GRUPO 1 UNMSM