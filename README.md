# =======================================================================
# INSTALACIÓN
# =======================================================================

# 1. Clona el repositorio y entra en la carpeta
git clone https://github.com/<TU-USUARIO>/uni-admission.git
cd uni-admission

# 2. (Opcional) crea y activa un entorno virtual
python -m venv .venv
#   Windows PowerShell
.\.venv\Scripts\Activate.ps1
#   Windows CMD
.\.venv\Scripts\activate.bat
#   macOS / Linux
source .venv/bin/activate

# 3. Instala las dependencias
pip install -r requirements.txt


# =======================================================================
# ENTRENAMIENTO
# =======================================================================

# ► Asegúrate de tener Ingresantes_UNI.xlsx en la raíz del proyecto.
#   (Si lo tienes en otro sitio, usa --data o la variable de entorno UNI_DATA_PATH)

# Entrenamiento con el Excel por defecto (./Ingresantes_UNI.xlsx)
python -m src.train

# Entrenamiento con un Excel en otra carpeta
python -m src.train --data "C:/ruta/Ingresantes_UNI.xlsx"
#   —o bien—
set UNI_DATA_PATH=C:\ruta\Ingresantes_UNI.xlsx     # Windows
export UNI_DATA_PATH=/ruta/Ingresantes_UNI.xlsx    # macOS / Linux
python -m src.train

# Resultados:
#   artefacts/xgb_uni_pipeline.joblib      ← modelo + pipeline de preprocesamiento
#   artefacts/feature_importances.png      ← gráfico de variables más importantes


# =======================================================================
# PREDICCIÓN
# =======================================================================

# 1) Registro individual (JSON escrito en línea)
python -m src.predict --json "{
  \"MODALIDAD\":\"Ordinario\",
  \"COLEGIO_DEPA\":\"LIMA\",
  \"COLEGIO_ANIO_EGRESO\":2023,
  \"SEXO\":\"M\",
  \"ANIO_NACIMIENTO\":2005
}"

# 2) Registro individual desde archivo JSON
python -m src.predict --json-file postulante.json
