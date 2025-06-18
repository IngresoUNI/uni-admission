"""
src/train.py
------------
Entrena un XGBoost dentro de un Pipeline y guarda modelo + gráfico.

Ejecutar:
    python -m src.train
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from .config import DATA_PATH, MODEL_OUT, PLOT_OUT, RANDOM_STATE
from .preprocess import build_preprocessor, load_and_split

# ------------------------------------------------------------------ #
#  Logging básico                                                    #
# ------------------------------------------------------------------ #
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  Modelo (Pipeline)                                                 #
# ------------------------------------------------------------------ #
def build_model() -> Pipeline:
    preprocessor = build_preprocessor()
    xgb = XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="auc",
        n_jobs=4,
        random_state=RANDOM_STATE,
    )
    return Pipeline([("prep", preprocessor), ("clf", xgb)])

# ------------------------------------------------------------------ #
#  Plot de importancias                                              #
# ------------------------------------------------------------------ #
def save_feature_plot(model: Pipeline, out_path: Path, top_n: int = 20) -> None:
    booster = model.named_steps["clf"].get_booster()
    gains = booster.get_score(importance_type="gain")
    importances = (
        pd.DataFrame.from_dict(gains, orient="index", columns=["gain"])
        .sort_values("gain", ascending=False)
        .head(top_n)
    )
    plt.figure(figsize=(8, 6))
    plt.barh(importances.index[::-1], importances["gain"][::-1])
    plt.title(f"Top {top_n} Feature Importances (gain)")
    plt.xlabel("Gain")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

# ------------------------------------------------------------------ #
#  Proceso principal de entrenamiento                                #
# ------------------------------------------------------------------ #
def train(data_path: Path, model_out: Path, plot_out: Path) -> None:
    # 1) Cargar dataset
    df = pd.read_excel(data_path)

    # 2) Transformar etiqueta y separar
    X, y = load_and_split(df)

    # 3) Split 80 / 20 estratificado
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    # 4) Construir y entrenar modelo
    model = build_model()
    logger.info("Training…")
    model.fit(X_tr, y_tr)

    # 5) Evaluar
    proba = model.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, proba)
    acc = accuracy_score(y_te, (proba >= 0.5).astype(int))
    logger.info("AUC=%.3f | Accuracy=%.3f", auc, acc)
    logger.info("\n%s", classification_report(y_te, (proba >= 0.5)))

    # 6) Guardar artefactos
    joblib.dump(model, model_out)
    logger.info("Model saved to %s", model_out)

    save_feature_plot(model, plot_out)
    logger.info("Importance plot saved to %s", plot_out)

# ------------------------------------------------------------------ #
#  CLI                                                               #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--model-out", type=Path, default=MODEL_OUT)
    parser.add_argument("--plot-out", type=Path, default=PLOT_OUT)
    args = parser.parse_args()

    train(args.data, args.model_out, args.plot_out)
