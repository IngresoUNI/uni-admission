from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, RocCurveDisplay,
    precision_recall_curve, PrecisionRecallDisplay
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from .config import DATA_PATH, MODEL_OUT, PLOT_OUT, RANDOM_STATE, ARTEFACT_DIR
from .preprocess import build_preprocessor, load_and_split

# -------------------- LOGGING -------------------- #
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# -------------------- FUNCIONES PARA GRÁFICOS -------------------- #

def save_feature_plot(model: XGBClassifier, preprocessor, out_path: Path, top_n: int = 20) -> None:
    """
    Guarda un gráfico de las variables más importantes según gain (XGBoost).
    """
    feature_names = preprocessor.get_feature_names_out()
    booster = model.get_booster()
    gains = booster.get_score(importance_type="gain")

    readable_importances = {
        feature_names[int(k[1:])]: v for k, v in gains.items() if k.startswith("f")
    }

    importances_df = (
        pd.DataFrame.from_dict(readable_importances, orient="index", columns=["gain"])
        .sort_values("gain", ascending=False)
        .head(top_n)
    )

    plt.figure(figsize=(12, 10))
    plt.barh(importances_df.index[::-1], importances_df["gain"][::-1], height=0.6)
    plt.title(f"Top {top_n} Importancia de Variables (gain)", fontsize=14)
    plt.xlabel("Ganancia (gain)", fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=9)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.subplots_adjust(left=0.35)
    plt.savefig(out_path)
    plt.close()

    # También guardamos el CSV con las importancias
    importances_df.to_csv(ARTEFACT_DIR / "feature_importances.csv")


def save_confusion_matrix(y_true, y_pred, out_path: Path) -> None:
    """
    Guarda la matriz de confusión como imagen.
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title("Matriz de Confusión")
    plt.savefig(out_path)
    plt.close()


def save_roc_curve(y_true, y_score, out_path: Path) -> None:
    """
    Guarda la curva ROC como imagen.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr)
    roc_display.plot()
    plt.title("Curva ROC")
    plt.savefig(out_path)
    plt.close()


def save_precision_recall_curve(y_true, y_score, out_path: Path) -> None:
    """
    Guarda la curva Precisión-Recall como imagen.
    """
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    pr_display = PrecisionRecallDisplay(precision=prec, recall=rec)
    pr_display.plot()
    plt.title("Curva Precisión vs Recall")
    plt.savefig(out_path)
    plt.close()


def save_global_metrics_plot(y_true, y_pred, out_path: Path) -> None:
    """
    Gráfico de barras comparando precisión, recall y F1-score por clase.
    """
    report = classification_report(y_true, y_pred, output_dict=True)
    metrics = ["precision", "recall", "f1-score"]
    labels = ["No (0)", "Sí (1)"]
    data = [[report["0"][m] for m in metrics], [report["1"][m] for m in metrics]]

    df = pd.DataFrame(data, index=labels, columns=metrics)
    df.plot(kind="bar", figsize=(8, 6))
    plt.title("Métricas por Clase")
    plt.ylabel("Valor")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# -------------------- ENTRENAMIENTO PRINCIPAL -------------------- #

def train(data_path: Path, model_out: Path, plot_out: Path) -> None:
    """
    Entrena un modelo XGBoost usando los datos especificados.

    Guarda:
    - modelo + pipeline en formato joblib
    - gráficos de evaluación en artefacts/
    """
    df = pd.read_excel(data_path)
    X, y = load_and_split(df)

    # División de entrenamiento y prueba
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    # Preprocesamiento
    preprocessor = build_preprocessor()
    X_tr_trans = preprocessor.fit_transform(X_tr)
    X_te_trans = preprocessor.transform(X_te)

    # Modelo
    model = XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="auc",
        n_jobs=4,
        random_state=RANDOM_STATE,
        scale_pos_weight=5.09  # ajustar desequilibrio
    )

    logger.info("Training…")
    model.fit(X_tr_trans, y_tr)

    # Evaluación
    proba = model.predict_proba(X_te_trans)[:, 1]
    threshold = 0.485
    y_pred = (proba >= threshold).astype(int)

    auc = roc_auc_score(y_te, proba)
    acc = accuracy_score(y_te, y_pred)
    logger.info("AUC=%.3f | Accuracy=%.3f", auc, acc)
    logger.info("\n%s", classification_report(y_te, y_pred))

    # Guardar modelo y gráficos
    joblib.dump((preprocessor, model), model_out)
    logger.info("Modelo guardado como tupla en %s", model_out)

    save_feature_plot(model, preprocessor, plot_out)
    save_confusion_matrix(y_te, y_pred, ARTEFACT_DIR / "confusion_matrix.png")
    save_roc_curve(y_te, proba, ARTEFACT_DIR / "roc_curve.png")
    save_precision_recall_curve(y_te, proba, ARTEFACT_DIR / "precision_recall_curve.png")
    save_global_metrics_plot(y_te, y_pred, ARTEFACT_DIR / "global_metrics.png")
    logger.info("Gráficos guardados en artefacts/")


# -------------------- EJECUCIÓN COMO SCRIPT -------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--model-out", type=Path, default=MODEL_OUT)
    parser.add_argument("--plot-out", type=Path, default=PLOT_OUT)
    args = parser.parse_args()

    train(args.data, args.model_out, args.plot_out)
