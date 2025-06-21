# src/train.py
# -----------------------------------------------------------------------------
# Entrenamiento de un modelo XGBoost para predecir ingreso a la UNI
# Incorpora:
#   • Eliminación de duplicados
#   • Split temporal opcional (out-of-time)
#   • Validación cruzada estratificada (AUC)
#   • Gráficas con valores visibles y zoom vertical
# -----------------------------------------------------------------------------
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    RocCurveDisplay,
    precision_recall_curve,
    PrecisionRecallDisplay,
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from .config import DATA_PATH, MODEL_OUT, PLOT_OUT, RANDOM_STATE, ARTEFACT_DIR
from .preprocess import build_preprocessor, load_and_split

# ------------------------- LOGGING GLOBAL ----------------------------------- #
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# -------------------- FUNCIONES AUXILIARES DE GRÁFICA ----------------------- #
def save_feature_plot(model: XGBClassifier, preprocessor, out_path: Path,
                      top_n: int = 20) -> None:
    """Graba un gráfico de importancia de variables basado en 'gain'."""
    feature_names = preprocessor.get_feature_names_out()
    gains = model.get_booster().get_score(importance_type="gain")

    readable = {feature_names[int(k[1:])]: v
                for k, v in gains.items() if k.startswith("f")}
    df = (pd.DataFrame.from_dict(readable, orient="index", columns=["gain"])
            .sort_values("gain", ascending=False)
            .head(top_n))

    plt.figure(figsize=(12, 10))
    plt.barh(df.index[::-1], df["gain"][::-1], height=0.6)
    plt.title(f"Top {top_n} Importancia de Variables (gain)")
    plt.xlabel("Ganancia (gain)")
    plt.tight_layout()
    plt.subplots_adjust(left=0.35)
    plt.savefig(out_path)
    plt.close()

    # CSV adicional
    df.to_csv(ARTEFACT_DIR / "feature_importances.csv")


def save_confusion_matrix(y_true, y_pred, out_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title("Matriz de Confusión")
    plt.savefig(out_path)
    plt.close()


def save_roc_curve(y_true, y_score, out_path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
    plt.title("Curva ROC")
    plt.savefig(out_path)
    plt.close()


def save_precision_recall_curve(y_true, y_score, out_path: Path) -> None:
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    PrecisionRecallDisplay(precision=prec, recall=rec).plot()
    plt.title("Curva Precisión vs Recall")
    plt.savefig(out_path)
    plt.close()


def save_global_metrics_plot(y_true, y_pred, out_path: Path) -> None:
    """Barras de precisión, recall, F1 con valores y zoom vertical."""
    report = classification_report(y_true, y_pred, output_dict=True)
    metrics = ["precision", "recall", "f1-score"]
    labels = ["No (0)", "Sí (1)"]
    data = [[report["0"][m] for m in metrics],
            [report["1"][m] for m in metrics]]

    df = pd.DataFrame(data, index=labels, columns=metrics)
    ax = df.plot(kind="bar", figsize=(8, 6))
    for p in ax.patches:
        ax.text(p.get_x() + p.get_width()/2,
                p.get_height() + 0.002,
                f"{p.get_height():.3f}",
                ha="center", va="bottom", fontsize=8)
    plt.title("Métricas por Clase")
    plt.ylabel("Valor")
    plt.ylim(0.95, 1.0)        # Zoom para distinguir valores altos
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

# ------------------------- ENTRENAMIENTO PRINCIPAL -------------------------- #
def train(data_path: Path,
          model_out: Path,
          plot_out: Path,
          year_cut: int | None = None) -> None:
    """
    Entrena un XGBClassifier + preprocesador.

    Args
    ----
    data_path : ruta al archivo Excel (o UNI_DATA_PATH).
    model_out : ruta de salida .joblib (tupla preprocessor, model).
    plot_out  : imagen con importancias.
    year_cut  : si se indica, realiza split temporal:
                train = ANIO_POSTULA ≤ year_cut, test = > year_cut.
    """
    df = pd.read_excel(data_path)

    # 1) Eliminar duplicados exactos
    dup = df.duplicated().sum()
    if dup:
        logger.warning("Se eliminarán %d duplicados", dup)
        df = df.drop_duplicates()

    # 2) Limpiar y obtener X, y
    X, y = load_and_split(df)

    # 3) Split: temporal o aleatorio estratificado
    if year_cut is not None:
        mask_train = X["ANIO_POSTULA"] <= year_cut
        mask_test = X["ANIO_POSTULA"] > year_cut
        if mask_train.sum() == 0 or mask_test.sum() == 0:
            raise ValueError("El corte temporal deja un conjunto vacío.")
        X_tr, X_te = X[mask_train], X[mask_test]
        y_tr, y_te = y[mask_train], y[mask_test]
        logger.info("Split temporal: train ≤ %d (%d filas) | test > %d (%d filas)",
                    year_cut, len(X_tr), year_cut, len(X_te))
    else:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
        )

    # 4) Preprocesamiento
    preprocessor = build_preprocessor()
    X_tr_trans = preprocessor.fit_transform(X_tr)
    X_te_trans = preprocessor.transform(X_te)

    # 5) Modelo
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
        scale_pos_weight=5.09     # ajusta desbalance
    )

    logger.info("Entrenando modelo …")
    model.fit(X_tr_trans, y_tr)

    # 6) Validación final
    proba = model.predict_proba(X_te_trans)[:, 1]
    threshold = 0.485
    y_pred = (proba >= threshold).astype(int)

    auc = roc_auc_score(y_te, proba)
    acc = accuracy_score(y_te, y_pred)
    logger.info("AUC=%.3f | Accuracy=%.3f", auc, acc)
    logger.info("\n%s", classification_report(y_te, y_pred))

    # 7) Validación cruzada estratificada (full pipeline)
    pipeline_full = Pipeline(steps=[("pre", preprocessor), ("clf", model)])
    cv = StratifiedKFold(n_splits=5, shuffle=True,
                         random_state=RANDOM_STATE)
    scores = cross_val_score(pipeline_full, X, y,
                             cv=cv, scoring="roc_auc", n_jobs=-1)
    logger.info("AUC CV: %.3f ± %.3f", scores.mean(), scores.std())

    # 8) Guardar modelo y gráficas
    joblib.dump((preprocessor, model), model_out)
    logger.info("Modelo guardado en %s", model_out)

    save_feature_plot(model, preprocessor, plot_out)
    save_confusion_matrix(y_te, y_pred, ARTEFACT_DIR / "confusion_matrix.png")
    save_roc_curve(y_te, proba, ARTEFACT_DIR / "roc_curve.png")
    save_precision_recall_curve(y_te, proba,
                                ARTEFACT_DIR / "precision_recall_curve.png")
    save_global_metrics_plot(y_te, y_pred,
                             ARTEFACT_DIR / "global_metrics.png")
    logger.info("Gráficas guardadas en %s", ARTEFACT_DIR)

# ----------------------------- CLI ------------------------------------------ #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento XGBoost UNI")
    parser.add_argument("--data", type=Path, default=DATA_PATH,
                        help="Archivo con datos (por defecto DATA_PATH)")
    parser.add_argument("--model-out", type=Path, default=MODEL_OUT,
                        help="Destino del modelo joblib")
    parser.add_argument("--plot-out", type=Path, default=PLOT_OUT,
                        help="Imagen con importancias")
    parser.add_argument("--year-cut", type=int, default=None,
                        help="Corte temporal: año máximo para TRAIN "
                             "(test usa > year-cut)")
    args = parser.parse_args()

    train(args.data, args.model_out, args.plot_out, args.year_cut)
