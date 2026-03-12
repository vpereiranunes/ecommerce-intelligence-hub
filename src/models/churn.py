"""
churn.py - Predição de churn com XGBoost + Optuna tuning.
"""
import pandas as pd
import numpy as np
import joblib
import optuna
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, roc_auc_score,
    RocCurveDisplay, ConfusionMatrixDisplay, precision_recall_curve
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from loguru import logger

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
FIGURES_DIR = Path("dashboard/figures")
MODELS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "recency", "frequency", "monetary",
    "n_products", "avg_basket",
    "avg_days_between_purchases",
    "r_score", "f_score", "m_score",
]
TARGET = "churn"


def load_data():
    df = pd.read_parquet(PROCESSED_DIR / "rfm_segmented.parquet")
    X = df[FEATURES].fillna(0)
    y = df[TARGET]
    return X, y, df


def tune_hyperparameters(X_train, y_train, n_trials: int = 30) -> dict:
    """Optuna para busca de hiperparâmetros."""
    logger.info(f"Iniciando tuning com {n_trials} trials...")

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1, 10),
            "eval_metric": "logloss",
            "random_state": 42,
            "verbosity": 0,
        }
        model = XGBClassifier(**params)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
        return scores.mean()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logger.success(f"Melhor AUC-ROC (CV): {study.best_value:.4f}")
    return study.best_params


def find_best_threshold(model, X_val, y_val) -> float:
    """Ajusta threshold para maximizar F1 no conjunto de validação."""
    probs = model.predict_proba(X_val)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_val, probs)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores[:-1])
    best_threshold = thresholds[best_idx]
    logger.info(f"Threshold ótimo: {best_threshold:.3f} (F1: {f1_scores[best_idx]:.3f})")
    return best_threshold


def evaluate_model(model, X_test, y_test, threshold: float = 0.5):
    """Gera métricas e gráficos de avaliação."""
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)

    auc = roc_auc_score(y_test, probs)
    logger.info(f"\nAUC-ROC: {auc:.4f}")
    logger.info(f"\n{classification_report(y_test, preds, target_names=['Ativo', 'Churn'])}")

    # Plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    RocCurveDisplay.from_predictions(y_test, probs, ax=axes[0], name="XGBoost")
    axes[0].plot([0, 1], [0, 1], "k--")
    axes[0].set_title(f"Curva ROC (AUC = {auc:.3f})")

    ConfusionMatrixDisplay.from_predictions(
        y_test, preds, ax=axes[1],
        display_labels=["Ativo", "Churn"], colorbar=False
    )
    axes[1].set_title("Matriz de Confusão")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "churn_evaluation.png", dpi=150, bbox_inches="tight")
    plt.close()

    return auc


def plot_feature_importance(model, feature_names: list):
    importance = pd.Series(model.feature_importances_, index=feature_names).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    importance.plot(kind="barh", ax=ax, color="#3498db")
    ax.set_title("Feature Importance - Churn Model")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()


def run(tune: bool = True, n_trials: int = 30):
    logger.info("=== MODELO: CHURN PREDICTION ===")
    X, y, df = load_data()

    # Split temporal (mais realista que random split)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # SMOTE para balancear classes
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    logger.info(f"Após SMOTE: {y_res.value_counts().to_dict()}")

    if tune:
        best_params = tune_hyperparameters(X_res, y_res, n_trials)
    else:
        best_params = {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.05,
                       "scale_pos_weight": 3, "random_state": 42, "verbosity": 0}

    model = XGBClassifier(**best_params, eval_metric="logloss")
    model.fit(X_res, y_res)

    threshold = find_best_threshold(model, X_test, y_test)
    auc = evaluate_model(model, X_test, y_test, threshold)
    plot_feature_importance(model, FEATURES)

    # Salva modelo e adiciona predições
    joblib.dump({"model": model, "threshold": threshold, "features": FEATURES},
                MODELS_DIR / "churn_model.joblib")

    probs = model.predict_proba(X)[:, 1]
    df["churn_probability"] = probs
    df["churn_predicted"] = (probs >= threshold).astype(int)
    df.to_parquet(PROCESSED_DIR / "final_dataset.parquet", index=False)

    logger.success(f"Modelo salvo. AUC-ROC final: {auc:.4f}")
    return model, auc


if __name__ == "__main__":
    run(tune=True, n_trials=20)
