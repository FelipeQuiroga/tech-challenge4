"""Etapa 1 — treino do classificador base de risco de crédito.

Treina em cima do Dataset de Referência (já validado pelo contrato de
dados), usando um Pipeline sklearn (imputação + one-hot) + XGBoost.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from . import config

logger = logging.getLogger(__name__)

MODEL_PATH = config.MODELS_DIR / "baseline_model.joblib"
METRICS_PATH = config.MODELS_DIR / "baseline_metrics.json"


def build_pipeline() -> Pipeline:
    numeric_transformer = SimpleImputer(strategy="median")
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, config.NUMERIC_FEATURES),
            ("cat", categorical_transformer, config.CATEGORICAL_FEATURES),
        ]
    )

    try:
        from xgboost import XGBClassifier

        classifier = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="auc",
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        )
    except ImportError:  # pragma: no cover - fallback de robustez
        from sklearn.ensemble import HistGradientBoostingClassifier

        logger.warning("xgboost indisponível, usando HistGradientBoostingClassifier.")
        classifier = HistGradientBoostingClassifier(random_state=config.RANDOM_STATE)

    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """KS entre as distribuições de score de bons e maus pagadores."""
    from scipy.stats import ks_2samp

    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(ks_2samp(pos, neg).statistic)


def train_baseline(reference_df: pd.DataFrame) -> dict:
    X = reference_df[config.FEATURE_COLUMNS]
    y = reference_df[config.TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=config.RANDOM_STATE, stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    proba_test = pipeline.predict_proba(X_test)[:, 1]
    metrics = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "auc": float(roc_auc_score(y_test, proba_test)),
        "ks": ks_statistic(y_test.to_numpy(), proba_test),
        "default_rate_train": float(y_train.mean()),
        "default_rate_test": float(y_test.mean()),
    }

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    with open(METRICS_PATH, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, ensure_ascii=False)

    logger.info("Baseline treinado: AUC=%.4f KS=%.4f", metrics["auc"], metrics["ks"])
    return metrics


def load_model() -> Pipeline:
    return joblib.load(MODEL_PATH)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from .contracts import enforce_contract
    from .data_prep import load_or_build

    ref_df, _ = load_or_build()
    enforce_contract(ref_df, batch_name="reference_training_batch")
    metrics = train_baseline(ref_df)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
