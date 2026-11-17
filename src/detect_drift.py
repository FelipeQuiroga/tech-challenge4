"""Etapa 2 — detecção estatística de Data/Concept Drift com Evidently AI.

Para cada lote de produção (já com drift simulado em `simulate_drift.py`),
compara a distribuição de cada feature numérica contra o Dataset de
Referência usando PSI e teste de Kolmogorov-Smirnov (via Evidently), e mede
a degradação da performance do modelo (AUC) — evidenciando tanto Data Drift
quanto Concept Drift.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.metrics import ValueDrift
from evidently.presets import DataDriftPreset
from sklearn.metrics import roc_auc_score

from . import config

logger = logging.getLogger(__name__)


def _data_definition() -> DataDefinition:
    return DataDefinition(
        numerical_columns=config.NUMERIC_FEATURES,
        categorical_columns=config.CATEGORICAL_FEATURES,
    )


def _to_dataset(df: pd.DataFrame) -> Dataset:
    cols = config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES
    return Dataset.from_pandas(df[cols].copy(), data_definition=_data_definition())


@dataclass
class FeatureDriftResult:
    feature: str
    feature_type: str
    psi: float | None
    ks_or_test_pvalue: float | None
    drifted: bool


def compute_feature_drift(reference_df: pd.DataFrame, batch_df: pd.DataFrame) -> pd.DataFrame:
    """PSI + teste estatístico (KS para numéricas, Z/Chi2 para categóricas)
    por feature, usando o Evidently como engine de cálculo.
    """
    ref_ds = _to_dataset(reference_df)
    cur_ds = _to_dataset(batch_df)

    # PSI é usado como critério PRIMÁRIO de decisão (numéricas e
    # categóricas): é uma métrica de "effect size", pouco sensível ao
    # tamanho da amostra — ao contrário de testes de p-valor (KS, qui-
    # quadrado/Z), que com n grande (aqui 60k vs 10k) acusam "significância"
    # para diferenças mínimas de puro ruído de amostragem (ver README,
    # seção de Governança/Causalidade, sobre essa armadilha estatística).
    # O p-valor de KS ainda é calculado e reportado de forma informativa
    # para as features numéricas, por ser explicitamente citado no enunciado.
    metrics = []
    for col in config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES:
        metrics.append(ValueDrift(column=col, method="psi"))
    for col in config.NUMERIC_FEATURES:
        metrics.append(ValueDrift(column=col, method="ks"))

    report = Report(metrics=metrics)
    snapshot = report.run(reference_data=ref_ds, current_data=cur_ds)
    result = snapshot.dict()

    rows: dict[str, dict] = {}
    for m in result["metrics"]:
        cfg = m["config"]
        col = cfg.get("column")
        method = cfg.get("method")
        value = m["value"]
        rows.setdefault(col, {"feature": col})
        if method == "psi":
            rows[col]["psi"] = float(value)
        elif method == "ks":
            rows[col]["ks_pvalue"] = float(value)

    out = pd.DataFrame(rows.values())
    if "ks_pvalue" not in out.columns:
        out["ks_pvalue"] = pd.NA
    out["drifted"] = out["psi"] > config.PSI_ALERT_THRESHOLD
    return out.sort_values("feature").reset_index(drop=True)


def compute_model_performance(model, batch_df: pd.DataFrame) -> dict:
    X = batch_df[config.FEATURE_COLUMNS]
    y = batch_df[config.TARGET_COLUMN]
    proba = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, proba) if y.nunique() > 1 else float("nan")
    return {"auc": float(auc), "n_rows": len(batch_df), "default_rate": float(y.mean())}


def analyze_batches(
    model,
    reference_df: pd.DataFrame,
    batches: list[pd.DataFrame],
    reference_auc: float,
) -> pd.DataFrame:
    """Roda a detecção de drift + performance para cada lote de produção."""
    records = []
    for i, batch in enumerate(batches):
        feature_drift = compute_feature_drift(reference_df, batch)
        perf = compute_model_performance(model, batch)
        n_drifted = int(feature_drift["drifted"].sum())
        auc_drop = reference_auc - perf["auc"]

        record = {
            "batch_index": i,
            "n_rows": perf["n_rows"],
            "default_rate": perf["default_rate"],
            "auc": perf["auc"],
            "auc_drop_vs_reference": auc_drop,
            "n_features_drifted": n_drifted,
            "drifted_features": ",".join(feature_drift.loc[feature_drift["drifted"], "feature"]),
            "performance_alert": auc_drop > config.AUC_DROP_ALERT_PP,
        }
        records.append(record)
        logger.info(
            "Lote %s: AUC=%.4f (queda=%.4f) | %s features com drift: %s",
            i, perf["auc"], auc_drop, n_drifted, record["drifted_features"],
        )
    return pd.DataFrame(records)


def generate_evidently_html_report(
    reference_df: pd.DataFrame, most_drifted_batch: pd.DataFrame, output_path
) -> None:
    """Gera o relatório HTML do Evidently (Etapa 2 — entregável obrigatório)."""
    ref_ds = _to_dataset(reference_df)
    cur_ds = _to_dataset(most_drifted_batch)
    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(reference_data=ref_ds, current_data=cur_ds)
    snapshot.save_html(str(output_path))
    logger.info("Relatório Evidently salvo em %s", output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from .data_prep import load_or_build
    from .simulate_drift import build_drifted_batches
    from .train import load_model

    ref_df, prod_df = load_or_build()
    model = load_model()
    batches = build_drifted_batches(prod_df)

    import json

    with open(config.MODELS_DIR / "baseline_metrics.json", encoding="utf-8") as fh:
        reference_auc = json.load(fh)["auc"]

    results = analyze_batches(model, ref_df, batches, reference_auc)
    results.to_csv(config.REPORTS_DIR / "drift_metrics.csv", index=False)
    print(results.to_string(index=False))

    generate_evidently_html_report(ref_df, batches[-1], config.REPORTS_DIR / "drift_report.html")
