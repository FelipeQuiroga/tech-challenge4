"""Etapa 3 — Observabilidade de Pipelines e Modelos.

Centraliza:
  * Logs estruturados JSON (cada etapa do pipeline grava um evento em
    `logs/pipeline.jsonl` — já usado por `contracts.py`).
  * Rastreamento de experimentos/métricas no MLflow (runs para o baseline
    e para cada lote de drift analisado).
  * Geração de um dashboard HTML estático consolidando: status do
    pipeline, métricas operacionais e alertas de drift/qualidade.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import mlflow
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import config

logger = logging.getLogger(__name__)


def log_pipeline_event(event_type: str, **fields) -> None:
    """Grava um evento estruturado de observabilidade do pipeline."""
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event_type,
        **fields,
    }
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.PIPELINE_LOG_FILE, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, ensure_ascii=False) + "\n")


def read_pipeline_events() -> pd.DataFrame:
    if not config.PIPELINE_LOG_FILE.exists():
        return pd.DataFrame()
    rows = []
    with open(config.PIPELINE_LOG_FILE, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def setup_mlflow() -> None:
    # MLflow aposentou o backend puro de arquivos (./mlruns); usamos SQLite
    # local, que também não exige um servidor MLflow rodando.
    db_path = config.MLRUNS_DIR / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{db_path.as_posix()}")
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)


def log_training_run(metrics: dict) -> str:
    """Registra o treino do baseline como um run do MLflow."""
    setup_mlflow()
    with mlflow.start_run(run_name="baseline_training") as run:
        mlflow.log_params(
            {
                "model": "XGBClassifier",
                "n_features": len(config.FEATURE_COLUMNS),
                "random_state": config.RANDOM_STATE,
            }
        )
        mlflow.log_metrics(
            {
                "auc": metrics["auc"],
                "ks": metrics["ks"],
                "default_rate_train": metrics["default_rate_train"],
                "default_rate_test": metrics["default_rate_test"],
            }
        )
        run_id = run.info.run_id
    log_pipeline_event("model_training", status="success", auc=metrics["auc"], ks=metrics["ks"], mlflow_run_id=run_id)
    return run_id


def log_drift_batch_run(batch_index: int, record: dict, n_drifted_threshold: int = 1) -> str:
    """Registra a análise de drift de um lote de produção como um run do MLflow."""
    setup_mlflow()
    with mlflow.start_run(run_name=f"drift_batch_{batch_index}") as run:
        mlflow.log_params({"batch_index": batch_index})
        mlflow.log_metrics(
            {
                "auc": record["auc"],
                "auc_drop_vs_reference": record["auc_drop_vs_reference"],
                "n_features_drifted": record["n_features_drifted"],
                "default_rate": record["default_rate"],
            }
        )
        mlflow.set_tag("drifted_features", record["drifted_features"])
        mlflow.set_tag("performance_alert", str(record["performance_alert"]))
        run_id = run.info.run_id

    log_pipeline_event(
        "drift_detection",
        status="alert" if record["performance_alert"] or record["n_features_drifted"] >= n_drifted_threshold else "ok",
        batch_index=batch_index,
        auc=record["auc"],
        auc_drop_vs_reference=record["auc_drop_vs_reference"],
        n_features_drifted=record["n_features_drifted"],
        drifted_features=record["drifted_features"],
        mlflow_run_id=run_id,
    )
    return run_id


def build_dashboard_html(
    drift_results: pd.DataFrame,
    baseline_metrics: dict,
    events_df: pd.DataFrame,
    output_path,
) -> None:
    """Gera um dashboard HTML estático (Plotly) com as métricas de saúde do
    pipeline/modelo e os alertas de drift — Etapa 3 (entregável obrigatório).
    """
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "AUC do modelo por lote de produção",
            "Taxa de default observada por lote",
            "Nº de features com drift (PSI) por lote",
            "Eventos do pipeline por status",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=drift_results["batch_index"],
            y=drift_results["auc"],
            mode="lines+markers",
            name="AUC produção",
            line=dict(color="#2563eb"),
        ),
        row=1,
        col=1,
    )
    fig.add_hline(
        y=baseline_metrics["auc"],
        line_dash="dash",
        line_color="#16a34a",
        annotation_text="AUC referência",
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=drift_results["batch_index"],
            y=drift_results["default_rate"],
            name="Taxa de default",
            marker_color="#ea580c",
        ),
        row=1,
        col=2,
    )

    colors = ["#dc2626" if alert else "#16a34a" for alert in drift_results["performance_alert"]]
    fig.add_trace(
        go.Bar(
            x=drift_results["batch_index"],
            y=drift_results["n_features_drifted"],
            name="Features com drift",
            marker_color=colors,
        ),
        row=2,
        col=1,
    )

    if not events_df.empty and "status" in events_df.columns:
        status_counts = events_df["status"].fillna("info").value_counts()
        fig.add_trace(
            go.Bar(x=status_counts.index, y=status_counts.values, name="Eventos", marker_color="#7c3aed"),
            row=2,
            col=2,
        )

    fig.update_layout(
        title_text="Dashboard de Observabilidade — Credit Scoring (Sustentação & Confiabilidade)",
        height=800,
        showlegend=False,
    )
    fig.update_xaxes(title_text="Lote (mês simulado)", row=1, col=1)
    fig.update_xaxes(title_text="Lote (mês simulado)", row=1, col=2)
    fig.update_xaxes(title_text="Lote (mês simulado)", row=2, col=1)

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path), include_plotlyjs="cdn")
    logger.info("Dashboard salvo em %s", output_path)
