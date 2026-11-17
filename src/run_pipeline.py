"""Orquestrador ponta a ponta do pipeline de Sustentação e Confiabilidade.

Executa, em sequência, as Etapas 1–3 do Tech Challenge:
  1. Validação de dados (contrato) + treino do baseline.
  2. Simulação de drift + detecção estatística (PSI/KS) nos lotes de produção.
  3. Observabilidade: logs estruturados, MLflow e dashboard HTML.

Uso:
    python -m src.run_pipeline
"""
from __future__ import annotations

import json
import logging
import time

from . import config
from .contracts import enforce_contract, make_dirty_batch
from .data_prep import load_or_build
from .detect_drift import analyze_batches, generate_evidently_html_report
from .observability import (
    build_dashboard_html,
    log_drift_batch_run,
    log_pipeline_event,
    log_training_run,
    read_pipeline_events,
)
from .simulate_drift import build_drifted_batches
from .train import train_baseline

logger = logging.getLogger(__name__)


def run() -> None:
    t0 = time.time()
    log_pipeline_event("pipeline_start")

    # --- Etapa 1: dados + contrato + baseline -----------------------------
    t_step = time.time()
    ref_df, prod_df = load_or_build()
    log_pipeline_event(
        "data_ingestion", status="success", n_reference=len(ref_df), n_production=len(prod_df),
        latency_seconds=round(time.time() - t_step, 2),
    )

    enforce_contract(ref_df, batch_name="reference_training_batch")

    # Demonstração obrigatória: lote sujo é bloqueado pelo contrato.
    dirty_batch = make_dirty_batch(ref_df, n=50, seed=123)
    try:
        enforce_contract(dirty_batch, batch_name="demo_dirty_batch")
    except ValueError as exc:
        logger.info("Bloqueio esperado do lote sujo demonstrado com sucesso: %s", exc)

    t_step = time.time()
    metrics = train_baseline(ref_df)
    log_training_run(metrics)
    log_pipeline_event(
        "model_training_step", status="success", auc=metrics["auc"],
        latency_seconds=round(time.time() - t_step, 2),
    )

    # --- Etapa 2: drift ------------------------------------------------
    t_step = time.time()
    batches = build_drifted_batches(prod_df)
    from .train import load_model

    model = load_model()
    results = analyze_batches(model, ref_df, batches, reference_auc=metrics["auc"])
    results.to_csv(config.REPORTS_DIR / "drift_metrics.csv", index=False)

    for _, row in results.iterrows():
        log_drift_batch_run(int(row["batch_index"]), row.to_dict())

    generate_evidently_html_report(ref_df, batches[-1], config.REPORTS_DIR / "drift_report.html")
    log_pipeline_event(
        "drift_analysis_step",
        status="success",
        n_batches=len(batches),
        n_alerts=int(results["performance_alert"].sum()),
        latency_seconds=round(time.time() - t_step, 2),
    )

    # --- Etapa 3: observabilidade ---------------------------------------
    events_df = read_pipeline_events()
    build_dashboard_html(results, metrics, events_df, config.REPORTS_DIR / "dashboard.html")

    log_pipeline_event(
        "pipeline_end", status="success", total_latency_seconds=round(time.time() - t0, 2)
    )
    logger.info("Pipeline completo em %.2fs", time.time() - t0)
    print(json.dumps({"baseline_metrics": metrics, "n_batches": len(batches)}, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run()
