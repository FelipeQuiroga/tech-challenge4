"""Data Contracts (Etapa 1) — contrato de qualidade de dados com Pandera.

Garante que "lixo não entre no modelo": valida um lote recebido contra regras
rígidas de negócio/schema. Se o lote falhar, a ingestão é bloqueada e o lote
é movido para `data/quarantine/` junto com um relatório dos erros.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema
from pandera.errors import SchemaErrors

from . import config

logger = logging.getLogger(__name__)

VALID_TERMS = {36, 60}
VALID_HOME_OWNERSHIP = {"RENT", "OWN", "MORTGAGE", "ANY", "NONE", "OTHER"}
VALID_VERIFICATION = {"Verified", "Source Verified", "Not Verified"}

# --------------------------------------------------------------------------
# Contrato: >= 3 regras rígidas exigidas pelo enunciado; aqui implementamos
# um conjunto mais amplo cobrindo nulos, ranges de negócio e domínios.
# --------------------------------------------------------------------------
LOAN_BATCH_SCHEMA = DataFrameSchema(
    columns={
        "loan_amnt": Column(float, Check.in_range(500, 100_000), nullable=False),
        "term": Column(int, Check.isin(VALID_TERMS), nullable=False),
        "int_rate": Column(float, Check.in_range(3.0, 40.0), nullable=False),
        "installment": Column(float, Check.greater_than(0), nullable=False),
        "emp_length": Column(float, Check.in_range(0, 10), nullable=True),
        "home_ownership": Column(str, Check.isin(VALID_HOME_OWNERSHIP), nullable=False),
        "annual_inc": Column(float, Check.in_range(0.01, 500_000), nullable=False),
        "verification_status": Column(str, Check.isin(VALID_VERIFICATION), nullable=False),
        "purpose": Column(str, nullable=False),
        "dti": Column(float, Check.in_range(0, 100), nullable=False),
        "delinq_2yrs": Column(float, Check.greater_than_or_equal_to(0), nullable=True),
        "fico_range_low": Column(float, Check.in_range(300, 850), nullable=False),
        "open_acc": Column(float, Check.greater_than_or_equal_to(0), nullable=True),
        "pub_rec": Column(float, Check.greater_than_or_equal_to(0), nullable=True),
        "revol_bal": Column(float, Check.greater_than_or_equal_to(0), nullable=True),
        "revol_util": Column(float, Check.in_range(0, 200), nullable=True),
        "total_acc": Column(float, Check.greater_than_or_equal_to(0), nullable=True),
    },
    checks=[
        # Regra extra: nenhuma linha totalmente duplicada no lote.
        Check(lambda df: ~df.duplicated().any(), error="lote contém linhas duplicadas"),
    ],
    strict=False,  # permite colunas auxiliares (target, issue_date) sem rejeitar o lote
    coerce=False,
)


def validate_batch(df: pd.DataFrame) -> tuple[bool, pd.DataFrame | None, str | None]:
    """Valida um lote contra o contrato.

    Retorna (ok, lazy_errors_df_ou_None, mensagem_resumo_ou_None).
    """
    try:
        LOAN_BATCH_SCHEMA.validate(df, lazy=True)
        return True, None, None
    except SchemaErrors as exc:
        failure_cases = exc.failure_cases
        summary = (
            f"{len(failure_cases)} violações de contrato encontradas "
            f"em {failure_cases['column'].nunique()} colunas distintas."
        )
        return False, failure_cases, summary


def enforce_contract(
    df: pd.DataFrame,
    batch_name: str = "batch",
    quarantine_dir: Path | None = None,
) -> pd.DataFrame:
    """Aplica o contrato; bloqueia e envia para quarentena se inválido.

    Levanta ValueError se o lote violar o contrato (ingestão bloqueada).
    Em caso de sucesso, retorna o próprio DataFrame validado.
    """
    quarantine_dir = Path(quarantine_dir or config.QUARANTINE_DIR)
    ok, failure_cases, summary = validate_batch(df)

    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": "data_contract_validation",
        "batch_name": batch_name,
        "n_rows": len(df),
        "passed": ok,
        "summary": summary,
    }

    if not ok:
        quarantine_dir.mkdir(parents=True, exist_ok=True)
        bad_path = quarantine_dir / f"{batch_name}_rejected.csv"
        errors_path = quarantine_dir / f"{batch_name}_errors.csv"
        df.to_csv(bad_path, index=False)
        failure_cases.to_csv(errors_path, index=False)
        event["quarantine_path"] = str(bad_path)
        event["errors_path"] = str(errors_path)
        logger.error("Contrato de dados FALHOU para lote '%s': %s", batch_name, summary)
        _log_event(event)
        raise ValueError(f"Lote '{batch_name}' rejeitado pelo contrato de dados: {summary}")

    logger.info("Contrato de dados OK para lote '%s' (%s linhas).", batch_name, len(df))
    _log_event(event)
    return df


def _log_event(event: dict) -> None:
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.PIPELINE_LOG_FILE, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, ensure_ascii=False) + "\n")


def make_dirty_batch(clean_df: pd.DataFrame, n: int = 50, seed: int = 0) -> pd.DataFrame:
    """Gera um lote corrompido propositalmente, para demonstrar o bloqueio
    do contrato (Etapa 1: 'forçar a passagem de um lote de dados com erros').
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    sample = clean_df.sample(n=min(n, len(clean_df)), random_state=seed).reset_index(drop=True)
    dirty = sample.copy()

    # 1) nulos em coluna obrigatória
    idx_null = rng.choice(dirty.index, size=max(1, n // 10), replace=False)
    dirty.loc[idx_null, "annual_inc"] = None

    # 2) renda negativa (viola range de negócio)
    idx_neg = rng.choice(dirty.index, size=max(1, n // 10), replace=False)
    dirty.loc[idx_neg, "annual_inc"] = -1000.0

    # 3) idade/FICO fora do range plausível
    idx_fico = rng.choice(dirty.index, size=max(1, n // 10), replace=False)
    dirty.loc[idx_fico, "fico_range_low"] = 10.0

    # 4) term fora do domínio permitido {36, 60}
    idx_term = rng.choice(dirty.index, size=max(1, n // 10), replace=False)
    dirty.loc[idx_term, "term"] = 12

    # 5) duplicatas exatas
    dup_rows = dirty.iloc[: max(1, n // 10)]
    dirty = pd.concat([dirty, dup_rows], ignore_index=True)

    return dirty
