"""Configuração central do pipeline de Credit Scoring (Tech Challenge Fase 4).

Mantém em um único lugar: caminhos, colunas usadas, janelas temporais de
referência/produção e limiares de alerta de drift. Nenhum dado sensível
(PII) é referenciado aqui — ver README seção de Governança/LGPD.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Caminhos
# --------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
QUARANTINE_DIR = ROOT_DIR / "data" / "quarantine"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
LOGS_DIR = ROOT_DIR / "logs"
MLRUNS_DIR = ROOT_DIR / "mlruns"

for _dir in (DATA_RAW_DIR, DATA_PROCESSED_DIR, QUARANTINE_DIR, MODELS_DIR, REPORTS_DIR, LOGS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# Fonte do dataset Lending Club (zip baixado manualmente do Kaggle pelo usuário).
# Caminho pode ser sobrescrito pela variável de ambiente LENDING_CLUB_ZIP.
import os  # noqa: E402

LENDING_CLUB_ZIP = Path(
    os.environ.get("LENDING_CLUB_ZIP", r"C:\Users\fequi\Downloads\dataset_att_5.zip")
)
ACCEPTED_CSV_NAME_IN_ZIP = "accepted_2007_to_2018Q4.csv.gz"

REFERENCE_PARQUET = DATA_PROCESSED_DIR / "reference.parquet"
PRODUCTION_PARQUET = DATA_PROCESSED_DIR / "production.parquet"
PIPELINE_LOG_FILE = LOGS_DIR / "pipeline.jsonl"

# --------------------------------------------------------------------------
# Esquema de dados — apenas colunas necessárias (privacy by design: nada de
# id, member_id, zip_code, addr_state, emp_title, desc, url, etc.)
# --------------------------------------------------------------------------
NUMERIC_FEATURES = [
    "loan_amnt",
    "int_rate",
    "installment",
    "annual_inc",
    "dti",
    "delinq_2yrs",
    "fico_range_low",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
]
CATEGORICAL_FEATURES = [
    "term",
    "emp_length",
    "home_ownership",
    "verification_status",
    "purpose",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Colunas extras só para split temporal / rótulo — descartadas antes de treinar.
AUX_COLUMNS = ["issue_d", "loan_status"]
ALL_USECOLS = FEATURE_COLUMNS + AUX_COLUMNS

TARGET_COLUMN = "target"  # 1 = Charged Off (default), 0 = Fully Paid
VALID_LOAN_STATUS = {"Fully Paid": 0, "Charged Off": 1}

# --------------------------------------------------------------------------
# Janela temporal base: referência e produção-base vêm do MESMO ano-safra
# (2015), amostradas sem sobreposição. Isso isola o efeito do drift
# sintético injetado em `simulate_drift.py` — sem essa separação, o drift
# "natural" real entre safras diferentes da Lending Club (mudanças de
# política de crédito ao longo dos anos) se misturaria com o drift
# intencional e poluiria a demonstração estatística. A passagem do tempo em
# produção é simulada fatiando esse pool em lotes e aplicando perturbação
# crescente (ver simulate_drift.py), não usando anos-calendário diferentes.
# --------------------------------------------------------------------------
BASE_ISSUE_YEARS = {2015}

SAMPLE_SIZE_REFERENCE = 60_000
SAMPLE_SIZE_PRODUCTION = 60_000
RANDOM_STATE = 42

# Número de lotes ("meses") em que o dataset de produção é fatiado para
# simular a passagem do tempo e a degradação progressiva.
N_PRODUCTION_BATCHES = 6

# --------------------------------------------------------------------------
# Limiares de alerta de drift / qualidade
# --------------------------------------------------------------------------
PSI_ALERT_THRESHOLD = 0.2          # PSI > 0.2 -> mudança significativa
KS_PVALUE_ALERT_THRESHOLD = 0.05   # p-valor < 0.05 -> distribuições diferentes
AUC_DROP_ALERT_PP = 0.03           # queda de 3 pontos percentuais de AUC vs. referência

MLFLOW_EXPERIMENT_NAME = "credit-scoring-sustentacao"
