"""Preparação dos datasets de Referência e Produção a partir do Lending Club.

Lê o CSV `accepted_2007_to_2018Q4.csv.gz` diretamente de dentro do zip baixado
do Kaggle (sem extrair o arquivo gigante inteiro para disco), em streaming por
chunks, usando apenas as colunas necessárias (privacy by design — nenhuma
coluna de identificação/PII é lida).

Saídas:
    data/processed/reference.parquet   -> usado para treinar/validar o baseline
    data/processed/production.parquet  -> usado para simular e detectar drift
"""
from __future__ import annotations

import gzip
import logging
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

logger = logging.getLogger(__name__)

CHUNKSIZE = 200_000


def _open_accepted_stream(zip_path: Path):
    """Abre o CSV.gz do dataset `accepted` de dentro do zip, como stream."""
    zf = zipfile.ZipFile(zip_path)
    raw = zf.open(config.ACCEPTED_CSV_NAME_IN_ZIP)
    return gzip.GzipFile(fileobj=raw)


def _parse_term(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.extract(r"(\d+)")[0], errors="coerce")


def _parse_emp_length(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    out = s.str.extract(r"(\d+)")[0]
    out = pd.to_numeric(out, errors="coerce")
    out[s.str.contains("< 1", na=False)] = 0
    return out


def _parse_int_rate(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        series = series.astype(str).str.replace("%", "", regex=False)
    return pd.to_numeric(series, errors="coerce")


def _clean_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    chunk = chunk.copy()
    chunk["issue_date"] = pd.to_datetime(chunk["issue_d"], format="%b-%Y", errors="coerce")
    chunk["issue_year"] = chunk["issue_date"].dt.year
    chunk["term"] = _parse_term(chunk["term"])
    chunk["emp_length"] = _parse_emp_length(chunk["emp_length"])
    chunk["int_rate"] = _parse_int_rate(chunk["int_rate"])
    chunk["revol_util"] = _parse_int_rate(chunk["revol_util"])  # também vem com '%'
    chunk = chunk[chunk["loan_status"].isin(config.VALID_LOAN_STATUS)]
    chunk[config.TARGET_COLUMN] = chunk["loan_status"].map(config.VALID_LOAN_STATUS).astype(int)

    # Limpeza de outliers grosseiros (erros de digitação/captura na fonte),
    # mantendo o Dataset de Referência/Produção "limpo" conforme o enunciado.
    # A demonstração do contrato falhando usa um lote sintético à parte
    # (ver contracts.make_dirty_batch), não o dataset real completo.
    chunk = chunk[chunk["dti"].between(0, 100) | chunk["dti"].isna()]
    chunk = chunk[chunk["term"].isin([36, 60])]
    chunk["term"] = chunk["term"].astype(int)
    return chunk


def _sample_df(df: pd.DataFrame, n: int, random_state: int) -> pd.DataFrame:
    if len(df) <= n:
        return df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    # amostragem estratificada pelo target para preservar a taxa de default
    frac = n / len(df)
    sampled = (
        df.groupby(config.TARGET_COLUMN, group_keys=False)
        .apply(lambda g: g.sample(frac=frac, random_state=random_state), include_groups=False)
    )
    sampled[config.TARGET_COLUMN] = df.loc[sampled.index, config.TARGET_COLUMN]
    return sampled.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def build_reference_and_production(
    zip_path: Path | None = None,
    sample_size_reference: int = config.SAMPLE_SIZE_REFERENCE,
    sample_size_production: int = config.SAMPLE_SIZE_PRODUCTION,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lê o zip em streaming e monta os DataFrames de referência e produção."""
    zip_path = Path(zip_path or config.LENDING_CLUB_ZIP)
    if not zip_path.exists():
        raise FileNotFoundError(
            f"Dataset Lending Club não encontrado em {zip_path}. "
            "Baixe o zip do Kaggle e aponte LENDING_CLUB_ZIP para ele."
        )

    usecols = config.ALL_USECOLS
    ref_chunks: list[pd.DataFrame] = []
    prod_chunks: list[pd.DataFrame] = []
    max_year_seen = 0
    rows_read = 0

    target_ref = sample_size_reference * 4
    target_prod = sample_size_production * 4
    max_prod_year = max(config.PRODUCTION_ISSUE_YEARS)

    with _open_accepted_stream(zip_path) as stream:
        reader = pd.read_csv(
            stream,
            usecols=usecols,
            chunksize=CHUNKSIZE,
            low_memory=False,
            on_bad_lines="skip",
        )
        for chunk in reader:
            rows_read += len(chunk)
            chunk = _clean_chunk(chunk)
            if chunk.empty:
                continue
            max_year_seen = max(max_year_seen, int(chunk["issue_year"].max(skipna=True) or 0))

            ref_part = chunk[chunk["issue_year"].isin(config.REFERENCE_ISSUE_YEARS)]
            if not ref_part.empty:
                ref_chunks.append(ref_part)

            prod_part = chunk[chunk["issue_year"].isin(config.PRODUCTION_ISSUE_YEARS)]
            if not prod_part.empty:
                prod_chunks.append(prod_part)

            have_enough = (
                sum(len(c) for c in ref_chunks) >= target_ref
                and sum(len(c) for c in prod_chunks) >= target_prod
            )
            if have_enough and max_year_seen > max_prod_year:
                logger.info("Coleta suficiente após %s linhas lidas; parando leitura.", rows_read)
                break

    if not ref_chunks or not prod_chunks:
        raise RuntimeError("Não foi possível coletar linhas suficientes para referência/produção.")

    ref_df = pd.concat(ref_chunks, ignore_index=True)
    prod_df = pd.concat(prod_chunks, ignore_index=True)

    ref_df = _sample_df(ref_df, sample_size_reference, config.RANDOM_STATE)
    prod_df = _sample_df(prod_df, sample_size_production, config.RANDOM_STATE)
    prod_df = prod_df.sort_values("issue_date").reset_index(drop=True)

    keep_cols = config.FEATURE_COLUMNS + [config.TARGET_COLUMN, "issue_date", "issue_year"]
    ref_df = ref_df[keep_cols]
    prod_df = prod_df[keep_cols]

    return ref_df, prod_df


def save_datasets(ref_df: pd.DataFrame, prod_df: pd.DataFrame) -> None:
    config.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    ref_df.to_parquet(config.REFERENCE_PARQUET, index=False)
    prod_df.to_parquet(config.PRODUCTION_PARQUET, index=False)
    logger.info("Referência: %s linhas -> %s", len(ref_df), config.REFERENCE_PARQUET)
    logger.info("Produção: %s linhas -> %s", len(prod_df), config.PRODUCTION_PARQUET)


def load_or_build() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carrega os parquets já processados, ou os constrói se ainda não existirem."""
    if config.REFERENCE_PARQUET.exists() and config.PRODUCTION_PARQUET.exists():
        return (
            pd.read_parquet(config.REFERENCE_PARQUET),
            pd.read_parquet(config.PRODUCTION_PARQUET),
        )
    ref_df, prod_df = build_reference_and_production()
    save_datasets(ref_df, prod_df)
    return ref_df, prod_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ref_df, prod_df = build_reference_and_production()
    save_datasets(ref_df, prod_df)
    print(ref_df[config.TARGET_COLUMN].value_counts(normalize=True))
    print(prod_df[config.TARGET_COLUMN].value_counts(normalize=True))
