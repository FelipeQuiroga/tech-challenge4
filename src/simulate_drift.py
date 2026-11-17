"""Etapa 2 — simulação de Data Drift e Concept Drift ao longo do tempo.

Fatiamos o Dataset de Produção (já ordenado por data de emissão) em
`N_PRODUCTION_BATCHES` lotes sequenciais, simulando a passagem de "meses".
A cada lote aplicamos uma perturbação crescente:

  * Data Drift: `annual_inc` sobe (efeito de inflação/mudança de perfil) e
    `dti` / `int_rate` também sobem — a distribuição das features muda.
  * Concept Drift: a relação entre as features e o default muda — para o
    mesmo perfil de entrada, a probabilidade real de default aumenta ao
    longo dos lotes (ex.: crise econômica torna clientes antes "bons" mais
    arriscados), o que é simulado re-sorteando o target com uma
    probabilidade ajustada por lote.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def split_into_batches(df: pd.DataFrame, n_batches: int = config.N_PRODUCTION_BATCHES) -> list[pd.DataFrame]:
    """Divide o dataset de produção (ordenado no tempo) em N lotes sequenciais."""
    df_sorted = df.sort_values("issue_date").reset_index(drop=True)
    idx_splits = np.array_split(np.arange(len(df_sorted)), n_batches)
    return [df_sorted.iloc[idx].reset_index(drop=True) for idx in idx_splits]


def apply_data_drift(batch: pd.DataFrame, batch_idx: int, n_batches: int) -> pd.DataFrame:
    """Aplica Data Drift crescente conforme o índice do lote (0 = sem drift)."""
    batch = batch.copy()
    intensity = batch_idx / max(1, n_batches - 1)  # 0.0 .. 1.0

    # Inflação: renda nominal cresce, mas não acompanha custo de vida real.
    income_growth = 1.0 + 0.35 * intensity
    batch["annual_inc"] = batch["annual_inc"] * income_growth

    # Endividamento e taxa de desemprego implícita sobem (dti como proxy).
    batch["dti"] = (batch["dti"] * (1.0 + 0.5 * intensity)).clip(upper=100)

    # Custo de crédito sobe (juros mais altos no mercado).
    batch["int_rate"] = (batch["int_rate"] + 6.0 * intensity).clip(upper=40)

    return batch


def apply_concept_drift(
    batch: pd.DataFrame,
    batch_idx: int,
    n_batches: int,
    seed: int = config.RANDOM_STATE,
) -> pd.DataFrame:
    """Concept drift: a RELAÇÃO feature->default muda ao longo do tempo,
    não apenas a distribuição das features (isso seria só Data Drift).

    Clientes com perfil antes considerado "bom" (target=0) passam a ter
    uma probabilidade crescente de virar default conforme dti sobe e renda
    relativa cai — ou seja, o MESMO perfil de entrada passa a ser mais
    arriscado. Loans já marcados como default permanecem default. Em
    `batch_idx=0` (intensity=0) o lote é um passe neutro exato (sem
    nenhuma alteração de rótulo), garantindo um baseline limpo para a
    detecção de drift comparar.
    """
    batch = batch.copy()
    rng = np.random.default_rng(seed + batch_idx)
    intensity = batch_idx / max(1, n_batches - 1)

    target = batch[config.TARGET_COLUMN].to_numpy().copy()
    if intensity > 0:
        dti_norm = (batch["dti"] / 100).clip(0, 1).fillna(0.5).to_numpy()
        income_rank = batch["annual_inc"].rank(pct=True).fillna(0.5).to_numpy()
        flip_prob = intensity * 0.45 * (0.5 * dti_norm + 0.5 * (1 - income_rank))

        currently_good = target == 0
        flips = rng.random(len(batch)) < flip_prob
        target[currently_good & flips] = 1

    batch[config.TARGET_COLUMN] = target
    return batch


def build_drifted_batches(production_df: pd.DataFrame) -> list[pd.DataFrame]:
    """Gera a sequência de lotes de produção com drift crescente aplicado."""
    batches = split_into_batches(production_df)
    n = len(batches)
    drifted = []
    for i, batch in enumerate(batches):
        batch = apply_data_drift(batch, i, n)
        batch = apply_concept_drift(batch, i, n)
        batch["batch_index"] = i
        drifted.append(batch)
    return drifted


if __name__ == "__main__":
    from .data_prep import load_or_build

    _, prod_df = load_or_build()
    batches = build_drifted_batches(prod_df)
    for i, b in enumerate(batches):
        print(
            f"lote {i}: n={len(b)} annual_inc_mean={b['annual_inc'].mean():.0f} "
            f"dti_mean={b['dti'].mean():.2f} default_rate={b[config.TARGET_COLUMN].mean():.3f}"
        )
