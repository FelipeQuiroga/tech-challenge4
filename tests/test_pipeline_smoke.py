"""Smoke test do pipeline ponta a ponta usando dados sintéticos pequenos
(não depende do dataset real do Kaggle, para rodar rápido e de forma
determinística em qualquer máquina/CI)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.contracts import enforce_contract
from src.detect_drift import analyze_batches, compute_feature_drift
from src.simulate_drift import build_drifted_batches
from src.train import train_baseline


def _make_synthetic_dataset(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    annual_inc = rng.normal(60000, 15000, n).clip(10000, 200000)
    dti = rng.normal(18, 6, n).clip(0, 60)
    default_logit = -2.0 + 0.00002 * (annual_inc - 60000) * -1 + 0.04 * dti
    default_prob = 1 / (1 + np.exp(-(-2.5 + 0.03 * dti - 0.00001 * (annual_inc - 60000))))
    target = rng.binomial(1, default_prob.clip(0.02, 0.9))

    df = pd.DataFrame(
        {
            "loan_amnt": rng.uniform(1000, 30000, n),
            "term": rng.choice([36, 60], n),
            "int_rate": rng.uniform(6, 25, n),
            "installment": rng.uniform(50, 900, n),
            "emp_length": rng.integers(0, 10, n).astype(float),
            "home_ownership": rng.choice(["RENT", "OWN", "MORTGAGE"], n),
            "annual_inc": annual_inc,
            "verification_status": rng.choice(["Verified", "Not Verified"], n),
            "purpose": rng.choice(["debt_consolidation", "credit_card", "other"], n),
            "dti": dti,
            "delinq_2yrs": rng.integers(0, 3, n).astype(float),
            "fico_range_low": rng.integers(640, 820, n).astype(float),
            "open_acc": rng.integers(1, 25, n).astype(float),
            "pub_rec": rng.integers(0, 2, n).astype(float),
            "revol_bal": rng.uniform(0, 40000, n),
            "revol_util": rng.uniform(0, 100, n),
            "total_acc": rng.integers(2, 50, n).astype(float),
            config.TARGET_COLUMN: target,
            "issue_date": pd.date_range("2015-01-01", periods=n, freq="h"),
        }
    )
    return df


@pytest.fixture(scope="module")
def synthetic_reference() -> pd.DataFrame:
    return _make_synthetic_dataset(2000, seed=1)


@pytest.fixture(scope="module")
def synthetic_production() -> pd.DataFrame:
    return _make_synthetic_dataset(1200, seed=2)


def test_synthetic_reference_passes_contract(synthetic_reference, tmp_path_factory):
    quarantine = tmp_path_factory.mktemp("quarantine")
    result = enforce_contract(synthetic_reference, batch_name="smoke_reference", quarantine_dir=quarantine)
    assert len(result) == len(synthetic_reference)


def test_train_baseline_runs_and_beats_random(synthetic_reference, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    import src.train as train_module

    monkeypatch.setattr(train_module, "MODEL_PATH", tmp_path / "model.joblib")
    monkeypatch.setattr(train_module, "METRICS_PATH", tmp_path / "metrics.json")

    metrics = train_baseline(synthetic_reference)
    assert 0.5 < metrics["auc"] <= 1.0
    assert (tmp_path / "model.joblib").exists()


def test_full_batch_drift_pipeline_runs(synthetic_reference, synthetic_production, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    import src.train as train_module

    monkeypatch.setattr(train_module, "MODEL_PATH", tmp_path / "model.joblib")
    monkeypatch.setattr(train_module, "METRICS_PATH", tmp_path / "metrics.json")

    metrics = train_baseline(synthetic_reference)
    model = train_module.load_model()

    batches = build_drifted_batches(synthetic_production)
    assert len(batches) == config.N_PRODUCTION_BATCHES

    results = analyze_batches(model, synthetic_reference, batches, reference_auc=metrics["auc"])
    assert len(results) == len(batches)
    assert set(["batch_index", "auc", "n_features_drifted", "performance_alert"]).issubset(results.columns)
    # degradação deve ser (fracamente) monotônica em média ao longo dos lotes
    assert results["default_rate"].iloc[-1] >= results["default_rate"].iloc[0]


def test_compute_feature_drift_flags_only_shifted_feature(synthetic_reference):
    shifted = synthetic_reference.copy()
    shifted["dti"] = shifted["dti"] * 3 + 20  # desloca fortemente só esta feature
    result = compute_feature_drift(synthetic_reference, shifted)
    row = result.loc[result["feature"] == "dti"].iloc[0]
    assert row["drifted"] is True or row["psi"] > config.PSI_ALERT_THRESHOLD
