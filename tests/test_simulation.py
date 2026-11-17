import numpy as np
import pandas as pd
import pytest

from src import config
from src.simulate_drift import apply_concept_drift, apply_data_drift, build_drifted_batches, split_into_batches


@pytest.fixture
def toy_production_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 600
    df = pd.DataFrame(
        {
            "annual_inc": rng.normal(60000, 15000, n).clip(min=1000),
            "dti": rng.normal(18, 6, n).clip(0, 100),
            "int_rate": rng.normal(13, 4, n).clip(3, 35),
            config.TARGET_COLUMN: rng.binomial(1, 0.2, n),
            "issue_date": pd.date_range("2015-01-01", periods=n, freq="D"),
        }
    )
    return df


def test_split_into_batches_preserves_total_rows(toy_production_df):
    batches = split_into_batches(toy_production_df, n_batches=6)
    assert len(batches) == 6
    assert sum(len(b) for b in batches) == len(toy_production_df)


def test_data_drift_increases_income_dti_and_rate(toy_production_df):
    batches = split_into_batches(toy_production_df, n_batches=6)
    untouched = batches[0]["annual_inc"].mean()
    drifted = apply_data_drift(batches[0], batch_idx=5, n_batches=6)
    assert drifted["annual_inc"].mean() > untouched
    assert drifted["dti"].mean() >= batches[0]["dti"].mean()
    assert drifted["int_rate"].mean() > batches[0]["int_rate"].mean()


def test_no_drift_at_batch_zero(toy_production_df):
    batches = split_into_batches(toy_production_df, n_batches=6)
    batch0 = apply_data_drift(batches[0], batch_idx=0, n_batches=6)
    pd.testing.assert_series_equal(batch0["annual_inc"], batches[0]["annual_inc"])


def test_concept_drift_increases_default_rate_over_batches(toy_production_df):
    batches = build_drifted_batches(toy_production_df)
    default_rates = [b[config.TARGET_COLUMN].mean() for b in batches]
    # taxa de default deve ser (fracamente) crescente do primeiro ao último lote
    assert default_rates[-1] > default_rates[0]


def test_build_drifted_batches_has_batch_index(toy_production_df):
    batches = build_drifted_batches(toy_production_df)
    for i, b in enumerate(batches):
        assert (b["batch_index"] == i).all()
