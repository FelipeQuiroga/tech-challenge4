import pandas as pd
import pytest

from src.contracts import enforce_contract, make_dirty_batch, validate_batch

CLEAN_ROW = {
    "loan_amnt": 10000.0,
    "term": 36,
    "int_rate": 12.5,
    "installment": 320.0,
    "emp_length": 5.0,
    "home_ownership": "RENT",
    "annual_inc": 55000.0,
    "verification_status": "Verified",
    "purpose": "debt_consolidation",
    "dti": 18.5,
    "delinq_2yrs": 0.0,
    "fico_range_low": 700.0,
    "open_acc": 8.0,
    "pub_rec": 0.0,
    "revol_bal": 5000.0,
    "revol_util": 45.0,
    "total_acc": 20.0,
    "target": 0,
}


@pytest.fixture
def clean_df() -> pd.DataFrame:
    df = pd.DataFrame([CLEAN_ROW] * 30)
    # evita colidir com a regra "sem duplicatas" do contrato
    df["loan_amnt"] = df["loan_amnt"] + df.index
    return df


def test_clean_batch_passes_contract(clean_df):
    ok, errors, summary = validate_batch(clean_df)
    assert ok is True
    assert errors is None


def test_dirty_batch_is_rejected(clean_df, tmp_path):
    dirty = make_dirty_batch(clean_df, n=20, seed=1)
    ok, errors, summary = validate_batch(dirty)
    assert ok is False
    assert errors is not None
    assert len(errors) > 0


def test_enforce_contract_blocks_ingestion_and_quarantines(clean_df, tmp_path):
    dirty = make_dirty_batch(clean_df, n=20, seed=2)
    with pytest.raises(ValueError, match="rejeitado pelo contrato"):
        enforce_contract(dirty, batch_name="unit_test_dirty", quarantine_dir=tmp_path)

    rejected_files = list(tmp_path.glob("unit_test_dirty_*"))
    assert len(rejected_files) == 2  # rejected.csv + errors.csv


def test_enforce_contract_allows_clean_batch(clean_df, tmp_path):
    result = enforce_contract(clean_df, batch_name="unit_test_clean", quarantine_dir=tmp_path)
    assert len(result) == len(clean_df)
    assert list(tmp_path.glob("unit_test_clean_*")) == []


@pytest.mark.parametrize(
    "column,bad_value",
    [
        ("annual_inc", None),
        ("annual_inc", -500.0),
        ("fico_range_low", 10.0),
        ("term", 12),
        ("dti", 150.0),
    ],
)
def test_individual_rule_violations_are_caught(clean_df, column, bad_value):
    bad_df = clean_df.copy()
    bad_df.loc[0, column] = bad_value
    ok, errors, _ = validate_batch(bad_df)
    assert ok is False
