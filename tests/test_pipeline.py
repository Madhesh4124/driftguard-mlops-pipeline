import os
import pytest
import pandas as pd
from utils.config_loader import load_config
from utils.schema import validate_record, validate_batch

def test_config_loader():
    config = load_config()
    assert config is not None
    assert "kafka" in config
    assert "drift" in config
    assert "mlflow" in config

def test_single_record_validation():
    # Valid record
    valid_record = {
        "LIMIT_BAL": 20000.0,
        "AGE": 24.0,
        "SEX": 2.0,
        "EDUCATION": 2.0,
        "MARRIAGE": 1.0,
        "PAY_0": 2.0,
        "PAY_2": 2.0,
        "PAY_3": -1.0,
        "PAY_4": -1.0,
        "PAY_5": -2.0,
        "PAY_6": -2.0,
        "BILL_AMT1": 3913.0,
        "BILL_AMT2": 3102.0,
        "BILL_AMT3": 689.0,
        "BILL_AMT4": 0.0,
        "BILL_AMT5": 0.0,
        "BILL_AMT6": 0.0,
        "PAY_AMT1": 0.0,
        "PAY_AMT2": 689.0,
        "PAY_AMT3": 0.0,
        "PAY_AMT4": 0.0,
        "PAY_AMT5": 0.0,
        "PAY_AMT6": 0.0
    }
    is_valid, err = validate_record(valid_record)
    assert is_valid is True
    assert err is None

    # Invalid record (negative age)
    invalid_record = valid_record.copy()
    invalid_record["AGE"] = -5.0
    is_valid, err = validate_record(invalid_record)
    assert is_valid is False
    assert "AGE" in err

def test_batch_validation():
    records = [
        # Valid
        {
            "LIMIT_BAL": 50000.0, "AGE": 30.0, "SEX": 1.0, "EDUCATION": 1.0, "MARRIAGE": 2.0,
            "PAY_0": 0.0, "PAY_2": 0.0, "PAY_3": 0.0, "PAY_4": 0.0, "PAY_5": 0.0, "PAY_6": 0.0,
            "BILL_AMT1": 1000.0, "BILL_AMT2": 1000.0, "BILL_AMT3": 1000.0, "BILL_AMT4": 1000.0, "BILL_AMT5": 1000.0, "BILL_AMT6": 1000.0,
            "PAY_AMT1": 100.0, "PAY_AMT2": 100.0, "PAY_AMT3": 100.0, "PAY_AMT4": 100.0, "PAY_AMT5": 100.0, "PAY_AMT6": 100.0
        },
        # Invalid (negative LIMIT_BAL)
        {
            "LIMIT_BAL": -1000.0, "AGE": 30.0, "SEX": 1.0, "EDUCATION": 1.0, "MARRIAGE": 2.0,
            "PAY_0": 0.0, "PAY_2": 0.0, "PAY_3": 0.0, "PAY_4": 0.0, "PAY_5": 0.0, "PAY_6": 0.0,
            "BILL_AMT1": 1000.0, "BILL_AMT2": 1000.0, "BILL_AMT3": 1000.0, "BILL_AMT4": 1000.0, "BILL_AMT5": 1000.0, "BILL_AMT6": 1000.0,
            "PAY_AMT1": 100.0, "PAY_AMT2": 100.0, "PAY_AMT3": 100.0, "PAY_AMT4": 100.0, "PAY_AMT5": 100.0, "PAY_AMT6": 100.0
        }
    ]
    
    valid_batch, invalid_batch = validate_batch(records)
    
    assert len(valid_batch) == 1
    assert len(invalid_batch) == 1
    assert valid_batch[0]["LIMIT_BAL"] == 50000.0
    assert invalid_batch[0][0]["LIMIT_BAL"] == -1000.0
    assert "LIMIT_BAL" in invalid_batch[0][1]
