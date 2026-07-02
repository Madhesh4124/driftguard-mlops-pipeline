import pandera.pandas as pa
import pandas as pd

# Define the schema for a single record (represented as a DataFrame row)
credit_schema = pa.DataFrameSchema(
    columns={
        "LIMIT_BAL": pa.Column(float, checks=pa.Check.greater_than(0)),
        "AGE": pa.Column(float, checks=pa.Check.greater_than_or_equal_to(18)),
        "SEX": pa.Column(float, checks=pa.Check.isin([1.0, 2.0])),
        "EDUCATION": pa.Column(float, checks=pa.Check.isin([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])),
        "MARRIAGE": pa.Column(float, checks=pa.Check.isin([0.0, 1.0, 2.0, 3.0])),
        "PAY_0": pa.Column(float),
        "PAY_2": pa.Column(float),
        "PAY_3": pa.Column(float),
        "PAY_4": pa.Column(float),
        "PAY_5": pa.Column(float),
        "PAY_6": pa.Column(float),
        "BILL_AMT1": pa.Column(float),
        "BILL_AMT2": pa.Column(float),
        "BILL_AMT3": pa.Column(float),
        "BILL_AMT4": pa.Column(float),
        "BILL_AMT5": pa.Column(float),
        "BILL_AMT6": pa.Column(float),
        "PAY_AMT1": pa.Column(float, checks=pa.Check.greater_than_or_equal_to(0)),
        "PAY_AMT2": pa.Column(float, checks=pa.Check.greater_than_or_equal_to(0)),
        "PAY_AMT3": pa.Column(float, checks=pa.Check.greater_than_or_equal_to(0)),
        "PAY_AMT4": pa.Column(float, checks=pa.Check.greater_than_or_equal_to(0)),
        "PAY_AMT5": pa.Column(float, checks=pa.Check.greater_than_or_equal_to(0)),
        "PAY_AMT6": pa.Column(float, checks=pa.Check.greater_than_or_equal_to(0)),
        "default.payment.next.month": pa.Column(float, checks=pa.Check.isin([0.0, 1.0]), required=False)
    },
    coerce=True,
    strict=False  # Allow other columns (e.g. ID)
)

def validate_record(record: dict):
    """
    Validates a single record dictionary against the Pandera schema.
    Returns (True, None) if valid, or (False, error_message) if invalid.
    """
    df = pd.DataFrame([record])
    try:
        credit_schema.validate(df)
        return True, None
    except Exception as e:
        return False, str(e)

def validate_batch(records: list):
    """
    Validates a list of record dictionaries against the Pandera schema.
    Returns (valid_records, list of (invalid_record, error_message)).
    """
    df = pd.DataFrame(records)
    valid_records = []
    invalid_records = []
    try:
        # Validate with lazy=True to collect all errors
        credit_schema.validate(df, lazy=True)
        return records, []
    except pa.errors.SchemaErrors as exc:
        failed_indices = set(exc.failure_cases["index"].dropna().astype(int).tolist())
        error_map = {}
        for _, row in exc.failure_cases.iterrows():
            idx_val = row.get("index")
            if pd.notna(idx_val):
                idx = int(idx_val)
                reason = f"Column '{row.get('column')}' failed check: {row.get('check')}"
                error_map.setdefault(idx, []).append(reason)
        
        for i, record in enumerate(records):
            if i in failed_indices:
                err_msg = "; ".join(error_map.get(i, ["Schema validation failed"]))
                invalid_records.append((record, err_msg))
            else:
                valid_records.append(record)
        return valid_records, invalid_records
    except Exception as e:
        return [], [(r, str(e)) for r in records]
