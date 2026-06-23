import pandera as pa
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
        "BILL_AMT1": pa.Column(float),
        "BILL_AMT2": pa.Column(float),
        "PAY_AMT1": pa.Column(float, checks=pa.Check.greater_than_or_equal_to(0)),
        "PAY_AMT2": pa.Column(float, checks=pa.Check.greater_than_or_equal_to(0)),
        "default.payment.next.month": pa.Column(float, checks=pa.Check.isin([0.0, 1.0]), required=False)
    },
    coerce=True,
    strict=False  # Allow other columns (e.g. ID, other BILL_AMTs)
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
