"""
data_prep.py
Loading and feature engineering for the claim denial prediction task.


"""
from __future__ import annotations

import numpy as np
import pandas as pd

ID_COL = "claim_id"
TARGET_COL = "is_denied"
LEAKY_COLS = ["denial_reason", "split"]  # never usable as model inputs

CATEGORICAL_COLS = ["payer_id", "payer_type", "visit_type"]
NUMERIC_BASE_COLS = [
    "total_billed",
    "expected_payment",
    "num_procedures",
    "num_diagnoses",
    "days_to_submit",
]
BINARY_COLS = [
    "prior_auth_required",
    "has_prior_auth",
    "is_in_network",
    "missing_documentation_flag",
    "eligibility_verified",
    "referral_required",
    "referral_present",
]


def load_history(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def load_current(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features. Safe to call on both history and current data
    since it only ever looks at a single row's own columns."""
    out = df.copy()

    # Payment / billing shape
    out["payment_ratio"] = out["expected_payment"] / out["total_billed"].replace(0, np.nan)
    out["payment_ratio"] = out["payment_ratio"].fillna(0)
    out["procedures_per_diagnosis"] = out["num_procedures"] / out["num_diagnoses"].replace(0, np.nan)
    out["procedures_per_diagnosis"] = out["procedures_per_diagnosis"].fillna(0)

    # Domain "gap" flags -- the actual risk signal a human reviewer would look for
    out["prior_auth_gap"] = ((out["prior_auth_required"] == 1) & (out["has_prior_auth"] == 0)).astype(int)
    out["referral_gap"] = ((out["referral_required"] == 1) & (out["referral_present"] == 0)).astype(int)
    out["eligibility_gap"] = (out["eligibility_verified"] == 0).astype(int)
    out["network_gap"] = (out["is_in_network"] == 0).astype(int)
    out["late_submission_flag"] = (out["days_to_submit"] > 30).astype(int)

    # Count of distinct risk gaps present on the claim -- a simple composite score
    out["total_risk_gaps"] = (
        out["prior_auth_gap"]
        + out["referral_gap"]
        + out["eligibility_gap"]
        + out["network_gap"]
        + out["missing_documentation_flag"]
        + out["late_submission_flag"]
    )

    # Seasonality without leaking year
    month_num = pd.to_datetime(out["service_month"], format="%Y-%m").dt.month
    out["month_sin"] = np.sin(2 * np.pi * month_num / 12)
    out["month_cos"] = np.cos(2 * np.pi * month_num / 12)

    return out


def get_engineered_feature_names() -> list[str]:
    return [
        "payment_ratio",
        "procedures_per_diagnosis",
        "prior_auth_gap",
        "referral_gap",
        "eligibility_gap",
        "network_gap",
        "late_submission_flag",
        "total_risk_gaps",
        "month_sin",
        "month_cos",
    ]


def get_feature_columns() -> dict:
    """Returns the column groups used to build the sklearn ColumnTransformer."""
    return {
        "categorical": CATEGORICAL_COLS,
        "numeric": NUMERIC_BASE_COLS + get_engineered_feature_names(),
        "binary": BINARY_COLS,
    }


def build_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Full pipeline: engineer features and return only model-input columns
    (+ claim_id for joining back later). Drops target/leaky columns if present."""
    engineered = engineer_features(df)
    cols = get_feature_columns()
    feature_cols = cols["categorical"] + cols["numeric"] + cols["binary"]
    keep = [ID_COL] + feature_cols
    return engineered[keep]
