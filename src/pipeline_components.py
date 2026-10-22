"""
Customer Churn Intelligence Platform
Reusable Pipeline Components

Encapsulates domain feature engineering and column transformations into
scikit-learn BaseEstimator / TransformerMixin classes.
Guarantees fold-safe cross-validation with zero data leakage and enables
1-step end-to-end inference directly from raw customer dictionary/DataFrame.
"""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np
import pandas as pd


class TelcoFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Applies audited, row-level domain feature engineering:
    1. TotalCharges numeric conversion & zero-tenure handling (0.0).
    2. Tenure dynamics: tenure_years, is_first_year (<=12m), is_long_term (>=48m).
    3. Safe historical monthly billing & charge discrepancy.
    4. Service ecosystem add-on depth & retention anchors.
    5. Payment friction indicators (electronic check, auto-payment, manual paperless).
    6. Demographic household vulnerability flags.
    7. Standardizes binary flags to 0/1 integers.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        # Row-level deterministic transformations require no parameter fitting
        return self

    def transform(self, X):
        # Convert to DataFrame if not already
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        df = X.copy()

        # Drop identifiers and targets if accidentally passed
        for col_to_remove in ["customerID", "Churn", "Churn_bin"]:
            if col_to_remove in df.columns:
                df = df.drop(columns=[col_to_remove])

        # 1. TotalCharges numeric cleaning
        if "TotalCharges" in df.columns:
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)

        # 2. Tenure Dynamics
        if "tenure" in df.columns:
            df["tenure_years"] = (df["tenure"] / 12.0).round(2)
            df["is_first_year"] = (df["tenure"] <= 12).astype(int)
            df["is_long_term"] = (df["tenure"] >= 48).astype(int)

            # 3. Financial Metrics & Pricing Discrepancy
            safe_tenure = np.where(df["tenure"] == 0, 1, df["tenure"])
            if "TotalCharges" in df.columns and "MonthlyCharges" in df.columns:
                df["avg_historical_monthly"] = np.where(
                    df["tenure"] == 0,
                    df["MonthlyCharges"],
                    (df["TotalCharges"] / safe_tenure)
                ).round(2)
                df["charge_discrepancy"] = (df["MonthlyCharges"] - df["avg_historical_monthly"]).round(2)

        # 4. Service Ecosystem & Retention Anchors
        addon_cols = [
            "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies"
        ]
        present_addons = [c for c in addon_cols if c in df.columns]
        if present_addons:
            df["total_addons"] = df[present_addons].apply(lambda row: sum(row == "Yes"), axis=1)
        else:
            df["total_addons"] = 0

        if "OnlineSecurity" in df.columns and "TechSupport" in df.columns:
            df["has_security_or_support"] = (
                (df["OnlineSecurity"] == "Yes") | (df["TechSupport"] == "Yes")
            ).astype(int)

        if "StreamingTV" in df.columns and "StreamingMovies" in df.columns:
            df["has_streaming_services"] = (
                (df["StreamingTV"] == "Yes") | (df["StreamingMovies"] == "Yes")
            ).astype(int)

        # 5. Payment Friction
        if "PaymentMethod" in df.columns:
            df["is_electronic_check"] = (df["PaymentMethod"] == "Electronic check").astype(int)
            df["is_auto_payment"] = df["PaymentMethod"].astype(str).str.contains("automatic", case=False).astype(int)
            if "PaperlessBilling" in df.columns:
                df["is_paperless_and_manual"] = (
                    (df["PaperlessBilling"] == "Yes") & (df["is_auto_payment"] == 0)
                ).astype(int)

        # 6. Household & Vulnerability
        if "SeniorCitizen" in df.columns and "Partner" in df.columns and "Dependents" in df.columns:
            df["senior_living_alone"] = (
                (df["SeniorCitizen"] == 1) & (df["Partner"] == "No") & (df["Dependents"] == "No")
            ).astype(int)
            df["family_support_household"] = (
                (df["Partner"] == "Yes") & (df["Dependents"] == "Yes")
            ).astype(int)

        # 7. Standardize binary columns to 0/1 integers
        binary_maps = {
            "gender": {"Male": 1, "Female": 0},
            "Partner": {"Yes": 1, "No": 0},
            "Dependents": {"Yes": 1, "No": 0},
            "PhoneService": {"Yes": 1, "No": 0},
            "PaperlessBilling": {"Yes": 1, "No": 0}
        }
        for col, mapping in binary_maps.items():
            if col in df.columns:
                df[f"{col}_bin"] = df[col].map(mapping).fillna(0).astype(int)
                df = df.drop(columns=[col])

        return df


def get_column_transformer() -> ColumnTransformer:
    """Builds the scikit-learn ColumnTransformer for scaling and encoding."""
    numeric_features = [
        "tenure", "MonthlyCharges", "TotalCharges", "tenure_years",
        "avg_historical_monthly", "charge_discrepancy", "total_addons"
    ]

    categorical_features = [
        "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
        "Contract", "PaymentMethod"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), categorical_features)
        ],
        remainder="passthrough",
        verbose_feature_names_out=False
    )
    return preprocessor


def create_full_pipeline(classifier) -> Pipeline:
    """
    Creates an end-to-end Pipeline combining:
    1. TelcoFeatureEngineer()
    2. ColumnTransformer (StandardScaler + OneHotEncoder)
    3. Classifier model

    Guarantees 100% fold isolation during CV and 1-step raw input inference.
    """
    return Pipeline([
        ("feature_engineer", TelcoFeatureEngineer()),
        ("preprocessor", get_column_transformer()),
        ("classifier", classifier)
    ])
