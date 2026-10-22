"""
Customer Churn Intelligence Platform
Automated Unit, Integration, and Regression Test Suite (v2.0.0 Release)
"""

import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

src_dir = Path(__file__).parent.parent / "src"
if str(src_dir.resolve()) not in sys.path:
    sys.path.insert(0, str(src_dir.resolve()))

from pipeline_components import TelcoFeatureEngineer, get_column_transformer, create_full_pipeline
from api import app


@pytest.fixture
def sample_customer():
    return pd.DataFrame([{
        "customerID": "1234-TESTX",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.50,
        "TotalCharges": 1026.00
    }])


@pytest.fixture
def zero_tenure_customer():
    return pd.DataFrame([{
        "customerID": "0000-NEWXX",
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 0,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "One year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Mailed check",
        "MonthlyCharges": 45.00,
        "TotalCharges": " "  # Whitespace blank
    }])


@pytest.fixture
def champion_pipeline():
    pipeline_path = Path("models/phase2_churn_pipeline.joblib")
    if not pipeline_path.exists():
        pipeline_path = Path("models/final_churn_pipeline.joblib")
    assert pipeline_path.exists(), "Trained model pipeline not found!"
    return joblib.load(pipeline_path)


@pytest.fixture
def baseline_pipeline():
    pipeline_path = Path("models/final_churn_pipeline.joblib")
    assert pipeline_path.exists(), "Baseline model pipeline not found!"
    return joblib.load(pipeline_path)


@pytest.fixture
def api_client():
    return TestClient(app)


# ==============================================================================
# PIPELINE UNIT TESTS
# ==============================================================================

def test_tenure_zero_handling(zero_tenure_customer):
    """Verify that tenure = 0 and blank TotalCharges are handled gracefully."""
    fe = TelcoFeatureEngineer()
    transformed = fe.transform(zero_tenure_customer)

    assert transformed["TotalCharges"].iloc[0] == 0.0
    assert transformed["avg_historical_monthly"].iloc[0] == 45.00
    assert transformed["charge_discrepancy"].iloc[0] == 0.0
    assert transformed["is_first_year"].iloc[0] == 1


def test_feature_engineering_deterministic(sample_customer):
    """Verify feature engineering produces identical output on repeated calls."""
    fe = TelcoFeatureEngineer()
    res1 = fe.transform(sample_customer)
    res2 = fe.transform(sample_customer)

    pd.testing.assert_frame_equal(res1, res2)


def test_champion_prediction_shape_and_range(champion_pipeline, sample_customer):
    """Verify champion pipeline output dimensions and valid probability bounds [0, 1]."""
    probs = champion_pipeline.predict_proba(sample_customer)

    assert probs.shape == (1, 2)
    assert 0.0 <= probs[0, 0] <= 1.0
    assert 0.0 <= probs[0, 1] <= 1.0
    assert np.isclose(probs[0, 0] + probs[0, 1], 1.0)


def test_baseline_prediction_shape_and_range(baseline_pipeline, sample_customer):
    """Verify baseline pipeline output dimensions and valid probability bounds [0, 1]."""
    probs = baseline_pipeline.predict_proba(sample_customer)

    assert probs.shape == (1, 2)
    assert 0.0 <= probs[0, 0] <= 1.0
    assert 0.0 <= probs[0, 1] <= 1.0
    assert np.isclose(probs[0, 0] + probs[0, 1], 1.0)


def test_pipeline_isolation_of_extra_columns(champion_pipeline, sample_customer):
    """Verify extraneous columns like customerID or Churn do not impact prediction."""
    prob_clean = champion_pipeline.predict_proba(sample_customer)[0, 1]

    dirty_customer = sample_customer.copy()
    dirty_customer["customerID"] = "UNSEEN-ID-999"
    dirty_customer["Churn"] = "Yes"
    dirty_customer["random_extra_metric"] = 99999.0

    prob_dirty = champion_pipeline.predict_proba(dirty_customer)[0, 1]
    assert np.isclose(prob_clean, prob_dirty), "Pipeline output altered by extra columns!"


def test_unseen_category_handling(champion_pipeline, sample_customer):
    """Verify pipeline does not crash when encountering an unseen category."""
    novel_customer = sample_customer.copy()
    novel_customer["PaymentMethod"] = "Cryptocurrency Wallet"
    novel_customer["InternetService"] = "Satellite Starlink"

    probs = champion_pipeline.predict_proba(novel_customer)
    assert probs.shape == (1, 2)
    assert 0.0 <= probs[0, 1] <= 1.0


def test_threshold_decision_logic():
    """Verify threshold boundary logic."""
    threshold = 0.30
    probs = np.array([0.10, 0.29, 0.30, 0.31, 0.85])
    preds = (probs >= threshold).astype(int)

    expected = np.array([0, 0, 1, 1, 1])
    np.testing.assert_array_equal(preds, expected)


def test_model_regression_benchmark(champion_pipeline):
    """Assert champion model generalization quality on holdout benchmark."""
    raw_df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, average_precision_score

    X = raw_df.drop(columns=["customerID", "Churn"])
    y = (raw_df["Churn"] == "Yes").astype(int)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    probs = champion_pipeline.predict_proba(X_test)[:, 1]
    roc = roc_auc_score(y_test, probs)
    ap = average_precision_score(y_test, probs)

    assert roc >= 0.84, f"Holdout ROC-AUC degraded below 0.84 (actual: {roc:.4f})"
    assert ap >= 0.63, f"Holdout AP degraded below 0.63 (actual: {ap:.4f})"


# ==============================================================================
# FASTAPI INTEGRATION TESTS
# ==============================================================================

def test_api_health_endpoint(api_client):
    res = api_client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"


def test_api_ready_endpoint(api_client):
    res = api_client.get("/ready")
    assert res.status_code == 200
    assert res.json()["model_loaded"] is True


def test_api_model_info_endpoint(api_client):
    res = api_client.get("/model-info")
    assert res.status_code == 200
    data = res.json()
    assert "metadata" in data
    assert "metrics" in data


def test_api_predict_valid_record(api_client):
    payload = {
        "customerID": "TEST-001",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.50,
        "TotalCharges": 1026.00
    }
    res = api_client.post("/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert data["churn_prediction"] in [0, 1]
    assert data["risk_tier"] in ["LOW RISK", "MODERATE RISK", "HIGH RISK"]


def test_api_predict_invalid_schema_rejected(api_client):
    payload = {
        "gender": "UnknownGender",
        "SeniorCitizen": 99,
        "tenure": -10
    }
    res = api_client.post("/predict", json=payload)
    assert res.status_code == 422  # Unprocessable Entity
