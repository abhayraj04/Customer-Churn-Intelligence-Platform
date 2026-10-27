"""
Customer Churn Intelligence Platform
FastAPI REST API Service (v2.0.0 Production Release)

Provides programmatic inference endpoints for integration with external
CRM systems, customer success tools, and automated marketing platforms.
"""

import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("churn-api")

# Ensure src is accessible
src_dir = Path(__file__).parent
if str(src_dir.resolve()) not in sys.path:
    sys.path.insert(0, str(src_dir.resolve()))

from pipeline_components import TelcoFeatureEngineer, get_column_transformer

# Initialize FastAPI application
app = FastAPI(
    title="Customer Churn Intelligence API",
    description="Enterprise REST API for customer churn probability prediction and risk stratification.",
    version="2.0.0"
)

# Model Artifact Locations (Prefers Champion v2 XGBoost, fallbacks gracefully to Baseline v1 LogisticRegression)
CHAMPION_MODEL_PATH = Path("models/phase2_churn_pipeline.joblib")
CHAMPION_METRICS_PATH = Path("models/phase2_metrics.json")
BASELINE_MODEL_PATH = Path("models/final_churn_pipeline.joblib")
BASELINE_METRICS_PATH = Path("models/model_metrics.json")

model_metadata = {
    "version": "unknown",
    "name": "unknown",
    "threshold": 0.30
}

try:
    if CHAMPION_MODEL_PATH.exists() and CHAMPION_METRICS_PATH.exists():
        pipeline = joblib.load(CHAMPION_MODEL_PATH)
        with open(CHAMPION_METRICS_PATH, "r") as f:
            metrics_data = json.load(f)
        OPTIMAL_THRESHOLD = float(metrics_data["threshold_tuning"]["best_threshold"])
        model_metadata = {
            "version": "2.0.0",
            "name": metrics_data.get("selected_model", "XGBoost (Tuned)"),
            "threshold": OPTIMAL_THRESHOLD,
            "artifact": str(CHAMPION_MODEL_PATH)
        }
        logger.info(f"Loaded Champion Pipeline: {model_metadata['name']} (threshold={OPTIMAL_THRESHOLD})")
    elif BASELINE_MODEL_PATH.exists():
        pipeline = joblib.load(BASELINE_MODEL_PATH)
        with open(BASELINE_METRICS_PATH, "r") as f:
            metrics_data = json.load(f)
        OPTIMAL_THRESHOLD = float(metrics_data["threshold_tuning"]["best_threshold"])
        model_metadata = {
            "version": "1.0.0",
            "name": metrics_data.get("selected_model", "Logistic Regression (Default)"),
            "threshold": OPTIMAL_THRESHOLD,
            "artifact": str(BASELINE_MODEL_PATH)
        }
        logger.info(f"Loaded Baseline Pipeline: {model_metadata['name']} (threshold={OPTIMAL_THRESHOLD})")
    else:
        pipeline = None
        metrics_data = {}
        OPTIMAL_THRESHOLD = 0.30
        logger.warning("No model artifacts found at startup.")
except Exception as e:
    pipeline = None
    metrics_data = {}
    OPTIMAL_THRESHOLD = 0.30
    logger.error(f"Error loading model artifacts: {e}", exc_info=True)


class CustomerRecord(BaseModel):
    """Pydantic schema for raw customer input validation with strict constraints."""
    customerID: Optional[str] = Field(None, json_schema_extra={"example": "7590-VHVEG"})
    gender: str = Field(..., json_schema_extra={"example": "Female"})
    SeniorCitizen: int = Field(..., ge=0, le=1, json_schema_extra={"example": 0})
    Partner: str = Field(..., json_schema_extra={"example": "Yes"})
    Dependents: str = Field(..., json_schema_extra={"example": "No"})
    tenure: int = Field(..., ge=0, le=120, json_schema_extra={"example": 12})
    PhoneService: str = Field(..., json_schema_extra={"example": "Yes"})
    MultipleLines: str = Field(..., json_schema_extra={"example": "No"})
    InternetService: str = Field(..., json_schema_extra={"example": "Fiber optic"})
    OnlineSecurity: str = Field(..., json_schema_extra={"example": "No"})
    OnlineBackup: str = Field(..., json_schema_extra={"example": "Yes"})
    DeviceProtection: str = Field(..., json_schema_extra={"example": "No"})
    TechSupport: str = Field(..., json_schema_extra={"example": "No"})
    StreamingTV: str = Field(..., json_schema_extra={"example": "Yes"})
    StreamingMovies: str = Field(..., json_schema_extra={"example": "Yes"})
    Contract: str = Field(..., json_schema_extra={"example": "Month-to-month"})
    PaperlessBilling: str = Field(..., json_schema_extra={"example": "Yes"})
    PaymentMethod: str = Field(..., json_schema_extra={"example": "Electronic check"})
    MonthlyCharges: float = Field(..., ge=0.0, le=500.0, json_schema_extra={"example": 85.50})
    TotalCharges: Optional[float] = Field(None, ge=0.0, json_schema_extra={"example": 1026.0})

    @field_validator("gender")
    @classmethod
    def validate_gender(cls, v: str) -> str:
        if v not in ["Male", "Female"]:
            raise ValueError("gender must be 'Male' or 'Female'")
        return v

    @field_validator("Contract")
    @classmethod
    def validate_contract(cls, v: str) -> str:
        valid_contracts = ["Month-to-month", "One year", "Two year"]
        if v not in valid_contracts:
            raise ValueError(f"Contract must be one of {valid_contracts}")
        return v


class PredictionResponse(BaseModel):
    customerID: Optional[str]
    churn_probability: float
    churn_prediction: int
    risk_tier: str
    decision_threshold_used: float
    recommended_action: str
    model_version: str


def assign_risk_and_action(prob: float, threshold: float) -> tuple:
    """Classify probability into standard risk tier and prescriptive action."""
    if prob >= 0.60:
        tier = "HIGH RISK"
        action = "Urgent: Assign Customer Success Specialist for 1-on-1 account review, bundle upgrade, or price discount."
    elif prob >= threshold:
        tier = "MODERATE RISK"
        action = "Proactive: Trigger automated retention email with discount on long-term contract or free TechSupport bundle."
    else:
        tier = "LOW RISK"
        action = "Standard: Maintain routine product updates, value-add communications, and loyalty rewards."
    
    pred = 1 if prob >= threshold else 0
    return pred, tier, action


@app.get("/health", status_code=status.HTTP_200_OK)
def health():
    """Liveness probe: returns OK if web service process is alive."""
    return {"status": "healthy", "service": "Customer Churn Intelligence API"}


@app.get("/ready", status_code=status.HTTP_200_OK)
def ready():
    """Readiness probe: returns OK if inference pipeline and artifacts are loaded."""
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference pipeline is not ready or failed to load."
        )
    return {
        "status": "ready",
        "model_loaded": True,
        "model_version": model_metadata["version"],
        "model_name": model_metadata["name"]
    }


@app.get("/", status_code=status.HTTP_200_OK)
def root():
    return {
        "platform": "Customer Churn Intelligence Platform",
        "status": "Operational",
        "model_loaded": pipeline is not None,
        "model_version": model_metadata["version"],
        "optimal_threshold": OPTIMAL_THRESHOLD,
        "docs_url": "/docs"
    }


@app.get("/model-info", status_code=status.HTTP_200_OK)
def get_model_info():
    """Return model architecture, active version, threshold, and holdout evaluation metrics."""
    if not metrics_data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model metrics not available.")
    return {
        "metadata": model_metadata,
        "metrics": metrics_data
    }


@app.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
def predict_single(customer: CustomerRecord):
    """Predict churn probability for an individual customer record."""
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference model pipeline is currently unavailable."
        )

    try:
        df_in = pd.DataFrame([customer.model_dump()])
        prob = float(pipeline.predict_proba(df_in)[0, 1])
        prob_bounded = max(0.0, min(1.0, prob))
        pred, tier, action = assign_risk_and_action(prob_bounded, OPTIMAL_THRESHOLD)

        return PredictionResponse(
            customerID=customer.customerID,
            churn_probability=round(prob_bounded, 4),
            churn_prediction=pred,
            risk_tier=tier,
            decision_threshold_used=OPTIMAL_THRESHOLD,
            recommended_action=action,
            model_version=model_metadata["version"]
        )
    except Exception as e:
        logger.error(f"Inference error during single prediction: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Model inference failed.")


@app.post("/predict-batch", response_model=List[PredictionResponse], status_code=status.HTTP_200_OK)
def predict_batch(customers: List[CustomerRecord]):
    """Predict churn probability for a batch of customer records (max 1,000 per request)."""
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference model pipeline is currently unavailable."
        )

    if len(customers) > 1000:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Batch size exceeds maximum limit of 1,000 records (received {len(customers)})."
        )

    if not customers:
        return []

    try:
        df_in = pd.DataFrame([c.model_dump() for c in customers])
        probs = pipeline.predict_proba(df_in)[:, 1]

        responses = []
        for c, p in zip(customers, probs):
            prob_bounded = max(0.0, min(1.0, float(p)))
            pred, tier, action = assign_risk_and_action(prob_bounded, OPTIMAL_THRESHOLD)
            responses.append(PredictionResponse(
                customerID=c.customerID,
                churn_probability=round(prob_bounded, 4),
                churn_prediction=pred,
                risk_tier=tier,
                decision_threshold_used=OPTIMAL_THRESHOLD,
                recommended_action=action,
                model_version=model_metadata["version"]
            ))

        return responses
    except Exception as e:
        logger.error(f"Inference error during batch prediction: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Batch inference failed.")
