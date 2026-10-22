# Production Deployment Pre-Flight Checklist
**Platform:** Customer Churn Intelligence System  
**Release Target:** v2.0.0 Production Portfolio  
**Audit Date:** 2026-10-22  

---

## Pre-Deployment Verification Matrix

| Checklist Item | Status | Verification Detail / Command |
| :--- | :---: | :--- |
| **Model Artifacts Integrity** | **PASS** | `models/phase2_churn_pipeline.joblib` and `models/final_churn_pipeline.joblib` verified loadable with `joblib.load()`. |
| **Metrics Artifacts Verification**| **PASS** | `models/phase2_metrics.json` and `models/model_metrics.json` verified valid JSON with identical schema. |
| **Zero Test-Set Leakage** | **PASS** | All preprocessing and hyperparameter search fitted strictly on training partition ($N=5,634$). Test set ($N=1,409$) evaluated once. |
| **Automated Test Suite** | **PASS** | `pytest tests/test_pipeline.py -v`: **13 passed in 2.40s** (100% pass rate). |
| **FastAPI REST Service Verification**| **PASS** | Verified endpoints: `GET /health` (200), `GET /ready` (200), `GET /model-info` (200), `POST /predict` (200), `POST /predict` invalid schema (422), `POST /predict-batch` (200). |
| **Streamlit Dashboard Verification**| **PASS** | `app.py` successfully imported and executed across all 8 modules (A through H). |
| **Batch CLI Utility Verification**| **PASS** | `python src/batch_score.py` executed successfully: 7,043 accounts scored, output verified in `reports/batch_scored_customers.csv`. |
| **Docker Build Configuration** | **PASS** | `Dockerfile` and `.dockerignore` structured with multi-port support (8501 for UI, 8000 for API). |
| **Environment Template** | **PASS** | `.env.example` created with variable placeholders. Zero secret credentials present. |
| **Privacy & Security Review** | **PASS** | Zero PII columns detected in raw data or outputs. Anonymized alphanumeric IDs preserved. |
| **License & Documentation** | **PASS** | `LICENSE` (MIT) and comprehensive `README.md` with Mermaid architecture diagrams created. |

---

## Hosting Deployment Options & Exact Steps

### Option A: Streamlit Community Cloud (Recommended for Dashboard)
1. Push this repository to GitHub:
   ```bash
   git init
   git add .
   git commit -m "feat: release v2.0.0 production churn intelligence platform"
   git branch -M main
   git remote add origin https://github.com/<your-username>/Churn_prediction.git
   git push -u origin main
   ```
2. Navigate to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
3. Select **"New app"**, select your repository, branch `main`, and main file path `app.py`.
4. Click **Deploy!** Streamlit Community Cloud will automatically build dependencies from `requirements.txt` and provide a public URL (`https://<your-app-name>.streamlit.app`).

### Option B: Render / Railway / Fly.io (Recommended for FastAPI Service)
1. In your chosen cloud dashboard (e.g., [render.com](https://render.com/)), choose **"New Web Service"** connected to your repository.
2. Select runtime: **Python 3**.
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn src.api:app --host 0.0.0.0 --port $PORT`
5. Click **Create Web Service**. Your live API and Swagger docs will be accessible at `https://<service-name>.onrender.com/docs`.

### Option C: Single-Node Self-Hosted Docker
```bash
docker build -t churn-intelligence:latest .
docker run -d -p 8501:8501 -p 8000:8000 --name churn-platform churn-intelligence:latest
```
