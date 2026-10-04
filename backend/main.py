"""
backend/main.py — FastAPI Backend

Serves:
- POST /predict/manual — single customer JSON → prediction JSON
- POST /predict/batch — CSV upload → results table
- GET /model/insights — global SHAP importance, model metrics, survival curves
- GET /health — health check

All model artifacts are loaded at startup for fast inference.
"""

import os
import sys
import json
import io
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from predict import ChurnPredictor, get_customer_schema

# ──────────────────────────────────────────────────────────────────────────────
# App Setup
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Telecom Churn AI",
    description="Churn Prediction & Retention Intelligence System",
    version="1.0.0",
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor instance
predictor_instance: Optional[ChurnPredictor] = None

def get_predictor() -> ChurnPredictor:
    """Get or lazily load the global ChurnPredictor instance."""
    global predictor_instance
    if predictor_instance is None or not predictor_instance.loaded:
        models_dir = Path(__file__).resolve().parent.parent / 'models'
        try:
            predictor_instance = ChurnPredictor(str(models_dir))
            print(f"✔ ChurnPredictor loaded from {models_dir}")
        except Exception as e:
            print(f"✗ Failed to load models from {models_dir}: {e}")
            raise HTTPException(status_code=503, detail=f"Models not loaded. Error: {e}")
    return predictor_instance


# ──────────────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        pred = get_predictor()
        loaded = pred is not None and pred.loaded
        err = None
    except Exception as e:
        loaded = False
        err = str(e)

    return {
        "status": "healthy" if loaded else "unhealthy",
        "models_loaded": loaded,
        "error": err,
    }


@app.get("/schema")
async def get_schema():
    """Return the customer input schema for the frontend form."""
    return get_customer_schema()


@app.post("/predict/manual")
async def predict_manual(customer_data: dict):
    """
    Single customer prediction.

    Accepts JSON body with customer features matching the IBM Telco schema.
    Returns full prediction with churn probability, reason codes, action, CLV.
    """
    pred = get_predictor()
    try:
        result = pred.predict_single(customer_data)
        if result is None:
            raise HTTPException(status_code=400, detail="Prediction failed")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
async def predict_batch(file: UploadFile = File(...)):
    """
    Batch prediction from CSV upload.

    Accepts CSV file with customer data.
    Returns JSON array of predictions sorted by priority score.
    Also returns a download link for the results CSV.
    """
    pred = get_predictor()
    try:
        # Read uploaded CSV
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        if len(df) == 0:
            raise HTTPException(status_code=400, detail="Empty CSV file")

        if len(df) > 10000:
            raise HTTPException(status_code=400, detail="Maximum 10,000 rows per batch")

        # Predict
        results = pred.predict_batch(df)
        results_df = pred.predict_to_dataframe(df)

        # Save results CSV for download
        csv_buffer = io.StringIO()
        results_df.to_csv(csv_buffer, index=False)

        return {
            "predictions": results,
            "summary": {
                "total_customers": len(results),
                "high_risk": sum(1 for r in results if r['churn_risk'] == 'HIGH'),
                "medium_risk": sum(1 for r in results if r['churn_risk'] == 'MEDIUM'),
                "low_risk": sum(1 for r in results if r['churn_risk'] == 'LOW'),
                "avg_churn_probability": round(np.mean([r['churn_probability'] for r in results]), 4),
                "total_clv_at_risk": round(sum(r['clv_estimate'] for r in results if r['churn_risk'] in ['HIGH', 'MEDIUM']), 2),
            },
            "csv_data": csv_buffer.getvalue(),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predict/batch/download")
async def download_results():
    """Download the latest batch results as CSV."""
    results_path = Path(__file__).parent.parent / 'data' / 'processed' / 'latest_batch_results.csv'
    if not results_path.exists():
        raise HTTPException(status_code=404, detail="No batch results available")

    return StreamingResponse(
        io.BytesIO(results_path.read_bytes()),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=churn_predictions.csv"}
    )


@app.get("/model/insights")
async def model_insights():
    """
    Return global model insights for the dashboard.

    Includes:
    - Training metrics (AUC, F1, precision, recall for each model)
    - Global SHAP feature importance
    - Model architecture info
    """
    pred = get_predictor()
    insights = pred.get_model_insights()

    # Add SHAP plot paths
    models_dir = Path(__file__).parent.parent / 'models'
    plot_paths = {}
    for plot_name in ['shap_summary_bar.png', 'shap_summary_beeswarm.png', 'survival_curves.png']:
        plot_path = models_dir / plot_name
        if plot_path.exists():
            plot_paths[plot_name.replace('.png', '')] = str(plot_path)

    insights['plot_paths'] = plot_paths

    return insights


# ──────────────────────────────────────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
