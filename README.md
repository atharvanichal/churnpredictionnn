# 📡 Telecom Customer Churn Prediction & Retention Intelligence System

An end-to-end ML/AI system that predicts telecom customer churn, explains why each customer is at risk, predicts when they'll churn, recommends specific retention actions, and prioritizes customers by lifetime value.

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Frontend                        │
│  ┌──────────┐  ┌──────────────┐  ┌───────────────────┐     │
│  │CSV Upload│  │ Manual Entry │  │ Model Insights    │     │
│  └────┬─────┘  └──────┬───────┘  │ Dashboard         │     │
│       │               │          └───────────────────┘     │
├───────┼───────────────┼─────────────────────────────────────┤
│       └───────┬───────┘                                     │
│               ▼                                             │
│        FastAPI Backend                                      │
│  POST /predict/manual  POST /predict/batch  GET /insights   │
├─────────────────────────────────────────────────────────────┤
│                  Inference Pipeline (predict.py)             │
│  ┌──────────────┐ ┌────────┐ ┌────────┐ ┌────────────────┐ │
│  │Feature Eng.  │→│Churn   │→│SHAP    │→│Retention       │ │
│  │Pipeline      │ │Predict │ │Explain │ │Engine          │ │
│  └──────────────┘ └────────┘ └────────┘ └────────────────┘ │
│  ┌──────────────┐ ┌────────┐ ┌──────────────────────────┐  │
│  │Survival      │→│CLV     │→│Priority Scoring +        │  │
│  │(Cox PH)      │ │Compute │ │Value-at-Risk Tiering     │  │
│  └──────────────┘ └────────┘ └──────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## 📊 Dual-Dataset Design Decision

This system uses **two distinct datasets** with different roles:

### IBM Telco (PRIMARY) — 7,043 labeled customers
- **Role**: Core training data for all supervised models
- **Strengths**: Complete churn labels (26.5% churn rate), rich service-subscription columns (Online Security, Tech Support, Streaming, etc.) that map directly to retention actions
- **Used for**: Churn classifier training, survival model, retention engine, CLV calculation

### Cell2Cell (SECONDARY/ENRICHMENT) — 20,000 customers, NO labels
- **Critical finding**: This is a **holdout/test set** — all 20,000 churn labels are NaN
- **Role**: Behavioral feature distribution templates
- **Strengths**: 58 columns with deep usage/behavioral data (call minutes, overage, support calls, retention calls, network quality metrics)
- **Used for**: Extracting realistic distribution parameters to **simulate** usage behavior features for IBM Telco customers, informing uplift model treatment simulation parameters

### Why not merge them?
1. **Different customer populations**: Different telecom providers, different time periods
2. **Cell2Cell has no labels**: Cannot train any supervised model on it
3. **Complementary strengths**: IBM has labels + service features; Cell2Cell has behavioral depth
4. **Our approach**: Use Cell2Cell's distributional statistics to realistically simulate behavioral features (usage decline, support calls, overage) for IBM Telco customers, conditioned on tenure, contract type, and churn status

## 🧠 Model Choices

### 1. Churn Classifier: LightGBM (Calibrated)
- **Why LightGBM over XGBoost**: Faster training, native categorical handling, comparable accuracy
- **Why calibration (isotonic)**: Tree models produce uncalibrated probabilities; we need reliable P(churn) for CLV math
- **Why Optuna over GridSearchCV**: Bayesian optimization finds better hyperparameters in fewer trials
- **Cost-sensitive**: False negatives (missed churners) weighted 5× more than false positives — retention is cheaper than acquisition

### 2. Explainability: SHAP TreeExplainer
- **Why SHAP over LIME**: TreeExplainer gives exact (not approximate) Shapley values for tree models
- **Human-readable mapping**: Every feature name is mapped to a plain-English explanation
- **Per-prediction**: Top 5 reason codes explain *why* each specific customer is at risk

### 3. Survival Model: Cox Proportional Hazards
- **Why Cox PH**: Semi-parametric (no distributional assumption on baseline hazard), interpretable hazard ratios
- **Duration**: Tenure Months; Event: Churn (1=churned, 0=still active/censored)
- **Output**: Predicted median time-to-churn in months + survival probability curves

### 4. Uplift Model: T-Learner (⚠ Simulated)
- **Why T-Learner**: Simple, effective approach when treatment/control groups are available
- **⚠ SIMULATED**: No real intervention data exists — treatment assignment and response are simulated using distributions informed by Cell2Cell's `RetentionCalls` and `RetentionOffersAccepted`
- **Purpose**: Demonstrates the uplift framework; in production, replace with real A/B test data
- **Customer segments**: Persuadable (treatment helps), Sure Thing (would stay anyway), Lost Cause (will leave regardless), Sleeping Dog (don't disturb)

### 5. Retention Engine: Rule-based + ROI
- Maps SHAP reason codes to a 12-action playbook (free trials, discounts, escalations)
- Each action has estimated cost, expected success rate, and ROI calculation
- Priority Score = churn_prob × CLV × (1 - retention_success_prob)

## 🧪 Synthetic Data Validation

Synthetic data (generated via CTGAN or bootstrap+noise) is validated against real data:
- **KS Test**: Kolmogorov-Smirnov test comparing real vs. synthetic distributions for every numeric feature
- **Quality threshold**: p > 0.01 (features with p ≤ 0.01 are flagged)
- **No leakage**: Synthetic data is used ONLY for training; the final test set is 100% real held-out data
- **Validation report**: Saved to `data/processed/synthetic_validation_report.csv`

## 📁 Project Structure

```
├── data/
│   ├── raw/                    # Original datasets
│   ├── synthetic/              # CTGAN-generated data
│   └── processed/              # Processed features, validation reports
├── src/
│   ├── features.py             # Feature engineering pipeline (training & inference)
│   ├── train.py                # Churn classifier training
│   ├── explain.py              # SHAP explainability layer
│   ├── survival.py             # Cox PH time-to-churn model
│   ├── uplift.py               # T-Learner uplift model (simulated)
│   ├── retention_engine.py     # Retention action recommendation engine
│   └── predict.py              # Unified inference pipeline
├── models/                     # Saved model artifacts (joblib, JSON, plots)
├── backend/
│   └── main.py                 # FastAPI backend
├── frontend/
│   └── app.py                  # Streamlit frontend
├── run_training.py             # Master training orchestrator
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Models
```bash
python run_training.py
```
This runs the full pipeline: data loading → synthetic generation → feature engineering → model training → SHAP → survival → uplift → retention engine → saves all artifacts.

### 3. Start Backend (optional)
```bash
python backend/main.py
```
FastAPI runs on `http://localhost:8000`.

### 4. Start Frontend
```bash
streamlit run frontend/app.py
```
Opens in browser at `http://localhost:8501`.

## 📈 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/schema` | GET | Input schema for manual entry |
| `/predict/manual` | POST | Single customer JSON → prediction |
| `/predict/batch` | POST | CSV upload → batch predictions |
| `/model/insights` | GET | Global SHAP, metrics, model info |

## 📋 Output Per Customer

Each prediction includes:
1. **Churn Probability** — calibrated P(churn) from LightGBM
2. **Risk Level** — HIGH (>70%), MEDIUM (40-70%), LOW (<40%)
3. **Time to Churn** — predicted months from Cox PH survival model
4. **Top 5 Reason Codes** — human-readable SHAP explanations
5. **Recommended Action** — specific retention offer from the playbook
6. **CLV Estimate** — MonthlyCharges × expected remaining months
7. **Priority Score** — churn_prob × CLV × (1 - retention_success)
8. **Value-at-Risk Tier** — HIGH/MEDIUM/LOW based on priority score

## ⚙ Feature Engineering (44 features)

| Category | Count | Examples |
|----------|-------|---------|
| Raw IBM Telco | 19 | Gender, Tenure, Contract, Monthly Charges |
| Engineered | 9 | service_count, high_risk_contract_flag, charge_trend |
| Behavioral (simulated) | 9 | usage_decline_slope, support_ticket_count |
| Survival outputs | 4 | median_survival_months, survival_prob_3/6/12mo |
| CLV/Priority | 3 | clv_estimate, priority_score, value_risk_tier |
