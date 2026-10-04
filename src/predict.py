"""
src/predict.py — Unified Inference Pipeline

This module is the single entry point for all predictions.
It loads saved model artifacts and produces per-customer:
1. Calibrated churn probability
2. Predicted time-to-churn (months)
3. Top 3-5 SHAP reason codes (human-readable)
4. Recommended retention action
5. CLV estimate
6. Priority score and value-at-risk tier

Accepts CSV (batch) or single-customer dict (manual entry).
Uses the EXACT same feature engineering pipeline as training (loaded from joblib).
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from features import (
    build_preprocessing_pipeline, FeatureSelector,
    load_pipeline, transform_raw_data
)
from explain import ChurnExplainer, build_explainer
from survival import ChurnSurvivalModel
from retention_engine import RetentionEngine
from uplift import UpliftModel

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / 'models'


class ChurnPredictor:
    """
    Unified inference pipeline that combines all models into a single
    prediction call. Loads all artifacts at initialization.
    """

    def __init__(self, models_dir=None):
        self.models_dir = Path(models_dir) if models_dir else MODELS_DIR
        self.loaded = False
        self._load_artifacts()

    def _load_artifacts(self):
        """Load all model artifacts from disk."""
        try:
            # Core artifacts
            self.preprocess_pipeline, self.column_transformer, self.feature_names = \
                load_pipeline(str(self.models_dir))

            # Churn classifier (calibrated)
            self.classifier = joblib.load(self.models_dir / 'calibrated_model.joblib')

            # SHAP explainer
            self.explainer = ChurnExplainer(
                self.classifier, self.feature_names
            )

            # Survival model
            try:
                self.survival_model = ChurnSurvivalModel.load(str(self.models_dir))
            except Exception:
                self.survival_model = None
                print("⚠ Survival model not found — using fallback")

            # Uplift model
            try:
                self.uplift_model = UpliftModel.load(str(self.models_dir))
            except Exception:
                self.uplift_model = None
                print("⚠ Uplift model not found — skipping uplift")

            # Retention engine
            self.retention_engine = RetentionEngine()

            # Training metrics
            try:
                with open(self.models_dir / 'training_metrics.json', 'r') as f:
                    self.training_metrics = json.load(f)
            except Exception:
                self.training_metrics = {}

            self.loaded = True
            print("✔ All model artifacts loaded successfully")

        except Exception as e:
            print(f"✗ Error loading artifacts: {e}")
            raise

    def predict_single(self, customer_data):
        """
        Predict for a single customer.

        Args:
            customer_data: dict with customer features matching IBM Telco schema

        Returns:
            dict with full prediction results
        """
        # Convert to DataFrame (single row)
        df = pd.DataFrame([customer_data])
        results = self.predict_batch(df)
        return results[0] if results else None

    def predict_batch(self, df):
        """
        Predict for a batch of customers (DataFrame or CSV path).

        Args:
            df: pandas DataFrame or path to CSV file

        Returns:
            List of prediction dicts, sorted by priority score (descending)
        """
        if isinstance(df, (str, Path)):
            df = pd.read_csv(df)

        # Store original customer IDs if available
        customer_ids = df['CustomerID'].values if 'CustomerID' in df.columns else range(len(df))

        # Step 1: Preprocess + engineer features
        df_processed = self.preprocess_pipeline.transform(df)

        # Step 2: Select features
        selector = FeatureSelector()
        df_selected = selector.transform(df_processed)

        numeric_features = [c for c in FeatureSelector.FINAL_NUMERIC_FEATURES if c in df_selected.columns]
        categorical_features = [c for c in FeatureSelector.CATEGORICAL_FEATURES if c in df_selected.columns]

        # Step 3: Transform with column transformer
        X = self.column_transformer.transform(df_selected)

        # Step 4: Predict churn probability (calibrated)
        churn_proba = self.classifier.predict_proba(X)[:, 1]

        # Step 5: Predict time-to-churn
        if self.survival_model is not None:
            try:
                survival_results = self.survival_model.predict_time_to_churn(df_processed)
            except Exception:
                survival_results = self._fallback_survival(df_processed, churn_proba)
        else:
            survival_results = self._fallback_survival(df_processed, churn_proba)

        # Step 6: Get SHAP explanations
        # Step 7: Get retention recommendations
        # Step 8: Compute CLV and priority

        results = []
        for i in range(len(df)):
            # SHAP reason codes
            try:
                reason_codes = self.explainer.get_reason_codes(X[i], top_n=5)
            except Exception:
                reason_codes = [{'feature': 'unknown', 'shap_value': 0, 'reason': 'Unable to compute', 'impact': 'unknown', 'direction': 'unknown'}]

            # Monthly charges for CLV
            monthly_charges = df_processed['Monthly Charges'].iloc[i] if 'Monthly Charges' in df_processed.columns else 50.0

            # Survival metrics
            median_survival = float(survival_results['median_survival_months'].iloc[i])
            if np.isnan(median_survival) or np.isinf(median_survival):
                median_survival = 24.0

            # CLV
            clv = self.retention_engine.compute_clv(
                monthly_charges, median_survival, float(churn_proba[i])
            )

            # Retention recommendation
            recommendation = self.retention_engine.recommend(
                reason_codes, float(churn_proba[i]), clv
            )

            # Uplift (if available)
            uplift_info = {}
            if self.uplift_model is not None:
                try:
                    uplift_result = self.uplift_model.predict_uplift(X[i:i+1])
                    uplift_info = {
                        'uplift_score': float(uplift_result['uplift'].iloc[0]),
                        'treatment_segment': uplift_result['segment'].iloc[0],
                        'is_simulated': True,
                    }
                except Exception:
                    pass

            result = {
                'customer_id': str(customer_ids[i]),
                'churn_probability': round(float(churn_proba[i]), 4),
                'churn_risk': 'HIGH' if churn_proba[i] > 0.7 else 'MEDIUM' if churn_proba[i] > 0.4 else 'LOW',
                'predicted_time_to_churn_months': round(median_survival, 1),
                'survival_prob_3mo': round(float(survival_results['survival_prob_3mo'].iloc[i]), 3),
                'survival_prob_6mo': round(float(survival_results['survival_prob_6mo'].iloc[i]), 3),
                'survival_prob_12mo': round(float(survival_results['survival_prob_12mo'].iloc[i]), 3),
                'top_reason_codes': reason_codes,
                'recommended_action': recommendation['primary_action'],
                'action_description': recommendation['primary_description'],
                'action_category': recommendation['primary_category'],
                'action_urgency': recommendation['urgency'],
                'all_recommended_actions': recommendation['all_actions'],
                'clv_estimate': clv,
                'priority_score': recommendation['priority_score'],
                'value_risk_tier': recommendation['value_risk_tier'],
                'retention_success_probability': recommendation['retention_success_probability'],
                **uplift_info,
            }
            results.append(result)

        # Sort by priority score (highest first)
        results.sort(key=lambda x: x['priority_score'], reverse=True)

        return results

    def _fallback_survival(self, df_processed, churn_proba):
        """
        Fallback survival estimation when the Cox PH model isn't available.
        Uses a simple heuristic based on tenure and churn probability.
        """
        n = len(df_processed)
        tenure = df_processed.get('Tenure Months', pd.Series([12] * n))

        # Simple heuristic: inverse relationship between churn prob and survival
        median_survival = (1 - churn_proba) * 36 + 3  # 3-39 months

        return pd.DataFrame({
            'median_survival_months': median_survival.clip(1, 120),
            'survival_prob_3mo': (1 - churn_proba * 0.3).clip(0, 1),
            'survival_prob_6mo': (1 - churn_proba * 0.5).clip(0, 1),
            'survival_prob_12mo': (1 - churn_proba * 0.8).clip(0, 1),
        })

    def predict_to_dataframe(self, df):
        """
        Predict and return results as a flat DataFrame (for CSV export).
        """
        results = self.predict_batch(df)

        # Flatten for CSV
        flat_results = []
        for r in results:
            flat = {
                'customer_id': r['customer_id'],
                'churn_probability': r['churn_probability'],
                'churn_risk': r['churn_risk'],
                'time_to_churn_months': r['predicted_time_to_churn_months'],
                'survival_3mo': r['survival_prob_3mo'],
                'survival_6mo': r['survival_prob_6mo'],
                'survival_12mo': r['survival_prob_12mo'],
                'clv_estimate': r['clv_estimate'],
                'priority_score': r['priority_score'],
                'value_risk_tier': r['value_risk_tier'],
                'recommended_action': r['recommended_action'],
                'action_urgency': r['action_urgency'],
                'retention_success_prob': r['retention_success_probability'],
            }
            # Add top 5 reasons
            for j, rc in enumerate(r['top_reason_codes'][:5]):
                flat[f'reason_{j+1}'] = rc['reason']
                flat[f'reason_{j+1}_impact'] = rc['impact']

            flat_results.append(flat)

        return pd.DataFrame(flat_results)

    def get_model_insights(self):
        """
        Return global model insights for the dashboard.
        """
        insights = {
            'training_metrics': self.training_metrics,
            'feature_importance': [],
            'model_info': {
                'classifier': 'LightGBM (Calibrated)',
                'survival': 'Cox Proportional Hazards',
                'uplift': 'T-Learner (Simulated)',
                'n_features': len(self.feature_names),
            }
        }

        # Global feature importance
        try:
            importance_path = self.models_dir / 'global_feature_importance.json'
            if importance_path.exists():
                with open(importance_path) as f:
                    insights['feature_importance'] = json.load(f)
        except Exception:
            pass

        return insights


# ──────────────────────────────────────────────────────────────────────────────
# Schema for manual entry form
# ──────────────────────────────────────────────────────────────────────────────

CUSTOMER_SCHEMA = {
    'Gender': {'type': 'select', 'options': ['Male', 'Female'], 'default': 'Male'},
    'Senior Citizen': {'type': 'select', 'options': ['Yes', 'No'], 'default': 'No'},
    'Partner': {'type': 'select', 'options': ['Yes', 'No'], 'default': 'No'},
    'Dependents': {'type': 'select', 'options': ['Yes', 'No'], 'default': 'No'},
    'Tenure Months': {'type': 'number', 'min': 0, 'max': 72, 'default': 12},
    'Phone Service': {'type': 'select', 'options': ['Yes', 'No'], 'default': 'Yes'},
    'Multiple Lines': {'type': 'select', 'options': ['Yes', 'No', 'No phone service'], 'default': 'No'},
    'Internet Service': {'type': 'select', 'options': ['DSL', 'Fiber optic', 'No'], 'default': 'Fiber optic'},
    'Online Security': {'type': 'select', 'options': ['Yes', 'No', 'No internet service'], 'default': 'No'},
    'Online Backup': {'type': 'select', 'options': ['Yes', 'No', 'No internet service'], 'default': 'No'},
    'Device Protection': {'type': 'select', 'options': ['Yes', 'No', 'No internet service'], 'default': 'No'},
    'Tech Support': {'type': 'select', 'options': ['Yes', 'No', 'No internet service'], 'default': 'No'},
    'Streaming TV': {'type': 'select', 'options': ['Yes', 'No', 'No internet service'], 'default': 'No'},
    'Streaming Movies': {'type': 'select', 'options': ['Yes', 'No', 'No internet service'], 'default': 'No'},
    'Contract': {'type': 'select', 'options': ['Month-to-month', 'One year', 'Two year'], 'default': 'Month-to-month'},
    'Paperless Billing': {'type': 'select', 'options': ['Yes', 'No'], 'default': 'Yes'},
    'Payment Method': {'type': 'select', 'options': ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], 'default': 'Electronic check'},
    'Monthly Charges': {'type': 'number', 'min': 0, 'max': 200, 'default': 70.0},
    'Total Charges': {'type': 'number', 'min': 0, 'max': 10000, 'default': 500.0},
}


def get_customer_schema():
    """Return the schema for manual entry form."""
    return CUSTOMER_SCHEMA


if __name__ == '__main__':
    print("Inference pipeline module loaded.")
    print(f"Customer schema has {len(CUSTOMER_SCHEMA)} fields.")
