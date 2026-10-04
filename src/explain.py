"""
src/explain.py — SHAP Explainability Layer

DESIGN DECISIONS:
- TreeExplainer for LightGBM: exact SHAP values, no approximation needed
- Per-prediction: top 3-5 human-readable reason codes mapped from feature names
- Global: SHAP summary/importance plot saved for dashboard
- Human-readable mapping: converts feature names like "Online Security" to
  natural language like "No online security subscription"
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / 'models'

# ──────────────────────────────────────────────────────────────────────────────
# Human-Readable Feature Name Mapping
# ──────────────────────────────────────────────────────────────────────────────

# Maps raw feature names → human-readable explanations
# The direction (positive/negative SHAP) determines the final message
FEATURE_EXPLANATIONS = {
    # Service features (binary: 0=No, 1=Yes)
    'Online Security': {
        'negative': 'No online security subscription',
        'positive': 'Has online security (protective factor)',
    },
    'Tech Support': {
        'negative': 'No tech support subscription',
        'positive': 'Has tech support (protective factor)',
    },
    'Device Protection': {
        'negative': 'No device protection plan',
        'positive': 'Has device protection (protective factor)',
    },
    'Online Backup': {
        'negative': 'No online backup service',
        'positive': 'Has online backup (protective factor)',
    },
    'Streaming TV': {
        'negative': 'No streaming TV subscription',
        'positive': 'Subscribed to streaming TV',
    },
    'Streaming Movies': {
        'negative': 'No streaming movies subscription',
        'positive': 'Subscribed to streaming movies',
    },
    'Multiple Lines': {
        'positive': 'Has multiple phone lines',
        'negative': 'Single phone line',
    },

    # Contract & billing
    'Contract_Month-to-month': {
        'positive': 'Month-to-month contract (no commitment)',
        'negative': 'Not on month-to-month contract',
    },
    'Contract_One year': {
        'positive': 'One-year contract commitment',
        'negative': 'Not on one-year contract',
    },
    'Contract_Two year': {
        'positive': 'Two-year contract commitment (strong retention)',
        'negative': 'No two-year contract',
    },
    'Payment Method_Electronic check': {
        'positive': 'Paying via electronic check (high friction)',
        'negative': 'Not using electronic check',
    },
    'Payment Method_Bank transfer (automatic)': {
        'positive': 'Automatic bank transfer payment',
        'negative': 'No automatic bank transfer',
    },
    'Payment Method_Credit card (automatic)': {
        'positive': 'Automatic credit card payment',
        'negative': 'No automatic credit card payment',
    },
    'Payment Method_Mailed check': {
        'positive': 'Paying via mailed check',
        'negative': 'Not using mailed check',
    },
    'Paperless Billing': {
        'positive': 'Paperless billing enabled',
        'negative': 'Paper billing (traditional)',
    },

    # Internet service
    'Internet Service_Fiber optic': {
        'positive': 'Fiber optic internet (higher charges)',
        'negative': 'No fiber optic internet',
    },
    'Internet Service_DSL': {
        'positive': 'DSL internet service',
        'negative': 'No DSL internet',
    },
    'Internet Service_No': {
        'positive': 'No internet service',
        'negative': 'Has internet service',
    },

    # Numeric features
    'Tenure Months': {
        'positive': 'Long tenure (loyal customer)',
        'negative': 'Short tenure (new customer, higher risk)',
    },
    'Monthly Charges': {
        'positive': 'High monthly charges',
        'negative': 'Low monthly charges',
    },
    'Total Charges': {
        'positive': 'High total lifetime charges',
        'negative': 'Low total charges (new or low-spend)',
    },

    # Demographics
    'Senior Citizen': {
        'positive': 'Senior citizen',
        'negative': 'Non-senior citizen',
    },
    'Partner': {
        'positive': 'Has a partner',
        'negative': 'No partner',
    },
    'Dependents': {
        'positive': 'Has dependents',
        'negative': 'No dependents',
    },

    # Engineered features
    'service_count': {
        'positive': 'Many subscribed services (high stickiness)',
        'negative': 'Few subscribed services (low engagement)',
    },
    'high_risk_contract_flag': {
        'positive': 'High-risk: month-to-month + no security/support',
        'negative': 'Not in high-risk contract segment',
    },
    'charge_trend': {
        'positive': 'Charges increasing over time',
        'negative': 'Charges decreasing or stable',
    },
    'payment_friction_flag': {
        'positive': 'High payment friction (e-check + paperless)',
        'negative': 'Low payment friction',
    },
    'no_protection_flag': {
        'positive': 'No protection suite (security+support+device)',
        'negative': 'Has some protection services',
    },
    'tenure_bucket': {
        'positive': 'In a longer tenure bucket',
        'negative': 'In a shorter tenure bucket (higher risk)',
    },
    'avg_charge_per_service': {
        'positive': 'High charge per service (poor value perception)',
        'negative': 'Low charge per service (good value)',
    },
    'monthly_charges_per_tenure': {
        'positive': 'High charge intensity relative to tenure',
        'negative': 'Low charge intensity relative to tenure',
    },

    # Simulated behavioral features
    'usage_decline_slope': {
        'positive': 'Usage increasing',
        'negative': 'Usage declining (disengagement)',
    },
    'revenue_change_pct': {
        'positive': 'Revenue trending up',
        'negative': 'Revenue trending down',
    },
    'support_ticket_count': {
        'positive': 'High support ticket volume',
        'negative': 'Low support ticket volume',
    },
    'support_ticket_trend': {
        'positive': 'Increasing support calls',
        'negative': 'Decreasing or stable support calls',
    },
    'overage_minutes': {
        'positive': 'High overage minutes (bill shock risk)',
        'negative': 'Low overage minutes',
    },
    'dropped_calls_rate': {
        'positive': 'High dropped call rate (network quality issue)',
        'negative': 'Low dropped call rate',
    },
    'call_minutes_decline': {
        'positive': 'Call minutes declining (disengagement)',
        'negative': 'Call minutes stable or increasing',
    },
    'retention_call_flag': {
        'positive': 'Called retention team (actively considering leaving)',
        'negative': 'Has not called retention team',
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# SHAP Explainer
# ──────────────────────────────────────────────────────────────────────────────

class ChurnExplainer:
    """
    SHAP-based explainability for churn predictions.

    Provides:
    1. Per-prediction top-N reason codes in human-readable format
    2. Global feature importance
    3. SHAP summary plot for dashboard
    """

    def __init__(self, model, feature_names, background_data=None):
        """
        Args:
            model: Trained model (LightGBM or calibrated wrapper)
            feature_names: List of feature names matching model input
            background_data: Sample of training data for SHAP background
        """
        self.model = model
        self.feature_names = feature_names
        self.background_data = background_data

        # Use TreeExplainer for LightGBM (exact SHAP, fast)
        # For calibrated models, extract the base estimator
        base_model = model
        if hasattr(model, 'calibrated_classifiers_') and len(model.calibrated_classifiers_) > 0:
            cc = model.calibrated_classifiers_[0]
            if hasattr(cc, 'estimator'):
                base_model = cc.estimator
            elif hasattr(cc, 'base_estimator'):
                base_model = cc.base_estimator
        elif hasattr(model, 'estimators_') and len(model.estimators_) > 0:
            est = model.estimators_[0]
            if hasattr(est, 'estimator'):
                base_model = est.estimator
            else:
                base_model = est
        elif hasattr(model, 'estimator'):
            base_model = model.estimator

        try:
            self.explainer = shap.TreeExplainer(base_model)
            self.explainer_type = 'tree'
        except Exception as e:
            # Fallback to KernelExplainer (slower)
            if background_data is not None:
                bg = shap.sample(background_data, 50)
            else:
                bg = None
            self.explainer = shap.KernelExplainer(model.predict_proba, bg)
            self.explainer_type = 'kernel'

    def get_shap_values(self, X):
        """Compute SHAP values for input data."""
        if self.explainer_type == 'tree':
            shap_values = self.explainer.shap_values(X)
            # For binary classification, TreeExplainer returns list of 2 arrays
            if isinstance(shap_values, list):
                return shap_values[1]  # SHAP values for positive class (churn)
            return shap_values
        else:
            shap_values = self.explainer.shap_values(X)
            if isinstance(shap_values, list):
                return shap_values[1]
            return shap_values

    def get_reason_codes(self, X_single, top_n=5):
        """
        Generate human-readable reason codes for a single prediction.

        Returns list of dicts:
        [
            {
                'feature': 'Contract_Month-to-month',
                'shap_value': 0.34,
                'direction': 'positive',  # increases churn risk
                'reason': 'Month-to-month contract (no commitment)',
                'impact': 'high',
            },
            ...
        ]
        """
        if len(X_single.shape) == 1:
            X_single = X_single.reshape(1, -1)

        shap_values = self.get_shap_values(X_single)[0]  # First (only) row

        # Get top features by absolute SHAP value
        feature_impacts = list(zip(self.feature_names, shap_values))
        feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)

        reason_codes = []
        for feat_name, shap_val in feature_impacts[:top_n]:
            direction = 'positive' if shap_val > 0 else 'negative'

            # Look up human-readable explanation
            if feat_name in FEATURE_EXPLANATIONS:
                reason = FEATURE_EXPLANATIONS[feat_name][direction]
            else:
                # Fallback: generate a generic explanation
                if shap_val > 0:
                    reason = f"{feat_name} is contributing to churn risk"
                else:
                    reason = f"{feat_name} is reducing churn risk"

            # Impact magnitude
            abs_shap = abs(shap_val)
            if abs_shap > 0.5:
                impact = 'high'
            elif abs_shap > 0.2:
                impact = 'medium'
            else:
                impact = 'low'

            reason_codes.append({
                'feature': feat_name,
                'shap_value': round(float(shap_val), 4),
                'direction': 'increases_churn' if shap_val > 0 else 'decreases_churn',
                'reason': reason,
                'impact': impact,
            })

        return reason_codes

    def get_global_importance(self):
        """Get global feature importance based on mean |SHAP| values."""
        if self.background_data is not None:
            shap_values = self.get_shap_values(self.background_data)
            importance = np.abs(shap_values).mean(axis=0)
        else:
            importance = np.zeros(len(self.feature_names))

        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance,
        }).sort_values('importance', ascending=False)

        return importance_df

    def save_shap_plots(self, X_sample, save_dir=None):
        """Generate and save SHAP summary plots for the dashboard."""
        if save_dir is None:
            save_dir = str(MODELS_DIR)
        os.makedirs(save_dir, exist_ok=True)

        shap_values = self.get_shap_values(X_sample)

        # 1. Summary bar plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values, X_sample,
            feature_names=self.feature_names,
            plot_type='bar',
            show=False,
            max_display=20,
        )
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'shap_summary_bar.png'), dpi=150, bbox_inches='tight')
        plt.close()

        # 2. Summary beeswarm plot
        plt.figure(figsize=(12, 10))
        shap.summary_plot(
            shap_values, X_sample,
            feature_names=self.feature_names,
            show=False,
            max_display=20,
        )
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'shap_summary_beeswarm.png'), dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✔ SHAP plots saved to {save_dir}")

    def save_global_importance(self, save_dir=None):
        """Save global feature importance as JSON."""
        if save_dir is None:
            save_dir = str(MODELS_DIR)
        importance_df = self.get_global_importance()
        importance_dict = importance_df.to_dict('records')

        with open(os.path.join(save_dir, 'global_feature_importance.json'), 'w') as f:
            json.dump(importance_dict, f, indent=2)

        print(f"✔ Global feature importance saved")
        return importance_df


def build_explainer(model_path=None, feature_names_path=None, background_data=None):
    """Factory function to build a ChurnExplainer from saved artifacts."""
    if model_path is None:
        model_path = MODELS_DIR / 'calibrated_model.joblib'
    if feature_names_path is None:
        feature_names_path = MODELS_DIR / 'feature_names.joblib'

    model = joblib.load(model_path)
    feature_names = joblib.load(feature_names_path)

    return ChurnExplainer(model, feature_names, background_data)


if __name__ == '__main__':
    print("Explainability module loaded successfully.")
    print(f"Feature explanations defined for {len(FEATURE_EXPLANATIONS)} features.")
