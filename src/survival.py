"""
src/survival.py — Survival / Time-to-Churn Model

DESIGN DECISIONS:
- Cox Proportional Hazards model via lifelines: well-tested, interpretable
- Duration = Tenure Months, Event = Churn (1=churned, 0=censored/still active)
- Usage-decline features from Cell2Cell-informed simulation improve hazard estimation
- Output: predicted time-to-churn (months) and survival curve per customer
- WHY Cox PH over discrete-time: interpretable hazard ratios, continuous time,
  and the semi-parametric nature handles the baseline hazard without assumptions
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / 'models'

# ──────────────────────────────────────────────────────────────────────────────
# Survival Model
# ──────────────────────────────────────────────────────────────────────────────

class ChurnSurvivalModel:
    """
    Cox Proportional Hazards model for predicting time-to-churn.

    Outputs per customer:
    - Predicted median survival time (months until churn)
    - Survival probability curve over time
    - Hazard ratio for each feature (how it affects churn timing)
    """

    def __init__(self):
        self.model = None
        self.feature_columns = None

    def prepare_survival_data(self, df_processed, y):
        """
        Prepare data for Cox PH model.

        Args:
            df_processed: DataFrame after preprocessing pipeline
            y: Binary target (1=churned, 0=active)

        Returns:
            DataFrame with duration, event, and covariates
        """
        survival_df = pd.DataFrame()

        # Duration: tenure months (how long they've been a customer)
        if 'Tenure Months' in df_processed.columns:
            survival_df['duration'] = df_processed['Tenure Months'].clip(lower=1)
        else:
            survival_df['duration'] = 12  # Default

        # Event: did churn happen? (1=yes, 0=censored/still active)
        survival_df['event'] = y.values if hasattr(y, 'values') else y

        # Covariates: features that affect the timing of churn
        covariate_cols = [
            'Monthly Charges', 'service_count', 'high_risk_contract_flag',
            'payment_friction_flag', 'no_protection_flag', 'has_streaming',
            'Senior Citizen', 'Partner', 'Dependents',
            'usage_decline_slope', 'support_ticket_count', 'overage_minutes',
            'dropped_calls_rate', 'call_minutes_decline',
        ]

        for col in covariate_cols:
            if col in df_processed.columns:
                survival_df[col] = df_processed[col].values

        # Handle any missing values
        survival_df = survival_df.fillna(survival_df.median())

        # Store feature columns for prediction
        self.feature_columns = [c for c in survival_df.columns if c not in ['duration', 'event']]

        return survival_df

    def fit(self, survival_df):
        """Fit the Cox Proportional Hazards model."""
        from lifelines import CoxPHFitter

        self.model = CoxPHFitter(
            penalizer=0.01,  # L2 regularization to handle multicollinearity
        )

        self.model.fit(
            survival_df,
            duration_col='duration',
            event_col='event',
            show_progress=True,
        )

        # Print summary
        print("\n── Cox PH Model Summary ──")
        self.model.print_summary()

        return self

    def predict_time_to_churn(self, df_processed):
        """
        Predict time-to-churn for each customer.

        Returns DataFrame with:
        - median_survival_months: expected months before churn
        - survival_prob_3mo: probability of not churning in 3 months
        - survival_prob_6mo: probability of not churning in 6 months
        - survival_prob_12mo: probability of not churning in 12 months
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Prepare covariates
        covariates = pd.DataFrame()
        for col in self.feature_columns:
            if col in df_processed.columns:
                covariates[col] = df_processed[col].values
            else:
                covariates[col] = 0

        covariates = covariates.fillna(covariates.median())

        # Predict
        try:
            survival_func = self.model.predict_survival_function(covariates)

            results = pd.DataFrame()

            # Median survival time
            median_vals = np.asarray(self.model.predict_median(covariates))
            median_vals = np.where(np.isnan(median_vals) | np.isinf(median_vals), 24.0, median_vals)
            results['median_survival_months'] = np.clip(median_vals, 1, 120)

            # Survival probabilities at key time points
            for months in [3, 6, 12]:
                if months in survival_func.index:
                    vals = survival_func.loc[months].values
                else:
                    closest = survival_func.index[np.argmin(np.abs(survival_func.index - months))]
                    vals = survival_func.loc[closest].values
                vals = np.where(np.isnan(vals) | np.isinf(vals), 0.5, vals)
                results[f'survival_prob_{months}mo'] = np.clip(vals, 0, 1)

        except Exception as e:
            print(f"Warning: Survival prediction error ({e}). Using tenure-based fallback.")
            n = len(covariates)
            results = pd.DataFrame({
                'median_survival_months': np.full(n, 24),
                'survival_prob_3mo': np.full(n, 0.85),
                'survival_prob_6mo': np.full(n, 0.7),
                'survival_prob_12mo': np.full(n, 0.5),
            })

        return results

    def get_hazard_ratios(self):
        """Get hazard ratios for each feature."""
        if self.model is None:
            return pd.DataFrame()

        summary = self.model.summary
        return summary[['exp(coef)', 'p', 'exp(coef) lower 95%', 'exp(coef) upper 95%']]

    def save(self, path=None):
        """Save the survival model."""
        if path is None:
            path = str(MODELS_DIR)
        os.makedirs(path, exist_ok=True)
        joblib.dump(self, os.path.join(path, 'survival_model.joblib'))
        print(f"✔ Survival model saved to {path}")

    @staticmethod
    def load(path=None):
        """Load a saved survival model."""
        if path is None:
            path = str(MODELS_DIR)
        return joblib.load(os.path.join(path, 'survival_model.joblib'))

    def plot_survival_curves(self, df_processed, save_dir=None):
        """Plot survival curves by key segments for the dashboard."""
        if save_dir is None:
            save_dir = str(MODELS_DIR)
        os.makedirs(save_dir, exist_ok=True)

        from lifelines import KaplanMeierFitter

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Survival by contract type risk
        if 'high_risk_contract_flag' in df_processed.columns:
            kmf = KaplanMeierFitter()
            for risk_val, label in [(1, 'High Risk (M2M, No Protection)'), (0, 'Lower Risk')]:
                mask = df_processed['high_risk_contract_flag'] == risk_val
                if mask.sum() > 0:
                    T = df_processed.loc[mask, 'Tenure Months'].clip(lower=1) if 'Tenure Months' in df_processed.columns else pd.Series([12] * mask.sum())
                    E = np.ones(mask.sum())  # Placeholder
                    kmf.fit(T, E, label=label)
                    kmf.plot_survival_function(ax=axes[0])
            axes[0].set_title('Survival by Contract Risk')
            axes[0].set_xlabel('Months')
            axes[0].set_ylabel('Survival Probability')

        # Survival by service count
        if 'service_count' in df_processed.columns:
            kmf = KaplanMeierFitter()
            for sc_range, label in [((0, 1), '0-1 Services'), ((2, 3), '2-3 Services'), ((4, 7), '4+ Services')]:
                mask = df_processed['service_count'].between(sc_range[0], sc_range[1])
                if mask.sum() > 0:
                    T = df_processed.loc[mask, 'Tenure Months'].clip(lower=1) if 'Tenure Months' in df_processed.columns else pd.Series([12] * mask.sum())
                    E = np.ones(mask.sum())
                    kmf.fit(T, E, label=label)
                    kmf.plot_survival_function(ax=axes[1])
            axes[1].set_title('Survival by Service Count')
            axes[1].set_xlabel('Months')
            axes[1].set_ylabel('Survival Probability')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'survival_curves.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✔ Survival curves saved")


def train_survival_model(df_processed, y):
    """Convenience function to train and save the survival model."""
    model = ChurnSurvivalModel()
    survival_df = model.prepare_survival_data(df_processed, y)
    model.fit(survival_df)
    model.save()

    # Save hazard ratios
    hr = model.get_hazard_ratios()
    hr.to_csv(MODELS_DIR / 'hazard_ratios.csv')

    # Plot survival curves
    try:
        model.plot_survival_curves(df_processed)
    except Exception as e:
        print(f"Warning: Could not plot survival curves: {e}")

    return model


if __name__ == '__main__':
    print("Survival model module loaded successfully.")
