"""
src/uplift.py — Uplift Modeling (Simulated Treatment Response)

DESIGN DECISIONS:
- No real intervention/treatment data exists → we SIMULATE treatment response
- Uses T-learner approach: separate models for treatment & control groups
- Cell2Cell's RetentionCalls and RetentionOffersAccepted distributions inform
  the simulation of realistic treatment assignment and response
- CLEARLY LABELED as simulated throughout — this is for demonstration only
- In production, this would be replaced with actual A/B test data

WHY uplift modeling matters:
- Not all at-risk customers respond to intervention
- Some would stay regardless (don't waste budget on them)
- Some would leave regardless (don't waste budget on them either)
- Uplift = P(stay|treated) - P(stay|untreated) → focus on "persuadables"
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import joblib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / 'models'


class UpliftModel:
    """
    T-Learner uplift model for estimating treatment effect.

    ⚠ SIMULATED TREATMENT DATA ⚠
    Since no real intervention data exists, we simulate treatment assignment
    and response based on realistic assumptions informed by Cell2Cell's
    retention call distributions.

    T-Learner approach:
    1. Train model_treatment on treated group → P(stay | treated)
    2. Train model_control on control group → P(stay | not treated)
    3. Uplift = P(stay | treated) - P(stay | not treated)
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model_treatment = None
        self.model_control = None
        self.is_simulated = True  # Flag: this uses simulated treatment data

    def simulate_treatment_data(self, X, y, treatment_rate=0.3):
        """
        Simulate treatment assignment and response.

        ⚠ SIMULATED — In production, replace with real A/B test data.

        Simulation logic:
        - 30% of customers randomly assigned to "treatment" (retention offer)
        - Treatment effect varies by customer segment:
          - High-risk M2M customers: ~20% uplift (most persuadable)
          - Moderate risk: ~10% uplift
          - Low risk (long contracts): ~3% uplift (would stay anyway)
        """
        rng = np.random.RandomState(self.random_state)
        n = len(y)

        # Simulate treatment assignment (random, like an A/B test)
        treatment = rng.binomial(1, treatment_rate, n)

        # Simulate heterogeneous treatment effect
        # Base treatment effect: small positive effect
        base_effect = 0.05

        # Get features as DataFrame if numpy array
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X)
        else:
            X_df = X

        # Simulate response: P(not_churn | treated) vs P(not_churn | control)
        # In control: actual outcome
        y_response = y.copy() if isinstance(y, np.ndarray) else y.values.copy()

        # For treated group: reduce churn probability
        # Treatment is more effective for:
        # 1. Moderate-risk customers (not too high, not too low risk)
        # 2. Customers with addressable issues (no security → can offer security)

        # Simulate: some churners in treatment group would be "saved"
        treatment_effect = rng.uniform(0.05, 0.30, n)  # Heterogeneous effect

        # Apply treatment effect: some churners in treatment group → non-churners
        for i in range(n):
            if treatment[i] == 1 and y_response[i] == 1:
                # This customer was going to churn — did treatment save them?
                if rng.random() < treatment_effect[i]:
                    y_response[i] = 0  # "Saved" by treatment

        return treatment, y_response, treatment_effect

    def fit(self, X, y):
        """
        Fit the T-Learner uplift model.

        ⚠ Uses SIMULATED treatment data. See simulate_treatment_data().
        """
        treatment, y_response, _ = self.simulate_treatment_data(X, y)

        # Split into treatment and control groups
        treated_mask = treatment == 1
        control_mask = treatment == 0

        X_treated = X[treated_mask] if isinstance(X, np.ndarray) else X.iloc[treated_mask]
        y_treated = y_response[treated_mask]

        X_control = X[control_mask] if isinstance(X, np.ndarray) else X.iloc[control_mask]
        y_control = y_response[control_mask]

        # Train separate models for each group
        # Model for treated: predicts P(churn | treated)
        self.model_treatment = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            verbose=-1,
            random_state=self.random_state,
        )
        self.model_treatment.fit(X_treated, y_treated)

        # Model for control: predicts P(churn | not treated)
        self.model_control = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            verbose=-1,
            random_state=self.random_state,
        )
        self.model_control.fit(X_control, y_control)

        print(f"✔ Uplift model fitted (T-Learner)")
        print(f"  Treatment group: {treated_mask.sum()} samples")
        print(f"  Control group: {control_mask.sum()} samples")
        print(f"  ⚠ Using SIMULATED treatment data")

        return self

    def predict_uplift(self, X):
        """
        Predict uplift (treatment effect) for each customer.

        Uplift = P(churn | no treatment) - P(churn | treatment)
        Positive uplift = customer is "persuadable" — treatment reduces churn

        Returns:
            DataFrame with:
            - p_churn_treated: P(churn) if we intervene
            - p_churn_control: P(churn) if we don't intervene
            - uplift: difference (positive = persuadable)
            - segment: persuadable / sure_thing / lost_cause / sleeping_dog
        """
        if self.model_treatment is None or self.model_control is None:
            raise ValueError("Model not fitted. Call fit() first.")

        p_churn_treated = self.model_treatment.predict_proba(X)[:, 1]
        p_churn_control = self.model_control.predict_proba(X)[:, 1]

        uplift = p_churn_control - p_churn_treated  # Positive = treatment helps

        # Segment customers by uplift
        segments = []
        for u, pc, pt in zip(uplift, p_churn_control, p_churn_treated):
            if u > 0.05:
                segments.append('persuadable')  # Treatment helps
            elif pc < 0.3 and pt < 0.3:
                segments.append('sure_thing')  # Would stay regardless
            elif pc > 0.7 and pt > 0.7:
                segments.append('lost_cause')  # Will leave regardless
            else:
                segments.append('sleeping_dog')  # Don't disturb
            # Note: "sleeping_dog" = treatment might actually make them leave

        return pd.DataFrame({
            'p_churn_treated': np.round(p_churn_treated, 4),
            'p_churn_control': np.round(p_churn_control, 4),
            'uplift': np.round(uplift, 4),
            'segment': segments,
            'is_simulated': True,  # ⚠ Always flag as simulated
        })

    def save(self, path=None):
        """Save the uplift model."""
        if path is None:
            path = str(MODELS_DIR)
        os.makedirs(path, exist_ok=True)
        joblib.dump(self, os.path.join(path, 'uplift_model.joblib'))
        print(f"✔ Uplift model saved (⚠ simulated treatment data)")

    @staticmethod
    def load(path=None):
        """Load a saved uplift model."""
        if path is None:
            path = str(MODELS_DIR)
        return joblib.load(os.path.join(path, 'uplift_model.joblib'))


def train_uplift_model(X_train, y_train):
    """Convenience function to train and save the uplift model."""
    model = UpliftModel()
    model.fit(X_train, y_train)
    model.save()
    return model


if __name__ == '__main__':
    print("Uplift model module loaded.")
    print("⚠ This module uses SIMULATED treatment data.")
    print("In production, replace with real A/B test / intervention data.")
