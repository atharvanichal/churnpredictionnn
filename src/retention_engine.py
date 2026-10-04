"""
src/retention_engine.py — Retention Action Recommendation Engine

DESIGN DECISIONS:
- Maps SHAP reason codes to a concrete retention playbook
- Each reason code triggers a specific, actionable retention offer
- Priority is determined by CLV × churn_probability × (1 - retention_success_probability)
- Actions are ranked by expected ROI: cost of intervention vs. CLV saved

THE RETENTION PLAYBOOK:
Based on telecom industry best practices and the specific service columns
available in IBM Telco. Each action maps to a known churn driver.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / 'models'

# ──────────────────────────────────────────────────────────────────────────────
# Retention Playbook
# ──────────────────────────────────────────────────────────────────────────────

# Maps churn driver (from SHAP) → specific retention action
# Each action has: description, estimated cost, expected success rate, urgency
RETENTION_PLAYBOOK = {
    # Service-related actions (directly from IBM Telco service columns)
    'no_tech_support': {
        'action': 'Offer 3-month free trial of Tech Support',
        'description': 'Customer has no Tech Support subscription. A free trial '
                      'reduces friction and increases service stickiness.',
        'estimated_monthly_cost': 10,
        'expected_success_rate': 0.35,
        'category': 'service_upgrade',
        'urgency': 'high',
    },
    'no_online_security': {
        'action': 'Offer 3-month free trial of Online Security',
        'description': 'Customer lacks online security. Adding this protective '
                      'service increases switching cost and perceived value.',
        'estimated_monthly_cost': 10,
        'expected_success_rate': 0.30,
        'category': 'service_upgrade',
        'urgency': 'high',
    },
    'no_device_protection': {
        'action': 'Offer free Device Protection for 6 months',
        'description': 'No device protection plan. Protecting their device '
                      'creates a tangible benefit that discourages switching.',
        'estimated_monthly_cost': 8,
        'expected_success_rate': 0.25,
        'category': 'service_upgrade',
        'urgency': 'medium',
    },
    'no_online_backup': {
        'action': 'Offer free Online Backup for 3 months',
        'description': 'Adding backup service creates data lock-in.',
        'estimated_monthly_cost': 5,
        'expected_success_rate': 0.20,
        'category': 'service_upgrade',
        'urgency': 'low',
    },

    # Contract-related actions
    'month_to_month': {
        'action': 'Offer 15% discount for annual contract upgrade',
        'description': 'Month-to-month contract is the #1 churn predictor. '
                      'Incentivize commitment with a meaningful discount.',
        'estimated_monthly_cost': 15,  # ~15% of avg monthly charge
        'expected_success_rate': 0.40,
        'category': 'contract_upgrade',
        'urgency': 'high',
    },
    'high_risk_contract': {
        'action': 'Priority: Bundle discount (annual contract + security + support)',
        'description': 'High-risk customer: month-to-month + no security + no support. '
                      'Offer a comprehensive bundle at 20% discount.',
        'estimated_monthly_cost': 25,
        'expected_success_rate': 0.45,
        'category': 'bundle_offer',
        'urgency': 'critical',
    },

    # Billing-related actions
    'payment_friction': {
        'action': 'Offer $5/month autopay discount + setup assistance',
        'description': 'Electronic check + paperless billing = high friction. '
                      'Autopay reduces involuntary churn and improves experience.',
        'estimated_monthly_cost': 5,
        'expected_success_rate': 0.50,
        'category': 'billing_optimization',
        'urgency': 'medium',
    },
    'high_charges': {
        'action': 'Offer loyalty discount or usage-based plan migration',
        'description': 'Monthly charges are high relative to tenure/services. '
                      'A 10% loyalty discount or right-sized plan reduces bill shock.',
        'estimated_monthly_cost': 12,
        'expected_success_rate': 0.35,
        'category': 'pricing_adjustment',
        'urgency': 'medium',
    },

    # Usage-related actions (from Cell2Cell-informed features)
    'usage_decline': {
        'action': 'Proactive re-engagement outreach + usage incentive',
        'description': 'Usage is declining — this often precedes churn by 2-3 months. '
                      'Proactive outreach with bonus data/minutes can reverse the trend.',
        'estimated_monthly_cost': 8,
        'expected_success_rate': 0.25,
        'category': 'engagement',
        'urgency': 'high',
    },
    'high_support_calls': {
        'action': 'Assign dedicated support rep + satisfaction follow-up',
        'description': 'High support ticket volume indicates frustration. '
                      'Dedicated support resolves issues faster and shows care.',
        'estimated_monthly_cost': 15,
        'expected_success_rate': 0.30,
        'category': 'support_escalation',
        'urgency': 'high',
    },
    'network_quality': {
        'action': 'Network quality investigation + service credit',
        'description': 'High dropped call rate indicates network issues in their area. '
                      'Investigate and offer a service credit while resolving.',
        'estimated_monthly_cost': 20,
        'expected_success_rate': 0.30,
        'category': 'network_remediation',
        'urgency': 'medium',
    },
    'retention_call': {
        'action': 'Immediate warm transfer to retention specialist + best available offer',
        'description': 'Customer has already called the retention team — they are '
                      'actively considering leaving. Escalate immediately.',
        'estimated_monthly_cost': 30,
        'expected_success_rate': 0.55,
        'category': 'immediate_escalation',
        'urgency': 'critical',
    },

    # Fallback
    'general_churn_risk': {
        'action': 'Personalized loyalty call + thank-you credit',
        'description': 'General churn risk without a clear single driver. '
                      'A personal call acknowledging their loyalty can help.',
        'estimated_monthly_cost': 10,
        'expected_success_rate': 0.20,
        'category': 'relationship_building',
        'urgency': 'medium',
    },
}

# Maps SHAP feature names to playbook keys
FEATURE_TO_PLAYBOOK = {
    'Tech Support': 'no_tech_support',
    'Online Security': 'no_online_security',
    'Device Protection': 'no_device_protection',
    'Online Backup': 'no_online_backup',
    'Contract_Month-to-month': 'month_to_month',
    'high_risk_contract_flag': 'high_risk_contract',
    'payment_friction_flag': 'payment_friction',
    'Monthly Charges': 'high_charges',
    'avg_charge_per_service': 'high_charges',
    'monthly_charges_per_tenure': 'high_charges',
    'usage_decline_slope': 'usage_decline',
    'call_minutes_decline': 'usage_decline',
    'support_ticket_count': 'high_support_calls',
    'support_ticket_trend': 'high_support_calls',
    'dropped_calls_rate': 'network_quality',
    'retention_call_flag': 'retention_call',
    'no_protection_flag': 'no_device_protection',
}


# ──────────────────────────────────────────────────────────────────────────────
# Retention Recommender
# ──────────────────────────────────────────────────────────────────────────────

class RetentionEngine:
    """
    Recommends specific retention actions based on SHAP reason codes.

    The engine:
    1. Takes SHAP reason codes (top churn drivers per customer)
    2. Maps each driver to a retention action from the playbook
    3. Selects the best action(s) based on expected ROI
    4. Computes a simulated retention success probability
    """

    def __init__(self):
        self.playbook = RETENTION_PLAYBOOK
        self.feature_map = FEATURE_TO_PLAYBOOK

    def recommend(self, reason_codes, churn_probability, clv_estimate):
        """
        Generate retention recommendation for a single customer.

        Args:
            reason_codes: List of dicts from ChurnExplainer.get_reason_codes()
            churn_probability: Calibrated P(churn)
            clv_estimate: Estimated customer lifetime value

        Returns:
            Dict with recommended actions, priority score, and ROI estimate
        """
        # Map reason codes to playbook actions
        actions = []
        seen_playbook_keys = set()

        for reason in reason_codes:
            feat = reason['feature']
            shap_val = reason['shap_value']

            # Only consider features that INCREASE churn risk
            if shap_val <= 0:
                continue

            playbook_key = self.feature_map.get(feat, None)

            # Check if the service feature value actually indicates absence
            if feat in ['Tech Support', 'Online Security', 'Device Protection', 'Online Backup']:
                # These are binary (0/1) — positive SHAP when 0 means "not having it increases risk"
                # The SHAP direction should indicate the feature pushing toward churn
                pass

            if playbook_key and playbook_key not in seen_playbook_keys:
                action_info = self.playbook[playbook_key].copy()
                action_info['triggered_by'] = reason['reason']
                action_info['shap_impact'] = shap_val
                actions.append(action_info)
                seen_playbook_keys.add(playbook_key)

        # If no specific actions found, use general fallback
        if not actions:
            fallback = self.playbook['general_churn_risk'].copy()
            fallback['triggered_by'] = 'General churn risk'
            fallback['shap_impact'] = 0
            actions.append(fallback)

        # Ensure numeric inputs are clean
        if clv_estimate is None or np.isnan(clv_estimate) or np.isinf(clv_estimate):
            clv_estimate = 500.0
        if churn_probability is None or np.isnan(churn_probability) or np.isinf(churn_probability):
            churn_probability = 0.5

        # Sort by expected ROI
        for action in actions:
            roi = (
                clv_estimate * churn_probability * action['expected_success_rate']
                - action['estimated_monthly_cost'] * 3  # 3-month intervention cost
            )
            action['expected_roi'] = 0.0 if (np.isnan(roi) or np.isinf(roi)) else roi

        actions.sort(key=lambda x: x['expected_roi'], reverse=True)

        # Primary recommendation = highest ROI action
        primary = actions[0]

        # Simulated retention success probability (weighted average of action success rates)
        retention_success_prob = min(
            sum(a['expected_success_rate'] for a in actions[:3]) / max(len(actions[:3]), 1),
            0.85  # Cap at 85%
        )

        # Priority score: churn_prob × CLV × (1 - retention_success_prob)
        priority_score = churn_probability * clv_estimate * (1 - retention_success_prob)
        if np.isnan(priority_score) or np.isinf(priority_score):
            priority_score = 100.0

        # Value-at-risk tier
        if priority_score > 500:
            tier = 'HIGH'
        elif priority_score > 200:
            tier = 'MEDIUM'
        else:
            tier = 'LOW'

        return {
            'primary_action': primary['action'],
            'primary_description': primary['description'],
            'primary_category': primary['category'],
            'urgency': primary['urgency'],
            'all_actions': [
                {
                    'action': a['action'],
                    'category': a['category'],
                    'urgency': a['urgency'],
                    'triggered_by': a['triggered_by'],
                    'expected_roi': round(float(a['expected_roi']), 2),
                    'success_rate': a['expected_success_rate'],
                }
                for a in actions
            ],
            'retention_success_probability': round(float(retention_success_prob), 3),
            'priority_score': round(float(priority_score), 2),
            'value_risk_tier': tier,
        }

    def compute_clv(self, monthly_charges, median_survival_months, churn_probability):
        """
        Compute Customer Lifetime Value.

        CLV = MonthlyCharges × expected_remaining_tenure_months
        Where expected_remaining = median_survival × (1 - churn_prob) + min_survival × churn_prob
        """
        if monthly_charges is None or np.isnan(monthly_charges) or np.isinf(monthly_charges):
            monthly_charges = 65.0
        if median_survival_months is None or np.isnan(median_survival_months) or np.isinf(median_survival_months):
            median_survival_months = 24.0
        if churn_probability is None or np.isnan(churn_probability) or np.isinf(churn_probability):
            churn_probability = 0.5

        # Expected remaining months (weighted by churn probability)
        expected_remaining = (
            median_survival_months * (1 - churn_probability)
            + 3 * churn_probability  # If they churn, assume ~3 months left
        )
        clv = monthly_charges * expected_remaining
        if np.isnan(clv) or np.isinf(clv):
            clv = monthly_charges * 12.0
        return max(round(float(clv), 2), 0.0)

    def save_playbook(self, path=None):
        """Save the retention playbook as JSON for reference."""
        if path is None:
            path = str(MODELS_DIR)
        with open(Path(path) / 'retention_playbook.json', 'w') as f:
            json.dump(self.playbook, f, indent=2)
        print(f"✔ Retention playbook saved")


# ──────────────────────────────────────────────────────────────────────────────
# Convenience
# ──────────────────────────────────────────────────────────────────────────────

def create_retention_engine():
    """Factory function."""
    engine = RetentionEngine()
    engine.save_playbook()
    return engine


if __name__ == '__main__':
    print("Retention engine module loaded.")
    print(f"Playbook contains {len(RETENTION_PLAYBOOK)} action templates.")
    print(f"Feature-to-playbook mappings: {len(FEATURE_TO_PLAYBOOK)}")
