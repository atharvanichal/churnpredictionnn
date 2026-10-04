import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from predict import ChurnPredictor

predictor = ChurnPredictor('models')
print("Predictor loaded:", predictor.loaded)

# Sample customer matching schema
customer = {
    'Gender': 'Female',
    'Senior Citizen': 'No',
    'Partner': 'No',
    'Dependents': 'No',
    'Tenure Months': 2,
    'Phone Service': 'Yes',
    'Multiple Lines': 'No',
    'Internet Service': 'Fiber optic',
    'Online Security': 'No',
    'Online Backup': 'No',
    'Device Protection': 'No',
    'Tech Support': 'No',
    'Streaming TV': 'Yes',
    'Streaming Movies': 'Yes',
    'Contract': 'Month-to-month',
    'Paperless Billing': 'Yes',
    'Payment Method': 'Electronic check',
    'Monthly Charges': 89.85,
    'Total Charges': 179.7,
}

result = predictor.predict_single(customer)
print("\n--- SAMPLE PREDICTION RESULT ---")
import json
print(json.dumps(result, indent=2))
