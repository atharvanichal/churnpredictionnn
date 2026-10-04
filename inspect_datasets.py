"""
Step 0: Dataset Inspection Script
Loads both datasets, prints schema info, and identifies overlapping/mappable columns.
"""
import pandas as pd
import sys
import os
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None

# ── 1. Load both datasets ──────────────────────────────────────────────
print("=" * 80)
print("LOADING DATASETS")
print("=" * 80)

ibm_path = r"c:\Users\anath\OneDrive\Desktop\AntiGravity Projects\Churn Prediction\Telco_customer_churn.xlsx"
cell2cell_path = r"c:\Users\anath\OneDrive\Desktop\AntiGravity Projects\Churn Prediction\cell2cellholdout.csv"

try:
    ibm = pd.read_excel(ibm_path)
    print(f"\n✓ IBM Telco loaded from .xlsx")
except Exception as e:
    print(f"✗ Error loading IBM Telco: {e}")
    sys.exit(1)

try:
    cell2cell = pd.read_csv(cell2cell_path)
    print(f"✓ Cell2Cell loaded from .csv")
except Exception as e:
    print(f"✗ Error loading Cell2Cell: {e}")
    sys.exit(1)

# ── 2. Shape ────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("DATASET SHAPES")
print("=" * 80)
print(f"IBM Telco:  {ibm.shape[0]:,} rows × {ibm.shape[1]} columns")
print(f"Cell2Cell:  {cell2cell.shape[0]:,} rows × {cell2cell.shape[1]} columns")

# ── 3. IBM Telco — info, head, target ───────────────────────────────────
print("\n" + "=" * 80)
print("IBM TELCO — df.info()")
print("=" * 80)
ibm.info()

print("\n" + "=" * 80)
print("IBM TELCO — df.head()")
print("=" * 80)
print(ibm.head().to_string())

# Find churn/target column
ibm_cols_lower = {c.lower(): c for c in ibm.columns}
ibm_churn_col = None
for candidate in ['churn', 'churn label', 'churnlabel', 'churn value', 'churnvalue', 'churn_label']:
    if candidate in ibm_cols_lower:
        ibm_churn_col = ibm_cols_lower[candidate]
        break
if ibm_churn_col is None:
    # Try partial match
    for c in ibm.columns:
        if 'churn' in c.lower():
            ibm_churn_col = c
            break

print(f"\n{'=' * 80}")
print(f"IBM TELCO — Target column: '{ibm_churn_col}'")
print(f"{'=' * 80}")
if ibm_churn_col:
    print(ibm[ibm_churn_col].value_counts())
    print(f"\nChurn rate: {ibm[ibm_churn_col].value_counts(normalize=True).to_string()}")
else:
    print("⚠ Could not find churn column! Columns containing 'churn':")
    print([c for c in ibm.columns if 'churn' in c.lower()])

# ── 4. Cell2Cell — info, head, target ───────────────────────────────────
print("\n" + "=" * 80)
print("CELL2CELL — df.info()")
print("=" * 80)
cell2cell.info()

print("\n" + "=" * 80)
print("CELL2CELL — df.head()")
print("=" * 80)
print(cell2cell.head().to_string())

cell2cell_cols_lower = {c.lower(): c for c in cell2cell.columns}
cell2cell_churn_col = None
for candidate in ['churn', 'churner', 'churn_label']:
    if candidate in cell2cell_cols_lower:
        cell2cell_churn_col = cell2cell_cols_lower[candidate]
        break
if cell2cell_churn_col is None:
    for c in cell2cell.columns:
        if 'churn' in c.lower():
            cell2cell_churn_col = c
            break

print(f"\n{'=' * 80}")
print(f"CELL2CELL — Target column: '{cell2cell_churn_col}'")
print(f"{'=' * 80}")
if cell2cell_churn_col:
    print(cell2cell[cell2cell_churn_col].value_counts())
    print(f"\nChurn rate: {cell2cell[cell2cell_churn_col].value_counts(normalize=True).to_string()}")
else:
    print("⚠ Could not find churn column! Columns containing 'churn':")
    print([c for c in cell2cell.columns if 'churn' in c.lower()])

# ── 5. Column names side by side ────────────────────────────────────────
print("\n" + "=" * 80)
print("COLUMN NAMES — SIDE BY SIDE")
print("=" * 80)
ibm_cols = list(ibm.columns)
c2c_cols = list(cell2cell.columns)
max_len = max(len(ibm_cols), len(c2c_cols))

print(f"{'IBM Telco':<45} | {'Cell2Cell':<45}")
print("-" * 45 + " | " + "-" * 45)
for i in range(max_len):
    ibm_c = ibm_cols[i] if i < len(ibm_cols) else ""
    c2c_c = c2c_cols[i] if i < len(c2c_cols) else ""
    print(f"{ibm_c:<45} | {c2c_c:<45}")

# ── 6. Identify overlapping / mappable columns ─────────────────────────
print("\n" + "=" * 80)
print("COLUMN MAPPING ANALYSIS")
print("=" * 80)

# Exact matches (case-insensitive)
ibm_set = set(c.lower().replace(" ", "").replace("_", "") for c in ibm.columns)
c2c_set = set(c.lower().replace(" ", "").replace("_", "") for c in cell2cell.columns)
exact_overlap = ibm_set & c2c_set
print(f"\nExact matches (normalized): {exact_overlap if exact_overlap else 'None'}")

# Semantic mapping
print("\nProposed Semantic Column Mapping:")
mappings = [
    ("tenure / Tenure Months", "MonthsInService", "Account tenure"),
    ("MonthlyCharges / Monthly Charges", "MonthlyRevenue", "Monthly billing"),
    ("TotalCharges / Total Charges", "TotalRevenue / LifetimeRevenue", "Total revenue"),
    ("Contract", "ContractTerm (inferred)", "Contract type"),
    ("Churn Label / Churn Value", "Churn", "Target variable"),
    ("gender / Gender", "—", "Demographics"),
    ("SeniorCitizen", "—", "Demographics"),
    ("Partner / Dependents", "MaritalStatus / ChildrenInHH", "Family status"),
    ("InternetService", "—", "Service type"),
    ("OnlineSecurity / TechSupport / etc.", "—", "Add-on services (IBM only)"),
    ("PaymentMethod", "CreditRating / HandsetPrice", "Billing/payment info"),
    ("—", "MonthlyMinutes / MeanMonthlyMinutes", "Usage minutes (Cell2Cell only)"),
    ("—", "OverageMinutes / PercChangeMinutes", "Usage trends (Cell2Cell only)"),
    ("—", "CustomerCareCalls / RetentionCalls", "Support interactions (Cell2Cell only)"),
    ("—", "HandsetRefurbished / HandsetWebCapable", "Device info (Cell2Cell only)"),
    ("—", "DirectorAssistedCalls / CallForwardingCalls", "Call behavior (Cell2Cell only)"),
    ("—", "RoamingCalls / DroppedCalls", "Network quality (Cell2Cell only)"),
    ("—", "AdjustmentsToCreditRating / ReferralsMadeBySubscriber", "Engagement (Cell2Cell only)"),
]

print(f"\n{'IBM Telco Column':<45} | {'Cell2Cell Column':<40} | {'Concept'}")
print("-" * 45 + " | " + "-" * 40 + " | " + "-" * 25)
for ibm_c, c2c_c, concept in mappings:
    print(f"{ibm_c:<45} | {c2c_c:<40} | {concept}")

# ── 7. Data types summary ──────────────────────────────────────────────
print("\n" + "=" * 80)
print("DATA TYPES SUMMARY")
print("=" * 80)
print("\nIBM Telco dtypes:")
print(ibm.dtypes.value_counts().to_string())
print("\nCell2Cell dtypes:")
print(cell2cell.dtypes.value_counts().to_string())

# ── 8. Missing values ──────────────────────────────────────────────────
print("\n" + "=" * 80)
print("MISSING VALUES")
print("=" * 80)
ibm_missing = ibm.isnull().sum()
ibm_missing = ibm_missing[ibm_missing > 0]
print(f"\nIBM Telco missing values ({len(ibm_missing)} columns):")
if len(ibm_missing) > 0:
    print(ibm_missing.to_string())
else:
    print("  No missing values")

c2c_missing = cell2cell.isnull().sum()
c2c_missing = c2c_missing[c2c_missing > 0]
print(f"\nCell2Cell missing values ({len(c2c_missing)} columns with missing):")
if len(c2c_missing) > 0:
    print(c2c_missing.to_string())
else:
    print("  No missing values")

# ── 9. Unique values in key categorical columns ────────────────────────
print("\n" + "=" * 80)
print("KEY CATEGORICAL COLUMN VALUES")
print("=" * 80)

# IBM key cats
ibm_cat_candidates = ['Contract', 'PaymentMethod', 'InternetService', 'gender', 'Gender',
                       'OnlineSecurity', 'TechSupport', 'DeviceProtection', 'StreamingTV',
                       'StreamingMovies', 'OnlineBackup', 'MultipleLines', 'PhoneService',
                       'PaperlessBilling']
print("\nIBM Telco categorical uniques:")
for col in ibm_cat_candidates:
    # Try exact and case-insensitive
    actual = None
    if col in ibm.columns:
        actual = col
    else:
        for c in ibm.columns:
            if c.lower().replace(" ", "") == col.lower().replace(" ", ""):
                actual = c
                break
    if actual:
        print(f"  {actual}: {ibm[actual].unique().tolist()}")

print("\nAll IBM Telco columns:")
for i, col in enumerate(ibm.columns):
    print(f"  [{i:2d}] {col} ({ibm[col].dtype}): nunique={ibm[col].nunique()}")

print("\nAll Cell2Cell columns:")
for i, col in enumerate(cell2cell.columns):
    print(f"  [{i:2d}] {col} ({cell2cell[col].dtype}): nunique={cell2cell[col].nunique()}")

print("\n" + "=" * 80)
print("INSPECTION COMPLETE")
print("=" * 80)
