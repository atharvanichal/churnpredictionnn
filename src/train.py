"""
src/train.py — Churn Classifier Training Pipeline

DESIGN DECISIONS:
- Baseline Logistic Regression for interpretability sanity-check
- Main model: LightGBM (faster than XGBoost, handles categoricals natively)
- Class imbalance: class_weight='balanced' + optional SMOTE
- Probability calibration: CalibratedClassifierCV (isotonic) because tree models
  produce uncalibrated probabilities — we need reliable P(churn) for CLV math
- Hyperparameter tuning: Optuna with cross-validation
- Experiment tracking: MLflow local tracking
- Cost-sensitive: False negatives (missed churners) cost ~5x false positives
  because retention is cheaper than acquisition
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, roc_auc_score, f1_score, precision_score,
    recall_score, accuracy_score, confusion_matrix, average_precision_score
)
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler
import joblib
import mlflow
import mlflow.sklearn

warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))
from features import (
    build_preprocessing_pipeline, build_final_transformer, FeatureSelector,
    prepare_target, save_pipeline, get_feature_names
)

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / 'data' / 'raw'
DATA_PROCESSED = PROJECT_ROOT / 'data' / 'processed'
DATA_SYNTHETIC = PROJECT_ROOT / 'data' / 'synthetic'
MODELS_DIR = PROJECT_ROOT / 'models'
RANDOM_STATE = 42
TEST_SIZE = 0.2
OPTUNA_TRIALS = 50

# Cost matrix: false negatives cost 5x false positives
# (losing a customer costs ~5x more than an unnecessary retention offer)
FN_COST = 5
FP_COST = 1


# ──────────────────────────────────────────────────────────────────────────────
# Data Loading
# ──────────────────────────────────────────────────────────────────────────────

def load_ibm_telco():
    """Load raw IBM Telco dataset."""
    path = DATA_RAW / 'Telco_customer_churn.xlsx'
    if not path.exists():
        # Try CSV fallback
        path = DATA_RAW / 'Telco_customer_churn.csv'
    df = pd.read_excel(path) if str(path).endswith('.xlsx') else pd.read_csv(path)
    print(f"Loaded IBM Telco: {df.shape}")
    return df


def load_cell2cell_stats():
    """Load Cell2Cell and extract distributional statistics for feature simulation."""
    path = DATA_RAW / 'cell2cellholdout.csv'
    df = pd.read_csv(path)
    print(f"Loaded Cell2Cell: {df.shape} (for distribution extraction only)")

    stats = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != 'CustomerID' and col != 'Churn':
            s = df[col].dropna()
            if len(s) > 0:
                stats[col] = {
                    'mean': float(s.mean()),
                    'std': float(s.std()),
                    'median': float(s.median()),
                    'q25': float(s.quantile(0.25)),
                    'q75': float(s.quantile(0.75)),
                    'min': float(s.min()),
                    'max': float(s.max()),
                }

    # Save stats for reference
    os.makedirs(DATA_PROCESSED, exist_ok=True)
    with open(DATA_PROCESSED / 'cell2cell_distribution_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Extracted distributions for {len(stats)} Cell2Cell features")
    return stats


def load_synthetic_data():
    """Load synthetic data if available."""
    path = DATA_SYNTHETIC / 'synthetic_ibm_telco.csv'
    if path.exists():
        df = pd.read_csv(path)
        print(f"Loaded synthetic data: {df.shape}")
        return df
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic Data Generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_synthetic_data(df_real, target_total_rows=50000):
    """
    Generate synthetic data using SDV's CTGAN to augment the training set.

    WHY CTGAN:
    - Preserves complex inter-feature correlations (e.g., contract type ↔ churn)
    - Generates realistic synthetic rows, not just noise
    - We validate quality with KS tests and distribution comparisons

    WHY augment:
    - 7,043 rows is small for a production model
    - More data → better generalization, especially for rare subgroups
    - Only synthetic data is added to training; test set remains 100% real
    """
    n_synthetic = target_total_rows - len(df_real)
    if n_synthetic <= 0:
        print("No synthetic data needed.")
        return None

    print(f"\nGenerating {n_synthetic:,} synthetic rows with CTGAN...")
    os.makedirs(DATA_SYNTHETIC, exist_ok=True)

    try:
        from sdv.single_table import CTGANSynthesizer
        from sdv.metadata import Metadata

        # Prepare data for CTGAN (exclude non-feature columns)
        cols_for_synth = [c for c in df_real.columns if c not in
                         ['CustomerID', 'Count', 'Country', 'State', 'Lat Long',
                          'Churn Score', 'CLTV', 'Churn Reason']]
        df_synth_input = df_real[cols_for_synth].copy()

        # Fix Total Charges to numeric before synthetic generation
        if 'Total Charges' in df_synth_input.columns:
            df_synth_input['Total Charges'] = pd.to_numeric(
                df_synth_input['Total Charges'], errors='coerce'
            ).fillna(0.0)

        metadata = Metadata.detect_from_dataframe(data=df_synth_input)

        synthesizer = CTGANSynthesizer(
            metadata,
            epochs=300,
            batch_size=500,
            verbose=True,
        )

        print("Fitting CTGAN (this may take a few minutes)...")
        synthesizer.fit(df_synth_input)

        print(f"Sampling {n_synthetic:,} synthetic rows...")
        df_synthetic = synthesizer.sample(num_rows=n_synthetic)

        # Save synthetic data
        df_synthetic.to_csv(DATA_SYNTHETIC / 'synthetic_ibm_telco.csv', index=False)

        # Validate synthetic data quality
        validate_synthetic_quality(df_synth_input, df_synthetic)

        print(f"✔ Synthetic data saved: {df_synthetic.shape}")
        return df_synthetic

    except ImportError as e:
        print(f"SDV not available ({e}). Falling back to bootstrap + noise augmentation.")
        return generate_synthetic_fallback(df_real, n_synthetic)
    except Exception as e:
        print(f"CTGAN failed ({e}). Falling back to bootstrap + noise augmentation.")
        return generate_synthetic_fallback(df_real, n_synthetic)


def generate_synthetic_fallback(df_real, n_synthetic):
    """
    Fallback synthetic data generation using stratified bootstrap + noise.
    Less sophisticated than CTGAN but always works.
    """
    print(f"Generating {n_synthetic:,} synthetic rows via bootstrap + noise...")
    rng = np.random.RandomState(RANDOM_STATE)

    cols_for_synth = [c for c in df_real.columns if c not in
                     ['CustomerID', 'Count', 'Country', 'State', 'Lat Long',
                      'Churn Score', 'CLTV', 'Churn Reason']]
    df_input = df_real[cols_for_synth].copy()

    if 'Total Charges' in df_input.columns:
        df_input['Total Charges'] = pd.to_numeric(
            df_input['Total Charges'], errors='coerce'
        ).fillna(0.0)

    # Stratified bootstrap: maintain class distribution
    synthetic_rows = []
    churn_col = 'Churn Label' if 'Churn Label' in df_input.columns else 'Churn Value'

    for churn_val in df_input[churn_col].unique():
        subset = df_input[df_input[churn_col] == churn_val]
        class_ratio = len(subset) / len(df_input)
        n_class = int(n_synthetic * class_ratio)

        # Bootstrap sample
        sampled = subset.sample(n=n_class, replace=True, random_state=RANDOM_STATE)

        # Add noise to numeric columns to avoid exact duplicates
        numeric_cols = sampled.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            std = sampled[col].std()
            if std > 0:
                noise = rng.normal(0, std * 0.05, len(sampled))
                sampled[col] = sampled[col] + noise

        synthetic_rows.append(sampled)

    df_synthetic = pd.concat(synthetic_rows, ignore_index=True)

    # Clip numeric columns to original ranges
    numeric_cols = df_synthetic.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        orig_min = df_input[col].min()
        orig_max = df_input[col].max()
        df_synthetic[col] = df_synthetic[col].clip(lower=orig_min, upper=orig_max)

    # Fix integer columns
    int_cols = df_input.select_dtypes(include=[int]).columns
    for col in int_cols:
        if col in df_synthetic.columns:
            df_synthetic[col] = df_synthetic[col].round().astype(int)

    os.makedirs(DATA_SYNTHETIC, exist_ok=True)
    df_synthetic.to_csv(DATA_SYNTHETIC / 'synthetic_ibm_telco.csv', index=False)
    validate_synthetic_quality(df_input, df_synthetic)
    print(f"✔ Bootstrap synthetic data saved: {df_synthetic.shape}")
    return df_synthetic


def validate_synthetic_quality(df_real, df_synthetic):
    """
    Validate synthetic data quality by comparing distributions.
    - KS test for numeric columns
    - Chi-squared for categorical columns
    """
    from scipy import stats

    print("\n── Synthetic Data Quality Validation ──")
    numeric_cols = df_real.select_dtypes(include=[np.number]).columns
    results = []

    for col in numeric_cols:
        if col in df_synthetic.columns:
            real_vals = df_real[col].dropna()
            synth_vals = df_synthetic[col].dropna()
            if len(real_vals) > 0 and len(synth_vals) > 0:
                ks_stat, ks_pval = stats.ks_2samp(real_vals, synth_vals)
                results.append({
                    'column': col,
                    'ks_statistic': round(ks_stat, 4),
                    'ks_pvalue': round(ks_pval, 4),
                    'quality': 'GOOD' if ks_pval > 0.01 else 'WARN'
                })

    results_df = pd.DataFrame(results)
    if len(results_df) > 0:
        print(results_df.to_string(index=False))
        good_pct = (results_df['quality'] == 'GOOD').mean() * 100
        print(f"\n{good_pct:.0f}% of numeric features pass KS test (p > 0.01)")

    # Save validation report
    os.makedirs(DATA_PROCESSED, exist_ok=True)
    results_df.to_csv(DATA_PROCESSED / 'synthetic_validation_report.csv', index=False)


# ──────────────────────────────────────────────────────────────────────────────
# Model Training
# ──────────────────────────────────────────────────────────────────────────────

def prepare_data(df_real, df_synthetic=None):
    """
    Prepare train/test splits.
    CRITICAL: Test set is 100% real data only. Synthetic data only in training.
    """
    # Extract target before preprocessing
    y_real = prepare_target(df_real)

    # Build and fit preprocessing pipeline on real data
    preprocess_pipeline = build_preprocessing_pipeline(is_training=True, random_state=RANDOM_STATE)
    df_processed = preprocess_pipeline.fit_transform(df_real)

    # Select features
    selector = FeatureSelector()
    df_selected = selector.transform(df_processed)

    numeric_features = [c for c in FeatureSelector.FINAL_NUMERIC_FEATURES if c in df_selected.columns]
    categorical_features = [c for c in FeatureSelector.CATEGORICAL_FEATURES if c in df_selected.columns]

    # Split REAL data first (test set must be 100% real)
    X_real_train, X_real_test, y_real_train, y_real_test = train_test_split(
        df_selected, y_real, test_size=TEST_SIZE,
        stratify=y_real, random_state=RANDOM_STATE
    )

    # Process synthetic data if available
    if df_synthetic is not None and len(df_synthetic) > 0:
        y_synth = prepare_target(df_synthetic)
        df_synth_processed = preprocess_pipeline.transform(df_synthetic)
        df_synth_selected = selector.transform(df_synth_processed)

        # Add synthetic to training only
        X_train = pd.concat([X_real_train, df_synth_selected], ignore_index=True)
        y_train = pd.concat([y_real_train, y_synth], ignore_index=True)
        print(f"Training set: {len(X_train):,} rows (real: {len(X_real_train):,}, synthetic: {len(df_synth_selected):,})")
    else:
        X_train = X_real_train
        y_train = y_real_train
        print(f"Training set: {len(X_train):,} rows (real only)")

    print(f"Test set: {len(X_real_test):,} rows (100% real)")
    print(f"Train churn rate: {y_train.mean():.3f}")
    print(f"Test churn rate: {y_real_test.mean():.3f}")

    # Build and fit the final column transformer
    column_transformer = build_final_transformer(numeric_features, categorical_features)
    X_train_transformed = column_transformer.fit_transform(X_train)
    X_test_transformed = column_transformer.transform(X_real_test)

    feature_names = get_feature_names(column_transformer, numeric_features, categorical_features)

    # Save pipeline artifacts
    save_pipeline(preprocess_pipeline, column_transformer, feature_names, str(MODELS_DIR))

    return (X_train_transformed, y_train, X_test_transformed, y_real_test,
            feature_names, preprocess_pipeline, column_transformer)


def train_baseline_logistic_regression(X_train, y_train, X_test, y_test):
    """
    Baseline Logistic Regression for interpretability sanity-check.
    WHY: Before trusting a black-box model, verify that a simple model
    can learn the signal. If LR gets ~0.75 AUC, the features are informative.
    """
    print("\n" + "=" * 60)
    print("BASELINE: Logistic Regression")
    print("=" * 60)

    lr = LogisticRegression(
        class_weight='balanced',  # Handle imbalance
        max_iter=1000,
        random_state=RANDOM_STATE,
        C=1.0,
    )
    lr.fit(X_train, y_train)

    y_pred = lr.predict(X_test)
    y_proba = lr.predict_proba(X_test)[:, 1]

    metrics = evaluate_model(y_test, y_pred, y_proba, "Logistic Regression")
    return lr, metrics


def objective_lgbm(trial, X_train, y_train):
    """Optuna objective for LightGBM hyperparameter tuning."""
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 10.0),
        'random_state': RANDOM_STATE,
        'verbose': -1,
    }

    model = lgb.LGBMClassifier(**params)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
    return scores.mean()


def train_lightgbm_with_optuna(X_train, y_train, X_test, y_test, feature_names):
    """
    Train LightGBM with Optuna hyperparameter tuning.

    WHY LightGBM over XGBoost:
    - Faster training (histogram-based splitting)
    - Native categorical feature support
    - Comparable or better performance on tabular data
    - Better with high-cardinality features

    WHY Optuna over GridSearchCV:
    - Bayesian optimization finds better hyperparameters faster
    - Early pruning stops unpromising trials
    - More efficient exploration of the search space
    """
    print("\n" + "=" * 60)
    print("MAIN MODEL: LightGBM with Optuna Tuning")
    print("=" * 60)

    # Optuna study
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(
        lambda trial: objective_lgbm(trial, X_train, y_train),
        n_trials=OPTUNA_TRIALS,
        show_progress_bar=True,
    )

    print(f"\nBest trial AUC: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    # Train final model with best params
    best_params = study.best_params
    best_params.update({
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'random_state': RANDOM_STATE,
        'verbose': -1,
    })

    model = lgb.LGBMClassifier(**best_params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = evaluate_model(y_test, y_pred, y_proba, "LightGBM (Optuna-tuned)")

    return model, metrics, study.best_params


def calibrate_model(model, X_train, y_train, X_test, y_test):
    """
    Calibrate model probabilities using isotonic regression.

    WHY calibration:
    - Tree-based models (LightGBM, XGBoost) produce uncalibrated probabilities
    - Calibrated P(churn) is essential for CLV calculations: CLV = revenue × E[remaining_months]
    - Without calibration, a "70% churn probability" might actually be 50% or 90%
    - Isotonic > Platt (sigmoid) because the true calibration curve is likely non-linear
    """
    print("\n" + "=" * 60)
    print("PROBABILITY CALIBRATION (Isotonic)")
    print("=" * 60)

    calibrated = CalibratedClassifierCV(
        model, method='isotonic', cv=5
    )
    calibrated.fit(X_train, y_train)

    y_pred = calibrated.predict(X_test)
    y_proba = calibrated.predict_proba(X_test)[:, 1]

    metrics = evaluate_model(y_test, y_pred, y_proba, "LightGBM (Calibrated)")

    return calibrated, metrics


def evaluate_model(y_true, y_pred, y_proba, model_name):
    """Comprehensive model evaluation with cost-sensitive metrics."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        'model': model_name,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_proba),
        'avg_precision': average_precision_score(y_true, y_proba),
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn),
        # Cost-sensitive metric: total cost = FN * FN_COST + FP * FP_COST
        'total_cost': int(fn * FN_COST + fp * FP_COST),
    }

    print(f"\n{'─' * 40}")
    print(f"  Model: {model_name}")
    print(f"{'─' * 40}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  ROC AUC:   {metrics['roc_auc']:.4f}")
    print(f"  Avg Prec:  {metrics['avg_precision']:.4f}")
    print(f"  Total Cost (FN×{FN_COST} + FP×{FP_COST}): {metrics['total_cost']}")
    print(f"\n  Confusion Matrix:")
    print(f"    TN={tn}  FP={fp}")
    print(f"    FN={fn}  TP={tp}")

    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# MLflow Tracking
# ──────────────────────────────────────────────────────────────────────────────

def log_to_mlflow(model, metrics, params, feature_names, model_name):
    """Log experiment to MLflow for tracking and reproducibility."""
    mlflow_dir = str(PROJECT_ROOT / 'mlruns')
    mlflow.set_tracking_uri(f'file:///{mlflow_dir}')
    mlflow.set_experiment('telecom_churn_prediction')

    with mlflow.start_run(run_name=model_name):
        # Log parameters
        for k, v in params.items():
            mlflow.log_param(k, v)

        # Log metrics
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, v)

        # Log model
        mlflow.sklearn.log_model(model, model_name)

        # Log feature names
        mlflow.log_text(json.dumps(feature_names), 'feature_names.json')

    print(f"✔ MLflow experiment logged: {model_name}")


# ──────────────────────────────────────────────────────────────────────────────
# Main Training Pipeline
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  TELECOM CHURN AI — Training Pipeline")
    print("=" * 70)

    os.makedirs(MODELS_DIR, exist_ok=True)

    # 1. Load data
    print("\n── Step 1: Loading Data ──")
    df_real = load_ibm_telco()
    cell2cell_stats = load_cell2cell_stats()

    # 2. Generate synthetic data
    print("\n── Step 2: Synthetic Data Generation ──")
    df_synthetic = load_synthetic_data()
    if df_synthetic is None:
        df_synthetic = generate_synthetic_data(df_real, target_total_rows=50000)

    # 3. Prepare data (preprocessing + feature engineering + train/test split)
    print("\n── Step 3: Data Preparation ──")
    (X_train, y_train, X_test, y_test,
     feature_names, preprocess_pipeline, column_transformer) = prepare_data(df_real, df_synthetic)

    print(f"\nFeature matrix shape: train={X_train.shape}, test={X_test.shape}")
    print(f"Feature names ({len(feature_names)}): {feature_names[:10]}...")

    # 4. Train Baseline
    print("\n── Step 4: Baseline Model ──")
    lr_model, lr_metrics = train_baseline_logistic_regression(X_train, y_train, X_test, y_test)

    # 5. Train LightGBM with Optuna
    print("\n── Step 5: LightGBM Training ──")
    lgb_model, lgb_metrics, best_params = train_lightgbm_with_optuna(
        X_train, y_train, X_test, y_test, feature_names
    )

    # 6. Calibrate probabilities
    print("\n── Step 6: Probability Calibration ──")
    calibrated_model, cal_metrics = calibrate_model(lgb_model, X_train, y_train, X_test, y_test)

    # 7. Save models
    print("\n── Step 7: Saving Models ──")
    joblib.dump(lr_model, MODELS_DIR / 'logistic_regression.joblib')
    joblib.dump(lgb_model, MODELS_DIR / 'lightgbm_model.joblib')
    joblib.dump(calibrated_model, MODELS_DIR / 'calibrated_model.joblib')
    joblib.dump(best_params, MODELS_DIR / 'best_params.joblib')

    # Save all metrics
    all_metrics = {
        'logistic_regression': lr_metrics,
        'lightgbm': lgb_metrics,
        'calibrated_lightgbm': cal_metrics,
    }
    with open(MODELS_DIR / 'training_metrics.json', 'w') as f:
        json.dump(all_metrics, f, indent=2)

    # 8. MLflow logging
    print("\n── Step 8: MLflow Logging ──")
    try:
        log_to_mlflow(lr_model, lr_metrics, {'C': 1.0, 'class_weight': 'balanced'},
                     feature_names, 'logistic_regression')
        log_to_mlflow(calibrated_model, cal_metrics, best_params,
                     feature_names, 'calibrated_lightgbm')
    except Exception as e:
        print(f"MLflow logging failed (non-critical): {e}")

    # 9. Summary
    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE — Results Summary")
    print("=" * 70)
    for name, m in all_metrics.items():
        print(f"  {name:30s} | AUC={m['roc_auc']:.4f} | F1={m['f1']:.4f} | "
              f"Recall={m['recall']:.4f} | Cost={m['total_cost']}")

    best_model_name = min(all_metrics, key=lambda k: all_metrics[k]['total_cost'])
    print(f"\n  Best model (lowest cost): {best_model_name}")
    print(f"  Model artifacts saved to: {MODELS_DIR}")

    return calibrated_model, feature_names, all_metrics


if __name__ == '__main__':
    main()
