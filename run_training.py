"""
run_training.py — Master Training Orchestrator

Runs the complete training pipeline end-to-end:
1. Data loading & synthetic generation
2. Feature engineering
3. Churn classifier training (LR baseline + LightGBM + calibration + Optuna)
4. SHAP explainability
5. Survival model (Cox PH)
6. Uplift model (simulated T-Learner)
7. Retention engine setup
8. Save all artifacts

Run: python run_training.py
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))

from features import (
    build_preprocessing_pipeline, build_final_transformer, FeatureSelector,
    prepare_target, save_pipeline, get_feature_names
)
from train import (
    load_ibm_telco, load_cell2cell_stats, load_synthetic_data,
    generate_synthetic_data, train_baseline_logistic_regression,
    train_lightgbm_with_optuna, calibrate_model, evaluate_model,
    RANDOM_STATE, TEST_SIZE, MODELS_DIR, DATA_PROCESSED
)

PROJECT_ROOT = Path(__file__).resolve().parent

def main():
    print("=" * 70)
    print("  TELECOM CHURN AI — Full Training Pipeline")
    print("=" * 70)

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(DATA_PROCESSED, exist_ok=True)

    # ── 1. Load Data ──────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  STEP 1: Loading Data")
    print("─" * 60)

    df_real = load_ibm_telco()
    cell2cell_stats = load_cell2cell_stats()

    # ── 2. Synthetic Data ─────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  STEP 2: Synthetic Data Generation")
    print("─" * 60)

    df_synthetic = load_synthetic_data()
    if df_synthetic is None:
        df_synthetic = generate_synthetic_data(df_real, target_total_rows=50000)

    # ── 3. Feature Engineering + Train/Test Split ─────────────────────────
    print("\n" + "─" * 60)
    print("  STEP 3: Feature Engineering & Data Preparation")
    print("─" * 60)

    y_real = prepare_target(df_real)

    # Build and fit preprocessing pipeline
    preprocess_pipeline = build_preprocessing_pipeline(is_training=True, random_state=RANDOM_STATE)
    df_processed = preprocess_pipeline.fit_transform(df_real)

    # Select features
    selector = FeatureSelector()
    df_selected = selector.transform(df_processed)

    numeric_features = [c for c in FeatureSelector.FINAL_NUMERIC_FEATURES if c in df_selected.columns]
    categorical_features = [c for c in FeatureSelector.CATEGORICAL_FEATURES if c in df_selected.columns]

    print(f"Numeric features ({len(numeric_features)}): {numeric_features}")
    print(f"Categorical features ({len(categorical_features)}): {categorical_features}")

    # Train/test split (REAL data only for test)
    from sklearn.model_selection import train_test_split

    X_real_train, X_real_test, y_real_train, y_real_test = train_test_split(
        df_selected, y_real, test_size=TEST_SIZE,
        stratify=y_real, random_state=RANDOM_STATE
    )

    # Keep processed version of real training data for survival model
    df_processed_train_idx = X_real_train.index
    df_processed_test_idx = X_real_test.index

    # Process synthetic data
    if df_synthetic is not None and len(df_synthetic) > 0:
        y_synth = prepare_target(df_synthetic)
        df_synth_processed = preprocess_pipeline.transform(df_synthetic)
        df_synth_selected = selector.transform(df_synth_processed)

        X_train = pd.concat([X_real_train, df_synth_selected], ignore_index=True)
        y_train = pd.concat([y_real_train, y_synth], ignore_index=True)
        print(f"Training set: {len(X_train):,} rows (real: {len(X_real_train):,}, synth: {len(df_synth_selected):,})")
    else:
        X_train = X_real_train
        y_train = y_real_train
        print(f"Training set: {len(X_train):,} rows (real only)")

    print(f"Test set: {len(X_real_test):,} rows (100% real)")

    # Build and fit column transformer
    column_transformer = build_final_transformer(numeric_features, categorical_features)
    X_train_transformed = column_transformer.fit_transform(X_train)
    X_test_transformed = column_transformer.transform(X_real_test)

    feature_names = get_feature_names(column_transformer, numeric_features, categorical_features)
    print(f"Total features after encoding: {len(feature_names)}")

    # Save pipeline artifacts
    save_pipeline(preprocess_pipeline, column_transformer, feature_names, str(MODELS_DIR))

    # ── 4. Baseline Model ─────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  STEP 4: Baseline Logistic Regression")
    print("─" * 60)

    lr_model, lr_metrics = train_baseline_logistic_regression(
        X_train_transformed, y_train, X_test_transformed, y_real_test
    )

    # ── 5. LightGBM + Optuna ─────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  STEP 5: LightGBM with Optuna Tuning")
    print("─" * 60)

    lgb_model, lgb_metrics, best_params = train_lightgbm_with_optuna(
        X_train_transformed, y_train, X_test_transformed, y_real_test, feature_names
    )

    # ── 6. Probability Calibration ────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  STEP 6: Probability Calibration")
    print("─" * 60)

    calibrated_model, cal_metrics = calibrate_model(
        lgb_model, X_train_transformed, y_train, X_test_transformed, y_real_test
    )

    # ── 7. SHAP Explainability ────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  STEP 7: SHAP Explainability")
    print("─" * 60)

    from explain import ChurnExplainer
    import joblib

    explainer = ChurnExplainer(
        calibrated_model, feature_names,
        background_data=X_train_transformed[:500]  # Sample for background
    )

    # Save SHAP plots
    try:
        explainer.save_shap_plots(X_test_transformed[:200], str(MODELS_DIR))
    except Exception as e:
        print(f"Warning: SHAP plots failed ({e})")

    # Save global importance
    try:
        explainer.save_global_importance(str(MODELS_DIR))
    except Exception as e:
        print(f"Warning: Global importance failed ({e})")

    # ── 8. Survival Model ─────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  STEP 8: Survival Model (Cox PH)")
    print("─" * 60)

    from survival import ChurnSurvivalModel

    survival_model = ChurnSurvivalModel()
    # Use real data for survival model (not synthetic, since survival needs real tenure/event)
    df_processed_real_train = df_processed.iloc[df_processed_train_idx].reset_index(drop=True)
    y_real_train_reset = y_real_train.reset_index(drop=True)

    try:
        survival_df = survival_model.prepare_survival_data(df_processed_real_train, y_real_train_reset)
        survival_model.fit(survival_df)
        survival_model.save(str(MODELS_DIR))

        # Plot survival curves
        try:
            survival_model.plot_survival_curves(df_processed_real_train, str(MODELS_DIR))
        except Exception as e:
            print(f"Warning: Survival curve plots failed ({e})")

        # Save hazard ratios
        hr = survival_model.get_hazard_ratios()
        hr.to_csv(MODELS_DIR / 'hazard_ratios.csv')
        print("✔ Survival model trained and saved")
    except Exception as e:
        print(f"Warning: Survival model failed ({e}). Predictions will use fallback.")

    # ── 9. Uplift Model ───────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  STEP 9: Uplift Model (Simulated T-Learner)")
    print("─" * 60)

    from uplift import UpliftModel

    try:
        uplift_model = UpliftModel(random_state=RANDOM_STATE)
        uplift_model.fit(X_train_transformed, y_train.values)
        uplift_model.save(str(MODELS_DIR))
        print("✔ Uplift model trained and saved (⚠ simulated treatment data)")
    except Exception as e:
        print(f"Warning: Uplift model failed ({e})")

    # ── 10. Retention Engine Setup ────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  STEP 10: Retention Engine Setup")
    print("─" * 60)

    from retention_engine import RetentionEngine
    retention = RetentionEngine()
    retention.save_playbook(str(MODELS_DIR))
    print("✔ Retention playbook saved")

    # ── 11. Save All Models & Metrics ─────────────────────────────────────
    print("\n" + "─" * 60)
    print("  STEP 11: Saving All Artifacts")
    print("─" * 60)

    import joblib
    joblib.dump(lr_model, MODELS_DIR / 'logistic_regression.joblib')
    joblib.dump(lgb_model, MODELS_DIR / 'lightgbm_model.joblib')
    joblib.dump(calibrated_model, MODELS_DIR / 'calibrated_model.joblib')
    joblib.dump(best_params, MODELS_DIR / 'best_params.joblib')

    all_metrics = {
        'logistic_regression': lr_metrics,
        'lightgbm': lgb_metrics,
        'calibrated_lightgbm': cal_metrics,
    }
    with open(MODELS_DIR / 'training_metrics.json', 'w') as f:
        json.dump(all_metrics, f, indent=2)

    # ── 12. MLflow Logging ────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  STEP 12: MLflow Logging")
    print("─" * 60)

    try:
        from train import log_to_mlflow
        log_to_mlflow(lr_model, lr_metrics, {'C': 1.0, 'class_weight': 'balanced'},
                     feature_names, 'logistic_regression')
        log_to_mlflow(calibrated_model, cal_metrics, best_params,
                     feature_names, 'calibrated_lightgbm')
        print("✔ MLflow experiments logged")
    except Exception as e:
        print(f"MLflow logging skipped ({e})")

    # ═══════════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ═══════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 70)
    print("  ✔ TRAINING COMPLETE — ALL ARTIFACTS SAVED")
    print("=" * 70)

    print("\n  Model Performance (on real held-out test set):")
    print(f"  {'Model':<30} {'AUC':>8} {'F1':>8} {'Recall':>8} {'Cost':>8}")
    print("  " + "─" * 62)
    for name, m in all_metrics.items():
        print(f"  {name:<30} {m['roc_auc']:>8.4f} {m['f1']:>8.4f} "
              f"{m['recall']:>8.4f} {m['total_cost']:>8}")

    print(f"\n  Artifacts saved to: {MODELS_DIR}")
    print(f"  Total features: {len(feature_names)}")
    print(f"  Training rows: {len(X_train):,} (real + synthetic)")
    print(f"  Test rows: {len(X_real_test):,} (100% real)")

    print("\n  Next steps:")
    print("  1. Start backend:  python backend/main.py")
    print("  2. Start frontend: streamlit run frontend/app.py")

    return all_metrics


if __name__ == '__main__':
    metrics = main()
