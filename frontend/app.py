"""
frontend/app.py — Streamlit Frontend

Simple, functional frontend for the Churn AI system.
Tabs:
1. CSV Upload — batch predictions with downloadable results
2. Manual Entry — single customer form
3. Model Insights — SHAP plots, metrics, feature importance
"""

import os
import sys
import json
import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / 'models'

# ──────────────────────────────────────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Telecom Churn AI",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────────────────────────────────────
# Load Predictor (cached)
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_predictor():
    """Load the prediction pipeline (cached across sessions)."""
    try:
        from predict import ChurnPredictor
        predictor = ChurnPredictor(str(MODELS_DIR))
        return predictor
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        return None


@st.cache_data
def load_metrics():
    """Load training metrics."""
    try:
        with open(MODELS_DIR / 'training_metrics.json') as f:
            return json.load(f)
    except Exception:
        return {}


@st.cache_data
def load_feature_importance():
    """Load global feature importance."""
    try:
        with open(MODELS_DIR / 'global_feature_importance.json') as f:
            return json.load(f)
    except Exception:
        return []


# ──────────────────────────────────────────────────────────────────────────────
# Styling
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        text-align: center;
        color: #888;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #333;
    }
    .risk-high { color: #ff4444; font-weight: bold; }
    .risk-medium { color: #ffaa00; font-weight: bold; }
    .risk-low { color: #00cc66; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="main-header">📡 Telecom Churn AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Predict. Explain. Retain.</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["📁 CSV Upload", "✏️ Manual Entry", "📊 Model Insights"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: CSV Upload
# ═══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.header("Batch Churn Prediction")
    st.write("Upload a CSV file with customer data to get predictions for all customers.")

    uploaded_file = st.file_uploader(
        "Upload Customer CSV",
        type=['csv'],
        help="CSV should match the IBM Telco schema (Gender, Tenure Months, Contract, etc.)"
    )

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(f"**Uploaded:** {len(df)} customers, {len(df.columns)} columns")

        with st.expander("Preview uploaded data", expanded=False):
            st.dataframe(df.head(10), use_container_width=True)

        if st.button("🔮 Run Predictions", type="primary", key="batch_predict"):
            predictor = load_predictor()
            if predictor is None:
                st.error("Models not loaded. Please train models first (run src/train.py).")
            else:
                with st.spinner("Running predictions..."):
                    try:
                        results_df = predictor.predict_to_dataframe(df)
                        results = predictor.predict_batch(df)

                        # Summary metrics
                        st.subheader("📊 Prediction Summary")
                        col1, col2, col3, col4, col5 = st.columns(5)

                        n_high = sum(1 for r in results if r['churn_risk'] == 'HIGH')
                        n_med = sum(1 for r in results if r['churn_risk'] == 'MEDIUM')
                        n_low = sum(1 for r in results if r['churn_risk'] == 'LOW')

                        col1.metric("Total Customers", len(results))
                        col2.metric("🔴 High Risk", n_high)
                        col3.metric("🟡 Medium Risk", n_med)
                        col4.metric("🟢 Low Risk", n_low)
                        col5.metric("Avg Churn Prob",
                                   f"{np.mean([r['churn_probability'] for r in results]):.1%}")

                        # CLV at risk
                        total_clv_risk = sum(r['clv_estimate'] for r in results if r['churn_risk'] in ['HIGH', 'MEDIUM'])
                        st.metric("💰 Total CLV at Risk", f"${total_clv_risk:,.0f}")

                        # Results table
                        st.subheader("📋 Detailed Results (sorted by priority)")
                        st.dataframe(
                            results_df.style.applymap(
                                lambda v: 'color: #ff4444' if v == 'HIGH'
                                else 'color: #ffaa00' if v == 'MEDIUM'
                                else 'color: #00cc66' if v == 'LOW' else '',
                                subset=['churn_risk', 'value_risk_tier']
                            ),
                            use_container_width=True,
                            height=400,
                        )

                        # Download button
                        csv_data = results_df.to_csv(index=False)
                        st.download_button(
                            "⬇️ Download Results CSV",
                            csv_data,
                            "churn_predictions.csv",
                            "text/csv",
                        )

                        # Churn risk distribution chart
                        st.subheader("Risk Distribution")
                        risk_counts = pd.DataFrame({
                            'Risk Level': ['HIGH', 'MEDIUM', 'LOW'],
                            'Count': [n_high, n_med, n_low],
                        })
                        fig = px.bar(
                            risk_counts, x='Risk Level', y='Count',
                            color='Risk Level',
                            color_discrete_map={'HIGH': '#ff4444', 'MEDIUM': '#ffaa00', 'LOW': '#00cc66'},
                            title='Customer Churn Risk Distribution'
                        )
                        fig.update_layout(template='plotly_dark', showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)

                        # Churn probability histogram
                        probs = [r['churn_probability'] for r in results]
                        fig2 = px.histogram(
                            x=probs, nbins=30,
                            labels={'x': 'Churn Probability'},
                            title='Churn Probability Distribution'
                        )
                        fig2.update_layout(template='plotly_dark')
                        st.plotly_chart(fig2, use_container_width=True)

                    except Exception as e:
                        st.error(f"Prediction failed: {e}")
                        st.exception(e)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: Manual Entry
# ═══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.header("Single Customer Prediction")
    st.write("Enter customer details to get an instant churn prediction.")

    with st.form("customer_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Demographics")
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Partner", ["No", "Yes"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)

        with col2:
            st.subheader("Services")
            phone = st.selectbox("Phone Service", ["Yes", "No"])
            multi_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
            security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
            protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
            tech = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
            tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

        with col3:
            st.subheader("Billing")
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)"
            ])
            monthly = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0, step=5.0)
            total = st.number_input("Total Charges ($)", 0.0, 10000.0, monthly * tenure, step=50.0)

        submitted = st.form_submit_button("🔮 Predict Churn Risk", type="primary")

    if submitted:
        predictor = load_predictor()
        if predictor is None:
            st.error("Models not loaded. Please train models first.")
        else:
            customer_data = {
                'Gender': gender, 'Senior Citizen': senior,
                'Partner': partner, 'Dependents': dependents,
                'Tenure Months': tenure, 'Phone Service': phone,
                'Multiple Lines': multi_lines, 'Internet Service': internet,
                'Online Security': security, 'Online Backup': backup,
                'Device Protection': protection, 'Tech Support': tech,
                'Streaming TV': tv, 'Streaming Movies': movies,
                'Contract': contract, 'Paperless Billing': paperless,
                'Payment Method': payment, 'Monthly Charges': monthly,
                'Total Charges': str(total),
            }

            with st.spinner("Analyzing..."):
                try:
                    result = predictor.predict_single(customer_data)

                    # Results display
                    st.markdown("---")
                    st.subheader("Prediction Results")

                    # Key metrics row
                    mc1, mc2, mc3, mc4 = st.columns(4)

                    risk_color = '#ff4444' if result['churn_risk'] == 'HIGH' else '#ffaa00' if result['churn_risk'] == 'MEDIUM' else '#00cc66'

                    mc1.metric("Churn Probability", f"{result['churn_probability']:.1%}")
                    mc2.metric("Risk Level", result['churn_risk'])
                    mc3.metric("Time to Churn", f"{result['predicted_time_to_churn_months']:.0f} months")
                    mc4.metric("CLV Estimate", f"${result['clv_estimate']:,.0f}")

                    col_left, col_right = st.columns(2)

                    with col_left:
                        st.subheader("⚠️ Top Churn Risk Factors")
                        for i, reason in enumerate(result['top_reason_codes'][:5]):
                            impact_emoji = '🔴' if reason['impact'] == 'high' else '🟡' if reason['impact'] == 'medium' else '🟢'
                            direction = '↑ Risk' if reason['direction'] == 'increases_churn' else '↓ Risk'
                            st.write(f"{impact_emoji} **{reason['reason']}** ({direction})")

                    with col_right:
                        st.subheader("💡 Recommended Action")
                        st.info(f"**{result['recommended_action']}**")
                        st.write(result['action_description'])
                        st.write(f"**Urgency:** {result['action_urgency'].upper()}")
                        st.write(f"**Priority Score:** {result['priority_score']:.0f}")
                        st.write(f"**Value-at-Risk Tier:** {result['value_risk_tier']}")

                    # Survival probabilities
                    st.subheader("📈 Survival Probabilities")
                    surv_data = pd.DataFrame({
                        'Time Horizon': ['3 months', '6 months', '12 months'],
                        'Survival Probability': [
                            result['survival_prob_3mo'],
                            result['survival_prob_6mo'],
                            result['survival_prob_12mo'],
                        ]
                    })
                    fig = px.bar(
                        surv_data, x='Time Horizon', y='Survival Probability',
                        title='Probability of Remaining a Customer',
                        color='Survival Probability',
                        color_continuous_scale='RdYlGn',
                        range_color=[0, 1],
                    )
                    fig.update_layout(template='plotly_dark', yaxis_range=[0, 1])
                    st.plotly_chart(fig, use_container_width=True)

                    # All actions
                    with st.expander("View all recommended actions"):
                        for action in result['all_recommended_actions']:
                            st.write(f"**{action['action']}** — {action['category']} "
                                    f"(Urgency: {action['urgency']}, ROI: ${action['expected_roi']:.0f})")

                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    st.exception(e)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: Model Insights
# ═══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.header("Model Insights Dashboard")

    # Training Metrics
    metrics = load_metrics()
    if metrics:
        st.subheader("🏆 Model Performance Comparison")

        metrics_rows = []
        for model_name, m in metrics.items():
            metrics_rows.append({
                'Model': model_name.replace('_', ' ').title(),
                'ROC AUC': m.get('roc_auc', 0),
                'F1 Score': m.get('f1', 0),
                'Precision': m.get('precision', 0),
                'Recall': m.get('recall', 0),
                'Accuracy': m.get('accuracy', 0),
                'Cost': m.get('total_cost', 0),
            })

        metrics_df = pd.DataFrame(metrics_rows)
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

        # Bar chart comparison
        fig = go.Figure()
        for _, row in metrics_df.iterrows():
            fig.add_trace(go.Bar(
                name=row['Model'],
                x=['ROC AUC', 'F1 Score', 'Precision', 'Recall'],
                y=[row['ROC AUC'], row['F1 Score'], row['Precision'], row['Recall']],
            ))
        fig.update_layout(
            title='Model Performance Comparison',
            barmode='group',
            template='plotly_dark',
            yaxis_range=[0, 1],
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No training metrics found. Run src/train.py first.")

    # Feature Importance
    importance = load_feature_importance()
    if importance:
        st.subheader("📊 Global Feature Importance (SHAP)")
        imp_df = pd.DataFrame(importance).head(20)
        fig = px.bar(
            imp_df, x='importance', y='feature',
            orientation='h',
            title='Top 20 Features by Mean |SHAP| Value',
            labels={'importance': 'Mean |SHAP| Value', 'feature': 'Feature'},
        )
        fig.update_layout(
            template='plotly_dark',
            yaxis={'categoryorder': 'total ascending'},
            height=600,
        )
        st.plotly_chart(fig, use_container_width=True)

    # SHAP Summary Plots
    st.subheader("🔍 SHAP Analysis Plots")
    shap_bar = MODELS_DIR / 'shap_summary_bar.png'
    shap_beeswarm = MODELS_DIR / 'shap_summary_beeswarm.png'
    survival_plot = MODELS_DIR / 'survival_curves.png'

    col1, col2 = st.columns(2)
    with col1:
        if shap_bar.exists():
            st.image(str(shap_bar), caption="SHAP Feature Importance", use_container_width=True)
        else:
            st.info("SHAP bar plot not generated yet.")
    with col2:
        if shap_beeswarm.exists():
            st.image(str(shap_beeswarm), caption="SHAP Beeswarm Plot", use_container_width=True)
        else:
            st.info("SHAP beeswarm plot not generated yet.")

    if survival_plot.exists():
        st.subheader("📈 Survival Curves")
        st.image(str(survival_plot), caption="Survival Analysis by Segment", use_container_width=True)

    # Retention Playbook
    st.subheader("📋 Retention Action Playbook")
    playbook_path = MODELS_DIR / 'retention_playbook.json'
    if playbook_path.exists():
        with open(playbook_path) as f:
            playbook = json.load(f)

        playbook_rows = []
        for key, action in playbook.items():
            playbook_rows.append({
                'Trigger': key.replace('_', ' ').title(),
                'Action': action['action'],
                'Category': action['category'],
                'Urgency': action['urgency'],
                'Est. Monthly Cost': f"${action['estimated_monthly_cost']}",
                'Success Rate': f"{action['expected_success_rate']:.0%}",
            })
        st.dataframe(pd.DataFrame(playbook_rows), use_container_width=True, hide_index=True)
    else:
        st.info("Retention playbook not generated yet.")


# ──────────────────────────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "Telecom Churn AI v1.0 | ML Pipeline: LightGBM + SHAP + Cox PH Survival + T-Learner Uplift"
    "</div>",
    unsafe_allow_html=True,
)
