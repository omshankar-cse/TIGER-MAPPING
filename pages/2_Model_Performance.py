import streamlit as st
import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import yaml

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Page settings
st.set_page_config(page_title="Model Performance Dashboard", page_icon="📈", layout="wide")

# Custom styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    .main-title {
        font-size: 32px;
        font-weight: 800;
        background: linear-gradient(90deg, #ff8c00 0%, #ff4500 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .kpi-card {
        background: rgba(17, 24, 39, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    .kpi-title {
        font-size: 13px;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 26px;
        font-weight: 700;
        color: #ff8c00;
    }
    .kpi-sub {
        font-size: 11px;
        color: #6b7280;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)

def load_overall_metrics():
    path = os.path.join("outputs", "metrics", "overall_test_metrics.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None

def load_year_wise_metrics():
    path = os.path.join("outputs", "metrics", "year_wise_metrics.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

def load_feature_importances():
    path = os.path.join("outputs", "metrics", "feature_importances.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

def load_cv_model_comparison():
    path = os.path.join("outputs", "metrics", "cv_model_comparison.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

def load_metadata():
    path = os.path.join("models", "model_metadata.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None

def main():
    st.markdown("<h1 class='main-title'>📈 Model Performance Dashboard</h1>", unsafe_allow_html=True)
    st.write("Rigorous temporal evaluation on unseen test years (2016–2020). Predictions are compared against actual tiger occurrence records.")
    st.write("---")
    
    # Load files
    metrics = load_overall_metrics()
    df_yw = load_year_wise_metrics()
    df_imp = load_feature_importances()
    df_compare = load_cv_model_comparison()
    metadata = load_metadata()
    
    if metrics is None or df_yw is None:
        st.error("⚠️ Model evaluation metrics not found. Please run the training and evaluation pipelines first: `python -m src.train` and `python -m src.evaluate`")
        return
        
    threshold_val = metadata.get("threshold", 0.5) if metadata else 0.5
    model_name_val = metadata.get("model_type", "Random Forest") if metadata else "Random Forest"
    
    # Top KPI cards
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Test Accuracy</div>
            <div class="kpi-value">{metrics['accuracy'] * 100:.1f}%</div>
            <div class="kpi-sub">Overall Correct Predictions</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Test ROC-AUC</div>
            <div class="kpi-value">{metrics['roc_auc']:.3f}</div>
            <div class="kpi-sub">Probability Discrimination Strength</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Test F1-Score</div>
            <div class="kpi-value">{metrics['f1'] * 100:.1f}%</div>
            <div class="kpi-sub">Harmonic Mean of Precision & Recall</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Calibrated Threshold</div>
            <div class="kpi-value">{threshold_val:.2f}</div>
            <div class="kpi-sub">Optimized Youden's J Threshold</div>
        </div>
        """, unsafe_allow_html=True)
    with col5:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Selected Model</div>
            <div class="kpi-value" style="font-size: 16px; padding: 5px 0;">{model_name_val}</div>
            <div class="kpi-sub">Best Spatial CV Performer</div>
        </div>
        """, unsafe_allow_html=True)

    st.write(" ")
    st.write(" ")
    
    # Model comparison table
    if df_compare is not None:
        st.subheader("Model Comparison (Training Cross-Validation Performance)")
        st.markdown("Metrics evaluated using 5-fold Spatial Block Cross-Validation on the training period (2001–2015):")
        # Format table for display
        df_comp_disp = df_compare.copy()
        for col in ["CV ROC-AUC", "CV F1", "CV Accuracy", "CV Brier Score"]:
            if col in df_comp_disp.columns:
                df_comp_disp[col] = df_comp_disp[col].map(lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A")
        st.dataframe(df_comp_disp, use_container_width=True, hide_index=True)
        st.write("---")

    # Metrics breakdown columns
    row1_col1, row1_col2 = st.columns([1, 1])
    
    with row1_col1:
        st.subheader("Model Discrimination & Calibration")
        st.markdown(f"""
        Below are the detailed test set metrics calculated across all combined test years:
        - **Balanced Accuracy**: `{metrics.get('balanced_accuracy', metrics['accuracy']):.4f}` (average of sensitivity and specificity)
        - **Precision (Positive Predictive Value)**: `{metrics['precision']:.4f}` (percentage of predicted suitable cells that actually had tiger occurrences)
        - **Recall (Sensitivity / True Positive Rate)**: `{metrics['recall']:.4f}` (percentage of actual occurrences correctly flagged as suitable)
        - **Specificity (True Negative Rate)**: `{metrics['specificity']:.4f}` (percentage of pseudo-absences correctly classified as unsuitable)
        - **PR-AUC (Precision-Recall Area Under Curve)**: `{metrics['pr_auc']:.4f}` (better indicator for imbalanced classification)
        - **Brier Score**: `{metrics['brier_score']:.4f}` (mean squared error of predicted suitability probability relative to binary outcome; lower is better)
        """)
        
    with row1_col2:
        st.subheader("Confusion Matrix")
        cm = metrics["confusion_matrix"]
        
        # Display confusion matrix as a heat map
        z = [[cm["tn"], cm["fp"]],
             [cm["fn"], cm["tp"]]]
        x = ["Predicted Absence (0)", "Predicted Presence (1)"]
        y = ["Actual Absence (0)", "Actual Presence (1)"]
        
        fig_cm = ff_create_annotated_heatmap(z, x=x, y=y, colorscale="YlOrRd")
        st.plotly_chart(fig_cm, use_container_width=True)
        
    st.write("---")
    
    # Year-wise metrics
    st.subheader("Year-by-Year Temporal Generalization (2016–2020)")
    col_table, col_chart = st.columns([2, 3])
    
    with col_table:
        st.markdown("Metrics evaluated independently for each unseen year:")
        # Format columns for display
        df_yw_disp = df_yw.copy()
        for col in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]:
            if col in df_yw_disp.columns:
                df_yw_disp[col] = df_yw_disp[col].map(lambda x: f"{x * 100:.1f}%" if not pd.isna(x) else "N/A")
        st.dataframe(df_yw_disp, use_container_width=True, hide_index=True)
        
    with col_chart:
        # Plot Plotly line chart for metrics over time
        df_yw_melt = df_yw.melt(id_vars=["Year"], value_vars=["Accuracy", "F1", "ROC-AUC"], var_name="Metric", value_name="Score")
        fig_line = px.line(
            df_yw_melt,
            x="Year",
            y="Score",
            color="Metric",
            markers=True,
            title="Performance Trends Over Test Years",
            color_discrete_sequence=["#ff8c00", "#ff4500", "#10b981"],
            height=300
        )
        fig_line.update_layout(yaxis_range=[0.4, 1.0], margin=dict(t=30, b=10, l=10, r=10))
        st.plotly_chart(fig_line, use_container_width=True)
        
    st.write("---")
    
    # Feature Importances & Explanation
    st.subheader("Model Interpretability & Feature Importances")
    col_feat_chart, col_feat_text = st.columns([3, 2])
    
    with col_feat_chart:
        if df_imp is not None:
            fig_bar = px.bar(
                df_imp,
                x="Importance",
                y="Feature",
                orientation="h",
                title="Climate Feature Importance Rankings",
                color="Importance",
                color_continuous_scale="Oranges",
                height=300
            )
            fig_bar.update_layout(yaxis=dict(autorange="reversed"), margin=dict(t=30, b=10, l=10, r=10))
            st.plotly_chart(fig_bar, use_container_width=True)
            
    with col_feat_text:
        st.write(" ")
        st.write(" ")
        st.markdown("""
        **🔍 Key Ecological Insights:**
        - **Precipitation / Log Precipitation**: These variables dominate model decisions, representing the critical importance of moisture regimes (forest canopy density, surface water availability) in tiger habitats.
        - **Temperature Mean / Limits**: Temperature limits (TMAX/TMIN) set biological constraints on vegetation growth (forest biomes) and prey distribution, which indirectly defines tiger habitat boundaries.
        """)

# Custom function to create annotated heatmaps without requiring extra scipy/plotly-figure-factory imports
def ff_create_annotated_heatmap(z, x, y, colorscale):
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=x,
        y=y,
        colorscale=colorscale,
        showscale=False
    ))
    
    # Add annotations
    for i, row in enumerate(z):
        for j, val in enumerate(row):
            fig.add_annotation(
                x=x[j],
                y=y[i],
                text=str(val),
                showarrow=False,
                font=dict(color="black" if val > np.mean(z) else "white", size=16)
            )
            
    fig.update_layout(
        xaxis=dict(side="bottom"),
        yaxis=dict(autorange="reversed"),
        margin=dict(t=20, b=20, l=20, r=20),
        height=280
    )
    return fig

if __name__ == "__main__":
    main()
