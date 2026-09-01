import streamlit as st
import os
import geopandas as gpd
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import yaml
import json
from shapely.geometry import Point

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Page settings
st.set_page_config(page_title="Habitat Suitability Map", page_icon="🗺️", layout="wide")

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
    .kpi-box {
        background: rgba(17, 24, 39, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        margin-bottom: 10px;
    }
    .kpi-val {
        font-size: 24px;
        font-weight: 700;
        color: #ff8c00;
    }
</style>
""", unsafe_allow_html=True)

# Compatibility wrappers for Plotly (supports Plotly v5, v6, and v7)
def create_scatter_map(df, **kwargs):
    if hasattr(px, "scatter_map"):
        if "mapbox_style" in kwargs:
            kwargs["map_style"] = kwargs.pop("mapbox_style")
        return px.scatter_map(df, **kwargs)
    else:
        return px.scatter_mapbox(df, **kwargs)

def create_map_trace(**kwargs):
    if hasattr(go, "Scattermap"):
        return go.Scattermap(**kwargs)
    else:
        return go.Scattermapbox(**kwargs)

# Helper cached functions to avoid slow reloading
@st.cache_resource
def load_ml_components():
    model_path = os.path.join("models", "best_model.pkl")
    if not os.path.exists(model_path):
        model_path = os.path.join("models", "trained_model.pkl")
    scaler_path = os.path.join("models", "scaler.pkl")
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model_data = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model_data["model"], scaler, model_data["features"]
    return None, None, None

@st.cache_data
def load_clean_occurrences():
    csv_path = os.path.join("data", "processed", "tiger_occurrences_clean.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return pd.DataFrame()

@st.cache_data
def generate_historical_predictions(year):
    """
    Dynamically generates suitability prediction for a historical year
    using the trained model and aggregates.
    """
    model, scaler, feature_cols = load_ml_components()
    if model is None:
        return pd.DataFrame()
        
    from src.spatial_processing import generate_india_prediction_grid, get_climate_value_at_points
    from src.feature_engineering import compute_engineered_features, scale_features
    
    # 2020 uses 2019 baseline climate
    target_climate_year = 2019 if year >= 2020 else year
    
    # Create grid
    df_grid = generate_india_prediction_grid()
    points = list(zip(df_grid["longitude"], df_grid["latitude"]))
    
    # Extract values
    df_sc = df_grid.copy()
    for var in config['features']['raw']:
        vals = get_climate_value_at_points(points, target_climate_year, var)
        df_sc[var] = vals
        
    # Impute NaNs
    for col in config['features']['raw']:
        if df_sc[col].isnull().any():
            df_sc[col] = df_sc[col].fillna(df_sc[col].median())
            
    # Features & scaling
    df_sc = compute_engineered_features(df_sc)
    df_sc_scaled = scale_features(df_sc, scaler)
    
    # Predict
    X_sc = df_sc_scaled[feature_cols]
    probs = model.predict_proba(X_sc)[:, 1]
    
    df_grid["suitability"] = probs
    return df_grid

@st.cache_data
def load_baseline_2020():
    """Loads pre-generated 2020 Baseline projection to calculate changes."""
    path = os.path.join("outputs", "predictions", "prediction_Baseline_2020.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return generate_historical_predictions(2020)


def main():
    st.markdown("<h1 class='main-title'>🗺️ Interactive Tiger Habitat Suitability Map</h1>", unsafe_allow_html=True)
    st.write("Explore historical occurrences alongside model projections up to 2050 under alternative climate scenarios.")
    st.write("---")
    
    # Check if models exist
    model, scaler, feature_cols = load_ml_components()
    if model is None:
        st.error("⚠️ Trained model or standard scaler not found in `models/` directory. Please run the training pipeline first: `python -m src.train`")
        return
        
    # Load occurrences
    df_occ = load_clean_occurrences()
    
    # Sidebar Controls
    st.sidebar.markdown("<h3 style='color: #ff8c00;'>Controls</h3>", unsafe_allow_html=True)
    
    # Slider
    year = st.sidebar.slider(
        "Prediction Year",
        min_value=2001,
        max_value=2050,
        value=2020,
        step=1
    )
    
    # Scenario
    is_future = year > 2020
    if is_future:
        scenario = st.sidebar.selectbox(
            "Climate Scenario",
            options=["Baseline", "Warmer", "Hotter_Drier", "Hotter_Wetter"],
            format_func=lambda x: x.replace("_", " + ")
        )
    else:
        scenario = "Baseline"
        st.sidebar.info("💡 Selected year is historical. Displaying historical learning mode.")
        
    # Load threshold from metadata if available
    meta_path = os.path.join("models", "model_metadata.json")
    default_thr = 0.35
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
                default_thr = float(meta.get("threshold", 0.35))
        except Exception:
            pass

    # Thresholds
    st.sidebar.markdown("---")
    st.sidebar.markdown("<h4 style='color: #ff8c00;'>Habitat Thresholds</h4>", unsafe_allow_html=True)
    low_thr = st.sidebar.slider("Low Suitability Max", 0.0, 1.0, 0.3, 0.05)
    high_thr = st.sidebar.slider("High Suitability Min", 0.0, 1.0, default_thr, 0.05)
    
    # Map Type selection
    st.sidebar.markdown("---")
    st.sidebar.markdown("<h4 style='color: #ff8c00;'>Map Visualization</h4>", unsafe_allow_html=True)
    map_type = st.sidebar.radio(
        "Select Map Type:",
        options=["Habitat Suitability Map", "Model Error / Validation Map"],
        index=0
    )
    
    # Load Predictions
    if is_future:
        pred_year = int(round(year / 5) * 5)
        pred_file = os.path.join("outputs", "predictions", f"prediction_{scenario}_{pred_year}.csv")
        if os.path.exists(pred_file):
            df_pred = pd.read_csv(pred_file)
            map_title = f"Future Projection ({scenario.replace('_', ' + ')} scenario, Year {pred_year})"
        else:
            st.warning(f"Pre-generated predictions not found for {scenario} {pred_year}. Simulating dynamically...")
            df_pred = generate_historical_predictions(year)
            map_title = f"Future Simulation ({scenario.replace('_', ' + ')} scenario, Year {year})"
    else:
        df_pred = generate_historical_predictions(year)
        map_title = f"Historical Suitability (Year {year})"

    # Calculate metrics
    if not df_pred.empty:
        mean_suit = df_pred["suitability"].mean()
        high_suit_mask = df_pred["suitability"] >= high_thr
        med_suit_mask = (df_pred["suitability"] >= low_thr) & (df_pred["suitability"] < high_thr)
        low_suit_mask = df_pred["suitability"] < low_thr
        
        pct_high = (high_suit_mask.sum() / len(df_pred)) * 100
        pct_med = (med_suit_mask.sum() / len(df_pred)) * 100
        pct_low = (low_suit_mask.sum() / len(df_pred)) * 100
        
        # Change from baseline (2020 Baseline)
        df_base = load_baseline_2020()
        if not df_base.empty:
            base_high_pct = (df_base["suitability"] >= high_thr).sum() / len(df_base) * 100
            diff_high = pct_high - base_high_pct
            diff_mean = mean_suit - df_base["suitability"].mean()
        else:
            diff_high = 0.0
            diff_mean = 0.0
    else:
        mean_suit = 0.0
        pct_high, pct_med, pct_low = 0.0, 0.0, 0.0
        diff_high, diff_mean = 0.0, 0.0

    # Layout: Metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="kpi-box">
            <div style="font-size:12px; color:#9ca3af;">MEAN HABITAT SUITABILITY</div>
            <div class="kpi-val">{mean_suit:.3f}</div>
            <div style="font-size:11px; color:{'#10b981' if diff_mean >= 0 else '#ef4444'}">{diff_mean:+.3f} from 2020</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="kpi-box">
            <div style="font-size:12px; color:#9ca3af;">HIGH SUITABILITY AREA</div>
            <div class="kpi-val">{pct_high:.1f}%</div>
            <div style="font-size:11px; color:{'#10b981' if diff_high >= 0 else '#ef4444'}">{diff_high:+.1f}% from 2020</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="kpi-box">
            <div style="font-size:12px; color:#9ca3af;">MODERATE SUITABILITY</div>
            <div class="kpi-val">{pct_med:.1f}%</div>
            <div style="font-size:11px; color:#6b7280;">Prob [{low_thr:.2f} - {high_thr:.2f}]</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="kpi-box">
            <div style="font-size:12px; color:#9ca3af;">LOW SUITABILITY AREA</div>
            <div class="kpi-val">{pct_low:.1f}%</div>
            <div style="font-size:11px; color:#6b7280;">Prob < {low_thr:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    # Plotly Map Section
    if map_type == "Model Error / Validation Map":
        st.subheader(f"Model Error / Validation Map (Year {year})")
        
        if year not in [2016, 2017, 2018, 2019, 2020]:
            st.info("💡 Validation Error/Audit map is only available for the unseen test years (2016–2020).")
            fig = create_scatter_map(
                pd.DataFrame(columns=["latitude", "longitude"]),
                lat="latitude",
                lon="longitude",
                zoom=4.2,
                center={"lat": 22.0, "lon": 80.0},
                map_style="carto-positron",
                height=650
            )
            fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
            st.plotly_chart(fig, use_container_width=True)
        else:
            test_path = os.path.join("data", "processed", "test_set_scaled.csv")
            if not os.path.exists(test_path):
                st.error("Test set scaled file not found. Please run predictions/evaluation first.")
            else:
                df_ts = pd.read_csv(test_path)
                df_ts_yr = df_ts[df_ts["year"] == year].copy()
                
                if df_ts_yr.empty:
                    st.info(f"No validation records available for the year {year}.")
                else:
                    probs_ts = model.predict_proba(df_ts_yr[feature_cols])[:, 1]
                    preds_ts = (probs_ts >= high_thr).astype(int)
                    
                    category_list = []
                    for act, pred in zip(df_ts_yr["presence"], preds_ts):
                        if act == 1 and pred == 1:
                            category_list.append("True Positive (Correct Presence)")
                        elif act == 0 and pred == 0:
                            category_list.append("True Negative (Correct Absence)")
                        elif act == 0 and pred == 1:
                            category_list.append("False Positive (False Alarm)")
                        else:
                            category_list.append("False Negative (Missed Occurrence)")
                            
                    df_ts_yr["Category"] = category_list
                    
                    fig = create_scatter_map(
                        df_ts_yr,
                        lat="latitude",
                        lon="longitude",
                        color="Category",
                        color_discrete_map={
                            "True Positive (Correct Presence)": "#10b981",
                            "True Negative (Correct Absence)": "#3b82f6",
                            "False Positive (False Alarm)": "#eab308",
                            "False Negative (Missed Occurrence)": "#ef4444"
                        },
                        zoom=4.2,
                        center={"lat": 22.0, "lon": 80.0},
                        map_style="carto-positron",
                        height=650
                    )
                    fig.update_traces(
                        marker=dict(size=10, opacity=0.9),
                        hoverinfo="text",
                        text=[f"<b>Validation Audit</b><br><br>Year: {r['year']}<br>Actual: {'Presence' if r['presence'] == 1 else 'Absence'}<br>Predicted Prob: {prob:.3f}<br>Category: {r['Category']}" for r, prob in zip(df_ts_yr.to_dict('records'), probs_ts)]
                    )
                    fig.update_layout(
                        margin={"r":0,"t":0,"l":0,"b":0},
                        legend=dict(
                            yanchor="top",
                            y=0.99,
                            xanchor="left",
                            x=0.01,
                            bgcolor="rgba(17,24,39,0.8)",
                            font=dict(color="#f3f4f6")
                        )
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.write("---")
                    st.markdown("""
                    **Legend Interpretation:**
                    - 🟢 **True Positive (Correct Presence)**: Tiger presence correctly predicted by the model.
                    - 🔵 **True Negative (Correct Absence)**: Pseudo-absence correctly predicted as unsuitable.
                    - 🟡 **False Positive (False Alarm)**: Pseudo-absence predicted as suitable.
                    - 🔴 **False Negative (Missed Occurrence)**: Tiger presence missed by the model.
                    """)
                    
    else:  # Habitat Suitability Map
        st.subheader(map_title)
        
        if df_pred.empty:
            st.warning("No prediction grid available for the selected settings.")
        else:
            fig = create_scatter_map(
                df_pred,
                lat="latitude",
                lon="longitude",
                color="suitability",
                color_continuous_scale="YlOrRd",
                range_color=[0.0, 1.0],
                opacity=0.6,
                zoom=4.2,
                center={"lat": 22.0, "lon": 80.0},
                map_style="carto-positron",
                labels={"suitability": "Habitat Suitability"},
                height=650
            )
            
            if not df_occ.empty:
                df_occ_yr = df_occ[df_occ["year"] == year]
                if not df_occ_yr.empty:
                    fig.add_trace(create_map_trace(
                        lat=df_occ_yr["latitude"],
                        lon=df_occ_yr["longitude"],
                        mode="markers",
                        marker=dict(
                            size=8,
                            color="#ff5500",
                            opacity=0.9
                        ),
                        name=f"Actual Occurrences ({year})",
                        hoverinfo="text",
                        text=[f"Observed Tiger Presence ({year})<br>Lat: {r['latitude']:.4f}, Lon: {r['longitude']:.4f}" for idx, r in df_occ_yr.iterrows()]
                    ))
                else:
                    st.info(f"ℹ️ No actual tiger observations recorded in the raw dataset for the year {year}.")
                    
            fig.update_layout(
                margin={"r":0,"t":0,"l":0,"b":0},
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01,
                    bgcolor="rgba(17,24,39,0.8)",
                    font=dict(color="#f3f4f6")
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)

    # Disclaimer
    st.markdown("""
    <div style="background-color: rgba(220, 38, 38, 0.1); border-left: 4px solid #dc2626; padding: 12px; border-radius: 4px; font-size: 12px; color: #ef4444; margin-top: 20px;">
        <strong>⚠️ Scientific Disclaimer:</strong> Predictions represent statistical habitat suitability derived from historical climate conditions and mapped boundaries. They are not direct estimates of population density or occupancy.
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
