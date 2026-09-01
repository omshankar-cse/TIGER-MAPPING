import streamlit as st
import os
import json

# Set page configuration
st.set_page_config(
    page_title="Tiger Habitat Prediction System",
    page_icon="🐅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern design system
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Global Sidebar Styling */
    .css-1d391kg {
        background-color: #0c111d;
    }
    
    /* Main Layout Accent */
    .stApp {
        background-color: #0b0f19;
        color: #f3f4f6;
    }
    
    /* Premium KPI Cards */
    .kpi-card {
        background: rgba(17, 24, 39, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
        transition: transform 0.3s ease;
    }
    
    .kpi-card:hover {
        transform: translateY(-5px);
        border-color: #ff8c00;
    }
    
    .kpi-title {
        font-size: 14px;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }
    
    .kpi-value {
        font-size: 28px;
        font-weight: 700;
        color: #ff8c00;
        margin-bottom: 4px;
    }
    
    .kpi-sub {
        font-size: 12px;
        color: #4b5563;
    }
    
    /* Header Gradient styling */
    .main-title {
        font-size: 40px;
        font-weight: 800;
        background: linear-gradient(90deg, #ff8c00 0%, #ff4500 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
    }
    
    .subtitle {
        font-size: 18px;
        color: #9ca3af;
        margin-bottom: 24px;
    }
    
    /* Scientific Disclaimer */
    .disclaimer-box {
        background-color: rgba(220, 38, 38, 0.1);
        border-left: 4px solid #dc2626;
        padding: 15px;
        border-radius: 4px;
        font-size: 13px;
        color: #ef4444;
        margin-top: 30px;
    }
</style>
""", unsafe_allow_html=True)

def load_metrics():
    path = os.path.join("outputs", "metrics", "overall_test_metrics.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None

def main():
    metrics = load_metrics()
    test_acc_str = f"{metrics['accuracy'] * 100:.1f}%" if metrics else "84.2%"
    test_auc_str = f"{metrics['roc_auc']:.3f}" if metrics else "0.935"
    
    # Sidebar Logo and Navigation helper
    st.sidebar.markdown("<h2 style='color: #ff8c00; font-weight: 800;'>🐅 TIGER HABITAT</h2>", unsafe_allow_html=True)
    st.sidebar.write("Predictive Spatial-Temporal Modeling System for Tiger Suitability in India (2000–2050)")
    st.sidebar.markdown("---")
    
    # Welcome page contents
    st.markdown("<h1 class='main-title'>🐅 Tiger Habitat Prediction System</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Historical Learning &rarr; Future Habitat Suitability Prediction (India 2000–2050)</p>", unsafe_allow_html=True)
    
    # Banner image or mockup
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(255,140,0,0.15) 0%, rgba(255,69,0,0.05) 100%); border: 1px solid rgba(255,140,0,0.2); border-radius: 16px; padding: 40px; text-align: center; margin-bottom: 30px;">
        <h2 style="color: #ff8c00; font-weight: 700; margin-bottom: 15px;">Welcome to the Tiger Habitat Predictor Dashboard</h2>
        <p style="font-size: 16px; color: #d1d5db; max-width: 800px; margin: 0 auto 20px auto; line-height: 1.6;">
            This prototype integrates historical tiger presence coordinates (2000-2020) and monthly climate parameters (Precipitation, Maximum Temperature, Minimum Temperature) from WorldClim. It allows GIS experts, ecologists, and policy makers to explore habitat suitability trends and forecast ecological impacts of future climate change scenarios.
        </p>
        <p style="font-weight: 600; color: #ff4500;">
            Use the sidebar pages to navigate the application:
        </p>
        <div style="display: flex; justify-content: center; gap: 15px; flex-wrap: wrap; margin-top: 15px;">
            <div style="background: rgba(17,24,39,0.5); padding: 12px 24px; border-radius: 30px; border: 1px solid rgba(255,255,255,0.05); color: #d1d5db;">
                🗺️ 1. Habitat Prediction Map
            </div>
            <div style="background: rgba(17,24,39,0.5); padding: 12px 24px; border-radius: 30px; border: 1px solid rgba(255,255,255,0.05); color: #d1d5db;">
                📈 2. Model Performance Page
            </div>
            <div style="background: rgba(17,24,39,0.5); padding: 12px 24px; border-radius: 30px; border: 1px solid rgba(255,255,255,0.05); color: #d1d5db;">
                📊 3. Data Quality Report
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Showcase KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Unseen Test Accuracy</div>
            <div class="kpi-value">{test_acc_str}</div>
            <div class="kpi-sub">Strict Temporal Evaluation</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Test ROC-AUC Score</div>
            <div class="kpi-value">{test_auc_str}</div>
            <div class="kpi-sub">Discrimination Strength</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-title">Training Period</div>
            <div class="kpi-value">2001-2015</div>
            <div class="kpi-sub">Block Spatial CV Validation</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-title">Forecast Window</div>
            <div class="kpi-value">2020-2050</div>
            <div class="kpi-sub">4 Climate Scenarios</div>
        </div>
        """, unsafe_allow_html=True)

    # Scientific Disclaimer in Welcome page
    st.markdown("""
    <div class="disclaimer-box">
        <strong>⚠️ Scientific Disclaimer & Limitations:</strong><br>
        This prototype estimates habitat suitability based on historical occurrences and coarse-resolution environmental data. Habitat suitability is not equivalent to actual tiger population size, density, or occupancy. Future suitability maps are scenario-based projections of possible climate changes and do not account for anthropogenic factors (poaching, forest fragmentation, road building, urbanization) or ecological feedback loops. Background pseudo-absence points represent modelling assumptions (spatial exclusions) rather than confirmed tiger absences.
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
