import os
import numpy as np
import pandas as pd
import joblib
import yaml
from src.spatial_processing import generate_india_prediction_grid, get_climate_value_at_points
from src.feature_engineering import compute_engineered_features, scale_features

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

OUT_DIR = os.path.join("outputs", "predictions")
os.makedirs(OUT_DIR, exist_ok=True)

SCENARIOS = {
    "Baseline": {"temp_adj": 0.0, "prec_mult": 1.0},
    "Warmer": {"temp_adj": 1.5, "prec_mult": 1.0},
    "Hotter_Drier": {"temp_adj": 3.0, "prec_mult": 0.85},
    "Hotter_Wetter": {"temp_adj": 3.0, "prec_mult": 1.15}
}

def load_baseline_climate_grid(df_grid):
    """
    Extracts baseline bioclimatic values (from year 2019) for all grid coordinates.
    """
    df_out = df_grid.copy()
    points = list(zip(df_out["longitude"], df_out["latitude"]))
    raw_vars = config['features']['raw']
    
    print("Extracting baseline (2019) bioclimatic layers for India prediction grid...")
    for var in raw_vars:
        vals = get_climate_value_at_points(points, 2019, var)
        df_out[var] = vals
        if df_out[var].isnull().any():
            df_out[var] = df_out[var].fillna(df_out[var].median())
            
    return df_out

def run_projections():
    print("==========================================================")
    print("=== STARTING FUTURE HABITAT PROJECTIONS (2020-2050) ===")
    print("==========================================================")
    
    model_path = os.path.join("models", "best_model.pkl")
    scaler_path = os.path.join("models", "scaler.pkl")
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError("Model or scaler not found. Please run training pipeline first.")
        
    model_data = joblib.load(model_path)
    model = model_data["model"]
    feature_cols = model_data["features"]
    scaler = joblib.load(scaler_path)
    
    # Generate prediction grid
    df_grid = generate_india_prediction_grid()
    df_baseline = load_baseline_climate_grid(df_grid)
    
    start_year = config.get('prediction', {}).get('start_year', 2020)
    end_year = config.get('prediction', {}).get('end_year', 2050)
    projection_years = list(range(start_year, end_year + 1, 5))
    
    for sc_name, adj in SCENARIOS.items():
        print(f"\nProcessing scenario: {sc_name}")
        temp_adj = adj["temp_adj"]
        prec_mult = adj["prec_mult"]
        
        for year in projection_years:
            # Linear progression from 2020 to 2050
            fraction = (year - 2020) / (2050 - 2020) if (2050 - 2020) > 0 else 0.0
            curr_temp_adj = temp_adj * fraction
            curr_prec_mult = 1.0 + (prec_mult - 1.0) * fraction
            
            df_sc = df_baseline.copy()
            if "Bio_Mean_Temp" in df_sc.columns:
                df_sc["Bio_Mean_Temp"] = df_sc["Bio_Mean_Temp"] + curr_temp_adj
            if "TMAX" in df_sc.columns:
                df_sc["TMAX"] = df_sc["TMAX"] + curr_temp_adj
            if "TMIN" in df_sc.columns:
                df_sc["TMIN"] = df_sc["TMIN"] + curr_temp_adj
            if "Precipitation" in df_sc.columns:
                df_sc["Precipitation"] = np.maximum(df_sc["Precipitation"] * curr_prec_mult, 0.0)
                
            # Compute engineered features
            df_sc = compute_engineered_features(df_sc)
            df_sc_scaled = scale_features(df_sc, scaler)
            
            # Predict probabilities
            X_sc = df_sc_scaled[feature_cols]
            probs = model.predict_proba(X_sc)[:, 1]
            
            df_pred = df_grid.copy()
            df_pred["suitability"] = probs
            
            out_file = os.path.join(OUT_DIR, f"prediction_{sc_name}_{year}.csv")
            df_pred.to_csv(out_file, index=False)
            
    print(f"\nAll projection rasters (2020-2050, 4 scenarios) successfully saved to `{OUT_DIR}`")

if __name__ == "__main__":
    run_projections()
