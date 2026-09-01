import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import yaml
from src.spatial_processing import get_climate_value_at_points

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

def extract_climate_features_for_dataset(df):
    """
    Extracts raw bioclimatic features for each coordinate point (longitude, latitude)
    corresponding strictly to its occurrence year.
    """
    df_out = df.copy()
    raw_vars = config.get('features', {}).get('raw', [
        "Precipitation", "TMAX", "TMIN", "Bio_Mean_Temp",
        "Bio_Temp_Range", "Bio_Temp_Seasonality", "Bio_Precip_Seasonality"
    ])
    
    for var in raw_vars:
        if var not in df_out.columns:
            df_out[var] = np.nan
            
    years = sorted(df_out["year"].unique())
    print(f"\n--- Extracting Bioclimatic Rasters for {len(years)} years ---")
    
    for year in years:
        mask = df_out["year"] == year
        df_year = df_out[mask]
        points = list(zip(df_year["longitude"], df_year["latitude"]))
        
        for var in raw_vars:
            vals = get_climate_value_at_points(points, year, var)
            df_out.loc[mask, var] = vals
            
        print(f"Year {year}: extracted {len(raw_vars)} bioclimatic variables for {len(df_year)} points.")
        
    return df_out

def compute_engineered_features(df):
    """
    Computes derived non-linear and interaction ecological features:
    - Bio_Precip_Log = log1p(Precipitation)
    - Temp_Precip_Interaction = Bio_Mean_Temp * Bio_Precip_Log
    - Bio_Temp_Sq = Bio_Mean_Temp ** 2
    """
    df_out = df.copy()
    
    # 1. Log precipitation
    if "Precipitation" in df_out.columns:
        df_out["Bio_Precip_Log"] = np.log1p(np.maximum(df_out["Precipitation"], 0.0))
        
    # 2. Temp-Precip Interaction
    if "Bio_Mean_Temp" in df_out.columns and "Bio_Precip_Log" in df_out.columns:
        df_out["Temp_Precip_Interaction"] = df_out["Bio_Mean_Temp"] * df_out["Bio_Precip_Log"]
        
    # 3. Quadratic Temperature term (ecological niche optimum curve)
    if "Bio_Mean_Temp" in df_out.columns:
        df_out["Bio_Temp_Sq"] = df_out["Bio_Mean_Temp"] ** 2
        
    # Median imputation for any residual NaNs
    feature_cols = config['features']['raw'] + config['features']['engineered']
    for col in feature_cols:
        if col in df_out.columns and df_out[col].isnull().any():
            med = df_out[col].median()
            if np.isnan(med):
                med = 0.0
            df_out[col] = df_out[col].fillna(med)
            
    return df_out

def compute_vif_report(df, feature_cols):
    """
    Calculates Variance Inflation Factor (VIF) for feature multicollinearity diagnostic.
    """
    from sklearn.linear_model import LinearRegression
    
    vif_data = []
    df_clean = df[feature_cols].dropna()
    
    for i, col in enumerate(feature_cols):
        X_other = df_clean.drop(columns=[col])
        y_target = df_clean[col]
        
        reg = LinearRegression().fit(X_other, y_target)
        r2 = reg.score(X_other, y_target)
        vif = 1.0 / (1.0 - r2) if r2 < 1.0 else np.inf
        vif_data.append({"Feature": col, "VIF": float(vif), "R2": float(r2)})
        
    df_vif = pd.DataFrame(vif_data).sort_values(by="VIF", ascending=False)
    return df_vif

def fit_and_save_scaler(df_train, out_dir="models"):
    """
    Fits a StandardScaler strictly on the training set features.
    """
    os.makedirs(out_dir, exist_ok=True)
    scaler_path = os.path.join(out_dir, "scaler.pkl")
    
    feature_cols = config['features']['raw'] + config['features']['engineered']
    scaler = StandardScaler()
    scaler.fit(df_train[feature_cols])
    
    joblib.dump(scaler, scaler_path)
    print(f"Fitted StandardScaler saved to {scaler_path}")
    return scaler

def scale_features(df, scaler):
    """
    Scales the features using the fitted scaler.
    """
    df_out = df.copy()
    feature_cols = config['features']['raw'] + config['features']['engineered']
    
    scaled = scaler.transform(df_out[feature_cols])
    for i, col in enumerate(feature_cols):
        df_out[col] = scaled[:, i]
        
    return df_out
