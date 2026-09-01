import os
import pandas as pd
import numpy as np
import yaml
from src.data_loader import load_and_sample_presence_data

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

PROCESSED_DIR = os.path.join("data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

def thin_spatial_coordinates(df, distance_km=3):
    """
    Applies spatial thinning per year so that no two points in the same year
    are closer than distance_km (converted to decimal degrees ~ 1 deg = 111 km).
    """
    if not config.get('spatial_thinning', {}).get('enabled', False):
        return df
        
    distance_deg = distance_km / 111.0
    thinned_dfs = []
    
    for year, df_year in df.groupby("year"):
        df_year = df_year.reset_index(drop=True)
        coords = df_year[["longitude", "latitude"]].values
        n = len(coords)
        if n == 0:
            continue
            
        kept_indices = []
        for i in range(n):
            lon, lat = coords[i]
            is_too_close = False
            for k_idx in kept_indices:
                k_lon, k_lat = coords[k_idx]
                dist = ((lon - k_lon)**2 + (lat - k_lat)**2)**0.5
                if dist < distance_deg:
                    is_too_close = True
                    break
            if not is_too_close:
                kept_indices.append(i)
                
        thinned_dfs.append(df_year.iloc[kept_indices])
        
    return pd.concat(thinned_dfs, ignore_index=True)

def clean_tiger_occurrence_data():
    """
    Extracts stratified presence coordinates across all years, cleans,
    validates boundaries, removes duplicates, and applies spatial thinning.
    """
    print("--- Extracting and Cleaning Stratified Presence Data ---")
    df_raw = load_and_sample_presence_data()
    
    # 1. Missing coordinate removal
    df_clean = df_raw.dropna(subset=["latitude", "longitude"]).copy()
    
    # 2. Coordinate range validation (India Bounds)
    valid_mask = (
        (df_clean["longitude"] >= 68.0) & 
        (df_clean["longitude"] <= 98.0) & 
        (df_clean["latitude"] >= 6.0) & 
        (df_clean["latitude"] <= 38.0)
    )
    df_clean = df_clean[valid_mask]
    print(f"Within India bounds [68-98°E, 6-38°N]: {len(df_clean)} records.")
    
    # 3. Duplicate removal (rounded to 4 decimal places ~ 11 meters)
    df_clean["lat_round"] = df_clean["latitude"].round(4)
    df_clean["lon_round"] = df_clean["longitude"].round(4)
    df_clean = df_clean.drop_duplicates(subset=["year", "lat_round", "lon_round"])
    df_clean = df_clean.drop(columns=["lat_round", "lon_round"])
    print(f"After duplicate deduplication: {len(df_clean)} records.")
    
    # 4. Spatial thinning
    thin_dist = config.get('spatial_thinning', {}).get('distance_km', 3)
    print(f"Applying spatial thinning ({thin_dist} km)...")
    df_thinned = thin_spatial_coordinates(df_clean, thin_dist)
    print(f"After spatial thinning: {len(df_thinned)} presence records.")
    
    out_csv = os.path.join(PROCESSED_DIR, "tiger_occurrences_clean.csv")
    df_thinned.to_csv(out_csv, index=False)
    print(f"Cleaned occurrences saved to {out_csv}")
    
    return df_thinned

if __name__ == "__main__":
    df = clean_tiger_occurrence_data()
    print("Cleaned occurrences summary:\n", df.groupby("year")["presence"].count())
