import os
import random
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, MultiPolygon, Polygon
from shapely.ops import unary_union
from shapely.prepared import prep
import yaml
from src.spatial_processing import get_india_boundary

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

def generate_pseudo_absences(occurrences_df):
    """
    Generates background (pseudo-absence) points across India, strictly outside
    a consolidated 20 km exclusion buffer around ALL historical presences (2000-2020).
    """
    india_boundary = get_india_boundary()
    ratio = config.get('pseudo_absence', {}).get('ratio', 2)
    min_dist_km = config.get('pseudo_absence', {}).get('minimum_distance_km', 20)
    random_state = config.get('model', {}).get('random_state', 42)
    
    rng = np.random.RandomState(random_state)
    
    print(f"\n--- Generating Pseudo-Absences (Ratio 1:{ratio}, Buffer: {min_dist_km} km) ---")
    
    # 1. Create consolidated buffer around ALL presences
    min_dist_deg = min_dist_km / 111.0
    all_pts = [Point(lon, lat) for lon, lat in zip(occurrences_df["longitude"], occurrences_df["latitude"])]
    presence_union = unary_union(all_pts)
    global_exclusion_buffer = presence_union.buffer(min_dist_deg)
    
    # 2. Subtract exclusion buffer from India boundary
    valid_background = india_boundary.difference(global_exclusion_buffer)
    if valid_background.is_empty:
        raise ValueError("Exclusion buffer completely covers India! Reduce buffer distance.")
        
    prepared_bg = prep(valid_background)
    min_lon, min_lat, max_lon, max_lat = valid_background.bounds
    
    all_records = []
    years = sorted(occurrences_df["year"].unique())
    
    for year in years:
        df_year_p = occurrences_df[occurrences_df["year"] == year].copy()
        n_presence = len(df_year_p)
        target_pa = int(n_presence * ratio)
        
        # Sample pseudo-absence points
        pa_points = []
        max_attempts = target_pa * 200
        attempts = 0
        
        while len(pa_points) < target_pa and attempts < max_attempts:
            attempts += 1
            x = rng.uniform(min_lon, max_lon)
            y = rng.uniform(min_lat, max_lat)
            p = Point(x, y)
            if prepared_bg.contains(p):
                pa_points.append({
                    "year": year,
                    "longitude": float(x),
                    "latitude": float(y),
                    "presence": 0,
                    "area_km2": 0.0,
                    "protected_km2": 0.0,
                    "eph_km2": 0.0,
                    "biome": "Background",
                    "ecoregion": "Background"
                })
                
        df_year_pa = pd.DataFrame(pa_points)
        
        # Merge presence and absence for this year
        year_dataset = pd.concat([df_year_p, df_year_pa], ignore_index=True)
        all_records.append(year_dataset)
        print(f"Year {year}: {n_presence} presences + {len(df_year_pa)} pseudo-absences = {len(year_dataset)} total.")
        
    df_final = pd.concat(all_records, ignore_index=True)
    out_path = os.path.join("data", "processed", "model_dataset_raw.csv")
    df_final.to_csv(out_path, index=False)
    print(f"\nConsolidated dataset saved to {out_path} (Total records: {len(df_final)})")
    return df_final

def create_model_dataset(occurrences_df):
    """
    Wrapper for pipeline consistency.
    """
    return generate_pseudo_absences(occurrences_df)

if __name__ == "__main__":
    from src.preprocessing import clean_tiger_occurrence_data
    occ = clean_tiger_occurrence_data()
    raw_df = generate_pseudo_absences(occ)
    print("Class balance:\n", raw_df["presence"].value_counts(normalize=True))
