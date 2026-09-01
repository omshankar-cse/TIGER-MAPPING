import os
import re
import random
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, MultiPolygon, Polygon
from shapely.ops import unary_union
from shapely.prepared import prep
import yaml

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

INDIA_BOUNDS = [68.0, 6.0, 98.0, 38.0]  # min_lon, min_lat, max_lon, max_lat

def find_tiger_dataset_dir():
    """
    Locates the tiger dataset directory in the workspace.
    """
    candidates = [
        os.path.join("downloaded dataset", "tiger dataset - Copy"),
        "tiger dataset - Copy",
        os.path.join("data", "raw", "tiger dataset")
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(f"Could not find tiger dataset directory in {candidates}")

def extract_year_from_foldername(name):
    """
    Extracts the 4-digit year from folder names like 'tiger_in_2001' or 'tiger_in_2020 (1)'.
    """
    match = re.search(r"(20\d{2})", name)
    if match:
        return int(match.group(1))
    return None

def sample_points_inside_polygon(geom, n_points, random_state=42):
    """
    Samples n_points uniformly at random inside a Shapely Polygon or MultiPolygon.
    """
    if geom is None or geom.is_empty or n_points <= 0:
        return []
        
    min_x, min_y, max_x, max_y = geom.bounds
    # Clamp to India bounds
    min_x = max(min_x, INDIA_BOUNDS[0])
    min_y = max(min_y, INDIA_BOUNDS[1])
    max_x = min(max_x, INDIA_BOUNDS[2])
    max_y = min(max_y, INDIA_BOUNDS[3])
    
    if min_x >= max_x or min_y >= max_y:
        return []
        
    prepared_geom = prep(geom)
    rng = np.random.RandomState(random_state)
    
    sampled = []
    max_attempts = n_points * 200
    attempts = 0
    
    while len(sampled) < n_points and attempts < max_attempts:
        attempts += 1
        x = rng.uniform(min_x, max_x)
        y = rng.uniform(min_y, max_y)
        p = Point(x, y)
        if prepared_geom.contains(p):
            sampled.append((float(x), float(y)))
            
    return sampled

def load_and_sample_presence_data(base_dir=None, target_points_per_year=None):
    """
    Discovers all yearly tiger presence datasets (2001-2020), loads presence polygons,
    and extracts stratified spatial presence points within verified polygons.
    """
    if base_dir is None:
        base_dir = find_tiger_dataset_dir()
        
    if target_points_per_year is None:
        target_points_per_year = config.get('presence_sampling', {}).get('points_per_year', 200)
        
    random_seed = config.get('model', {}).get('random_state', 42)
    
    year_folders = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    all_presence_records = []
    
    for folder in sorted(year_folders):
        year = extract_year_from_foldername(folder)
        if year is None:
            continue
            
        folder_path = os.path.join(base_dir, folder)
        species_path = os.path.join(folder_path, f"scl_species_{year}.geojson")
        
        if not os.path.exists(species_path):
            continue
            
        try:
            gdf = gpd.read_file(species_path)
            # Filter for India
            gdf_india = gdf[(gdf['country'] == 'India') | (gdf['iso2'] == 'IN')].copy()
            
            if gdf_india.empty:
                print(f"[{year}] No India features found in {species_path}")
                continue
                
            total_area = gdf_india.geometry.area.sum()
            if total_area <= 0:
                continue
                
            year_points = []
            
            # Stratified sampling across polygons proportional to area
            for idx, row in gdf_india.iterrows():
                geom = row.geometry
                poly_area = geom.area
                if poly_area <= 0:
                    continue
                    
                # Proportional allocation, at least 1 point per non-trivial polygon
                n_poly_points = max(1, int(round(target_points_per_year * (poly_area / total_area))))
                
                # Sample points inside polygon
                pts = sample_points_inside_polygon(
                    geom, 
                    n_poly_points, 
                    random_state=random_seed + year * 100 + int(idx)
                )
                
                # If polygon is small and random hit missed, add centroid
                if not pts and geom.centroid.is_valid and not geom.centroid.is_empty:
                    c = geom.centroid
                    if INDIA_BOUNDS[0] <= c.x <= INDIA_BOUNDS[2] and INDIA_BOUNDS[1] <= c.y <= INDIA_BOUNDS[3]:
                        pts.append((float(c.x), float(c.y)))
                        
                for lon, lat in pts:
                    year_points.append({
                        "year": year,
                        "longitude": lon,
                        "latitude": lat,
                        "presence": 1,
                        "area_km2": float(row.get("area", 0) or 0),
                        "protected_km2": float(row.get("protected", 0) or 0),
                        "eph_km2": float(row.get("eph", 0) or 0),
                        "biome": str(row.get("biome", "Unknown")),
                        "ecoregion": str(row.get("ecoregion", "Unknown"))
                    })
                    
            print(f"Year {year}: Sampled {len(year_points)} verified presence points across {len(gdf_india)} India polygons.")
            all_presence_records.extend(year_points)
            
        except Exception as e:
            print(f"Error reading presence file for {year}: {e}")
            
    df_presences = pd.DataFrame(all_presence_records)
    print(f"\nTotal raw stratified presence points collected: {len(df_presences)}")
    return df_presences

if __name__ == "__main__":
    df = load_and_sample_presence_data()
    print("Sample:\n", df.head())
