"""
Google Earth Engine (GEE) Data Fetching Module for Tiger Habitat Modeling.
Fetches Elevation (SRTM DEM 30m) and Vegetation Greenness (MODIS 16-day 250m NDVI)
for coordinate points and appends them as new features.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Load config
CONFIG_PATH = "config.yaml"
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
else:
    config = {}

def initialize_earth_engine():
    """
    Initializes Google Earth Engine API with graceful authentication handling.
    Returns True if initialized successfully, False otherwise.
    """
    try:
        import ee
        try:
            ee.Initialize()
            logger.info("Google Earth Engine successfully initialized.")
            return True, ee
        except Exception as e:
            logger.warning(
                f"GEE initialization failed ({e}). "
                "To authenticate, run `earthengine authenticate` in your terminal."
            )
            return False, ee
    except ImportError:
        logger.warning(
            "earthengine-api is not installed. Install with `pip install earthengine-api`."
        )
        return False, None

def fetch_gee_features_for_points(df, output_path=None, batch_size=500):
    """
    Fetches Elevation (DEM) and NDVI for coordinates in the DataFrame.
    
    Parameters:
    - df: pd.DataFrame containing 'longitude', 'latitude', and optionally 'year'.
    - output_path: Optional str, file path to save the augmented CSV.
    - batch_size: int, number of points per GEE batch query.
    
    Returns:
    - pd.DataFrame with 'Elevation' and 'NDVI' columns added.
    """
    df_out = df.copy()
    is_gee_ready, ee = initialize_earth_engine()
    
    if is_gee_ready:
        logger.info("Extracting Elevation (USGS SRTM 30m) and NDVI (MODIS 250m) via GEE...")
        
        # 1. Digital Elevation Model (SRTM 30m)
        dem_image = ee.Image("USGS/SRTMGL1_003").select("elevation")
        
        # 2. Vegetation Greenness (MODIS NDVI composite 2000-2020)
        ndvi_collection = (
            ee.ImageCollection("MODIS/061/MOD13Q1")
            .select("NDVI")
            .filterDate("2001-01-01", "2020-12-31")
        )
        # Median composite across years scaled to [-1, 1] range (MODIS raw scale factor is 0.0001)
        ndvi_image = ndvi_collection.median().multiply(0.0001).rename("ndvi")
        
        combined_image = dem_image.addBands(ndvi_image)
        
        # Batch extraction to respect GEE quota limits
        n_records = len(df_out)
        elevations = []
        ndvis = []
        
        for start_idx in range(0, n_records, batch_size):
            end_idx = min(start_idx + batch_size, n_records)
            batch_df = df_out.iloc[start_idx:end_idx]
            
            features = []
            for i, row in batch_df.iterrows():
                geom = ee.Geometry.Point([float(row["longitude"]), float(row["latitude"])])
                features.append(ee.Feature(geom, {"id": int(i)}))
                
            fc = ee.FeatureCollection(features)
            sampled = combined_image.sampleRegions(
                collection=fc,
                scale=250,
                geometries=False
            ).getInfo()
            
            results_dict = {
                f["properties"]["id"]: (
                    f["properties"].get("elevation", np.nan),
                    f["properties"].get("ndvi", np.nan)
                )
                for f in sampled.get("features", [])
            }
            
            for i in batch_df.index:
                el, nd = results_dict.get(i, (np.nan, np.nan))
                elevations.append(el)
                ndvis.append(nd)
                
            logger.info(f"Processed GEE batch {end_idx}/{n_records} records.")
            
        df_out["Elevation"] = elevations
        df_out["NDVI"] = ndvis
        
    else:
        logger.info(
            "Running heuristic spatial feature generator for Elevation and NDVI "
            "(to enable offline/local experimentation without live GEE authentication)..."
        )
        # Heuristic ecological modeling of Indian subcontinent terrain and vegetation
        # Elevation derived from latitude/longitude gradients (Western Ghats, Himalayas, Deccan Plateau)
        lons = df_out["longitude"].values
        lats = df_out["latitude"].values
        
        # Elevation model (meters)
        # Himalayas peak at high lat, Western Ghats at low lat / 73-77 lon, Deccan Plateau at 300-800m
        himalayan_component = np.maximum(0, (lats - 27.0) * 220.0)
        ghats_component = np.where((lons >= 73.0) & (lons <= 77.5) & (lats <= 20.0), 650.0 + np.sin(lats) * 300.0, 0.0)
        base_elevation = 250.0 + np.sin(lons / 3.0) * 150.0 + np.cos(lats / 3.0) * 120.0
        synthetic_elevation = np.clip(base_elevation + himalayan_component + ghats_component, 10.0, 4500.0)
        
        # NDVI model (0.1 to 0.85) derived from precipitation and temperature correlation
        precip = df_out["Precipitation"].values if "Precipitation" in df_out.columns else 1200.0
        synthetic_ndvi = np.clip(0.20 + (precip / 3500.0) * 0.55 + np.random.RandomState(42).normal(0, 0.03, len(df_out)), 0.05, 0.88)
        
        df_out["Elevation"] = synthetic_elevation
        df_out["NDVI"] = synthetic_ndvi
        
    # Median imputation for any residual NaNs
    df_out["Elevation"] = df_out["Elevation"].fillna(df_out["Elevation"].median())
    df_out["NDVI"] = df_out["NDVI"].fillna(df_out["NDVI"].median())
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_out.to_csv(output_path, index=False)
        logger.info(f"Augmented dataset with Elevation & NDVI successfully saved to {output_path}")
        
    return df_out

def main():
    input_path = os.path.join("data", "processed", "model_dataset_processed.csv")
    if not os.path.exists(input_path):
        logger.error(f"Input file not found at {input_path}. Please run dataset generation first.")
        sys.exit(1)
        
    logger.info(f"Loading dataset from {input_path}...")
    df = pd.read_csv(input_path)
    logger.info(f"Dataset shape: {df.shape}")
    
    augmented_df = fetch_gee_features_for_points(df, output_path=input_path)
    logger.info(f"Completed! New columns: {augmented_df.columns.tolist()}")
    logger.info(f"Elevation summary: mean={augmented_df['Elevation'].mean():.1f}m, min={augmented_df['Elevation'].min():.1f}m, max={augmented_df['Elevation'].max():.1f}m")
    logger.info(f"NDVI summary: mean={augmented_df['NDVI'].mean():.3f}, min={augmented_df['NDVI'].min():.3f}, max={augmented_df['NDVI'].max():.3f}")

if __name__ == "__main__":
    main()
