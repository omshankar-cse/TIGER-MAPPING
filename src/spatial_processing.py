import os
import glob
import numpy as np
import pandas as pd
import geopandas as gpd
from PIL import Image
from shapely.geometry import Point
from shapely.ops import unary_union
from shapely.prepared import prep
import yaml

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

INDIA_BOUNDS = [68.0, 6.0, 98.0, 38.0]  # min_lon, min_lat, max_lon, max_lat
CLIMATE_PROCESSED_DIR = os.path.join("data", "processed", "climate")
os.makedirs(CLIMATE_PROCESSED_DIR, exist_ok=True)

# Grid parameters for WorldClim 2.5m global rasters (4320 rows x 8640 cols)
# Lon: -180 to 180, Lat: 90 to -90
COL_START = int((INDIA_BOUNDS[0] + 180.0) / 360.0 * 8640)  # 5952
COL_END   = int((INDIA_BOUNDS[2] + 180.0) / 360.0 * 8640)  # 6672
ROW_START = int((90.0 - INDIA_BOUNDS[3]) / 180.0 * 4320)   # 1248
ROW_END   = int((90.0 - INDIA_BOUNDS[1]) / 180.0 * 4320)   # 2016

# In-memory cache for fast access
_CLIMATE_CACHE = {}
_INDIA_BOUNDARY = None

def find_worldclim_dir():
    candidates = [
        os.path.join("downloaded dataset", "worlclim"),
        "worlclim",
        os.path.join("data", "raw", "worlclim")
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(f"Could not find worldclim directory in {candidates}")

def find_states_boundary_file():
    candidates = [
        os.path.join("downloaded dataset", "tiger dataset - Copy", "tiger_in_2001", "scl_states_2001.geojson"),
        os.path.join("tiger dataset - Copy", "tiger_in_2001", "scl_states_2001.geojson")
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(f"Could not find scl_states_2001.geojson in {candidates}")

def get_india_boundary():
    """
    Load the state boundaries for India and merge them into a single polygon.
    """
    global _INDIA_BOUNDARY
    if _INDIA_BOUNDARY is not None:
        return _INDIA_BOUNDARY
        
    states_path = find_states_boundary_file()
    df_states = gpd.read_file(states_path)
    india_states = df_states[(df_states['country'] == 'India') | (df_states['iso2'] == 'IN')]
    if india_states.empty:
        raise ValueError("No India boundary features found in state file.")
        
    _INDIA_BOUNDARY = unary_union(india_states.geometry)
    return _INDIA_BOUNDARY

def read_geotiff_crop(file_path):
    """
    Reads a single GeoTIFF file using PIL and extracts the India bounding box slice.
    Returns a 2D numpy float32 array with shape (768, 720).
    """
    im = Image.open(file_path)
    arr = np.array(im, dtype=np.float32)
    crop = arr[ROW_START:ROW_END, COL_START:COL_END]
    # Replace nodata values (often < -999 or extreme values) with nan
    crop[crop < -999.0] = np.nan
    crop[crop > 100000.0] = np.nan
    return crop

def compute_and_cache_climate_for_year(year):
    """
    Computes all 7 bioclimatic layers for India for a given year using the 12 monthly rasters:
    - Precipitation (Annual sum)
    - TMAX (Mean monthly maximum temperature)
    - TMIN (Mean monthly minimum temperature)
    - Bio_Mean_Temp (Annual mean temperature)
    - Bio_Temp_Range (TMAX - TMIN)
    - Bio_Temp_Seasonality (Standard deviation of monthly mean temp * 100)
    - Bio_Precip_Seasonality (Coefficient of variation of monthly precipitation)
    """
    global _CLIMATE_CACHE
    if year in _CLIMATE_CACHE:
        return _CLIMATE_CACHE[year]
        
    npz_path = os.path.join(CLIMATE_PROCESSED_DIR, f"climate_stack_{year}.npz")
    if os.path.exists(npz_path):
        data = np.load(npz_path)
        stack = {k: data[k] for k in data.files}
        _CLIMATE_CACHE[year] = stack
        return stack

    # 2020 uses 2019 data as proxy
    effective_year = 2019 if year >= 2020 else year
    decade_dir_name = "2000-2009" if effective_year <= 2009 else "2010-2019"
    
    worldclim_root = find_worldclim_dir()
    decade_dir = os.path.join(worldclim_root, decade_dir_name)
    
    # Locate folders
    prec_dirs = [os.path.join(decade_dir, "precipitation"), os.path.join(decade_dir, "prep"), decade_dir]
    tmax_dirs = [os.path.join(decade_dir, "tmax"), decade_dir]
    tmin_dirs = [os.path.join(decade_dir, "tmin"), decade_dir]
    
    def find_12_files(dirs, var_pattern):
        for d in dirs:
            if not os.path.exists(d):
                continue
            files = sorted(glob.glob(os.path.join(d, f"*{var_pattern}_{effective_year}-*.tif")))
            if len(files) == 12:
                return files
        raise FileNotFoundError(f"Could not find 12 monthly TIFFs for {var_pattern} in year {effective_year}")
        
    prec_files = find_12_files(prec_dirs, "prec")
    tmax_files = find_12_files(tmax_dirs, "tmax")
    tmin_files = find_12_files(tmin_dirs, "tmin")
    
    # Read 12 months for each
    prec_months = np.stack([read_geotiff_crop(f) for f in prec_files], axis=0)  # (12, 768, 720)
    tmax_months = np.stack([read_geotiff_crop(f) for f in tmax_files], axis=0)  # (12, 768, 720)
    tmin_months = np.stack([read_geotiff_crop(f) for f in tmin_files], axis=0)  # (12, 768, 720)
    
    mean_temp_months = (tmax_months + tmin_months) / 2.0
    
    # 1. Annual Precipitation (BIO12)
    prec_annual = np.nansum(prec_months, axis=0)
    # Mask places where all months were nan
    all_nan_prec = np.all(np.isnan(prec_months), axis=0)
    prec_annual[all_nan_prec] = np.nan
    
    # 2. Mean TMAX & TMIN
    tmax_mean = np.nanmean(tmax_months, axis=0)
    tmin_mean = np.nanmean(tmin_months, axis=0)
    
    # 3. BIO1: Annual Mean Temp
    bio_mean_temp = np.nanmean(mean_temp_months, axis=0)
    
    # 4. BIO7: Temperature Range
    bio_temp_range = tmax_mean - tmin_mean
    
    # 5. BIO4: Temperature Seasonality (std dev of monthly mean temp * 100)
    bio_temp_seasonality = np.nanstd(mean_temp_months, axis=0) * 100.0
    
    # 6. BIO15: Precipitation Seasonality (CV = std / (mean + 1e-4) * 100)
    prec_mean_m = np.nanmean(prec_months, axis=0)
    prec_std_m = np.nanstd(prec_months, axis=0)
    bio_precip_seasonality = (prec_std_m / (prec_mean_m + 1e-4)) * 100.0
    
    stack = {
        "Precipitation": prec_annual,
        "TMAX": tmax_mean,
        "TMIN": tmin_mean,
        "Bio_Mean_Temp": bio_mean_temp,
        "Bio_Temp_Range": bio_temp_range,
        "Bio_Temp_Seasonality": bio_temp_seasonality,
        "Bio_Precip_Seasonality": bio_precip_seasonality
    }
    
    # Save to npz
    np.savez_compressed(npz_path, **stack)
    _CLIMATE_CACHE[year] = stack
    print(f"Aggregated & cached bioclimatic stack for {year} -> {npz_path} (Shape: {prec_annual.shape})")
    return stack

def coord_to_pixel(lon, lat):
    """
    Converts (lon, lat) to (row, col) in the India crop array.
    """
    col = int(round((lon - INDIA_BOUNDS[0]) / (INDIA_BOUNDS[2] - INDIA_BOUNDS[0]) * (COL_END - COL_START)))
    row = int(round((INDIA_BOUNDS[3] - lat) / (INDIA_BOUNDS[3] - INDIA_BOUNDS[1]) * (ROW_END - ROW_START)))
    return row, col

def get_climate_value_at_points(points, year, var_name):
    """
    Extracts variable values at coordinate points for a given year.
    points: List of (longitude, latitude) tuples.
    """
    stack = compute_and_cache_climate_for_year(year)
    grid = stack.get(var_name)
    if grid is None:
        raise ValueError(f"Variable {var_name} not found in climate stack for year {year}")
        
    num_rows, num_cols = grid.shape
    values = []
    
    for lon, lat in points:
        r, c = coord_to_pixel(lon, lat)
        if 0 <= r < num_rows and 0 <= c < num_cols:
            val = grid[r, c]
            if np.isnan(val):
                # Try 3x3 local neighborhood fallback
                r_min, r_max = max(0, r-1), min(num_rows, r+2)
                c_min, c_max = max(0, c-1), min(num_cols, c+2)
                sub = grid[r_min:r_max, c_min:c_max]
                val = np.nanmean(sub) if not np.all(np.isnan(sub)) else np.nan
            values.append(float(val) if not np.isnan(val) else np.nan)
        else:
            values.append(np.nan)
            
    return values

def generate_india_prediction_grid(resolution=None):
    """
    Generates a uniform grid of points strictly inside India's boundary.
    """
    if resolution is None:
        resolution = config.get('spatial_grid', {}).get('resolution', 0.1)
        
    india_boundary = get_india_boundary()
    min_lon, min_lat, max_lon, max_lat = india_boundary.bounds
    
    lons = np.arange(min_lon, max_lon + resolution / 2.0, resolution)
    lats = np.arange(min_lat, max_lat + resolution / 2.0, resolution)
    
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    grid_points = np.stack([lon_grid.ravel(), lat_grid.ravel()], axis=-1)
    
    prepared_boundary = prep(india_boundary)
    valid_points = []
    
    for pt in grid_points:
        p = Point(pt[0], pt[1])
        if prepared_boundary.contains(p):
            valid_points.append(pt)
            
    df_grid = pd.DataFrame(valid_points, columns=['longitude', 'latitude'])
    print(f"Generated India prediction grid: {len(df_grid)} points at resolution {resolution}°")
    return df_grid

if __name__ == "__main__":
    st = compute_and_cache_climate_for_year(2001)
    print("Climate stack 2001 keys:", list(st.keys()))
    val = get_climate_value_at_points([(77.5, 12.9), (88.3, 22.5)], 2001, "Precipitation")
    print("Sample values (Bangalore, Kolkata 2001):", val)
