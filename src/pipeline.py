import os
import sys
import pandas as pd
import numpy as np
import yaml
import json

# Import project pipeline modules
import src.train as train_module
import src.evaluate as evaluate_module
import src.predict as predict_module

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

def run_data_quality_audit(df_processed):
    """
    Performs comprehensive data quality and integrity checks and saves outputs/data_quality_report.csv.
    """
    print("\n--- Generating Data Quality and Integrity Report ---")
    
    total_records = len(df_processed)
    missing_vals = df_processed.isnull().sum().sum()
    
    duplicates = df_processed.duplicated(subset=["year", "latitude", "longitude"]).sum()
    unique_coords = df_processed.groupby(["latitude", "longitude"]).ngroups
    unique_years = df_processed["year"].nunique()
    
    class_balance = df_processed["presence"].value_counts(normalize=True).to_dict()
    
    min_lat, max_lat = df_processed["latitude"].min(), df_processed["latitude"].max()
    min_lon, max_lon = df_processed["longitude"].min(), df_processed["longitude"].max()
    
    feature_cols = config['features']['raw'] + config['features']['engineered']
    feature_ranges = {}
    for col in feature_cols:
        if col in df_processed.columns:
            feature_ranges[f"{col}_min"] = float(df_processed[col].min())
            feature_ranges[f"{col}_max"] = float(df_processed[col].max())
            feature_ranges[f"{col}_mean"] = float(df_processed[col].mean())
            
    report_data = {
        "metric": [
            "total_records",
            "missing_values",
            "duplicate_records",
            "unique_coordinate_points",
            "unique_years",
            "class_balance_presence",
            "class_balance_absence",
            "latitude_extent",
            "longitude_extent"
        ],
        "value": [
            str(total_records),
            str(missing_vals),
            str(duplicates),
            str(unique_coords),
            str(unique_years),
            f"{class_balance.get(1, 0.0) * 100:.2f}%",
            f"{class_balance.get(0, 0.0) * 100:.2f}%",
            f"{min_lat:.4f} to {max_lat:.4f}",
            f"{min_lon:.4f} to {max_lon:.4f}"
        ]
    }
    
    for k, v in feature_ranges.items():
        report_data["metric"].append(k)
        report_data["value"].append(f"{v:.4f}")
        
    df_report = pd.DataFrame(report_data)
    os.makedirs("outputs", exist_ok=True)
    report_path = os.path.join("outputs", "data_quality_report.csv")
    df_report.to_csv(report_path, index=False)
    print(f"Data quality report saved to `{report_path}`")

def run_sanity_checks():
    """
    Runs crucial sanity checks to ensure scientific validity of splits, coordinates, and targets.
    """
    print("\n--- Running Pipeline Sanity Checks ---")
    errors = []
    
    train_path = os.path.join("data", "processed", "train_set_scaled.csv")
    test_path = os.path.join("data", "processed", "test_set_scaled.csv")
    metadata_path = os.path.join("models", "model_metadata.json")
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        errors.append("Scaled train/test split files are missing.")
        return False
        
    df_tr = pd.read_csv(train_path)
    df_ts = pd.read_csv(test_path)
    
    # 1. No test-year data in training
    test_years_in_train = df_tr[df_tr["year"] >= 2016]["year"].unique()
    if len(test_years_in_train) > 0:
        errors.append(f"Training set contains test year observations: {test_years_in_train}")
        
    # 2. No training-year data in test
    train_years_in_test = df_ts[df_ts["year"] <= 2015]["year"].unique()
    if len(train_years_in_test) > 0:
        errors.append(f"Test set contains training year observations: {train_years_in_test}")
        
    # 3. Target boundaries (WGS84)
    invalid_tr_coords = df_tr[
        (df_tr["latitude"] < 6.0) | (df_tr["latitude"] > 38.0) |
        (df_tr["longitude"] < 68.0) | (df_tr["longitude"] > 98.0)
    ]
    if not invalid_tr_coords.empty:
        errors.append(f"Training set contains invalid coordinate bounds outside India: {len(invalid_tr_coords)} rows")
        
    # 4. Target values (binary check)
    unique_tr_targets = df_tr["presence"].unique()
    if not set(unique_tr_targets).issubset({0, 1}):
        errors.append(f"Presence target column contains non-binary values: {unique_tr_targets}")
        
    # 5. Metadata threshold check
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            meta = json.load(f)
        threshold = meta.get("threshold", 0.5)
        if threshold < 0.0 or threshold > 1.0:
            errors.append(f"Invalid optimized classification threshold: {threshold}")
            
    if errors:
        print("[FAILED] Sanity checks FAILED:")
        for err in errors:
            print(f"  - {err}")
        return False
    else:
        print("[PASSED] All scientific sanity checks PASSED!")
        return True

def main():
    print("==========================================================")
    print("=== STARTING MASTER REBUILD PIPELINE (python -m src.pipeline) ===")
    print("==========================================================")
    
    # Step 1: Preprocessing & Training (Tuning, Comparison, Calibration, Threshold)
    train_module.main()
    
    # Step 2: Data Quality Audit
    processed_path = os.path.join("data", "processed", "model_dataset_processed.csv")
    if os.path.exists(processed_path):
        df_processed = pd.read_csv(processed_path)
        run_data_quality_audit(df_processed)
        
    # Step 3: Evaluation & Spatial Metrics
    evaluate_module.main()
    
    # Step 4: Future Projections
    predict_module.run_projections()
    
    # Step 5: Sanity Checks
    success = run_sanity_checks()
    
    if success:
        print("\n==========================================================")
        print("=== PIPELINE EXECUTED SUCCESSFULLY WITH ZERO ERRORS ===")
        print("==========================================================")
    else:
        print("\n==========================================================")
        print("=== PIPELINE ENCOUNTERED SANITY ERRORS! PLEASE REVISE ===")
        print("==========================================================")
        sys.exit(1)

if __name__ == "__main__":
    main()
