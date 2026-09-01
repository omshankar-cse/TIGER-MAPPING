import os
import json
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss, confusion_matrix
)
import joblib
import yaml
from shapely.geometry import Point
from shapely.ops import unary_union
from shapely.prepared import prep

from src.feature_engineering import scale_features
from src.spatial_processing import (
    get_india_boundary, 
    generate_india_prediction_grid, 
    get_climate_value_at_points
)

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

METRICS_DIR = os.path.join("outputs", "metrics")
os.makedirs(METRICS_DIR, exist_ok=True)

def calculate_detailed_metrics(y_true, y_pred, y_prob):
    """
    Computes all standard classification & discrimination metrics.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    brier = brier_score_loss(y_true, y_prob)
    
    return {
        "accuracy": float(acc),
        "balanced_accuracy": float((rec + spec) / 2.0),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "specificity": float(spec),
        "sensitivity": float(rec),
        "brier_score": float(brier),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        }
    }

def run_memorization_test(df_train, df_test, best_model_type):
    """
    Compares Climate-only vs Climate + Coordinate models to test for geographic memorization.
    """
    from sklearn.ensemble import RandomForestClassifier
    try:
        import xgboost as xgb
        HAS_XGB = True
    except ImportError:
        HAS_XGB = False
        
    random_state = config.get('model', {}).get('random_state', 42)
    feature_cols_a = config['features']['raw'] + config['features']['engineered']
    feature_cols_b = feature_cols_a + ["longitude", "latitude"]
    
    if best_model_type == "xgb" and HAS_XGB:
        clf_a = xgb.XGBClassifier(random_state=random_state, eval_metric="logloss", max_depth=6)
        clf_b = xgb.XGBClassifier(random_state=random_state, eval_metric="logloss", max_depth=6)
    else:
        clf_a = RandomForestClassifier(random_state=random_state, n_estimators=150, max_depth=10, class_weight='balanced')
        clf_b = RandomForestClassifier(random_state=random_state, n_estimators=150, max_depth=10, class_weight='balanced')
        
    clf_a.fit(df_train[feature_cols_a], df_train["presence"])
    clf_b.fit(df_train[feature_cols_b], df_train["presence"])
    
    probs_a = clf_a.predict_proba(df_test[feature_cols_a])[:, 1]
    probs_b = clf_b.predict_proba(df_test[feature_cols_b])[:, 1]
    
    auc_a = roc_auc_score(df_test["presence"], probs_a)
    auc_b = roc_auc_score(df_test["presence"], probs_b)
    
    print("\n--- Geographic Memorization Audit Report ---")
    print(f"Experiment A (Climate Features Only)         -> Test ROC-AUC: {auc_a:.4f}")
    print(f"Experiment B (Climate + Spatial Coordinates)    -> Test ROC-AUC: {auc_b:.4f}")
    if auc_b - auc_a > 0.05:
        print("[WARNING]: Model B achieves noticeably higher score, indicating spatial memorization.")
    else:
        print("[SUCCESS]: Pure climate model generalizes robustly without relying on coordinate memorization.")

def load_historical_species_polygons(year):
    """
    Loads mapped species range polygons for India for a given year.
    """
    base_dir = "downloaded dataset/tiger dataset - Copy"
    if not os.path.exists(base_dir):
        base_dir = "tiger dataset - Copy"
        
    folder_name = f"tiger_in_{year}"
    if year == 2020:
        folder_name = "tiger_in_2020 (1)"
    path = os.path.join(base_dir, folder_name, f"scl_species_{year}.geojson")
    if os.path.exists(path):
        try:
            gdf = gpd.read_file(path)
            return gdf[(gdf["country"] == "India") | (gdf["iso2"] == "IN")]
        except Exception:
            return gpd.GeoDataFrame()
    return gpd.GeoDataFrame()

def evaluate_spatial_grid_metrics(year, model, scaler, threshold, feature_cols):
    """
    Computes spatial precision, recall, and IoU by comparing predictions on the India grid
    against the actual mapped presence polygon boundaries.
    """
    from src.feature_engineering import compute_engineered_features
    
    gdf_range = load_historical_species_polygons(year)
    if gdf_range.empty:
        return None
        
    df_grid = generate_india_prediction_grid(resolution=0.15)
    points = list(zip(df_grid["longitude"], df_grid["latitude"]))
    
    df_sc = df_grid.copy()
    target_year = min(year, 2019)
    for var in config['features']['raw']:
        df_sc[var] = get_climate_value_at_points(points, target_year, var)
        if df_sc[var].isnull().any():
            df_sc[var] = df_sc[var].fillna(df_sc[var].median())
            
    df_sc = compute_engineered_features(df_sc)
    df_sc_scaled = scale_features(df_sc, scaler)
    
    probs = model.predict_proba(df_sc_scaled[feature_cols])[:, 1]
    predicted_presence = (probs >= threshold).astype(int)
    
    union_geom = unary_union(gdf_range.geometry)
    prep_geom = prep(union_geom)
    actual_presence = np.array([1 if prep_geom.contains(Point(lon, lat)) else 0 for lon, lat in points])
    
    tp = np.sum((actual_presence == 1) & (predicted_presence == 1))
    fp = np.sum((actual_presence == 0) & (predicted_presence == 1))
    fn = np.sum((actual_presence == 1) & (predicted_presence == 0))
    
    sp_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    sp_rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    sp_iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    
    return {
        "spatial_precision": float(sp_prec),
        "spatial_recall": float(sp_rec),
        "spatial_iou": float(sp_iou)
    }

def main():
    print("==========================================================")
    print("=== EVALUATING UNSEEN TEST YEARS (2016-2020) ===")
    print("==========================================================")
    
    model_path = os.path.join("models", "best_model.pkl")
    metadata_path = os.path.join("models", "model_metadata.json")
    scaler_path = os.path.join("models", "scaler.pkl")
    dataset_path = os.path.join("data", "processed", "model_dataset_processed.csv")
    
    model_data = joblib.load(model_path)
    model = model_data["model"]
    feature_cols = model_data["features"]
    model_code = model_data["model_type"]
    scaler = joblib.load(scaler_path)
    
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
        
    threshold = float(metadata.get("threshold", 0.5))
    print(f"Loaded model: {metadata.get('model_type')} | Optimal Threshold: {threshold:.2f}")
    
    df_all = pd.read_csv(dataset_path)
    test_start = config.get('testing', {}).get('start_year', 2016)
    test_end = config.get('testing', {}).get('end_year', 2020)
    
    df_train = df_all[(df_all["year"] < test_start)].copy()
    df_test = df_all[(df_all["year"] >= test_start) & (df_all["year"] <= test_end)].copy()
    
    df_train_scaled = scale_features(df_train, scaler)
    df_test_scaled = scale_features(df_test, scaler)
    df_test_scaled.to_csv(os.path.join("data", "processed", "test_set_scaled.csv"), index=False)
    
    # Memorization test
    run_memorization_test(df_train_scaled, df_test_scaled, model_code)
    
    # Test Predictions
    X_test = df_test_scaled[feature_cols]
    y_test = df_test_scaled["presence"]
    
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)
    
    overall_metrics = calculate_detailed_metrics(y_test, preds, probs)
    print("\n==========================================================")
    print(f"=== OVERALL UNSEEN TEST METRICS ({test_start}-{test_end}) ===")
    print("==========================================================")
    print(f"  Test ROC-AUC:           {overall_metrics['roc_auc'] * 100:.2f}%")
    print(f"  Test Accuracy:          {overall_metrics['accuracy'] * 100:.2f}%")
    print(f"  Test Balanced Accuracy: {overall_metrics['balanced_accuracy'] * 100:.2f}%")
    print(f"  Test Precision:         {overall_metrics['precision'] * 100:.2f}%")
    print(f"  Test Recall (Sens):     {overall_metrics['recall'] * 100:.2f}%")
    print(f"  Test Specificity:       {overall_metrics['specificity'] * 100:.2f}%")
    print(f"  Test F1-Score:          {overall_metrics['f1']:.4f}")
    print(f"  Test PR-AUC:            {overall_metrics['pr_auc']:.4f}")
    print(f"  Test Brier Score:       {overall_metrics['brier_score']:.4f}")
    print(f"  Confusion Matrix:       {overall_metrics['confusion_matrix']}")
    
    with open(os.path.join(METRICS_DIR, "overall_test_metrics.json"), "w") as f:
        json.dump(overall_metrics, f, indent=4)
        
    # Year-wise Breakdown & Spatial Validation
    year_records = []
    test_years = sorted(df_test_scaled["year"].unique())
    print("\n--- Year-Wise Breakdown & Spatial IoU ---")
    
    for yr in test_years:
        df_yr = df_test_scaled[df_test_scaled["year"] == yr]
        X_yr = df_yr[feature_cols]
        y_yr = df_yr["presence"]
        
        p_yr = model.predict_proba(X_yr)[:, 1]
        pred_yr = (p_yr >= threshold).astype(int)
        
        yr_m = calculate_detailed_metrics(y_yr, pred_yr, p_yr)
        
        # Spatial polygon overlap
        sp_m = evaluate_spatial_grid_metrics(yr, model, scaler, threshold, feature_cols)
        sp_iou = sp_m["spatial_iou"] if sp_m else np.nan
        sp_prec = sp_m["spatial_precision"] if sp_m else np.nan
        sp_rec = sp_m["spatial_recall"] if sp_m else np.nan
        
        print(f"  Year {yr} -> Acc: {yr_m['accuracy']:.4f}, ROC-AUC: {yr_m['roc_auc']:.4f}, F1: {yr_m['f1']:.4f} | Spatial IoU: {sp_iou:.4f}")
        
        year_records.append({
            "Year": int(yr),
            "Accuracy": yr_m["accuracy"],
            "Balanced Accuracy": yr_m["balanced_accuracy"],
            "Precision": yr_m["precision"],
            "Recall": yr_m["recall"],
            "F1": yr_m["f1"],
            "ROC-AUC": yr_m["roc_auc"],
            "Specificity": yr_m["specificity"],
            "Brier Score": yr_m["brier_score"],
            "Spatial IoU": sp_iou,
            "Spatial Precision": sp_prec,
            "Spatial Recall": sp_rec
        })
        
    df_yw = pd.DataFrame(year_records)
    df_yw.to_csv(os.path.join(METRICS_DIR, "year_wise_metrics.csv"), index=False)
    
    # Feature Importances / Permutation Importance
    print("\n--- Computing Feature Importances ---")
    from sklearn.inspection import permutation_importance
    perm_res = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, scoring="roc_auc")
    df_imp = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": perm_res.importances_mean,
        "Std": perm_res.importances_std
    }).sort_values(by="Importance", ascending=False)
    
    print(df_imp.to_string(index=False))
    df_imp.to_csv(os.path.join(METRICS_DIR, "feature_importances.csv"), index=False)
    
    print("\nEvaluation pipeline successfully finished!")

if __name__ == "__main__":
    main()
