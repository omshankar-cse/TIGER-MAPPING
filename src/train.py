import os
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import RandomizedSearchCV, GroupKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, brier_score_loss, precision_score, recall_score
import joblib
import yaml

# Import project modules
from src.preprocessing import clean_tiger_occurrence_data
from src.pseudo_absence import create_model_dataset
from src.feature_engineering import (
    extract_climate_features_for_dataset,
    compute_engineered_features,
    compute_vif_report,
    fit_and_save_scaler,
    scale_features
)

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Import XGBoost if available
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

def create_spatial_groups(df, block_size=None):
    """
    Computes spatial block group IDs (e.g. 2.0° x 2.0° cells) to prevent spatial autocorrelation leakage.
    """
    if block_size is None:
        block_size = config.get('spatial_grid', {}).get('validation_block_size', 2.0)
        
    block_x = (df["longitude"] / block_size).astype(int)
    block_y = (df["latitude"] / block_size).astype(int)
    return block_x.astype(str) + "_" + block_y.astype(str)

def run_hyperparameter_tuning(X, y, groups, model_type="rf"):
    """
    Runs spatial GroupKFold RandomizedSearchCV for hyperparameter optimization.
    """
    random_state = config.get('model', {}).get('random_state', 42)
    n_splits = config.get('hyperparameter_tuning', {}).get('cv_splits', 5)
    n_iter = config.get('hyperparameter_tuning', {}).get('n_iter', 20)
    gkf = GroupKFold(n_splits=n_splits)
    
    if model_type == "rf":
        clf = RandomForestClassifier(random_state=random_state, class_weight='balanced')
        param_dist = {
            "n_estimators": [100, 150, 200, 300],
            "max_depth": [6, 10, 15, 20, None],
            "min_samples_split": [2, 4, 8],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", 0.7]
        }
    elif model_type == "hgb":
        clf = HistGradientBoostingClassifier(random_state=random_state, class_weight='balanced')
        param_dist = {
            "max_iter": [100, 150, 200],
            "max_depth": [4, 6, 10, 15, None],
            "learning_rate": [0.03, 0.05, 0.1, 0.15],
            "l2_regularization": [0.0, 0.1, 1.0, 5.0],
            "min_samples_leaf": [10, 20, 40]
        }
    elif model_type == "xgb" and HAS_XGB:
        clf = xgb.XGBClassifier(random_state=random_state, eval_metric="logloss")
        param_dist = {
            "n_estimators": [100, 150, 200, 300],
            "max_depth": [4, 6, 8, 10],
            "learning_rate": [0.03, 0.05, 0.1, 0.15],
            "subsample": [0.7, 0.85, 1.0],
            "colsample_bytree": [0.7, 0.85, 1.0],
            "min_child_weight": [1, 3, 5],
            "gamma": [0.0, 0.1, 0.5]
        }
    else:
        raise ValueError(f"Unknown or unsupported model type: {model_type}")
        
    search = RandomizedSearchCV(
        estimator=clf,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=list(gkf.split(X, y, groups)),
        random_state=random_state,
        n_jobs=-1,
        verbose=0
    )
    search.fit(X, y)
    print(f"[{model_type.upper()}] Best Params: {search.best_params_} | Spatial CV ROC-AUC: {search.best_score_:.4f}")
    return search.best_estimator_

def evaluate_and_calibrate_model(model, X, y, groups):
    """
    Evaluates model across spatial GroupKFold blocks with probability calibration.
    Returns calibrated classifier, out-of-fold probability predictions, and CV metrics.
    """
    gkf = GroupKFold(n_splits=5)
    
    cv_accs = []
    cv_f1s = []
    cv_aucs = []
    cv_briers = []
    oof_probs = np.zeros(len(y))
    
    for train_idx, val_idx in gkf.split(X, y, groups):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        cal_clf = CalibratedClassifierCV(estimator=model, method='sigmoid', cv=3)
        cal_clf.fit(X_tr, y_tr)
        
        probs = cal_clf.predict_proba(X_val)[:, 1]
        preds = (probs >= 0.5).astype(int)
        
        oof_probs[val_idx] = probs
        cv_accs.append(accuracy_score(y_val, preds))
        cv_f1s.append(f1_score(y_val, preds, zero_division=0))
        cv_aucs.append(roc_auc_score(y_val, probs))
        cv_briers.append(brier_score_loss(y_val, probs))
        
    final_calibrated = CalibratedClassifierCV(estimator=model, method='sigmoid', cv=5)
    final_calibrated.fit(X, y)
    
    metrics = {
        "acc": float(np.mean(cv_accs)),
        "f1": float(np.mean(cv_f1s)),
        "auc": float(np.mean(cv_aucs)),
        "brier": float(np.mean(cv_briers))
    }
    return final_calibrated, oof_probs, metrics

def optimize_threshold(y_true, oof_probs):
    """
    Computes the optimal probability classification threshold by maximizing Youden's J-statistic
    (J = Sensitivity + Specificity - 1) on out-of-fold validation probabilities.
    """
    thresholds = np.linspace(0.10, 0.90, 81)
    best_j = -1.0
    best_threshold = 0.5
    
    for thr in thresholds:
        preds = (oof_probs >= thr).astype(int)
        tp = np.sum((y_true == 1) & (preds == 1))
        fn = np.sum((y_true == 1) & (preds == 0))
        tn = np.sum((y_true == 0) & (preds == 0))
        fp = np.sum((y_true == 0) & (preds == 1))
        
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        j = sens + spec - 1.0
        
        if j > best_j:
            best_j = j
            best_threshold = thr
            
    return float(best_threshold)

def main():
    print("==========================================================")
    print("=== STARTING TIGER HABITAT RE-ARCHITECTED TRAINING PIPELINE ===")
    print("==========================================================")
    
    # 1. Stratified Presence Data
    occurrences_df = clean_tiger_occurrence_data()
    
    # 2. Pseudo-Absence Generation
    dataset_raw = create_model_dataset(occurrences_df)
    
    # 3. Bioclimatic Feature Extraction
    dataset_features = extract_climate_features_for_dataset(dataset_raw)
    
    # 4. Feature Engineering
    print("\n--- Step 4: Computing Engineered Features ---")
    dataset_engineered = compute_engineered_features(dataset_features)
    
    # Save full processed dataset
    os.makedirs(os.path.join("data", "processed"), exist_ok=True)
    dataset_engineered.to_csv(os.path.join("data", "processed", "model_dataset_processed.csv"), index=False)
    
    # Feature columns
    feature_cols = config['features']['raw'] + config['features']['engineered']
    
    # 5. Multicollinearity VIF Diagnostic
    print("\n--- Multicollinearity Diagnostic (VIF) ---")
    vif_df = compute_vif_report(dataset_engineered, feature_cols)
    os.makedirs(os.path.join("outputs", "metrics"), exist_ok=True)
    vif_df.to_csv(os.path.join("outputs", "metrics", "vif_report.csv"), index=False)
    print(vif_df.to_string(index=False))
    
    # 6. Temporal Split: Train (2001-2015) vs Test (2016-2020)
    train_start = config.get('training', {}).get('start_year', 2001)
    train_end = config.get('training', {}).get('end_year', 2015)
    
    df_train = dataset_engineered[
        (dataset_engineered["year"] >= train_start) & 
        (dataset_engineered["year"] <= train_end)
    ].copy()
    
    print(f"\n--- Train Set ({train_start}-{train_end}): {len(df_train)} records (Presences: {df_train['presence'].sum()}, Absences: {(df_train['presence']==0).sum()}) ---")
    
    # 7. Scaler Fitting (strictly on Train set)
    scaler = fit_and_save_scaler(df_train)
    df_train_scaled = scale_features(df_train, scaler)
    df_train_scaled.to_csv(os.path.join("data", "processed", "train_set_scaled.csv"), index=False)
    
    X_train = df_train_scaled[feature_cols]
    y_train = df_train_scaled["presence"]
    groups_train = create_spatial_groups(df_train_scaled)
    
    # 8. Model Selection, Spatial CV & Hyperparameter Tuning
    models_to_test = {
        "rf": "Random Forest",
        "hgb": "HistGradientBoosting"
    }
    if HAS_XGB:
        models_to_test["xgb"] = "XGBoost"
        
    estimators = {}
    calibrated_models = {}
    oof_predictions = {}
    cv_metrics = {}
    
    print("\n--- Step 8: Spatial Block CV & Hyperparameter Tuning ---")
    for m_code, m_name in models_to_test.items():
        print(f"\n[Model: {m_name}] Tuning on 5 Spatial Block Folds...")
        best_est = run_hyperparameter_tuning(X_train, y_train, groups_train, m_code)
        estimators[m_code] = best_est
        
        cal_model, oof_probs, metrics = evaluate_and_calibrate_model(best_est, X_train, y_train, groups_train)
        calibrated_models[m_code] = cal_model
        oof_predictions[m_code] = oof_probs
        cv_metrics[m_code] = metrics
        print(f"-> {m_name} Calibrated Spatial CV -> ROC-AUC: {metrics['auc']:.4f}, Accuracy: {metrics['acc']:.4f}, F1: {metrics['f1']:.4f}, Brier: {metrics['brier']:.4f}")
        
    # Soft Voting Ensemble
    ensemble_oof = np.mean([oof_predictions[m_code] for m_code in models_to_test.keys()], axis=0)
    ensemble_preds = (ensemble_oof >= 0.5).astype(int)
    cv_metrics["ensemble"] = {
        "acc": float(accuracy_score(y_train, ensemble_preds)),
        "f1": float(f1_score(y_train, ensemble_preds, zero_division=0)),
        "auc": float(roc_auc_score(y_train, ensemble_oof)),
        "brier": float(brier_score_loss(y_train, ensemble_oof))
    }
    print(f"\n-> Ensemble Spatial CV -> ROC-AUC: {cv_metrics['ensemble']['auc']:.4f}, Accuracy: {cv_metrics['ensemble']['acc']:.4f}, F1: {cv_metrics['ensemble']['f1']:.4f}")
    
    # Save CV comparison table
    comp_rows = []
    for m_code, m_name in models_to_test.items():
        comp_rows.append({
            "Model": m_name,
            "CV ROC-AUC": cv_metrics[m_code]["auc"],
            "CV Accuracy": cv_metrics[m_code]["acc"],
            "CV F1": cv_metrics[m_code]["f1"],
            "CV Brier Score": cv_metrics[m_code]["brier"]
        })
    comp_rows.append({
        "Model": "Ensemble (Calibrated Average)",
        "CV ROC-AUC": cv_metrics["ensemble"]["auc"],
        "CV Accuracy": cv_metrics["ensemble"]["acc"],
        "CV F1": cv_metrics["ensemble"]["f1"],
        "CV Brier Score": cv_metrics["ensemble"]["brier"]
    })
    df_compare = pd.DataFrame(comp_rows)
    df_compare.to_csv(os.path.join("outputs", "metrics", "cv_model_comparison.csv"), index=False)
    
    # Select best model
    best_model_code = max(models_to_test.keys(), key=lambda k: cv_metrics[k]["auc"])
    best_model_name = models_to_test[best_model_code]
    final_model = calibrated_models[best_model_code]
    best_oof_probs = oof_predictions[best_model_code]
    
    # 9. Youden's J Threshold Optimization
    optimal_threshold = optimize_threshold(y_train, best_oof_probs)
    print(f"\n=== Selected Champion Model: {best_model_name.upper()} ===")
    print(f"Spatial CV ROC-AUC: {cv_metrics[best_model_code]['auc']:.4f}")
    print(f"Optimal Youden J Decision Threshold: {optimal_threshold:.2f}")
    
    # Save model and metadata
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "best_model.pkl")
    model_data = {
        "model_type": best_model_code,
        "model_name": best_model_name,
        "model": final_model,
        "features": feature_cols
    }
    joblib.dump(model_data, model_path)
    joblib.dump(model_data, os.path.join("models", "trained_model.pkl"))
    
    metadata = {
        "training_period": f"{train_start}-{train_end}",
        "test_period": "2016-2020",
        "features": feature_cols,
        "model_type": best_model_name,
        "threshold": optimal_threshold,
        "random_state": config.get('model', {}).get('random_state', 42),
        "cv_roc_auc": cv_metrics[best_model_code]["auc"],
        "cv_accuracy": cv_metrics[best_model_code]["acc"]
    }
    with open(os.path.join("models", "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"\nTrained model and metadata successfully saved!")

if __name__ == "__main__":
    main()
