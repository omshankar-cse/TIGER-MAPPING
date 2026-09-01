# Model Audit Report: Tiger Habitat Prediction

This document presents a comprehensive audit of the baseline machine learning and geospatial pipeline for tiger habitat suitability modeling in India.

---

## 1. Baseline Model Specifications

### Current Model
- **Algorithm**: Random Forest Classifier (or HistGradientBoostingClassifier fallback).
- **Hyperparameters**: `n_estimators=100`, `max_depth=10`, `class_weight='balanced'`, `random_state=42`.

### Current Features
- **Raw Climatic features**: `Precipitation`, `TMAX` (Max Temperature), `TMIN` (Min Temperature).
- **Engineered features**: `Mean_Temperature` (`(TMAX+TMIN)/2`), `Temperature_Range` (`TMAX-TMIN`), `log_precipitation` (`log1p(Precipitation)`).
- *Note*: No physical land-cover, topography (DEM), or anthropogenic (human footprint) rasters are available in the project directories.

---

## 2. Dataset Statistics

### Sample Sizes & Class Distribution
- **Total Records**: 3,522 rows.
- **Presence Class (1)**: 1,174 records (33.3%).
- **Pseudo-Absence Class (0)**: 2,348 records (66.7%).
- **Imbalance Ratio**: 1 : 2 (Configured via `pseudo_absence.ratio = 2`).

### Missing Values
- **Percentage of missing values**: 0.0% in the final modeling dataset.
- *Imputation mechanism*: Median values are calculated per climate variable and filled in where coordinates fall outside raster coverage.

### Feature Correlations
- `TMAX`, `TMIN`, and `Mean_Temperature` are highly collinear ($r > 0.90$).
- `Precipitation` and `log_precipitation` are highly collinear ($r > 0.85$).
- High collinearity can inflate feature importances in standard Random Forests and cause coefficient instability in linear models.

---

## 3. Spatial & Temporal Distribution

### Spatial Distribution
- Presence centroids are clustered in five major tiger landscapes in India:
  1. Shivalik Hills & Gangetic Plains (Terai belt)
  2. Central India (Madhya Pradesh, Chhattisgarh, Maharashtra)
  3. Western Ghats (Karnataka, Kerala, Tamil Nadu)
  4. Eastern Ghats & Eastern Plains
  5. Sundarbans & Northeast Hills
- Significant spatial clustering represents *sampling bias* rather than pure habitat occupancy.

### Temporal Distribution
- **Training Period**: 2001–2015 (strictly used for fitting standard scalers, model cross-validation, and training).
- **Testing Period**: 2016–2020 (kept completely unseen during feature engineering, scale fitting, and model selection).

---

## 4. Modeling Methodology Audit

### Cross-Validation Strategy
- **Baseline**: 5-fold Spatial Block Cross-Validation using `GroupKFold` split on 2.0° × 2.0° block groups.
- *Strengths*: Highly effective at preventing spatial autocorrelation leakage (points in the same grid block are kept together in training or validation).
- *Weaknesses*: Hyperparameters and decision thresholds are not tuned during CV.

### Data Leakage Risks
- **Spatial Leakage**: If validation blocks are too small, nearby autocorrelated points leak environment values. Currently set to 2.0° (~220 km), which is adequate.
- **Temporal Leakage**: Occurrences and climate aggregates must be aligned exactly by year. Currently aligned correctly (Year $Y$ occurrences matched with Year $Y$ annual climate).
- **Pre-scaling Leakage**: Scaling fitted on the combined train+test set would leak variance. The baseline fits `StandardScaler` strictly on 2001-2015 and transforms both, which is correct.

---

## 5. Major Problems Discovered

1. **Spatial Sampling Bias**: Clustered presence points cause the tree models to learn specific coordinate zones rather than the environmental suitability. Spatial thinning is not currently implemented.
2. **Pseudo-Absence Contamination**:
   - Pseudo-absences are generated annually. A point selected as absence in 2010 might be within a few kilometers of a presence point in 2011, introducing label noise.
   - The spatial exclusion buffer is only 0.2° (~22 km) around presence points of *that specific year*, leaving other years' presences unbuffered.
3. **Absence of Hyperparameter Optimization**: The baseline models use default or arbitrary hyperparameters without systematic tuning (e.g. max_depth, learning_rate, min_samples_leaf).
4. **Arbitrary Suitability Thresholding**: The binary classification threshold is hard-coded to `0.5` instead of optimizing for Youden's J or balanced accuracy.
5. **No Probability Calibration**: Classifiers (especially XGBoost or shallow Random Forests) output uncalibrated probabilities that do not correspond directly to real-world occurrence rates.
6. **No Spatial Model Auditing**: We do not check whether the model is simply memorizing coordinates (latitude/longitude) rather than learning environmental ecology.

---

## 6. Proposed Improvements Plan

1. **Implement Spatial Thinning**: Filter presence points to ensure no two occurrences are closer than 5 km within the same year, reducing sampling bias.
2. **Rebuild Pseudo-Absence Generation**:
   - Exclude a strict 20 km (~0.18°) buffer around *all* presence coordinates (across all training years) rather than just the current year.
   - Support multiple configurable strategies in `config.yaml` (`random`, `accessible_area` using survey boundaries, and `buffered_m` using a large ecological accessible buffer around presences).
   - Test background ratios from 1:1 to 1:5.
3. **Train and Compare Multiple Models**:
   - Compare `RandomForestClassifier`, `XGBClassifier`, `ExtraTreesClassifier`, and `HistGradientBoostingClassifier`.
   - Create an Ensemble model that aggregates predictions using weights based on CV performance.
4. **Implement Hyperparameter Optimization**:
   - Use `RandomizedSearchCV` with GroupKFold split to tune model depth, leaf nodes, estimators, and learning rates.
5. **Perform Geographic Memorization Test**:
   - Train Model A (Climate features only) and Model B (Climate + Coordinates).
   - Verify that Model A retains generalization capability on the unseen test set, and prioritize it to prevent spatial overfitting.
6. **Optimize and Calibrate Probability Thresholds**:
   - Apply `CalibratedClassifierCV` on the selected best estimator.
   - Find the optimal probability threshold using Youden's J-index on the training CV predictions.
   - Save the threshold to `models/model_metadata.json`.
7. **Refactor Pipeline Commands**:
   - Retain modular scripts in `src/`.
   - Add `src/pipeline.py` to run the entire data cleaning, thinning, CV tuning, training, testing, and future projection sequence.
