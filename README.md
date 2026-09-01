# AI/ML-Based Tiger Habitat Prediction System — India (2000–2050)

A professional geospatial machine learning prototype designed to model and project tiger habitat suitability in India. The system integrates historical tiger occurrence coordinates (represented by species range centroids) and monthly WorldClim climate rasters (Precipitation, Maximum Temperature, Minimum Temperature), utilizing a spatial-temporal modeling approach.

---

## 1. Project Architecture

```text
tiger-habitat-prediction/
│
├── app.py                     # Streamlit Main Welcome / Landing Page
│
├── pages/                     # Streamlit Dashboard Pages
│   ├── 1_Habitat_Prediction.py# Forecast maps, Year Slider, Scenarios
│   ├── 2_Model_Performance.py # Validation and unseen test metrics
│   └── 3_Data_Quality.py      # Data checklist, bounds, and ISRO metadata
│
├── data/                      # Raw and Processed Datasets
│   ├── raw/                   # (Workspace root directories)
│   └── processed/             # Cleaned occurrences, aggregated annual rasters
│
├── models/                    # Serialized models and standard scalers
│   ├── trained_model.pkl      # Pickled Best ML Model (Random Forest)
│   └── scaler.pkl             # Fitted StandardScaler
│
├── outputs/                   # Exported evaluation metrics and forecasts
│   ├── predictions/           # Forecast grid CSVs (2020-2050)
│   └── metrics/               # Model accuracy reports, confusion matrices
│
├── src/                       # Modulized ML Pipeline Scripts
│   ├── data_loader.py         # GeoJSON loading utilities
│   ├── preprocessing.py       # Centroid extraction & cleaning
│   ├── spatial_processing.py  # Climate aggregation, cropping, grid generation
│   ├── pseudo_absence.py      # Spatial background point generation
│   ├── feature_engineering.py  # Derived features and scaling
│   ├── train.py               # Spatial block cross-validation and training
│   ├── evaluate.py            # Strict test years (2016-2020) evaluation
│   └── predict.py             # Future projections generator
│
├── config.yaml                # Global config parameters (Thresholds, Resolutions)
├── DATA_DICTIONARY.md         # Full schema, columns, and data details
├── requirements.txt           # Pip dependencies
├── environment.yml            # Conda environment file
└── README.md                  # Installation and methodology documentation
```

---

## 2. Installation & Setup

### Prerequisites
- Python 3.10, 3.11, or 3.12 (standard 64-bit).
- Virtual environment support.

### Windows Setup
1. Open PowerShell or Command Prompt in the project workspace:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Linux / macOS Setup
1. Open terminal:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Conda Option
```bash
conda env create -f environment.yml
conda activate tiger-habitat
```

---

## 3. Running the Pipeline

The project features a highly modular pipeline. Pre-calculated models and forecast grids are saved to disk, so the Streamlit dashboard loads instantly without retraining.

### Step 1: Preprocess and Train Model
This command cleans occurrences, aggregates monthly climate layers, generates pseudo-absences, runs 5-fold Spatial Block Cross-Validation, selects the best model (Random Forest), and saves it to `models/trained_model.pkl`:
```bash
python -m src.train
```

### Step 2: Run Evaluation
Generates overall and year-wise metrics on unseen test years (2016–2020) and computes feature importances:
```bash
python -m src.evaluate
```

### Step 3: Run Future Projections
Pre-generates gridded forecasts for years 2020–2050 at 5-year intervals under all climate scenarios:
```bash
python -m src.predict
```

### Step 4: Run Streamlit Application
Launches the interactive dashboard:
```bash
streamlit run app.py
```

---

## 4. Machine Learning & Modeling Methodology

### Spatial Centroid Extraction (Presence = 1)
Since occurrences are stored as range polygons (`scl_species_XXXX.geojson`), we extract the **centroids** of the features representing India (`country == "India"`) to get latitude/longitude coordinate points. 

### Pseudo-Absence Generation (Presence = 0)
Background points are generated randomly inside the administrative bounds of India (union of state boundaries in `scl_states_XXXX.geojson`). To avoid false negatives, a **0.2-degree (~22 km) buffer** is applied around all known presence coordinates, and pseudo-absences are only sampled outside these buffers. The default presence-to-absence ratio is **1:2**.

### Temporal Alignment
For an observation in year $Y$, climate variables correspond to the aggregated annual averages/sums of monthly WorldClim files for year $Y$. Because climate data is only available up to 2019, observations in 2020 use the **2019 climate layers** as the closest temporal proxy.

### Spatial Block Cross-Validation
Ecological spatial datasets are highly autocorrelated, which causes random cross-validation to overfit. We divide the coordinates into **2° × 2° spatial grid cells**. These grid cell IDs are used as group labels for a `GroupKFold` split, ensuring that spatial blocks in the validation set remain completely unseen during training.

### Selected Model
Evaluated using Spatial block CV on the training period (2001–2015):
- **Random Forest Classifier**: Selected as the final predictor because it achieved a significantly higher Validation F1-score (`0.481` vs. `0.338` for XGBoost) and demonstrated better robustness to spatial block partitions.

---

## 5. Performance Results (Unseen Test Years 2016–2020)

- **Overall Test Accuracy**: `67.5%`
- **ROC-AUC**: `0.784`
- **F1-Score**: `61.3%`
- **Recall (Sensitivity)**: `77.4%` (flagging a high proportion of actual occurrences)
- **Specificity**: `62.5%`
- **Brier Score**: `0.191`

### Year-by-Year Performance Breakdown

| Test Year | Accuracy | Precision | Recall (Sensitivity) | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **2016** | 70.9% | 53.6% | 78.7% | 63.7% | 0.788 |
| **2017** | 73.1% | 56.2% | 85.1% | 67.8% | 0.842 |
| **2018** | 67.3% | 50.8% | 82.5% | 62.7% | 0.794 |
| **2019** | 63.9% | 46.9% | 74.6% | 57.5% | 0.752 |
| **2020** | 65.2% | 48.0% | 74.3% | 58.3% | 0.754 |

---

## 6. Limitations & Future Research Extensions

### Limitations
1. **Scenario-Based Projections**: Projections for 2020–2050 are simulated scenario shifts (linear adjustments of baseline climate variables) and do not represent validated future IPCC CMIP6 models.
2. **Coarse Resolution**: Global 2.5 arc-minute rasters (approx. 4.5 km) do not capture micro-climates or local corridors.
3. **Absence of Anthropogenic Factors**: The prototype only models climate variables; it does not represent land cover changes, road networks, human-wildlife conflict, or poaching.

### Future Research Extensions
- **Environmental Variables**: Incorporate Elevation (DEM), Slope, NDVI (vegetation greenness), and Forest Cover fragmentation.
- **Anthropogenic Variables**: Integrate Human Footprint Index, Population Density, road buffer distances, and Protected Area boundaries.
- **Future Climate Models**: Connect CMIP6 climate model projections (such as SSP2-4.5 and SSP5-8.5 scenarios) for 2021–2050.
- **Model Ensembles**: Evaluate MaxEnt, LightGBM, and SHAP (Shapley Additive exPlanations) for explainable AI.
