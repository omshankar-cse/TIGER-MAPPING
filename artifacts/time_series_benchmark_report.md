# Time Series Environmental Forecasting Benchmark Report

## Executive Summary
This benchmark systematically compares three distinct architectural generations for forecasting multi-year macro-environmental climate trajectories (**Precipitation** in mm and **Mean Temperature** in °C) across India's tiger habitat landscape:
1. **Traditional Statistical**: Auto-Regressive Integrated Moving Average (ARIMA)
2. **Machine Learning**: XGBoost Regressor with autoregressive lag features ($t-1, t-2, t-3$)
3. **Modern Deep Learning**: PyTorch Recurrent Neural Network (LSTM)

Strict temporal isolation is enforced throughout the experimental pipeline: **2001–2015** (15 annual cycles) constitutes the training horizon, and **2016–2020** (5 annual cycles) forms the unseen out-of-sample holdout test horizon.

---

## 1. Experimental Setup & Temporal Isolation

- **Dataset**: `data/processed/model_dataset_processed.csv` (Aggregated nationally across all habitat coordinates by year)
- **Temporal Granularity**: Annual aggregates (2001 to 2020, $N=20$)
- **Training Period**: 2001 – 2015 ($N_{train} = 15$)
- **Testing Period**: 2016 – 2020 ($N_{test} = 5$)
- **Feature Engineering**: Autoregressive temporal lag features ($x_{t-1}, x_{t-2}, x_{t-3}$) with zero future-lookahead leakage
- **Forecast Horizon**: 5-step recursive out-of-sample forecast ($t+1$ through $t+5$)

---

## 2. Model Performance Benchmark Table

### Precipitation Forecasting (mm)
| Model Generation | Architecture / Order | RMSE (mm) | MAE (mm) | MAPE (%) | Performance Rank |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Machine Learning** | XGBoost Regressor (depth=3, n=100) | **165.892** | **149.390** | **11.89%** | 🥇 Best |
| **Modern Deep Learning** | PyTorch LSTM (seq=3, hidden=32) | 197.302 | 153.140 | 12.20% | 🥈 Runner-up |
| **Traditional Statistical** | ARIMA(0, 1, 1) | 208.597 | 160.722 | 11.89% | 🥉 Third |

### Mean Temperature Forecasting (°C)
| Model Generation | Architecture / Order | RMSE (°C) | MAE (°C) | MAPE (%) | Performance Rank |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Traditional Statistical** | ARIMA(0, 1, 2) | **0.207** | **0.190** | **0.79%** | 🥇 Best |
| **Modern Deep Learning** | PyTorch LSTM (seq=3, hidden=32) | 0.382 | 0.329 | 1.38% | 🥈 Runner-up |
| **Machine Learning** | XGBoost Regressor (depth=3, n=100) | 0.425 | 0.406 | 1.70% | 🥉 Third |

---

## 3. Detailed Yearly Predictions vs Ground Truth (2016–2020)

### Precipitation (mm)
| Year | Actual Value | ARIMA Forecast | XGBoost Forecast | PyTorch LSTM Forecast |
| :---: | :---: | :---: | :---: | :---: |
| **2016** | 1164.38 | 1156.07 | 1218.31 | 1345.29 |
| **2017** | 1256.46 | 1134.49 | 1185.10 | 1157.12 |
| **2018** | 1072.80 | 1112.91 | 1289.04 | 968.09 |
| **2019** | 1424.16 | 1091.33 | 1203.35 | 1418.98 |
| **2020** | 1370.14 | 1069.75 | 1185.51 | 994.57 |

### Mean Temperature (°C)
| Year | Actual Value | ARIMA Forecast | XGBoost Forecast | PyTorch LSTM Forecast |
| :---: | :---: | :---: | :---: | :---: |
| **2016** | 23.901 | 24.051 | 24.444 | 24.299 |
| **2017** | 23.876 | 23.991 | 24.390 | 24.303 |
| **2018** | 23.649 | 23.962 | 24.110 | 24.219 |
| **2019** | 24.045 | 23.933 | 23.788 | 24.047 |
| **2020** | 24.163 | 23.904 | 23.906 | 23.914 |

---

## 4. Key Architectural Insights & Findings

1. **Machine Learning (XGBoost Regressor)**:
   - XGBoost achieved strong accuracy on Precipitation (RMSE: 165.89 mm, MAE: 149.39 mm), effectively capturing non-linear precipitation swings based on three-year rainfall patterns.
2. **Traditional Statistical (ARIMA)**:
   - ARIMA demonstrated exceptional parsimony on Mean Temperature (RMSE: 0.2067 °C, MAE: 0.1899 °C), successfully capturing low-variance thermal autocorrelation without risk of neural overfitting on a compact dataset.
3. **Deep Learning Sequence Modeling (LSTM)**:
   - PyTorch LSTM with hidden gated states modeled gradual multi-year dynamics across both variables, offering a balanced architecture that scales effectively to higher-frequency spatiotemporal inputs.
4. **Spatial Pipeline Preservation**:
   - The time series benchmark operates autonomously in `src/time_series_benchmark.py` and does not alter or disrupt the spatial classification models or pipelines in `src/train.py` or `src/evaluate.py`.
