# Methodology

## Data Preprocessing (Phase 3)
- Loaded raw hourly meter readings from `data/raw/` using `src/data_loader.py`.
- Handled missing values: rows with >10 % missing data were removed; remaining gaps filled with linear interpolation.
- Resolved daylight‑saving‑time duplicate timestamps by averaging duplicate rows.
- Saved cleaned hourly data to `data/interim/cleaned_hourly.csv`.

## Daily Aggregation (Phase 4)
- Aggregated hourly consumption to daily totals per building by summing kWh values.
- Produced `data/processed/daily_energy_dataset.parquet` (8 MB) and CSV version.

## Feature Engineering (Phase 5)
- Generated 37 features covering:
  - Temporal (day of week, month, holiday flag)
  - Weather (daily min/max temperature, humidity, precipitation) – joined from external weather CSV.
  - Building characteristics (area, year built, primary use type).
  - Lag features (previous day/week consumption).
- Stored feature matrix in `data/processed/features_dataset.parquet`.

## Experimental Design (Phase 7)
- Applied a **Randomized Block Design (RBD)** to control for block effects (building and month).
- Assigned three treatments (Decision Tree, Random Forest, Gradient Boosting) to blocks using `src/experimental_design.py`.
- Mapping saved in `data/processed/experimental_design_info.csv`.

## Model Training (Phase 8)
- Implemented in `src/train_models_v2.py`.
- Hyper‑parameters:
  - Decision Tree: `max_depth=10`.
  - Random Forest: `n_estimators=100`, `max_depth=10`.
  - Gradient Boosting: `n_estimators=200`, `learning_rate=0.05`.
- Chronological split (70 % train, 15 % validation, 15 % test) to avoid leakage.
- Trained models saved as `models/decisiontree_model.pkl`, `models/randomforest_model.pkl`, `models/gradientboosting_model.pkl`.
- Summary metrics recorded in `results/tables/05_model_training_summary.csv`.

## Model Evaluation (Phase 9)
- Regression metrics: MAE, RMSE, R².
- Binary classification (above‑median consumption) metrics: Accuracy, Precision, Recall, F1‑score.
- Evaluation script `src/evaluate_models.py` and notebook `notebooks/05_model_performance_evaluation.ipynb` generated `results/tables/06_evaluation_summary.csv`.

All code is version‑controlled and reproducible; running the notebooks or scripts from the `src/` directory reproduces the results.
