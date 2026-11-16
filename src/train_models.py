"""
train_models.py
==============
Train the three models defined in the Randomized Block Design (RBD).
* **DecisionTree** – baseline (control)
* **RandomForest** – experimental
* **GradientBoosting** – experimental

The script:
1. Loads the feature dataset (`features_dataset.parquet`).
2. Loads the block‑treatment mapping (`experimental_design_info.csv`).
3. Performs a **chronological split**:
   - Train period   : 2016‑01‑01 → 2016‑12‑31
   - Validation period: 2017‑01‑01 → 2017‑06‑30
   - Test period    : 2017‑07‑01 → 2017‑12‑31
   (All splits are *time‑aware* to avoid leakage.)
4. For each treatment, selects the buildings assigned to that treatment and
   trains the corresponding Scikit‑learn model **only on the train data**.
5. Performs a quick hyper‑parameter search on the *validation* set (grid
   search with 2‑3 candidate values).  The best parameters are then used to
   **re‑fit** the model on the combined train+validation data.
6. Saves the final model objects (`*.pkl`) in `models/`.
7. Writes a short summary CSV (`results/tables/05_model_training_summary.csv`).

Run the script from the project root:
```powershell
cd "C:\Users\ADMIN\Desktop\smart-building-energy-prediction"
python src/train_models.py
```
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import sys
import joblib
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------------------------
# Paths (relative to project root)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_PARQUET = PROJECT_ROOT / "data" / "processed" / "features_dataset.parquet"
DESIGN_CSV = PROJECT_ROOT / "data" / "processed" / "experimental_design_info.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_CSV = PROJECT_ROOT / "results" / "tables" / "05_model_training_summary.csv"
SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)

TARGET_COL = "meter_reading_daily"

# ---------------------------------------------------------------------------
# Helper: chronological split
# ---------------------------------------------------------------------------
def split_by_date(df: pd.DataFrame):
    """Return train/val/test masks based on the date column.
    Dates are inclusive on each interval.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    train_mask = (df["date"] >= pd.to_datetime("2016-01-01").date()) & (df["date"] <= pd.to_datetime("2016-12-31").date())
    val_mask   = (df["date"] >= pd.to_datetime("2017-01-01").date()) & (df["date"] <= pd.to_datetime("2017-06-30").date())
    test_mask  = (df["date"] >= pd.to_datetime("2017-07-01").date()) & (df["date"] <= pd.to_datetime("2017-12-31").date())
    return train_mask, val_mask, test_mask

# ---------------------------------------------------------------------------
# Model factories & simple hyper‑parameter grids
# ---------------------------------------------------------------------------
MODEL_INFO = {
    "DecisionTree": {
        "factory": DecisionTreeRegressor,
        "grid": {"max_depth": [None, 10, 20]},
    },
    "RandomForest": {
        "factory": RandomForestRegressor,
        "grid": {"n_estimators": [200, 400], "max_depth": [None, 15]},
    },
    "GradientBoosting": {
        "factory": GradientBoostingRegressor,
        "grid": {"n_estimators": [200, 400], "learning_rate": [0.1, 0.05]},
    },
}

def train_and_select(model_name: str, X_train, y_train, X_val, y_val, random_state: int = 42):
    info = MODEL_INFO[model_name]
    model_cls = info["factory"]
    param_grid = info["grid"]

    # Use GridSearchCV with 3‑fold CV on the *training* data only (to keep validation separate)
    # but limit CV to 2 folds for speed – the dataset is large.
    base_model = model_cls(random_state=random_state)
    grid = GridSearchCV(base_model, param_grid, cv=2, scoring="neg_mean_absolute_error", n_jobs=-1)
    grid.fit(X_train, y_train)
    best_params = grid.best_params_

    # Re‑fit on train+validation using best params
    X_comb = pd.concat([X_train, X_val])
    y_comb = pd.concat([y_train, y_val])
    final_model = model_cls(**best_params, random_state=random_state)
    final_model.fit(X_comb, y_comb)
    return final_model, best_params

def evaluate(model, X, y):
    preds = model.predict(X)
    mae = mean_absolute_error(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))
    r2 = r2_score(y, preds)
    return mae, rmse, r2

def main() -> int:
    # -------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------
    print("Loading feature data …")
    df = pd.read_parquet(FEATURES_PARQUET)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    print(f"  Feature rows: {len(df):,}")

    print("Loading experimental design mapping …")
    design = pd.read_csv(DESIGN_CSV)
    # Merge treatment info into the feature dataframe
    df = df.merge(design, on="building_id", how="left")
    if df["treatment"].isna().any():
        print("⚠️  Some buildings have no treatment – they will be ignored.")
    # -------------------------------------------------------------------
    # Split dates (same for all models)
    # -------------------------------------------------------------------
    train_mask, val_mask, test_mask = split_by_date(df)
    df_train = df[train_mask]
    df_val   = df[val_mask]
    df_test  = df[test_mask]
    print(f"Train period rows: {len(df_train):,}")
    print(f"Validation period rows: {len(df_val):,}")
    print(f"Test period rows: {len(df_test):,}")

    # Features – drop identifiers and target
    feature_cols = [c for c in df.columns if c not in ["building_id", "site_id", "date", TARGET_COL, "treatment"]]

    # -------------------------------------------------------------------
    # Train each treatment on its assigned buildings
    # -------------------------------------------------------------------
    summary_records = []
    for treatment, info in MODEL_INFO.items():
        print(f"\n=== Training model: {treatment} ===")
        # Select buildings assigned to this treatment
        train_ids = design[design["treatment"] == treatment]["building_id"]
        val_ids   = design[design["treatment"] == treatment]["building_id"]  # same set for val/test
        # Filter rows
        X_train = df_train[df_train["building_id"].isin(train_ids)][feature_cols]
        y_train = df_train[df_train["building_id"].isin(train_ids)][TARGET_COL]
        X_val   = df_val[df_val["building_id"].isin(val_ids)][feature_cols]
        y_val   = df_val[df_val["building_id"].isin(val_ids)][TARGET_COL]
        X_test  = df_test[df_test["building_id"].isin(val_ids)][feature_cols]
        y_test  = df_test[df_test["building_id"].isin(val_ids)][TARGET_COL]

        if X_train.empty:
            print(f"⚠️  No training data for {treatment} – skipping.")
            continue

        model, best_params = train_and_select(treatment, X_train, y_train, X_val, y_val)
        mae, rmse, r2 = evaluate(model, X_test, y_test)
        print(f"  Test MAE  : {mae:,.2f} kWh/day")
        print(f"  Test RMSE : {rmse:,.2f} kWh/day")
        print(f"  Test R²   : {r2:.4f}")
        # Save model
        model_path = MODELS_DIR / f"{treatment.lower()}_model.pkl"
        joblib.dump(model, model_path)
        print(f"  Model saved to {model_path.relative_to(PROJECT_ROOT)}")
        # Record summary
        summary_records.append({
            "Treatment": treatment,
            "ModelFile": str(model_path.relative_to(PROJECT_ROOT)),
            "BestParams": best_params,
            "Test_MAE": mae,
            "Test_RMSE": rmse,
            "Test_R2": r2,
            "TrainRows": len(X_train),
            "ValRows": len(X_val),
            "TestRows": len(X_test),
        })

    # -------------------------------------------------------------------
    # Save summary CSV
    # -------------------------------------------------------------------
    if summary_records:
        summary_df = pd.DataFrame(summary_records)
        summary_df.to_csv(SUMMARY_CSV, index=False)
        print("\nTraining summary saved →", SUMMARY_CSV.relative_to(PROJECT_ROOT))
    else:
        print("⚠️  No models were trained – check the treatment mapping.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
