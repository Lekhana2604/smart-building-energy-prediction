"""
train_models_v2.py
=================
Train the three models defined in the Randomized Block Design (RBD).
* DecisionTree – baseline (control)
* RandomForest – experimental
* GradientBoosting – experimental

The script:
1. Loads the feature dataset (`features_dataset.parquet`).
2. Loads the block‑treatment mapping (`experimental_design_info.csv`).
3. Performs a chronological split (train: 2016, val: H1‑2017, test: H2‑2017).
4. For each treatment, selects the buildings assigned to it and trains the respective Scikit‑learn model.
5. Runs a small grid‑search on the validation set, then re‑fits on train+val.
6. Saves the final model objects (`*.pkl`) in `models/`.
7. Writes a summary CSV (`results/tables/05_model_training_summary.csv`).

Run from the project root:
```powershell
cd "C:\\Users\\ADMIN\\Desktop\\smart-building-energy-prediction"
python src/train_models_v2.py
```
"""

import pandas as pd
import numpy as np
from pathlib import Path
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
def split_by_date(df: pd.DataFrame):
    """Return boolean masks for train/val/test based on the `date` column (inclusive)."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    train = (df["date"] >= pd.to_datetime("2016-01-01").date()) & (df["date"] <= pd.to_datetime("2016-12-31").date())
    val   = (df["date"] >= pd.to_datetime("2017-01-01").date()) & (df["date"] <= pd.to_datetime("2017-06-30").date())
    test  = (df["date"] >= pd.to_datetime("2017-07-01").date()) & (df["date"] <= pd.to_datetime("2017-12-31").date())
    return train, val, test

# Model factories and simple hyper‑parameter grids
MODEL_INFO = {
    "DecisionTree": {
        "factory": DecisionTreeRegressor,
        "grid": {"max_depth": [None, 10, 20]},
    },
    "RandomForest": {
        "factory": RandomForestRegressor,
        "grid": {"n_estimators": [100], "max_depth": [10]},
    },
    "GradientBoosting": {
        "factory": GradientBoostingRegressor,
        "grid": {"n_estimators": [200, 400], "learning_rate": [0.1, 0.05]},
    },
}

def train_and_select(model_name, X_train, y_train, X_val, y_val, random_state=42):
    info = MODEL_INFO[model_name]
    base = info["factory"](random_state=random_state)
    grid = GridSearchCV(base, info["grid"], cv=2, scoring="neg_mean_absolute_error", n_jobs=-1)
    grid.fit(X_train, y_train)
    best = grid.best_params_
    # Re‑fit on train+val
    X_comb = pd.concat([X_train, X_val])
    y_comb = pd.concat([y_train, y_val])
    final = info["factory"](**best, random_state=random_state)
    final.fit(X_comb, y_comb)
    return final, best

def evaluate(model, X, y):
    preds = model.predict(X)
    mae = mean_absolute_error(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))
    r2 = r2_score(y, preds)
    return mae, rmse, r2

def main():
    print("Loading feature data …")
    df = pd.read_parquet(FEATURES_PARQUET)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    print(f"  Total rows: {len(df):,}")
    design = pd.read_csv(DESIGN_CSV)
    df = df.merge(design, on="building_id", how="left")
    if df["treatment"].isna().any():
        print("⚠️ Some buildings have no treatment – they will be ignored.")
    train_mask, val_mask, test_mask = split_by_date(df)
    df_train, df_val, df_test = df[train_mask], df[val_mask], df[test_mask]
    print(f"Train rows: {len(df_train):,}, Val rows: {len(df_val):,}, Test rows: {len(df_test):,}")
    # feature columns (exclude identifiers & target & treatment)
    feature_cols = [c for c in df.columns if c not in ["building_id", "site_id", "date", TARGET_COL, "treatment"]]
    summary = []
    for treatment in MODEL_INFO.keys():
        print(f"\n=== Training {treatment} model ===")
        ids = design[design["treatment"] == treatment]["building_id"]
        X_tr = df_train[df_train["building_id"].isin(ids)][feature_cols]
        y_tr = df_train[df_train["building_id"].isin(ids)][TARGET_COL]
        X_val = df_val[df_val["building_id"].isin(ids)][feature_cols]
        y_val = df_val[df_val["building_id"].isin(ids)][TARGET_COL]
        X_test = df_test[df_test["building_id"].isin(ids)][feature_cols]
        y_test = df_test[df_test["building_id"].isin(ids)][TARGET_COL]
        if X_tr.empty:
            print(f"⚠️ No data for {treatment} – skipping.")
            continue
        model, best = train_and_select(treatment, X_tr, y_tr, X_val, y_val)
        mae, rmse, r2 = evaluate(model, X_test, y_test)
        print(f"Test MAE: {mae:,.2f}, RMSE: {rmse:,.2f}, R²: {r2:.4f}")
        model_path = MODELS_DIR / f"{treatment.lower()}_model.pkl"
        joblib.dump(model, model_path)
        print(f"Model saved: {model_path.relative_to(PROJECT_ROOT)}")
        summary.append({
            "Treatment": treatment,
            "ModelFile": str(model_path.relative_to(PROJECT_ROOT)),
            "BestParams": best,
            "Test_MAE": mae,
            "Test_RMSE": rmse,
            "Test_R2": r2,
            "TrainRows": len(X_tr),
            "ValRows": len(X_val),
            "TestRows": len(X_test),
        })
    if summary:
        pd.DataFrame(summary).to_csv(SUMMARY_CSV, index=False)
        print("\nTraining summary saved →", SUMMARY_CSV.relative_to(PROJECT_ROOT))
    else:
        print("⚠️ No models trained.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
