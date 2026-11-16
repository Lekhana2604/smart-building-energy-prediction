"""
evaluate_models.py
===================
Evaluate the three trained models on the test set.
In addition to regression metrics (MAE, RMSE, R²) we also compute
binary‑classification metrics (accuracy, precision, recall, F1).
The binary label is defined as whether the daily consumption is **above the
median** consumption of the test period.
The script writes a detailed CSV to `results/tables/06_evaluation_summary.csv`.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# ---------------------------------------------------------------------------
# Paths (relative to project root)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_PARQUET = PROJECT_ROOT / "data" / "processed" / "features_dataset.parquet"
DESIGN_CSV = PROJECT_ROOT / "data" / "processed" / "experimental_design_info.csv"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_CSV = PROJECT_ROOT / "results" / "tables" / "06_evaluation_summary.csv"
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

TARGET_COL = "meter_reading_daily"

def split_by_date(df: pd.DataFrame):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    train = (df["date"] >= pd.to_datetime("2016-01-01").date()) & (df["date"] <= pd.to_datetime("2016-12-31").date())
    val   = (df["date"] >= pd.to_datetime("2017-01-01").date()) & (df["date"] <= pd.to_datetime("2017-06-30").date())
    test  = (df["date"] >= pd.to_datetime("2017-07-01").date()) & (df["date"] <= pd.to_datetime("2017-12-31").date())
    return train, val, test

def load_data():
    df = pd.read_parquet(FEATURES_PARQUET)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    design = pd.read_csv(DESIGN_CSV)
    df = df.merge(design, on="building_id", how="left")
    train_mask, val_mask, test_mask = split_by_date(df)
    return df[test_mask].reset_index(drop=True)

def eval_model(model_path: Path, X_test: pd.DataFrame, y_true: pd.Series):
    model = joblib.load(model_path)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_true, preds)
    rmse = np.sqrt(mean_squared_error(y_true, preds))
    r2 = r2_score(y_true, preds)
    # Binary classification: above median consumption?
    median_val = y_true.median()
    y_bin = (y_true >= median_val).astype(int)
    pred_bin = (preds >= median_val).astype(int)
    acc = accuracy_score(y_bin, pred_bin)
    prec = precision_score(y_bin, pred_bin, zero_division=0)
    rec = recall_score(y_bin, pred_bin, zero_division=0)
    f1 = f1_score(y_bin, pred_bin, zero_division=0)
    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
    }

def main():
    print("Loading test data …")
    test_df = load_data()
    # Feature columns (exclude identifiers, target, treatment)
    feature_cols = [c for c in test_df.columns if c not in ["building_id", "site_id", "date", TARGET_COL, "treatment"]]
    X_test = test_df[feature_cols]
    y_test = test_df[TARGET_COL]

    results = []
    for name in ["DecisionTree", "RandomForest", "GradientBoosting"]:
        model_file = MODELS_DIR / f"{name.lower()}_model.pkl"
        if not model_file.exists():
            print(f"⚠️  Model file missing: {model_file}")
            continue
        print(f"Evaluating {name} …")
        metrics = eval_model(model_file, X_test, y_test)
        metrics["Model"] = name
        results.append(metrics)
        print(f"  Done – MAE {metrics['MAE']:.2f}, R2 {metrics['R2']:.4f}, F1 {metrics['F1']:.4f}")

    if results:
        out_df = pd.DataFrame(results)
        out_df.to_csv(OUTPUT_CSV, index=False)
        print("\nEvaluation summary saved →", OUTPUT_CSV.relative_to(PROJECT_ROOT))
    else:
        print("⚠️  No results to write.")

if __name__ == "__main__":
    main()
