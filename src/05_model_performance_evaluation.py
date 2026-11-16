"""
05_model_performance_evaluation.py
================================
Standalone script that reproduces the notebook "05_model_performance_evaluation.ipynb".
It loads the test split, evaluates the three trained models (DecisionTree, RandomForest,
GradientBoosting) and prints a summary table with regression and binary‑classification
metrics (accuracy, precision, recall, F1). The results are also saved to
`results/tables/06_evaluation_summary.csv`.

Run from the project root:
    cd "C:\\Users\\ADMIN\\Desktop\\smart-building-energy-prediction"
    python src/05_model_performance_evaluation.py
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
# Paths (relative to the project root)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_PARQUET = PROJECT_ROOT / "data" / "processed" / "features_dataset.parquet"
DESIGN_CSV = PROJECT_ROOT / "data" / "processed" / "experimental_design_info.csv"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_CSV = PROJECT_ROOT / "results" / "tables" / "06_evaluation_summary.csv"
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

TARGET = "meter_reading_daily"

def split_by_date(df: pd.DataFrame):
    """Chronological train/val/test split (inclusive dates)."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    train = (df["date"] >= pd.to_datetime("2016-01-01").date()) & (df["date"] <= pd.to_datetime("2016-12-31").date())
    val   = (df["date"] >= pd.to_datetime("2017-01-01").date()) & (df["date"] <= pd.to_datetime("2017-06-30").date())
    test  = (df["date"] >= pd.to_datetime("2017-07-01").date()) & (df["date"] <= pd.to_datetime("2017-12-31").date())
    return train, val, test

def load_test_data():
    """Load the full feature set, merge treatment mapping, and return only the test portion."""
    df = pd.read_parquet(FEATURES_PARQUET)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    design = pd.read_csv(DESIGN_CSV)
    df = df.merge(design, on="building_id", how="left")
    _, _, test_mask = split_by_date(df)
    test_df = df[test_mask].reset_index(drop=True)
    return test_df

def evaluate_one_model(model_path: Path, X: pd.DataFrame, y: pd.Series):
    model = joblib.load(model_path)
    preds = model.predict(X)
    mae = mean_absolute_error(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))
    r2 = r2_score(y, preds)
    # Binary classification: is consumption >= median of the test set?
    median_val = y.median()
    y_bin = (y >= median_val).astype(int)
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
    test_df = load_test_data()
    # Build feature matrix (drop identifiers, target, and treatment column)
    feature_cols = [c for c in test_df.columns if c not in ["building_id", "site_id", "date", TARGET, "treatment"]]
    X_test = test_df[feature_cols]
    y_test = test_df[TARGET]
    print(f"Test rows: {len(test_df):,} | Features: {len(feature_cols)}")

    results = []
    for model_name in ["DecisionTree", "RandomForest", "GradientBoosting"]:
        model_file = MODELS_DIR / f"{model_name.lower()}_model.pkl"
        if not model_file.is_file():
            print(f"⚠️  Model file missing: {model_file}")
            continue
        print(f"Evaluating {model_name} …")
        metrics = evaluate_one_model(model_file, X_test, y_test)
        metrics["Model"] = model_name
        results.append(metrics)
        print(f"  Done – MAE {metrics['MAE']:.2f}, R2 {metrics['R2']:.4f}, F1 {metrics['F1']:.4f}")

    if not results:
        print("⚠️ No results to report.")
        return

    summary_df = pd.DataFrame(results)
    # Re‑order columns for readability
    summary_df = summary_df[["Model", "MAE", "RMSE", "R2", "Accuracy", "Precision", "Recall", "F1"]]
    print("\n=== Model Evaluation Summary ===")
    print(summary_df.to_string(index=False))
    summary_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved evaluation CSV → {OUTPUT_CSV.relative_to(PROJECT_ROOT)}")

if __name__ == "__main__":
    main()
