"""
feature_engineering.py
=======================
Phase 5 – Feature Engineering for Daily Electricity Consumption Prediction.

This module creates all prediction features from the daily aggregated dataset.
Every feature is strictly backward-looking — no future data is ever used.

Output: data/processed/features_dataset.csv
         data/processed/features_dataset.parquet  (smaller, faster to load)

Author : Smart Building ML Project Team
Dataset: BDG2 (Building Data Genome Project 2)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# ─────────────────────────── Paths ───────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR   = PROJECT_ROOT / "results" / "tables"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

INPUT_CSV   = PROCESSED_DIR / "daily_energy_dataset.csv"
OUTPUT_CSV  = PROCESSED_DIR / "features_dataset.csv"
OUTPUT_PARQ = PROCESSED_DIR / "features_dataset.parquet"


# ─────────────────────────── 1. Load Data ────────────────────────────────────
def load_daily_data(filepath: Path) -> pd.DataFrame:
    """
    Load the daily aggregated dataset produced in Phase 4.
    Parses 'date' as a proper datetime column right away.
    """
    print(f"[1/7] Loading daily dataset from: {filepath.name}")
    df = pd.read_csv(
        filepath,
        parse_dates=["date"],
        dtype={
            "building_id"       : str,
            "site_id"           : str,
            "primaryspaceusage" : str,
            "sub_primaryspaceusage": str,
            "timezone"          : str,
        }
    )
    print(f"      Loaded {len(df):,} rows × {df.shape[1]} columns.")
    print(f"      Date range  : {df['date'].min().date()}  →  {df['date'].max().date()}")
    print(f"      Buildings   : {df['building_id'].nunique():,}")
    return df


# ─────────────────────────── 2. Calendar Features ───────────────────────────
def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create calendar-based features from the 'date' column.

    All these features describe what kind of day it is in the calendar —
    weekday, month, season, etc. — which helps the model learn patterns
    like 'buildings use more energy on weekdays' or 'consumption peaks in summer'.

    Features added:
        day_of_week   – 0 = Monday … 6 = Sunday
        is_weekend    – 1 if Saturday or Sunday, else 0
        month         – 1 … 12
        season        – 0 = Winter, 1 = Spring, 2 = Summer, 3 = Autumn
        day_of_year   – 1 … 366 (useful for capturing yearly seasonality)
        week_of_year  – ISO week number (1 … 53)
    """
    print("[2/7] Adding calendar features …")

    df = df.copy()

    df["day_of_week"] = df["date"].dt.day_of_week         # 0 = Monday
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
    df["month"]       = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.day_of_year
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)

    # Season definition (Northern Hemisphere meteorological seasons):
    #   Winter : Dec, Jan, Feb  → 0
    #   Spring : Mar, Apr, May  → 1
    #   Summer : Jun, Jul, Aug  → 2
    #   Autumn : Sep, Oct, Nov  → 3
    season_map = {
        12: 0,  1: 0,  2: 0,   # Winter
         3: 1,  4: 1,  5: 1,   # Spring
         6: 2,  7: 2,  8: 2,   # Summer
         9: 3, 10: 3, 11: 3,   # Autumn
    }
    df["season"] = df["month"].map(season_map)

    print(f"      Calendar features added: day_of_week, is_weekend, month, season, "
          f"day_of_year, week_of_year")
    return df


# ─────────────────────────── 3. Lag & Rolling Features ──────────────────────
def add_lag_and_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create historical consumption features (lag + rolling window).

    WHY lags?
        The best predictor of today's consumption is yesterday's consumption.
        Lag features let the model learn from recent history.

    CRITICAL – Leakage prevention:
        • We GROUP BY building_id before computing any lag/rolling feature.
          This ensures building A's history never leaks into building B.
        • We SORT by date within each group.
        • Rolling windows use .shift(1) so day D only sees data up to day D-1.

    Features added:
        lag_1_day      – previous day's consumption (most important feature)
        lag_7_days     – same weekday last week (captures weekly cycle)
        rolling_7d_mean  – average of last 7 days (trend)
        rolling_14d_mean – average of last 14 days (longer trend)
        rolling_7d_std   – variability over last 7 days (stability signal)
    """
    print("[3/7] Adding lag and rolling window features …")
    print("      (Grouped by building_id — this may take 1–2 minutes for 1,440 buildings)")

    df = df.sort_values(["building_id", "date"]).reset_index(drop=True)

    # Group by building so shifts never bleed across buildings
    grp = df.groupby("building_id", sort=False)["meter_reading_daily"]

    # --- Lag features ----------------------------------------------------------
    df["lag_1_day"]   = grp.shift(1)   # previous day
    df["lag_7_days"]  = grp.shift(7)   # 7 days ago (same weekday last week)

    # --- Rolling features (shift(1) so day D does not include day D itself) ----
    df["rolling_7d_mean"]  = (grp
                               .transform(lambda x: x.shift(1).rolling(7, min_periods=4).mean()))
    df["rolling_14d_mean"] = (grp
                               .transform(lambda x: x.shift(1).rolling(14, min_periods=7).mean()))
    df["rolling_7d_std"]   = (grp
                               .transform(lambda x: x.shift(1).rolling(7, min_periods=4).std()))

    print("      Lag + rolling features added: lag_1_day, lag_7_days, "
          "rolling_7d_mean, rolling_14d_mean, rolling_7d_std")
    return df


# ─────────────────────────── 4. Derived Weather Feature ─────────────────────
def add_derived_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create Cooling Degree Day (CDD) — a simple but powerful weather proxy.

    What is CDD?
        CDD = max(0,  mean_temperature - balance_point)
        Balance point = 18 °C  (the temperature below which heating/cooling is minimal)
        When it's hotter than 18 °C, buildings need air-conditioning → more energy.
        CDD captures this in a single number.

    Example:
        If mean temp = 25 °C  →  CDD = 25 - 18 = 7
        If mean temp = 10 °C  →  CDD = max(0, 10 - 18) = 0  (no cooling load)

    Feature added:
        cooling_degree_day – CDD using 18 °C balance point
        temp_range         – daily temperature swing (max - min); high swings
                             often correlate with unusual energy behaviour
    """
    print("[4/7] Adding derived weather features …")

    BALANCE_POINT = 18.0   # °C — standard meteorological balance point

    df["cooling_degree_day"] = np.maximum(0.0, df["air_temperature_mean"] - BALANCE_POINT)
    df["temp_range"]         = df["air_temperature_max"] - df["air_temperature_min"]

    print(f"      Derived weather features added: cooling_degree_day (balance={BALANCE_POINT}°C), "
          f"temp_range")
    return df


# ─────────────────────────── 5. Building Meta Features ──────────────────────
def add_building_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical building metadata for use by ML models.

    ML models only understand numbers — they cannot directly process strings
    like 'Education' or 'Office'. We convert them using:

    One-Hot Encoding for primaryspaceusage:
        Creates one binary (0/1) column per category.
        e.g., 'Office' → usage_Office = 1, all others = 0
        drop_first=True removes one dummy column to avoid multicollinearity
        (the dropped category is encoded by all-zeros).

    Log-transformed sqm:
        Building size spans a very wide range (100 m² → 60,000 m²).
        Taking the natural log compresses this range so large buildings
        don't dominate the model.

    Features added:
        log_sqm        – log(1 + sqm)  — size in log scale
        yearbuilt_norm – year built, with NaNs filled by median
        usage_*        – one-hot columns for primaryspaceusage
    """
    print("[5/7] Adding building metadata features …")

    df = df.copy()

    # --- Log-transform building size ----------------------------------------
    # +1 avoids log(0) if any sqm value is 0 or missing
    df["log_sqm"] = np.log1p(df["sqm"].fillna(0))

    # --- Normalise year built -------------------------------------------------
    median_year = df["yearbuilt"].median()
    df["yearbuilt_norm"] = df["yearbuilt"].fillna(median_year)

    # --- One-Hot Encode primaryspaceusage ------------------------------------
    df["primaryspaceusage"] = df["primaryspaceusage"].fillna("Unknown").str.strip()
    usage_dummies = pd.get_dummies(
        df["primaryspaceusage"],
        prefix="usage",
        drop_first=True,   # drop first category to avoid perfect collinearity
        dtype=int
    )
    df = pd.concat([df, usage_dummies], axis=1)

    usage_cols = [c for c in df.columns if c.startswith("usage_")]
    print(f"      Building features added: log_sqm, yearbuilt_norm, "
          f"{len(usage_cols)} usage dummy columns")
    print(f"      Usage categories → {usage_cols}")
    return df


# ─────────────────────────── 6. Drop NaN rows from lag window ───────────────
def drop_warmup_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows where lag/rolling features are NaN.

    The first 14 rows per building have NaN in rolling_14d_mean because
    there is not enough history yet. These rows are useless for training
    (the model cannot learn from incomplete feature vectors).

    We drop any row where ALL five lag/rolling columns have NaN.
    This removes at most 14 rows per building — a tiny fraction of the data.
    """
    print("[6/7] Dropping warm-up rows with NaN lag features …")

    lag_cols = ["lag_1_day", "lag_7_days",
                "rolling_7d_mean", "rolling_14d_mean", "rolling_7d_std"]

    before = len(df)
    df = df.dropna(subset=lag_cols).reset_index(drop=True)
    after = len(df)

    print(f"      Dropped {before - after:,} warm-up rows  "
          f"({before:,} → {after:,} rows remaining)")
    return df


# ─────────────────────────── 7. Final Feature Set & Save ────────────────────
def save_feature_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select the final ML-ready feature columns, verify no leakage, and save.

    What we keep:
        - Identifier columns: building_id, site_id, date
        - Target            : meter_reading_daily
        - All engineered features (calendar, lag, rolling, weather, building)

    What we drop:
        - Raw columns already encoded (primaryspaceusage, sub_primaryspaceusage,
          timezone, sqm, sqft, yearbuilt — superseded by encoded/log versions)
        - is_complete_day, valid_hours (data quality flags, not prediction signals)
    """
    print("[7/7] Selecting final feature set and saving …")

    # Columns to keep
    id_cols     = ["building_id", "site_id", "date"]
    target_col  = ["meter_reading_daily"]
    calendar    = ["day_of_week", "is_weekend", "month", "season",
                   "day_of_year", "week_of_year"]
    lag_roll    = ["lag_1_day", "lag_7_days",
                   "rolling_7d_mean", "rolling_14d_mean", "rolling_7d_std"]
    weather     = ["air_temperature_mean", "air_temperature_min", "air_temperature_max",
                   "dew_temperature_mean", "wind_speed_mean",
                   "precip_depth_1hr_total", "sea_lvl_pressure_mean",
                   "cooling_degree_day", "temp_range"]
    building    = ["log_sqm", "yearbuilt_norm"]
    usage_cols  = sorted([c for c in df.columns if c.startswith("usage_")])

    all_keep = id_cols + target_col + calendar + lag_roll + weather + building + usage_cols

    # Keep only columns that actually exist in the dataframe
    all_keep = [c for c in all_keep if c in df.columns]

    df_final = df[all_keep].copy()

    # ── Leakage sanity check ──────────────────────────────────────────────────
    # Verify that for each building, lag_1_day on date D == meter_reading_daily on D-1
    sample_bldg = df_final["building_id"].iloc[0]
    sample_df   = df_final[df_final["building_id"] == sample_bldg].sort_values("date").head(20)
    target_shift = sample_df["meter_reading_daily"].shift(1).values
    lag_values   = sample_df["lag_1_day"].values
    mismatch     = np.nansum(np.abs(target_shift - lag_values) > 0.001)
    if mismatch == 0:
        print(f"      ✅  Leakage check PASSED for building '{sample_bldg}': "
              f"lag_1_day correctly mirrors previous day's consumption.")
    else:
        print(f"      ⚠️  Leakage check WARNING: {mismatch} mismatches detected. "
              f"Investigate before training.")

    # ── Save ─────────────────────────────────────────────────────────────────
    df_final.to_csv(OUTPUT_CSV, index=False)
    print(f"\n      ✅  CSV  saved → {OUTPUT_CSV}  ({OUTPUT_CSV.stat().st_size / 1e6:.2f} MB)")

    df_final.to_parquet(OUTPUT_PARQ, index=False)
    print(f"      ✅  Parquet saved → {OUTPUT_PARQ}  ({OUTPUT_PARQ.stat().st_size / 1e6:.2f} MB)")

    # ── Feature Summary Table ─────────────────────────────────────────────────
    summary_records = []
    feature_cols = [c for c in all_keep if c not in id_cols + target_col]
    for col in feature_cols:
        summary_records.append({
            "Feature"       : col,
            "Category"      : (
                "Calendar"  if col in calendar  else
                "Lag/Rolling" if col in lag_roll else
                "Weather"   if col in weather   else
                "Building"  if col in building  else
                "Usage Dummy"
            ),
            "Non-Null Count": int(df_final[col].notna().sum()),
            "NaN Count"     : int(df_final[col].isna().sum()),
            "Mean"          : round(float(df_final[col].mean()), 4) if df_final[col].dtype != object else "N/A",
            "Std"           : round(float(df_final[col].std()),  4) if df_final[col].dtype != object else "N/A",
            "Min"           : round(float(df_final[col].min()),  4) if df_final[col].dtype != object else "N/A",
            "Max"           : round(float(df_final[col].max()),  4) if df_final[col].dtype != object else "N/A",
        })

    summary_df = pd.DataFrame(summary_records)
    summary_path = RESULTS_DIR / "04_feature_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"      ✅  Feature summary saved → {summary_path}")

    # ── Final report ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PHASE 5 — FEATURE ENGINEERING COMPLETE")
    print("=" * 60)
    print(f"  Total rows         : {len(df_final):,}")
    print(f"  Total features     : {len(feature_cols)}")
    print(f"  Buildings          : {df_final['building_id'].nunique():,}")
    print(f"  Date range         : {df_final['date'].min().date()}  →  {df_final['date'].max().date()}")
    print(f"  Target column      : meter_reading_daily  (kWh/day)")
    print(f"  NaN in target      : {df_final['meter_reading_daily'].isna().sum()}")
    print(f"  NaN in lag_1_day   : {df_final['lag_1_day'].isna().sum()}")
    print("=" * 60)

    return df_final


# ─────────────────────────── Main Pipeline ───────────────────────────────────
def run_feature_engineering() -> pd.DataFrame:
    """
    Run the complete Phase 5 feature engineering pipeline in sequence.
    Returns the final ML-ready DataFrame.
    """
    print("\n" + "=" * 60)
    print("  STARTING PHASE 5 — FEATURE ENGINEERING")
    print("=" * 60 + "\n")

    df = load_daily_data(INPUT_CSV)
    df = add_calendar_features(df)
    df = add_lag_and_rolling_features(df)
    df = add_derived_weather_features(df)
    df = add_building_features(df)
    df = drop_warmup_rows(df)
    df = save_feature_dataset(df)

    return df


if __name__ == "__main__":
    run_feature_engineering()
