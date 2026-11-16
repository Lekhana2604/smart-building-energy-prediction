"""
daily_aggregator.py
===================
Pipeline for Phase 4: Daily Dataset Creation.

Transforms cleaned hourly electricity and weather observations from 'data/interim/'
into the primary processed daily modeling dataset:
    data/processed/daily_energy_dataset.csv

Mathematical Aggregation Logic:
- Meter type: Electricity
- Measurement unit: Kilowatt-hours (kWh) per 1-hour interval (energy, not power)
- Full Days (24 valid hours):
    Daily kWh = sum(Hourly kWh)
- Near-Complete Days (23 valid hours, e.g. DST Spring Forward or momentary drop):
    Daily kWh = sum(Hourly kWh) * (24 / 23)
- Incomplete Days (< 23 valid hours):
    Discarded (masked as NaN) to prevent artificial under-prediction bias.

Combines daily electricity consumption with:
- Daily meteorological features (mean, min, max temperature; mean dew point, wind, pressure; total rain)
- Static building attributes (primary space usage, gross floor area sqm, site ID, timezone)
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Union, Tuple, Optional
import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def aggregate_daily_weather(clean_weather_path: Path) -> pd.DataFrame:
    """
    Aggregate hourly meteorological observations by site_id and date.
    
    Produces:
    - air_temperature_mean (°C)
    - air_temperature_min (°C)
    - air_temperature_max (°C)
    - dew_temperature_mean (°C)
    - wind_speed_mean (m/s)
    - precip_depth_1hr_total (mm, daily cumulative precipitation)
    - sea_lvl_pressure_mean (hPa)
    """
    print(f"[Daily Weather] Loading cleaned hourly weather from: {clean_weather_path}...")
    weather = pd.read_csv(clean_weather_path, parse_dates=["timestamp"])
    weather["date"] = weather["timestamp"].dt.strftime("%Y-%m-%d")
    
    print("[Daily Weather] Grouping by site_id and date to compute daily summary statistics...")
    weather_daily = weather.groupby(["site_id", "date"]).agg(
        air_temperature_mean=("airTemperature", "mean"),
        air_temperature_min=("airTemperature", "min"),
        air_temperature_max=("airTemperature", "max"),
        dew_temperature_mean=("dewTemperature", "mean"),
        wind_speed_mean=("windSpeed", "mean"),
        precip_depth_1hr_total=("precipDepth1HR", "sum"),
        sea_lvl_pressure_mean=("seaLvlPressure", "mean")
    ).reset_index()
    
    # Round numerical metrics for clean storage
    round_cols = {
        "air_temperature_mean": 2,
        "air_temperature_min": 2,
        "air_temperature_max": 2,
        "dew_temperature_mean": 2,
        "wind_speed_mean": 2,
        "precip_depth_1hr_total": 2,
        "sea_lvl_pressure_mean": 2
    }
    weather_daily = weather_daily.round(round_cols)
    print(f"[Daily Weather] Aggregated weather records: {len(weather_daily)} site-days ({weather_daily['site_id'].nunique()} sites * 731 days)")
    return weather_daily


def aggregate_daily_electricity(clean_electricity_path: Path,
                                min_hours_threshold: int = 23) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregate hourly electricity matrix to daily energy consumption (kWh/day).
    
    Parameters:
    -----------
    clean_electricity_path : Path
        Path to 'data/interim/clean_electricity_hourly.csv' (17,544 rows x 1,440 buildings).
    min_hours_threshold : int, default 23
        Minimum number of valid hours required to accept a day as usable.
        
    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame]
        1. daily_energy_wide: (731 dates x 1,440 buildings) with normalized daily kWh (NaN for incomplete days).
        2. daily_counts_wide: (731 dates x 1,440 buildings) with valid hour counts (0 to 24).
    """
    print(f"[Daily Electricity] Loading cleaned hourly electricity from: {clean_electricity_path}...")
    t0 = time.time()
    elec = pd.read_csv(clean_electricity_path, index_col="timestamp")
    elec.index = pd.to_datetime(elec.index)
    print(f"[Daily Electricity] Loaded {elec.shape[0]} timestamps x {elec.shape[1]} buildings in {time.time()-t0:.2f}s.")
    
    print("[Daily Electricity] Resampling to daily frequency and counting valid hourly readings...")
    daily_counts = elec.notnull().resample("D").sum()
    daily_sums = elec.fillna(0.0).resample("D").sum()
    
    # Normalization: daily_sums * (24.0 / daily_counts)
    # For 24 hours: 24/24 = 1.0 (exact sum)
    # For 23 hours: 24/23 = 1.0435 (normalizes the single missing hour)
    daily_normalized = daily_sums * (24.0 / daily_counts)
    
    # Mask incomplete days (< min_hours_threshold) as NaN
    daily_normalized[daily_counts < min_hours_threshold] = np.nan
    
    total_cells = daily_normalized.size
    valid_cells = daily_normalized.notnull().sum().sum()
    print(f"[Daily Electricity] Aggregation complete: {valid_cells:,} usable building-days out of {total_cells:,} ({valid_cells/total_cells*100:.2f}%)")
    
    daily_normalized.index = daily_normalized.index.strftime("%Y-%m-%d")
    daily_counts.index = daily_counts.index.strftime("%Y-%m-%d")
    
    return daily_normalized, daily_counts


def create_daily_dataset(workspace_root: Optional[Union[str, Path]] = None,
                         min_hours_threshold: int = 23) -> Dict[str, Union[int, float, str]]:
    """
    Orchestrate Phase 4: build, validate, and save the long-format daily energy dataset.
    
    Outputs:
    - data/processed/daily_energy_dataset.csv
    - data/processed/daily_energy_dataset.parquet (for rapid I/O)
    - results/tables/03_daily_aggregation_summary.csv
    """
    t0 = time.time()
    root = Path(workspace_root) if workspace_root else PROJECT_ROOT
    interim_dir = root / "data" / "interim"
    processed_dir = root / "data" / "processed"
    tables_dir = root / "results" / "tables"
    processed_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("STARTING PHASE 4: DAILY DATASET CREATION")
    print(f"Workspace Root: {root}")
    print(f"Interim Directory: {interim_dir}")
    print(f"Processed Directory: {processed_dir}")
    print("=" * 70)
    
    # 1. Load clean metadata
    meta_path = interim_dir / "clean_metadata.csv"
    print(f"\n--- STEP 1: Loading Clean Metadata ({meta_path.name}) ---")
    meta = pd.read_csv(meta_path)
    print(f"Loaded metadata for {len(meta)} screened buildings.")
    
    # 2. Aggregate weather daily
    weather_path = interim_dir / "clean_weather_hourly.csv"
    print(f"\n--- STEP 2: Aggregating Weather to Daily Summary ({weather_path.name}) ---")
    weather_daily = aggregate_daily_weather(weather_path)
    
    # 3. Aggregate electricity daily
    elec_path = interim_dir / "clean_electricity_hourly.csv"
    print(f"\n--- STEP 3: Aggregating Electricity to Daily Energy ({elec_path.name}) ---")
    daily_energy_wide, daily_counts_wide = aggregate_daily_electricity(
        elec_path, min_hours_threshold=min_hours_threshold
    )
    
    # 4. Melt from wide matrix into long tabular format
    print("\n--- STEP 4: Transforming into Long Tabular Format (One row per building-day) ---")
    energy_long = daily_energy_wide.stack().reset_index()
    energy_long.columns = ["date", "building_id", "meter_reading_daily"]
    
    counts_long = daily_counts_wide.stack().reset_index()
    counts_long.columns = ["date", "building_id", "valid_hours"]
    
    # Merge electricity values and valid hour counts
    daily_df = pd.merge(energy_long, counts_long, on=["date", "building_id"])
    print(f"Melted daily electricity records: {len(daily_df):,} rows")
    
    # Filter out incomplete days (NaN in meter_reading_daily)
    daily_df = daily_df.dropna(subset=["meter_reading_daily"]).copy()
    daily_df["valid_hours"] = daily_df["valid_hours"].astype(int)
    daily_df["is_complete_day"] = (daily_df["valid_hours"] == 24).astype(int)
    daily_df["meter_reading_daily"] = daily_df["meter_reading_daily"].round(2)
    print(f"Usable daily electricity records (valid_hours >= {min_hours_threshold}): {len(daily_df):,} rows")
    
    # 5. Merge Building Metadata Attributes
    print("\n--- STEP 5: Merging Building Metadata Attributes ---")
    meta_cols = ["building_id", "site_id", "primaryspaceusage", "sub_primaryspaceusage", "sqm", "sqft", "timezone", "yearbuilt"]
    daily_df = pd.merge(daily_df, meta[meta_cols], on="building_id", how="left")
    
    # 6. Merge Daily Meteorological Features
    print("\n--- STEP 6: Merging Daily Site Weather Features ---")
    daily_df = pd.merge(daily_df, weather_daily, on=["site_id", "date"], how="left")
    
    # Organize column order
    ordered_cols = [
        "building_id", "site_id", "date", "meter_reading_daily",
        "valid_hours", "is_complete_day",
        "air_temperature_mean", "air_temperature_min", "air_temperature_max",
        "dew_temperature_mean", "wind_speed_mean", "precip_depth_1hr_total", "sea_lvl_pressure_mean",
        "primaryspaceusage", "sub_primaryspaceusage", "sqm", "sqft", "timezone", "yearbuilt"
    ]
    daily_df = daily_df[ordered_cols].sort_values(by=["building_id", "date"]).reset_index(drop=True)
    
    # 7. Data Quality & Sanity Verification
    print("\n--- STEP 7: Quality and Integrity Verification ---")
    neg_check = (daily_df["meter_reading_daily"] < 0).sum()
    null_target_check = daily_df["meter_reading_daily"].isnull().sum()
    null_weather_check = daily_df["air_temperature_mean"].isnull().sum()
    print(f"Verification: Negative daily readings = {neg_check}")
    print(f"Verification: Null target readings = {null_target_check}")
    print(f"Verification: Null weather features = {null_weather_check}")
    print(f"Verification: Date range = {daily_df['date'].min()} to {daily_df['date'].max()} ({daily_df['date'].nunique()} unique calendar dates)")
    print(f"Verification: Unique buildings = {daily_df['building_id'].nunique()}")
    
    # 8. Save Processed Dataset
    csv_out_path = processed_dir / "daily_energy_dataset.csv"
    parquet_out_path = processed_dir / "daily_energy_dataset.parquet"
    
    print(f"\n--- STEP 8: Saving Processed Datasets ---")
    print(f"Writing CSV: {csv_out_path}...")
    daily_df.to_csv(csv_out_path, index=False)
    csv_size_mb = csv_out_path.stat().st_size / (1024 * 1024)
    print(f"Saved: {csv_out_path} ({csv_size_mb:.2f} MB, {len(daily_df):,} rows)")
    
    print(f"Writing Parquet: {parquet_out_path}...")
    daily_df.to_parquet(parquet_out_path, index=False)
    parquet_size_mb = parquet_out_path.stat().st_size / (1024 * 1024)
    print(f"Saved: {parquet_out_path} ({parquet_size_mb:.2f} MB)")
    
    # 9. Save Summary Table
    summary_table = pd.DataFrame([
        {"Metric": "Target Variable", "Value": "Daily Electricity Consumption (meter_reading_daily)"},
        {"Metric": "Target Measurement Unit", "Value": "Kilowatt-hours per day (kWh/day)"},
        {"Metric": "Total Processed Rows", "Value": f"{len(daily_df):,}"},
        {"Metric": "Total Screened Buildings", "Value": f"{daily_df['building_id'].nunique()}"},
        {"Metric": "Total Unique Calendar Dates", "Value": f"{daily_df['date'].nunique()} (2016-01-01 to 2017-12-31)"},
        {"Metric": "Full 24-Hour Days", "Value": f"{(daily_df['valid_hours'] == 24).sum():,} ({(daily_df['valid_hours'] == 24).sum() / len(daily_df) * 100:.2f}%)"},
        {"Metric": "Normalized 23-Hour Days", "Value": f"{(daily_df['valid_hours'] == 23).sum():,} ({(daily_df['valid_hours'] == 23).sum() / len(daily_df) * 100:.2f}%)"},
        {"Metric": "Incomplete Days Discarded (<23h)", "Value": f"{1052640 - len(daily_df):,} ({(1052640 - len(daily_df)) / 1052640 * 100:.2f}%)"},
        {"Metric": "Mean Daily Electricity", "Value": f"{daily_df['meter_reading_daily'].mean():.2f} kWh/day"},
        {"Metric": "Median Daily Electricity", "Value": f"{daily_df['meter_reading_daily'].median():.2f} kWh/day"},
        {"Metric": "Min Daily Electricity", "Value": f"{daily_df['meter_reading_daily'].min():.2f} kWh/day"},
        {"Metric": "Max Daily Electricity", "Value": f"{daily_df['meter_reading_daily'].max():.2f} kWh/day"},
        {"Metric": "CSV File Size", "Value": f"{csv_size_mb:.2f} MB"},
        {"Metric": "Execution Runtime", "Value": f"{time.time() - t0:.2f} seconds"}
    ])
    summary_path = tables_dir / "03_daily_aggregation_summary.csv"
    summary_table.to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")
    
    print("\n" + "=" * 70)
    print("PHASE 4 COMPLETED SUCCESSFULLY!")
    print(f"Total Execution Time: {time.time() - t0:.2f}s")
    print("=" * 70)
    
    return {
        "rows": len(daily_df),
        "buildings": daily_df["building_id"].nunique(),
        "csv_size_mb": round(csv_size_mb, 2),
        "mean_daily_kwh": round(daily_df["meter_reading_daily"].mean(), 2),
        "elapsed_seconds": round(time.time() - t0, 2)
    }


if __name__ == "__main__":
    result = create_daily_dataset()
    print("\nExecution Output Summary:", result)
