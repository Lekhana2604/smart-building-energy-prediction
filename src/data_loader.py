"""
data_loader.py
==============
Modular data loading utility module for the Building Data Genome Project 2 (BDG2) dataset.

This module provides reusable, robust functions to locate and load the raw BDG2 dataset files:
1. Building Metadata (metadata.csv)
2. Weather Records (weather.csv)
3. Electricity Meter Readings (electricity.csv and electricity_cleaned.csv)

Designed for academic clarity, beginner-friendliness, and memory efficiency.
"""

import os
from pathlib import Path
from typing import List, Optional, Union, Tuple
import pandas as pd
import numpy as np


def get_raw_data_dir(workspace_root: Optional[Union[str, Path]] = None) -> Path:
    """
    Locate the extracted BDG2 raw data directory dynamically.
    
    Parameters:
    -----------
    workspace_root : str or Path, optional
        Root directory of the project. If None, resolves relative to this file.
        
    Returns:
    --------
    Path
        Absolute path to the directory containing 'metadata', 'meters', and 'weather'.
        
    Raises:
    -------
    FileNotFoundError
        If the raw dataset cannot be found at the expected location.
    """
    if workspace_root is None:
        # Default to current file's parent's parent: src/.. -> project_root
        current_dir = Path(__file__).resolve().parent
        project_root = current_dir.parent
    else:
        project_root = Path(workspace_root).resolve()

    raw_dir = project_root / "data" / "raw"
    
    # Candidate directory names inside data/raw
    # BDG2 Zenodo extract is commonly named buds-lab-building-data-genome-project-2-<commit_hash>
    candidate_folders = [f for f in raw_dir.iterdir() if f.is_dir() and "buds-lab" in f.name.lower()]
    
    if candidate_folders:
        target_dir = candidate_folders[0] / "data"
        if target_dir.exists():
            return target_dir
            
    # Direct check if data/raw/data exists
    if (raw_dir / "data").exists():
        return raw_dir / "data"
        
    # Check if raw_dir directly has metadata
    if (raw_dir / "metadata").exists():
        return raw_dir

    raise FileNotFoundError(
        f"BDG2 raw dataset not found inside '{raw_dir}'. "
        "Please ensure the Zenodo archive is extracted inside 'data/raw/'."
    )


def load_metadata(raw_data_dir: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Load the building metadata table (metadata.csv).
    
    Contains static building attributes including building_id, site_id, primary space usage,
    gross floor area (sqm/sqft), latitude, longitude, timezone, and meter indicators.
    
    Parameters:
    -----------
    raw_data_dir : str or Path, optional
        Path to raw data directory. Discovered automatically if None.
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with 1,636 building records and 32 attribute columns.
    """
    base_dir = Path(raw_data_dir) if raw_data_dir else get_raw_data_dir()
    meta_path = base_dir / "metadata" / "metadata.csv"
    
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found at: {meta_path}")
        
    df = pd.read_csv(meta_path)
    return df


def load_weather(raw_data_dir: Optional[Union[str, Path]] = None,
                 sites: Optional[List[str]] = None,
                 parse_dates: bool = True) -> pd.DataFrame:
    """
    Load the hourly weather data table (weather.csv).
    
    Contains hourly observations of outdoor air temperature (deg C), dew point,
    cloud coverage, wind speed/direction, and precipitation across 19 sites.
    
    Parameters:
    -----------
    raw_data_dir : str or Path, optional
        Path to raw data directory. Discovered automatically if None.
    sites : list of str, optional
        Optional list of site names to filter (e.g., ['Panther', 'Bull']).
    parse_dates : bool, default True
        Whether to parse the 'timestamp' column as datetime.
        
    Returns:
    --------
    pd.DataFrame
        Hourly weather observations.
    """
    base_dir = Path(raw_data_dir) if raw_data_dir else get_raw_data_dir()
    weather_path = base_dir / "weather" / "weather.csv"
    
    if not weather_path.exists():
        raise FileNotFoundError(f"Weather file not found at: {weather_path}")
        
    parse_cols = ["timestamp"] if parse_dates else None
    df = pd.read_csv(weather_path, parse_dates=parse_cols)
    
    if sites is not None:
        df = df[df["site_id"].isin(sites)].copy()
        
    return df


def load_electricity(raw_data_dir: Optional[Union[str, Path]] = None,
                     cleaned: bool = True,
                     buildings: Optional[List[str]] = None,
                     parse_dates: bool = True) -> pd.DataFrame:
    """
    Load hourly electricity meter readings in wide format.
    
    Each column represents a building_id, and rows represent hourly timestamps (2016-2017).
    Measurement unit is kilowatt-hours (kWh) per hourly interval.
    
    Parameters:
    -----------
    raw_data_dir : str or Path, optional
        Path to raw data directory. Discovered automatically if None.
    cleaned : bool, default True
        If True, loads 'electricity_cleaned.csv' (anomalies & zero-streaks filtered to NaN).
        If False, loads raw 'electricity.csv'.
    buildings : list of str, optional
        Subset of building IDs to load (for memory efficiency). Always includes 'timestamp'.
    parse_dates : bool, default True
        Whether to parse 'timestamp' as datetime and set it as index.
        
    Returns:
    --------
    pd.DataFrame
        Hourly electricity readings indexed by timestamp, with building columns.
    """
    base_dir = Path(raw_data_dir) if raw_data_dir else get_raw_data_dir()
    subfolder = "cleaned" if cleaned else "raw"
    filename = "electricity_cleaned.csv" if cleaned else "electricity.csv"
    file_path = base_dir / "meters" / subfolder / filename
    
    if not file_path.exists():
        raise FileNotFoundError(f"Electricity file not found at: {file_path}")
        
    # Read header row only to verify which buildings actually have electricity meters
    header_cols = pd.read_csv(file_path, nrows=0).columns.tolist()
    
    usecols = None
    if buildings is not None:
        valid_buildings = [b for b in buildings if b in header_cols and b != "timestamp"]
        missing_buildings = [b for b in buildings if b not in header_cols and b != "timestamp"]
        if missing_buildings:
            print(f"[Warning] The following {len(missing_buildings)} requested buildings do not have electricity meters in '{filename}': {missing_buildings[:5]}...")
        if not valid_buildings:
            raise ValueError(f"None of the requested buildings have electricity meters in '{filename}'.")
        usecols = ["timestamp"] + valid_buildings
        
    df = pd.read_csv(file_path, usecols=usecols)
    
    if parse_dates:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")
        
    return df


def get_dataset_inventory(raw_data_dir: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Scan all files in the BDG2 dataset and return a structured inventory table.
    
    Returns:
    --------
    pd.DataFrame
        Table with File, Category, Format, Size (MB), and Extension.
    """
    base_dir = Path(raw_data_dir) if raw_data_dir else get_raw_data_dir()
    records = []
    
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            full_path = Path(root) / f
            rel_path = full_path.relative_to(base_dir)
            size_mb = full_path.stat().st_size / (1024 * 1024)
            category = rel_path.parts[0] if len(rel_path.parts) > 1 else "root"
            records.append({
                "Category": category,
                "Relative Path": str(rel_path),
                "Size (MB)": round(size_mb, 2),
                "Extension": full_path.suffix
            })
            
    return pd.DataFrame(records).sort_values(by=["Category", "Size (MB)"], ascending=[True, False])


if __name__ == "__main__":
    print("Testing data_loader.py...")
    data_dir = get_raw_data_dir()
    print(f"Discovered Raw Data Directory: {data_dir}")
    
    meta = load_metadata(data_dir)
    print(f"Metadata loaded successfully: shape = {meta.shape}")
    
    weather_sample = load_weather(data_dir, sites=["Panther"])
    print(f"Weather (Panther) loaded successfully: shape = {weather_sample.shape}")
    
    # Select sample buildings that actually have electricity meters
    elec_buildings = meta[meta["electricity"] == "Yes"]["building_id"].head(3).tolist()
    print(f"Sample buildings with electricity meter: {elec_buildings}")
    elec_sample = load_electricity(data_dir, cleaned=True, buildings=elec_buildings)
    print(f"Cleaned Electricity (3 buildings) loaded: shape = {elec_sample.shape}")
    
    inventory = get_dataset_inventory(data_dir)
    print(f"\nInventory of BDG2 Files (Total: {len(inventory)} files):")
    print(inventory.to_string(index=False))
