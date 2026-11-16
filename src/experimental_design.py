"""
experimental_design.py
=======================
Randomized Block Design (RBD) setup for the smart‑building energy prediction project.

* **Blocks**   – each `building_id` (1,440 eligible buildings).
* **Treatments** – three modelling approaches:
    - `LinearRegression`   (control)
    - `RandomForest`
    - `GradientBoosting`

The script randomly assigns one treatment to each building (seed = 42 for reproducibility),
stores the mapping in `data/processed/experimental_design_info.csv`, and prints a brief
summary.

The mapping will be read later in Phase 8 so that each model is trained only on the
buildings allocated to it, fulfilling the RBD experimental protocol.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import sys

# ---------------------------------------------------------------------------
# Configuration – keep everything relative to the project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_PARQUET = PROJECT_ROOT / "data" / "processed" / "features_dataset.parquet"
OUTPUT_CSV = PROJECT_ROOT / "data" / "processed" / "experimental_design_info.csv"

TREATMENTS = ["DecisionTree", "RandomForest", "GradientBoosting"]
RANDOM_SEED = 42


def load_buildings(parquet_path: Path) -> pd.Index:
    """Load the feature dataset (only the `building_id` column) and return the unique IDs.
    Loading a single column is fast and memory‑efficient.
    """
    print(f"Loading building IDs from {parquet_path.name} …")
    df = pd.read_parquet(parquet_path, columns=["building_id"])
    unique_ids = df["building_id"].unique()
    print(f"  Found {len(unique_ids):,} unique building IDs (blocks).")
    return pd.Index(unique_ids)


def assign_treatments(buildings: pd.Index, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Randomly assign one of the three treatments to each building.
    The assignment is reproducible thanks to the fixed seed.
    Returns a DataFrame with columns `building_id` and `treatment`.
    """
    rng = np.random.default_rng(seed)
    assignments = rng.choice(TREATMENTS, size=len(buildings), replace=True)
    design_df = pd.DataFrame({"building_id": buildings, "treatment": assignments})
    # Ensure deterministic ordering for reproducibility
    design_df = design_df.sort_values("building_id").reset_index(drop=True)
    return design_df


def save_design(df: pd.DataFrame, out_path: Path) -> None:
    """Save the design mapping as CSV (UTF‑8, no index)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved experimental design to {out_path.relative_to(PROJECT_ROOT)} ({df.shape[0]} rows).")


def summary(df: pd.DataFrame) -> None:
    """Print a brief summary – counts per treatment and first few rows."""
    print("\nExperimental Design Summary")
    print("---------------------------")
    counts = df["treatment"].value_counts().reindex(TREATMENTS, fill_value=0)
    for trt, cnt in counts.items():
        print(f"  {trt:16s}: {cnt:,} buildings")
    print("\nSample assignment (first 10 rows):")
    print(df.head(10).to_string(index=False))


def main() -> int:
    parser = argparse.ArgumentParser(description="Create Randomized Block Design mapping for model comparison.")
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed for reproducible treatment assignment (default: 42).",
    )
    args = parser.parse_args()

    # 1. Load unique building IDs (blocks)
    try:
        buildings = load_buildings(FEATURES_PARQUET)
    except Exception as exc:
        print(f"❌ Error loading feature data: {exc}")
        return 1

    # 2. Assign treatments
    design_df = assign_treatments(buildings, seed=args.seed)

    # 3. Save mapping
    save_design(design_df, OUTPUT_CSV)

    # 4. Print summary
    summary(design_df)

    return 0

if __name__ == "__main__":
    sys.exit(main())
