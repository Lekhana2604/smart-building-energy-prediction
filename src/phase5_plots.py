"""
phase5_plots.py
===============
Generates all Phase 5 visualisation graphs directly (no Jupyter kernel needed).
Loads the already-saved features_dataset.parquet and saves 4 graphs to results/graphs/.

Run: python src/phase5_plots.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')
plt.rcParams['figure.dpi'] = 100

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GRAPHS_DIR   = PROJECT_ROOT / 'results' / 'graphs'
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

PARQ = PROJECT_ROOT / 'data' / 'processed' / 'features_dataset.parquet'

print("Loading features_dataset.parquet …")
# Load only the columns we need for plots — much faster than loading all 40 cols
PLOT_COLS = [
    'building_id', 'date', 'meter_reading_daily',
    'lag_1_day', 'lag_7_days', 'rolling_7d_mean',
    'day_of_week', 'month', 'season',
    'cooling_degree_day', 'temp_range',
]
df = pd.read_parquet(PARQ, columns=PLOT_COLS)
df['date'] = pd.to_datetime(df['date'])
print(f"Loaded {len(df):,} rows.  Buildings: {df['building_id'].nunique():,}")
print(f"NaN target: {df['meter_reading_daily'].isna().sum()}  |  NaN lag_1: {df['lag_1_day'].isna().sum()}")

# ── Leakage verification ──────────────────────────────────────────────────────
print("\nLeakage verification …")
for bldg in df['building_id'].unique()[:3]:
    bdf = df[df['building_id'] == bldg].sort_values('date').reset_index(drop=True)
    expected = bdf['meter_reading_daily'].shift(1)
    mismatch = (abs(bdf['lag_1_day'] - expected) > 0.001).sum()
    status = '✅ PASSED' if mismatch == 0 else f'❌ FAILED ({mismatch})'
    print(f"  {bldg[:45]:45s}  {status}")

# ── Graph 10: Lag feature verification ───────────────────────────────────────
print("\n[1/4] Graph 10: Lag feature verification …")
bldg = df['building_id'].iloc[0]
bdf  = df[df['building_id'] == bldg].sort_values('date').head(60)

fig, ax = plt.subplots(figsize=(13, 4))
ax.plot(bdf['date'], bdf['meter_reading_daily'], label='Actual (Day D)',      lw=2,   color='steelblue')
ax.plot(bdf['date'], bdf['lag_1_day'],           label='lag_1_day (Day D-1)', lw=1.5, ls='--', color='darkorange')
ax.plot(bdf['date'], bdf['rolling_7d_mean'],     label='rolling_7d_mean',     lw=1.5, ls=':',  color='green')
ax.set_title(f'Lag Feature Verification — {bldg}', fontsize=12, fontweight='bold')
ax.set_xlabel('Date');  ax.set_ylabel('kWh/day')
ax.legend();  ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(GRAPHS_DIR / '10_lag_feature_verification.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved → results/graphs/10_lag_feature_verification.png")

# ── Graph 11: Calendar features ───────────────────────────────────────────────
print("[2/4] Graph 11: Calendar features …")
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

day_means = df.groupby('day_of_week')['meter_reading_daily'].median()
axes[0].bar(['Mon','Tue','Wed','Thu','Fri','Sat','Sun'], day_means.values,
            color=['steelblue']*5 + ['coral']*2)
axes[0].set_title('Median Consumption by Weekday');  axes[0].set_ylabel('kWh/day')
axes[0].grid(axis='y', alpha=0.3)

month_means = df.groupby('month')['meter_reading_daily'].median()
axes[1].bar(range(1,13), month_means.values, color='steelblue')
axes[1].set_xticks(range(1,13))
axes[1].set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
axes[1].set_title('Median Consumption by Month');  axes[1].set_ylabel('kWh/day')
axes[1].grid(axis='y', alpha=0.3)

season_means = df.groupby('season')['meter_reading_daily'].median()
axes[2].bar(['Winter','Spring','Summer','Autumn'], season_means.values,
            color=['#4878CF','#6ACC65','#D65F5F','#B47CC7'])
axes[2].set_title('Median Consumption by Season');  axes[2].set_ylabel('kWh/day')
axes[2].grid(axis='y', alpha=0.3)

plt.suptitle('Calendar Features vs Energy Consumption', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(GRAPHS_DIR / '11_calendar_feature_distributions.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved → results/graphs/11_calendar_feature_distributions.png")

# ── Graph 12: Derived weather features ───────────────────────────────────────
print("[3/4] Graph 12: Derived weather features …")
sample = df.sample(n=5000, random_state=42)

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].scatter(sample['cooling_degree_day'], np.log1p(sample['meter_reading_daily']),
                alpha=0.3, s=5, color='steelblue')
axes[0].set_xlabel('Cooling Degree Day (°C above 18°C)')
axes[0].set_ylabel('log(1 + kWh/day)')
axes[0].set_title('Cooling Degree Day vs Log-Consumption')
axes[0].grid(True, alpha=0.3)

axes[1].scatter(sample['temp_range'], np.log1p(sample['meter_reading_daily']),
                alpha=0.3, s=5, color='coral')
axes[1].set_xlabel('Daily Temperature Range (°C)')
axes[1].set_ylabel('log(1 + kWh/day)')
axes[1].set_title('Temperature Range vs Log-Consumption')
axes[1].grid(True, alpha=0.3)

plt.suptitle('Derived Weather Features vs Energy Consumption', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(GRAPHS_DIR / '12_derived_weather_vs_consumption.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved → results/graphs/12_derived_weather_vs_consumption.png")

# ── Graph 13: Lag scatter ─────────────────────────────────────────────────────
print("[4/4] Graph 13: Lag scatter …")
sample = df.sample(n=8000, random_state=42)

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].scatter(np.log1p(sample['lag_1_day']), np.log1p(sample['meter_reading_daily']),
                alpha=0.2, s=4, color='steelblue')
axes[0].set_xlabel('log(1 + lag_1_day)');  axes[0].set_ylabel('log(1 + Actual kWh/day)')
axes[0].set_title('Lag 1 Day vs Actual');  axes[0].grid(True, alpha=0.3)

axes[1].scatter(np.log1p(sample['rolling_7d_mean']), np.log1p(sample['meter_reading_daily']),
                alpha=0.2, s=4, color='green')
axes[1].set_xlabel('log(1 + rolling_7d_mean)');  axes[1].set_ylabel('log(1 + Actual kWh/day)')
axes[1].set_title('7-Day Rolling Mean vs Actual');  axes[1].grid(True, alpha=0.3)

plt.suptitle('Lag Feature Strength (log scale)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(GRAPHS_DIR / '13_lag_feature_scatter.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved → results/graphs/13_lag_feature_scatter.png")

print("\n✅  All 4 Phase 5 graphs saved successfully.")
