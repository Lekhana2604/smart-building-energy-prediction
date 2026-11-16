from pathlib import Path
import pandas as pd

files = {
    'features_dataset.csv'    : 'data/processed/features_dataset.csv',
    'features_dataset.parquet': 'data/processed/features_dataset.parquet',
    'feature_summary.csv'     : 'results/tables/04_feature_summary.csv',
    'fe_report.md'            : 'docs/feature_engineering_report.md',
    'notebook'                : 'notebooks/04_feature_engineering.ipynb',
    'graph 10'                : 'results/graphs/10_lag_feature_verification.png',
    'graph 11'                : 'results/graphs/11_calendar_feature_distributions.png',
    'graph 12'                : 'results/graphs/12_derived_weather_vs_consumption.png',
    'graph 13'                : 'results/graphs/13_lag_feature_scatter.png',
}

print('Phase 5 Output Verification')
print('='*55)
all_ok = True
for name, path in files.items():
    p = Path(path)
    if p.exists():
        size = p.stat().st_size
        label = f'{size/1e6:.2f} MB' if size > 1_000_000 else f'{size:,} bytes'
        print(f'  OK  {name:30s}  {label}')
    else:
        print(f'  MISSING  {name}')
        all_ok = False

print()
df = pd.read_parquet('data/processed/features_dataset.parquet',
                     columns=['building_id', 'date', 'meter_reading_daily', 'lag_1_day'])
n_bldg = df['building_id'].nunique()
n_rows = len(df)
n_nan_target = int(df['meter_reading_daily'].isna().sum())
n_nan_lag    = int(df['lag_1_day'].isna().sum())
print(f'  Rows      : {n_rows:,}')
print(f'  Buildings : {n_bldg:,}')
print(f'  NaN target: {n_nan_target}')
print(f'  NaN lag_1 : {n_nan_lag}')
print()
print('  ALL OUTPUTS VERIFIED' if all_ok else '  SOME FILES MISSING!')
