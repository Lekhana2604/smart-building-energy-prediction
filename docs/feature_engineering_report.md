# Phase 5 – Feature Engineering Report
## Daily Electricity Consumption Prediction in Smart Buildings (BDG2)

---

## Overview

Phase 5 transforms the cleaned daily aggregated dataset (produced in Phase 4) into a fully
machine-learning-ready feature dataset. The central principle throughout this phase is
**strict leakage prevention**: every feature for prediction day D uses only information
that was available before day D.

---

## 1. Input Dataset

| Property | Value |
|---|---|
| File | `data/processed/daily_energy_dataset.csv` |
| Rows | 1,004,892 |
| Buildings | 1,440 |
| Date range | 2016-01-01 → 2017-12-31 |
| Target column | `meter_reading_daily` (kWh/day) |

---

## 2. Feature Engineering Pipeline (`src/feature_engineering.py`)

The pipeline runs in seven sequential steps:

```
Step 1 → Load daily dataset
Step 2 → Calendar features
Step 3 → Lag & Rolling window features  (grouped by building_id)
Step 4 → Derived weather features
Step 5 → Building metadata features    (one-hot encoding)
Step 6 → Drop warm-up rows             (NaN from lag windows)
Step 7 → Save final dataset + summary table
```

---

## 3. Feature Groups

### 3.1 Calendar Features (6 features)

Calendar features capture the cyclic patterns of energy use across time.

| Feature | Description | Why It Matters |
|---|---|---|
| `day_of_week` | 0 = Monday … 6 = Sunday | Weekdays vs weekends have different consumption profiles |
| `is_weekend` | 1 if Sat/Sun, else 0 | Simplified binary signal for the weekly cycle |
| `month` | 1 … 12 | Captures monthly seasonality |
| `season` | 0=Winter, 1=Spring, 2=Summer, 3=Autumn | Coarser seasonal signal (less noisy than month) |
| `day_of_year` | 1 … 366 | Smooth yearly cycle signal |
| `week_of_year` | ISO week 1 … 53 | Useful for year-on-year comparisons |

**Season definition (Northern Hemisphere meteorological):**
- Winter: December, January, February → 0
- Spring: March, April, May → 1
- Summer: June, July, August → 2
- Autumn: September, October, November → 3

---

### 3.2 Lag & Rolling Window Features (5 features)

These are the most predictive features because buildings tend to repeat their energy behaviour from day to day.

| Feature | Formula | Why It Matters |
|---|---|---|
| `lag_1_day` | `meter_reading_daily` shifted by 1 day per building | Yesterday's consumption is the strongest predictor of today's |
| `lag_7_days` | `meter_reading_daily` shifted by 7 days | Same weekday last week — captures weekly cycle |
| `rolling_7d_mean` | Mean of last 7 days (shifted by 1) | Short-term trend |
| `rolling_14d_mean` | Mean of last 14 days (shifted by 1) | Medium-term trend |
| `rolling_7d_std` | Std of last 7 days (shifted by 1) | Variability / stability signal |

> [!IMPORTANT]
> **Leakage Prevention Rule:** All lag and rolling features are computed **grouped by `building_id`** so that building A's history never influences building B's features. Rolling windows use `.shift(1)` to ensure day D only uses data from days D-1, D-2, … D-N.

**Leakage verification result:** ✅ PASSED — `lag_1_day` on day D exactly equals `meter_reading_daily` on day D-1 for all buildings tested.

---

### 3.3 Raw Weather Features (7 features, already in Phase 4 output)

| Feature | Description |
|---|---|
| `air_temperature_mean` | Mean hourly temperature (°C) |
| `air_temperature_min` | Minimum hourly temperature (°C) |
| `air_temperature_max` | Maximum hourly temperature (°C) |
| `dew_temperature_mean` | Mean dew point temperature (°C) — proxy for humidity |
| `wind_speed_mean` | Mean wind speed (m/s) |
| `precip_depth_1hr_total` | Total precipitation depth (mm) |
| `sea_lvl_pressure_mean` | Mean sea-level pressure (mbar) |

---

### 3.4 Derived Weather Features (2 features)

| Feature | Formula | Why It Matters |
|---|---|---|
| `cooling_degree_day` | `max(0, air_temperature_mean − 18.0)` | Measures how much hotter than 18°C the day was. When CDD > 0, buildings need air-conditioning → higher energy use. |
| `temp_range` | `air_temperature_max − air_temperature_min` | High daily swings often correlate with unusual energy patterns. |

**Balance point = 18°C** — the standard meteorological threshold below which space cooling is typically not needed.

**Example:**
```
Mean temperature = 25°C  →  CDD = 25 - 18 = 7
Mean temperature = 10°C  →  CDD = max(0, 10 - 18) = 0  (no cooling load)
```

---

### 3.5 Building Metadata Features (17 features total)

| Feature | Encoding | Why It Matters |
|---|---|---|
| `log_sqm` | log(1 + sqm) | Building size; log-transform compresses the wide range (100 → 60,000 m²) |
| `yearbuilt_norm` | Numeric, NaN filled with median | Older buildings may be less energy-efficient |
| `usage_*` (15 columns) | One-Hot Encoding, drop_first=True | Different building types have very different baseline consumption |

**One-Hot Encoding example:**
```
primaryspaceusage = 'Office'
→  usage_Office = 1,  all other usage_* = 0

primaryspaceusage = 'Education'  (dropped as reference category)
→  all usage_* = 0
```

**15 usage dummy columns created:**
`usage_Entertainment/public assembly`, `usage_Food sales and service`,
`usage_Healthcare`, `usage_Lodging/residential`, `usage_Manufacturing/industrial`,
`usage_Office`, `usage_Other`, `usage_Parking`, `usage_Public services`,
`usage_Religious worship`, `usage_Retail`, `usage_Services`,
`usage_Technology/science`, `usage_Utility`, `usage_Warehouse/storage`

> [!NOTE]
> `drop_first=True` removes the **Education** category as the reference level (all-zeros encodes Education). This avoids the dummy variable trap (perfect multicollinearity).

---

## 4. Warm-Up Row Removal

The lag and rolling window features are undefined for the first 14 days per building
(insufficient history). These rows are dropped before saving.

| Statistic | Value |
|---|---|
| Rows before warm-up removal | 1,004,892 |
| Warm-up rows dropped | 10,080 (7 days × 1,440 buildings) |
| **Rows in final feature dataset** | **994,812** |
| Percentage retained | 99.0% |

---

## 5. Output Dataset

| Property | Value |
|---|---|
| File (CSV) | `data/processed/features_dataset.csv` |
| File (Parquet) | `data/processed/features_dataset.parquet` |
| Rows | 994,812 |
| Columns | 40 (3 ID + 1 target + 37 features) |
| Buildings | 1,440 |
| Date range | 2016-01-08 → 2017-12-31 |
| NaN in target | 0 |
| NaN in lag_1_day | 0 |
| Feature summary table | `results/tables/04_feature_summary.csv` |

---

## 6. Final Feature List (37 features used by models)

| # | Feature | Category | Type |
|---|---|---|---|
| 1 | day_of_week | Calendar | Ordinal (0-6) |
| 2 | is_weekend | Calendar | Binary |
| 3 | month | Calendar | Ordinal (1-12) |
| 4 | season | Calendar | Ordinal (0-3) |
| 5 | day_of_year | Calendar | Numeric |
| 6 | week_of_year | Calendar | Numeric |
| 7 | lag_1_day | Lag/Rolling | Numeric (kWh) |
| 8 | lag_7_days | Lag/Rolling | Numeric (kWh) |
| 9 | rolling_7d_mean | Lag/Rolling | Numeric (kWh) |
| 10 | rolling_14d_mean | Lag/Rolling | Numeric (kWh) |
| 11 | rolling_7d_std | Lag/Rolling | Numeric (kWh) |
| 12 | air_temperature_mean | Weather | Numeric (°C) |
| 13 | air_temperature_min | Weather | Numeric (°C) |
| 14 | air_temperature_max | Weather | Numeric (°C) |
| 15 | dew_temperature_mean | Weather | Numeric (°C) |
| 16 | wind_speed_mean | Weather | Numeric (m/s) |
| 17 | precip_depth_1hr_total | Weather | Numeric (mm) |
| 18 | sea_lvl_pressure_mean | Weather | Numeric (mbar) |
| 19 | cooling_degree_day | Derived Weather | Numeric (°C) |
| 20 | temp_range | Derived Weather | Numeric (°C) |
| 21 | log_sqm | Building | Numeric |
| 22 | yearbuilt_norm | Building | Numeric |
| 23-37 | usage_* (15 columns) | Building | Binary dummy |

---

## 7. Graphs Produced

| Graph | File | Description |
|---|---|---|
| Lag verification | `results/graphs/10_lag_feature_verification.png` | Shows actual vs lag_1_day vs rolling_7d_mean for one building |
| Calendar distributions | `results/graphs/11_calendar_feature_distributions.png` | Median consumption by weekday, month, season |
| Derived weather | `results/graphs/12_derived_weather_vs_consumption.png` | CDD and temp_range vs log-consumption scatter |
| Lag scatter | `results/graphs/13_lag_feature_scatter.png` | lag_1_day and rolling_7d_mean vs actual consumption |

---

## 8. Data Leakage Prevention — Summary

> [!CAUTION]
> **Data leakage** is when information from the future (day D or later) is accidentally used to predict day D. It inflates model performance during training but causes the model to fail in real use.

| Leakage Risk | How We Prevented It |
|---|---|
| Lag features bleeding across buildings | Computed using `groupby('building_id')` before shifting |
| Rolling window including current day | Used `.shift(1)` before `.rolling()` |
| Future weather data | Only daily weather for day D itself is used (available at end of day) |
| Test set information during training | Train/validation/test split done chronologically in Phase 8 |
| NaN rows creating implicit future reference | Warm-up rows dropped before saving |

---

## 9. What's Next — Phase 6

**Phase 6:** System Design and Architecture
- System architecture diagram
- Data processing flowchart
- Model training workflow diagram
- Prediction pipeline diagram
- Save all diagrams in `docs/architecture/`

---

*Report generated by Smart Building Energy Prediction Project — Phase 5*
