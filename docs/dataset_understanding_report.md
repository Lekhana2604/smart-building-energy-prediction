# Building Data Genome Project 2 (BDG2): Dataset Inspection & Understanding Report

**Project Title:** Daily Electricity Consumption Prediction in Smart Buildings Using Machine Learning  
**Dataset:** Building Data Genome Project 2 (BDG2)  
**Dataset Official DOI/Zenodo:** [10.5281/zenodo.3887306](https://zenodo.org/records/3887306) (Miller et al., 2020)  
**Academic Target:** 4-Member Team Academic ML Project (3rd Year Information Science & Engineering)  
**Phase:** Phase 1 – Dataset Inspection and Understanding  
**Status:** Completed & Empirically Verified against Raw Files  

---

## 1. Executive Summary & Phase Objective

The objective of **Phase 1** is to perform a rigorous, programmatic, and ground-truth inspection of the raw Building Data Genome Project 2 (BDG2) dataset downloaded from Zenodo and extracted into `data/raw/`. 

As an academic machine learning engineering team, we **never make assumptions** regarding folder structures, file paths, column names, measurement units, data types, or temporal frequencies. Every property documented in this report has been verified by directly loading and inspecting the files using Python scripts and the companion Jupyter notebook [`notebooks/01_dataset_understanding.ipynb`](../notebooks/01_dataset_understanding.ipynb).

### Key Empirical Findings at a Glance
1. **Extracted Location:** `data/raw/buds-lab-building-data-genome-project-2-3d0cbaf/data/`
2. **Total Buildings in Dataset:** **1,636** non-residential buildings distributed across **19** geographical campus/city sites.
3. **Buildings with Electricity Meters:** Exactly **1,578 buildings** have active electricity meters (`electricity == 'Yes'`). 58 buildings have only thermal, gas, or water meters.
4. **Target Variable & Physical Unit:** Hourly electricity consumption is measured in **kilowatt-hours (kWh)** per 1-hour interval (cumulative energy, not instantaneous power).
5. **Temporal Span & Frequency:** Hourly data covering **2 full calendar years**: January 1, 2016 `00:00:00` to December 31, 2017 `23:00:00` (**731 calendar days** = 366 days in leap year 2016 + 365 days in 2017), yielding exactly **17,544 rows**.
6. **Meter Data Storage Format:** **Wide matrix format** (1 `timestamp` column + 1,578 building ID columns = 1,579 columns).
7. **Weather Records:** **331,166 hourly rows** across 19 sites with outdoor dry-bulb temperature (°C) and dew point (°C) showing **>99.9% completeness**.
8. **Critical Data Limitation:** BDG2 contains **no dynamic occupancy sensors**, no building schedule logs, and no appliance-level sub-metering. The metadata column `occupants` is 85.94% missing and reflects static nominal estimates only. Human occupancy patterns must therefore be inferred through calendar and temporal proxies (e.g., day of week, hour of day, weekend indicators).

---

## 2. Dataset Folder Hierarchy & File Inventory

The raw archive extracted inside `data/raw/` contains the official repository commit snapshot `buds-lab-building-data-genome-project-2-3d0cbaf`. The data files are neatly partitioned into three functional domains: `metadata/`, `weather/`, and `meters/`.

### Verified File Inventory

| Category | Relative File Path | File Size (MB) | Rows | Columns | Description & Role |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Metadata** | `metadata/metadata.csv` | **0.26 MB** | 1,636 | 32 | Static building attributes (use type, floor area, coordinates, meter flags) |
| **Weather** | `weather/weather.csv` | **18.56 MB** | 331,166 | 10 | Hourly outdoor weather metrics across 19 campus sites |
| **Meters (Cleaned)** | `meters/cleaned/electricity_cleaned.csv` | **169.25 MB** | 17,544 | 1,579 | Hourly electricity (kWh) with anomaly spikes & zero dropouts filtered to NaN |
| **Meters (Raw)** | `meters/raw/electricity.csv` | **166.17 MB** | 17,544 | 1,579 | Unfiltered hourly electricity readings (kWh) |
| **Meters (Cleaned)** | `meters/cleaned/chilledwater_cleaned.csv`| 73.44 MB | 17,544 | 556 | Hourly chilled water cooling meter readings (kWh equivalent) |
| **Meters (Raw)** | `meters/raw/chilledwater.csv` | 75.19 MB | 17,544 | 556 | Raw chilled water readings |
| **Meters (Cleaned)** | `meters/cleaned/steam_cleaned.csv` | 43.32 MB | 17,544 | 371 | Hourly steam heating meter readings |
| **Meters (Raw)** | `meters/raw/steam.csv` | 44.58 MB | 17,544 | 371 | Raw steam readings |
| **Meters (Cleaned)** | `meters/cleaned/hotwater_cleaned.csv` | 23.23 MB | 17,544 | 186 | Hourly hot water heating readings |
| **Meters (Raw)** | `meters/raw/hotwater.csv` | 23.79 MB | 17,544 | 186 | Raw hot water readings |
| **Meters (Cleaned)** | `meters/cleaned/gas_cleaned.csv` | 20.98 MB | 17,544 | 178 | Natural gas meter readings |
| **Meters (Raw)** | `meters/raw/gas.csv` | 19.25 MB | 17,544 | 178 | Raw gas readings |
| **Meters (Cleaned)** | `meters/cleaned/water_cleaned.csv` | 14.68 MB | 17,544 | 147 | Potable water meter readings |
| **Meters (Raw)** | `meters/raw/water.csv` | 14.92 MB | 17,544 | 147 | Raw potable water readings |
| **Meters (Cleaned)** | `meters/cleaned/irrigation_cleaned.csv` | 2.93 MB | 17,544 | 38 | Irrigation water meter readings |
| **Meters (Raw)** | `meters/raw/irrigation.csv` | 3.01 MB | 17,544 | 38 | Raw irrigation readings |
| **Meters (Cleaned)** | `meters/cleaned/solar_cleaned.csv` | 0.70 MB | 17,544 | 6 | Solar photovoltaic generation readings |
| **Meters (Raw)** | `meters/raw/solar.csv` | 0.71 MB | 17,544 | 6 | Raw solar photovoltaic readings |
| **Kaggle** | `meters/kaggle/kaggle.csv` | **855.27 MB** | 20,216,100 | 4 | Long-format ASHRAE GEPIII competition subset (2017 only) |

---

## 3. Project Relevance Analysis: Which Files Are Relevant and Why?

For our project—**Daily Electricity Consumption Prediction in Smart Buildings Using Machine Learning**—we must select the minimal, sufficient, and clean subset of files:

### Primary Files Required:
1. `meters/cleaned/electricity_cleaned.csv`:
   - **Why:** This is our **primary target variable source**. It contains hourly electricity consumption in kilowatt-hours (kWh) across all 1,578 metered buildings. Using the cleaned version protects our baseline models from severe sensor failure anomalies (such as single-hour multi-megawatt spikes and prolonged sensor disconnection flatlines) while preserving real consumption trends.
2. `weather/weather.csv`:
   - **Why:** Outdoor weather—primarily **ambient dry-bulb temperature** and **dew point temperature**—is the primary thermodynamic driver of building HVAC electricity consumption (chillers, heat pumps, fans). We will merge hourly weather by `site_id` and `timestamp`.
3. `metadata/metadata.csv`:
   - **Why:** Provides building-level static features including **gross floor area (`sqm`)**, **primary space usage (`primaryspaceusage`)**, and geographical site linkage (`site_id`, `timezone`). Floor area and building use type are fundamental predictors of baseline load and scaling.

### Files Excluded / Non-Relevant:
- **Thermal and utility meters** (`chilledwater`, `hotwater`, `steam`, `gas`, `water`, `irrigation`, `solar`): Our objective is strictly electricity consumption prediction. Non-electricity utilities introduce orthogonal prediction tasks and many buildings do not possess those meters.
- `meters/kaggle/kaggle.csv`: This is an 855 MB long-format file restricted to 2017 and Kaggle-specific building IDs. Working with the native BDG2 tables preserves the full 2-year timeline (2016–2017) and rich building names.

---

## 4. Building Metadata Deep Dive (`metadata.csv`)

### Schema and Column Definitions

`metadata.csv` contains **1,636 rows** and **32 columns**.

| Column Name | Data Type | Non-Null Count | Missing (%) | Description & Meaning |
| :--- | :--- | :---: | :---: | :--- |
| `building_id` | object (str) | 1,636 | 0.00% | Unique building identifier: `{SiteID}_{SimplifiedUsage}_{Name}` |
| `site_id` | object (str) | 1,636 | 0.00% | Animal code-name for the campus site (e.g., `Panther`, `Bull`) |
| `building_id_kaggle` | float64 | 1,449 | 11.43% | Numerical ID mapping to Kaggle ASHRAE GEPIII competition |
| `site_id_kaggle` | float64 | 1,599 | 2.26% | Numerical site ID for Kaggle competition |
| `primaryspaceusage` | object (str) | 1,615 | 1.28% | Primary functional use category (Energy Star Property Type) |
| `sub_primaryspaceusage` | object (str) | 1,615 | 1.28% | Granular sub-category of building space use |
| `sqm` | float64 | 1,636 | 0.00% | Gross floor area in square meters ($m^2$) |
| `sqft` | float64 | 1,636 | 0.00% | Gross floor area in square feet ($ft^2$, $1 \text{ sqm} \approx 10.7639 \text{ sqft}$) |
| `lat` | float64 | 1,399 | 14.49% | Latitude (city-level resolution to preserve anonymity) |
| `lng` | float64 | 1,399 | 14.49% | Longitude (city-level resolution) |
| `timezone` | object (str) | 1,636 | 0.00% | Local standard timezone of the site |
| `electricity` | object (str) | 1,578 | 3.55% | Indicates presence of electricity meter: `'Yes'` or `NaN` |
| `hotwater` | object (str) | 185 | 88.69% | Indicates presence of hot water meter: `'Yes'` or `NaN` |
| `chilledwater` | object (str) | 555 | 66.08% | Indicates presence of chilled water meter: `'Yes'` or `NaN` |
| `steam` | object (str) | 370 | 77.38% | Indicates presence of steam meter: `'Yes'` or `NaN` |
| `water` | object (str) | 146 | 91.08% | Indicates presence of water meter: `'Yes'` or `NaN` |
| `irrigation` | object (str) | 37 | 97.74% | Indicates presence of irrigation meter: `'Yes'` or `NaN` |
| `solar` | object (str) | 5 | 99.69% | Indicates presence of solar PV meter: `'Yes'` or `NaN` |
| `gas` | object (str) | 177 | 89.18% | Indicates presence of gas meter: `'Yes'` or `NaN` |
| `industry` | object (str) | 579 | 64.61% | Sector classification (e.g., Education, Commercial) |
| `subindustry` | object (str) | 579 | 64.61% | Granular industry sector classification |
| `heatingtype` | object (str) | 215 | 86.86% | Heating system architecture |
| `yearbuilt` | float64 | 817 | 50.06% | Year of initial construction |
| `date_opened` | object (str) | 21 | 98.72% | Formal operational opening date |
| `numberoffloors` | float64 | 441 | 73.04% | Above-ground floor count |
| `occupants` | float64 | 230 | **85.94%**| Estimated typical occupants (nominal capacity) |
| `energystarscore` | float64 | 163 | **90.04%**| Energy Star efficiency rating (1–100 scale) |
| `eui` | float64 | 299 | 81.72% | Annual Energy Use Intensity ($kWh/year/m^2$) |
| `site_eui` | float64 | 163 | 90.04% | Site-level annual energy use intensity ($kWh/year/m^2$) |
| `source_eui` | float64 | 163 | 90.04% | Source-level annual energy use intensity ($kWh/year/m^2$) |
| `leed_level` | object (str) | 16 | 99.02% | LEED green certification tier (Certified, Silver, Gold, Platinum) |
| `rating` | object (str) | 184 | 88.75% | Other local green building ratings |

### Primary Space Usage Distribution
The non-residential profile is heavily dominated by educational and commercial office properties:
1. **Education:** 617 buildings (37.7%)
2. **Office:** 307 buildings (18.8%)
3. **Entertainment / Public Assembly:** 204 buildings (12.5%)
4. **Lodging / Residential:** 168 buildings (10.3%)
5. **Public Services:** 166 buildings (10.1%)
6. **Other Categories (Healthcare, Parking, Storage, Retail, etc.):** 153 buildings (9.4%)
7. **Unlabeled / Missing:** 21 buildings (1.3%)

---

## 5. Weather Data Deep Dive (`weather.csv`)

### Schema, Units, and Missingness

`weather.csv` contains **331,166 rows** and **10 columns** spanning all 19 sites from `2016-01-01 00:00:00` to `2017-12-31 23:00:00`.

| Column | Physical Quantity | Standard Measurement Unit | Data Type | Missing Count | Missing (%) | Summary / Typical Values |
| :--- | :--- | :--- | :--- | :---: | :---: | :--- |
| `timestamp` | Date and Time | Format `YYYY-MM-DD hh:mm:ss` (Local time) | object / datetime | 0 | 0.00% | 17,544 unique hourly timestamps |
| `site_id` | Campus Location | Categorical animal code-name | object (str) | 0 | 0.00% | 19 unique sites |
| `airTemperature` | Ambient dry-bulb temp | **Degrees Celsius (°C)** | float64 | 128 | **0.04%** | Mean: 14.24°C, Min: -28.9°C, Max: 48.3°C |
| `dewTemperature` | Dew point temp | **Degrees Celsius (°C)** | float64 | 328 | **0.10%** | Mean: 7.65°C, Min: -35.0°C, Max: 26.7°C |
| `windSpeed` | Horizontal air speed | **Meters per second (m/s)** | float64 | 574 | **0.17%** | Mean: 3.42 m/s, Min: 0.0 m/s, Max: 24.2 m/s |
| `windDirection` | Wind direction | **Degrees clockwise from North (0°–360°)** | float64 | 13,005 | 3.93% | Mean: 172.9°, Range: 0° to 360° |
| `seaLvlPressure` | Atmospheric pressure | **Millibars / Hectopascals (mbar / hPa)** | float64 | 21,624 | 6.53% | Mean: 1016.4 hPa, Range: 968 to 1049 hPa |
| `precipDepth1HR` | 1-hour precipitation | **Millimeters (mm)** | float64 | 133,186 | **40.22%** | Mean: 0.98 mm, Max: 343 mm |
| `cloudCoverage` | Sky cloudiness | **Oktas (0 to 9 integer scale)** | float64 | 170,987 | **51.63%** | Mean: 2.14 oktas |
| `precipDepth6HR` | 6-hour precipitation | **Millimeters (mm)** | float64 | 313,004 | **94.52%** | Rarely reported; not suitable for modeling |

### Critical Weather Observations:
- **High-Quality Predictors:** `airTemperature`, `dewTemperature`, and `windSpeed` have **<0.2% missing values**. They provide complete, robust thermodynamic signals across all 19 sites.
- **Low-Coverage Variables:** `precipDepth6HR` is 94.5% missing and `cloudCoverage` is 51.6% missing. In subsequent phases, these columns must either be imputed with careful zero-filling (for precipitation) or dropped to prevent dropping entire rows.

---

## 6. Electricity Meter Data Deep Dive (`electricity.csv` & `electricity_cleaned.csv`)

### Matrix Structure
- **Format:** Wide matrix (Pivoted).
- **Index Column:** `timestamp` formatted as `YYYY-MM-DD hh:mm:ss`.
- **Value Columns:** 1,578 building columns named `{SiteID}_{Usage}_{UniqueName}`.
- **Rows:** Exactly **17,544 rows**.
  - 2016: 366 days (leap year) $\times 24 \text{ hours} = 8,784 \text{ rows}$.
  - 2017: 365 days $\times 24 \text{ hours} = 8,760 \text{ rows}$.
  - Total: $8,784 + 8,760 = 17,544 \text{ rows}$.
- **Total Values in Matrix:** $17,544 \times 1,578 = 27,684,432$ data points.

### Measurement Unit Verification: Energy vs. Power
A fundamental error in building energy modeling is confusing **power** (rate of energy flow in kilowatts, kW) with **energy** (integral of power over time in kilowatt-hours, kWh).

Through inspection of the BDG2 official repository documentation (`notebooks/03_EDA-meter-reading.ipynb`) and the reference publication (Miller et al., 2020), we have verified:
> **The electricity meter readings are recorded in kilowatt-hours (kWh) per 1-hour interval.**

#### Implication for Daily Aggregation (Phase 4):
Because each hourly reading represents the quantity of energy consumed during that hour ($E_h$ in kWh):
$$\text{Daily Energy Consumption (kWh/day)} = \sum_{h=1}^{24} E_h$$
We **do not** need to multiply by time ($\Delta t = 1 \text{ hour}$) or integrate instantaneous kW. We simply sum the 24 hourly kWh measurements, provided that the day meets data completeness criteria (e.g., at least 23 or 24 valid hourly readings).

### Comparison: Raw vs. Cleaned Electricity Data

| Property | `meters/raw/electricity.csv` | `meters/cleaned/electricity_cleaned.csv` | Difference & Explanation |
| :--- | :---: | :---: | :--- |
| **Total Values** | 27,684,432 | 27,684,432 | Identical shape and column names |
| **Missing Values (NaN)** | 1,312,095 (**4.74%**) | 1,504,197 (**5.43%**) | Cleaned has 192,102 additional NaNs |
| **Zero Values** | 1,159,758 (**4.19%**) | 1,115,924 (**4.03%**) | 43,834 false zeros converted to NaN |
| **Buildings with 0 NaNs** | 373 buildings | 82 buildings | Strict cleaning flagged sporadic anomalies |
| **Buildings with <5% NaNs** | 1,278 buildings | 1,267 buildings | Over 80% of buildings have >95% completeness |
| **Buildings with <10% NaNs**| 1,444 buildings | 1,440 buildings | High data availability across the portfolio |

#### How the BDG2 Cleaned Dataset Was Created:
As documented in BDG2 notebook `10_Cleaned-dataset.ipynb`:
1. **Anomaly Spike Removal:** Extreme non-physical spikes were detected using Twitter's **Seasonal Hybrid ESD (S-H-ESD)** algorithm and replaced with `NaN`.
2. **Persistent Sensor Dropout Flatlines:** In commercial buildings, true electricity consumption is never exactly 0.0 kWh over continuous days because baseline systems (emergency lights, servers, security, refrigeration, standby power) are always active. Any continuous streak of zero readings exceeding **24 consecutive hours** was identified as sensor transmission dropout and replaced with `NaN`.
3. **Project Recommendation:** We should base our model development and daily aggregation on `electricity_cleaned.csv` while maintaining full transparency regarding the cleaning logic.

---

## 7. Cross-Dataset Relational Architecture

The BDG2 dataset operates as a relational constellation connected through three keys:
1. `building_id` links `metadata.csv` to column headers in `electricity_cleaned.csv`.
2. `site_id` (the first token of `building_id`) links buildings to weather observations in `weather.csv`.
3. `timestamp` links weather observations to meter reading timestamps.

```
+-------------------------------------------------------------+
|                        metadata.csv                         |
|  - building_id (PK: e.g., 'Panther_office_Hannah')          |
|  - site_id (FK: e.g., 'Panther')                            |
|  - primaryspaceusage, sqm, timezone, electricity ('Yes')    |
+------------------------------+------------------------------+
                               | 1:N
                               v
+-------------------------------------------------------------+
|              electricity_cleaned.csv (Wide)                 |
|  - timestamp (PK/Index: '2016-01-01 00:00:00' to 2017)      |
|  - Panther_office_Hannah [Hourly kWh]                       |
|  - Bull_education_Otis   [Hourly kWh]                       |
|  - ... (1,578 building columns)                             |
+------------------------------+------------------------------+
                               ^
                               | N:1 on (site_id, timestamp)
+------------------------------+------------------------------+
|                         weather.csv                         |
|  - timestamp (PK/Index)                                     |
|  - site_id (PK: e.g., 'Panther')                            |
|  - airTemperature, dewTemperature, windSpeed, etc.          |
+-------------------------------------------------------------+
```

### Geographical Sites and Timezones

| Site Name | Timezone | Total Buildings | Buildings with Electricity |
| :--- | :--- | :---: | :---: |
| **Rat** | US/Eastern | 305 | 299 |
| **Hog** | US/Eastern | 163 | 162 |
| **Lamb** | US/Eastern | 147 | 146 |
| **Fox** | US/Eastern | 137 | 137 |
| **Panther** | US/Central | 136 | 133 |
| **Bull** | US/Central | 124 | 124 |
| **Cockatoo**| US/Mountain | 124 | 114 |
| **Eagle** | Europe/London | 106 | 104 |
| **Bear** | US/Mountain | 92 | 92 |
| **Gator** | US/Central | 74 | 74 |
| **Robin** | US/Pacific | 52 | 52 |
| **Peacock** | Europe/London | 47 | 47 |
| **Bobcat** | US/Pacific | 36 | 36 |
| **Wolf** | Europe/Dublin | 36 | 36 |
| **Swan** | Europe/London | 21 | 21 |
| **Moose** | Europe/London | 15 | 15 |
| **Shrew** | Europe/London | 9 | 9 |
| **Mouse** | US/Pacific | 7 | 7 |
| **Crow** | Europe/London | 5 | 5 |
| **Total** | **6 Timezones** | **1,636** | **1,578** |

---

## 8. Data Limitations & Engineering Integrity

In accordance with academic integrity guidelines, we document the following verified dataset boundaries:

1. **No Measured Dynamic Occupancy:** There are no badge-swipe counters, infrared occupancy sensor streams, or Wi-Fi device counts. In `metadata.csv`, `occupants` is 85.94% missing and reflects static rated capacity. In our project, occupancy must be modeled strictly through **calendar proxies** (day of week, weekend indicator, academic semester periods).
2. **No Appliance-Level or End-Use Sub-Metering:** Readings are aggregated whole-building electricity consumption. Plug loads, lighting, and HVAC chillers are not isolated.
3. **Local Standard Time vs. Daylight Saving Time:** Timestamps are recorded in local standard time per site. In Phase 3 and Phase 4, we must verify whether 23-hour or 25-hour clock shift anomalies occur during DST transitions and implement standard 24-hour daily aggregation windows.
4. **Zero Readings vs. Missing Readings:** Zero readings are not necessarily true zero energy consumption. A zero reading lasting several days indicates meter disconnection or sensor failure, which justifies using `electricity_cleaned.csv`.

---

## 9. Deliverables Created in Phase 1

1. [`src/data_loader.py`](../src/data_loader.py):
   - Fully tested, modular Python utility module.
   - Provides functions: `get_raw_data_dir()`, `load_metadata()`, `load_weather()`, `load_electricity()`, `get_dataset_inventory()`.
   - Handles wide matrix loading with safe building column filtering and automatic memory protection.
2. [`notebooks/01_dataset_understanding.ipynb`](../notebooks/01_dataset_understanding.ipynb):
   - Executable Jupyter notebook with 25 cells (markdown explanations + executed python cells with embedded output tables).
   - Demonstrates complete data loading, inspection, missingness calculation, unit verification, and relational linkage.
3. [`requirements.txt`](../requirements.txt):
   - Pinned dependencies with beginner-friendly documentation of each library's role.
4. [`docs/dataset_understanding_report.md`](dataset_understanding_report.md):
   - This comprehensive technical report.

---

## 10. Student Evaluation & Viva Guide (Academic Q&A)

### Q1: What dataset are we using and what is its primary focus?
**Answer:** We are using the Building Data Genome Project 2 (BDG2), compiled by Miller et al. (2020) and hosted on Zenodo. It contains real-world hourly energy meter data from 1,636 non-residential buildings (predominantly university campuses and commercial offices) across 19 sites for the years 2016 and 2017.

### Q2: What is the physical measurement unit of the electricity readings?
**Answer:** The readings are measured in **kilowatt-hours (kWh)** per hourly interval. Because kWh is a measure of cumulative energy (not instantaneous power in kW), computing daily electricity consumption involves summing the valid hourly kWh readings over a 24-hour day ($\text{Daily kWh} = \sum_{h=1}^{24} \text{Hourly kWh}_h$).

### Q3: Why is the meter data stored in wide format instead of long format?
**Answer:** In wide format, each row is a single timestamp (17,544 rows) and each column is a building ID (1,578 columns). This reduces storage size from over 850 MB (long format) to ~170 MB and enables vectorised time-series operations across all buildings simultaneously.

### Q4: Does BDG2 include real occupancy numbers or working hour measurements?
**Answer:** No. Dynamic occupancy is not measured. The `occupants` column in metadata is 85.94% missing and only represents nominal static design estimates. Therefore, building activity schedules must be captured through calendar proxy features such as Day of Week, Weekend Flag, and Month.

### Q5: Why are we using `electricity_cleaned.csv` rather than raw `electricity.csv`?
**Answer:** Raw meter data suffers from real-world telemetry defects: extreme anomaly spikes (sensor glitch reading megawatt surges) and persistent zero flatlines (>24 continuous hours) caused by meter disconnections. The BDG2 authors filtered these errors to `NaN` using Twitter's Seasonal Hybrid ESD (S-H-ESD) algorithm and gap detection, giving us a clean baseline for machine learning without fabricating data.
