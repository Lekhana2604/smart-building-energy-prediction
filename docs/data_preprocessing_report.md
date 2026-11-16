# Building Data Genome Project 2 (BDG2): Data Preprocessing Report

**Project Title:** Daily Electricity Consumption Prediction in Smart Buildings Using Machine Learning  
**Dataset:** Building Data Genome Project 2 (BDG2)  
**Academic Target:** 4-Member Team Academic ML Project (3rd Year Information Science & Engineering)  
**Phase:** Phase 3 – Data Preprocessing & Cleaning  
**Status:** Completed, Empirically Verified & Saved in `data/interim/`  
**Generated Artifacts:** 4 Clean Interim Datasets in `data/interim/`, 1 Audit Table in `results/tables/`, and Executed Notebook [`notebooks/03_data_preprocessing.ipynb`](../notebooks/03_data_preprocessing.ipynb)  

---

## 1. Executive Summary & Phase Objective

In raw telemetry from IoT smart meters and weather stations, missing values, sensor glitches, clock shifts, and incomplete days are ubiquitous. Data preprocessing is the systematic, mathematically justified process of converting raw, dirty observations into consistent, leak-free intermediate datasets.

The objective of **Phase 3** is to clean, standardize, and align the building metadata, meteorological observations, and electricity meter readings before daily aggregation and feature engineering.

### Key Preprocessing Achievements
1. **Building Screening:** Filtered raw metadata (1,636 buildings) down to **1,440 highly reliable, eligible non-residential buildings** by screening out buildings without electricity meters (58), missing space usage labels (19), or exceeding the 10% missingness threshold (119).
2. **Daylight Saving Time (DST) Resolution:** Identified and resolved the 1-hour "Fall Back" duplicate timestamp anomaly present in all 19 sites' weather records on the first Sunday of November, ensuring a strictly unique 1-to-1 temporal index.
3. **Meteorological Standardization:** Reindexed and interpolated all 19 site weather stations onto an exact, continuous **17,544-hour calendar grid** ($19 \times 17,544 = \mathbf{333,336\text{ records}}$). Eliminated 100% of missing values in primary thermal features (`airTemperature`, `dewTemperature`, `windSpeed`).
4. **Data Integrity Verification:** Verified across all 25 million electricity data points that **zero negative values** exist.
5. **Daily Completeness Audit:** Audited all $1,440 \times 731 = 1,052,640$ building-days for hourly completeness. Achieved an average **daily usability rate of 95.46%** across the portfolio, defining the exact threshold ($\ge 23$ valid hours/day) for Phase 4 daily aggregation.
6. **Integrity Rule Followed:** All intermediate outputs were saved into `data/interim/`. The original Zenodo archive in `data/raw/` was left **100% unmodified**.

---

## 2. In-Depth Preprocessing Decisions & Engineering Rationale

### Decision 1: Building Portfolio Screening Rules
**Problem:** Raw `metadata.csv` lists 1,636 buildings. However, building energy modeling requires complete categorical descriptors, positive floor areas, and active electricity meters with reliable time series.

**Our Multi-Stage Screening Filter:**
```
Raw Buildings in Metadata: 1,636
  ├── [Step 1] Filter 'electricity == Yes': 1,578 retained (58 dropped - no electricity meter)
  ├── [Step 2] Filter 'primaryspaceusage.notnull()' & 'sqm > 0': 1,559 retained (19 dropped - missing usage label)
  └── [Step 3] Filter 'missing_hourly_pct <= 10.0%': 1,440 retained (119 dropped - severe sensor outages)
Final Cleaned Building Portfolio: 1,440 buildings
```

- **Why 10% Missingness Cutoff?** Buildings with $>10\%$ missing hours (over 1,750 missing hours out of 17,544) suffer from long-term sensor disconnections or decommissioning. Including them would introduce artificial flatlines and distort rolling averages. Filtering at 10% retains **over 91% of metered buildings** while eliminating broken sensors.
- **Output:** Saved as [`data/interim/clean_metadata.csv`](../data/interim/clean_metadata.csv).

---

### Decision 2: Weather Data Alignment & Daylight Saving Time (DST) Handling
**Problem:** 
1. Across all 19 sites in `weather.csv`, exactly 2 duplicate timestamps were discovered:
   - `2016-11-06 01:00:00`
   - `2017-11-05 01:00:00`
   These dates correspond to the annual **Daylight Saving Time (DST) "Fall Back"** in North America and Europe, where local clocks are set back 1 hour, causing the 1:00 AM hour to occur twice.
2. Across the 19 sites, approximately 2,170 sporadic hourly readings were missing due to transmission glitches.
3. Feature `precipDepth6HR` had 94.52% missing data, and `precipDepth1HR` had 40.22% missing data.

**Our Engineering Solutions:**
1. **DST Duplicate Aggregation:** Duplicate timestamps for each site were grouped and averaged:
   $$\bar{W}_{t} = \frac{1}{K} \sum_{k=1}^K W_{t, k}$$
   where $W$ is any numerical weather variable and $K=2$ is the duplicate hour count. This preserves the measured thermodynamic conditions during both occurrences without creating duplicate index labels.
2. **Continuous Grid Reindexing:** Reindexed each site to the exact calendar range of 17,544 hours: `2016-01-01 00:00:00` to `2017-12-31 23:00:00`.
3. **Time-Based Linear Interpolation:** For continuous thermodynamic variables (`airTemperature`, `dewTemperature`, `windSpeed`, `seaLvlPressure`), missing values were imputed using time-weighted linear interpolation with a maximum consecutive gap limit of 24 hours, followed by backward/forward fill for boundary endpoints.
4. **Precipitation Handling:** Missing values in `precipDepth1HR` were filled with `0.0` (physical assumption: unrecorded precipitation represents dry weather without rainfall). `precipDepth6HR` was dropped to avoid discarding 95% of rows.
5. **Output:** Saved as [`data/interim/clean_weather_hourly.csv`](../data/interim/clean_weather_hourly.csv) ($333,336$ rows with $0$ missing temperature values).

---

### Decision 3: Electricity Meter Integrity & Negative Value Investigation
**Problem:** Smart meters can occasionally record negative values due to reverse power flow, incorrect calibration, or data logger corruption.

**Our Verification:**
- We queried all 17,544 rows across all 1,440 eligible buildings ($25,263,360$ total data points) in `electricity_cleaned.csv`.
- **Finding:** Exactly **0 negative values** were detected.
- **Output:** Subsampled clean matrix saved as [`data/interim/clean_electricity_hourly.csv`](../data/interim/clean_electricity_hourly.csv) (159.3 MB).

---

### Decision 4: Hourly Completeness & Daily Usability Thresholding
**Problem:** In Phase 4, we must aggregate hourly energy readings into daily consumption:
$$\text{Daily Consumption (kWh/day)} = \sum_{h=1}^{24} E_h$$
If a building has only 5 or 10 valid hours on a given day because of a power outage or sensor drop, summing those hours would artificially report a low daily consumption, creating severe target noise.

**Our Engineering Solution:**
We audited every single building-date combination ($1,440\text{ buildings} \times 731\text{ days} = \mathbf{1,052,640\text{ building-days}}$):
- **Full Day (24/24 valid hours):** 100% complete; directly summed.
- **Near-Complete Day (23/24 valid hours):** Exactly 1 hour missing (often due to DST "Spring Forward" or momentary telemetry drop). Classified as **Usable** and normalized:
  $$\text{Daily kWh}_{\text{normalized}} = \left(\sum_{h \in \text{valid}} E_h\right) \times \frac{24}{23}$$
- **Incomplete Day ($< 23$ valid hours):** Discarded in Phase 4 to prevent target distortion.
- **Audit Result:** Across all 1,440 buildings, the mean daily usability rate is **95.46%** (over 697 usable days out of 731 days per building).
- **Output:** Saved as [`data/interim/daily_completeness_audit.csv`](../data/interim/daily_completeness_audit.csv).

---

## 3. Preprocessing Audit Summary Table

Saved in [`results/tables/02_preprocessing_audit_summary.csv`](../results/tables/02_preprocessing_audit_summary.csv):

| Preprocessing Metric | Raw Initial Value | Cleaned Intermediate Value | Decision / Action Taken |
| :--- | :---: | :---: | :--- |
| **Total Buildings in Metadata** | 1,636 | **1,440** | Retained 88.0% of buildings meeting all quality filters |
| **Buildings with Electricity Meter** | 1,578 | **1,440** | Dropped 58 buildings with no electricity meter |
| **Buildings with Valid Space Usage** | 1,615 | **1,440** | Dropped 19 buildings with missing usage labels |
| **Buildings with Missingness $\le 10\%$**| 1,559 | **1,440** | Dropped 119 buildings with broken/missing telemetry |
| **Gross Floor Area Validity** | 1,636 | **1,440** | 100% of buildings have verified positive $m^2$ ($sqm > 0$) |
| **Weather Records Across 19 Sites** | 331,166 | **333,336** | Reindexed onto continuous 17,544-hour calendar grid |
| **Weather DST Duplicate Hours** | 38 duplicates | **0 duplicates** | Resolved via hourly numerical mean |
| **Air Temperature Missing Values** | 128 (0.04%) | **0 (0.00%)** | Time-interpolated ($100\%$ complete) |
| **Dew Point Missing Values** | 328 (0.10%) | **0 (0.00%)** | Time-interpolated ($100\%$ complete) |
| **Wind Speed Missing Values** | 574 (0.17%) | **0 (0.00%)** | Time-interpolated ($100\%$ complete) |
| **Electricity Negative Values** | 0 | **0** | Verified 0 negative readings across 25.2M values |
| **Mean Daily Usability Rate** | N/A | **95.46%** | Audited across 1,052,640 building-days |
| **Pipeline Execution Runtime** | N/A | **35.83 seconds** | Fully automated and reproducible |

---

## 4. Inventory of Intermediate Files in `data/interim/`

All intermediate data files have been created, formatted, and verified:

| File Name | Size (MB) | Rows | Cols | Content & Role |
| :--- | :---: | :---: | :---: | :--- |
| [`clean_metadata.csv`](../data/interim/clean_metadata.csv) | **0.18 MB** | 1,440 | 11 | Screened eligible buildings with verified use, floor area, site ID, and missingness rate |
| [`clean_weather_hourly.csv`](../data/interim/clean_weather_hourly.csv) | **20.96 MB** | 333,336 | 9 | Complete, gap-free hourly weather for all 19 sites across 17,544 hours |
| [`clean_electricity_hourly.csv`](../data/interim/clean_electricity_hourly.csv) | **159.30 MB** | 17,544 | 1,441 | Screened hourly electricity readings for all 1,440 eligible buildings |
| [`daily_completeness_audit.csv`](../data/interim/daily_completeness_audit.csv) | **0.06 MB** | 1,440 | 7 | Summary of usable dates ($\ge 23$ hours) per building over the 731 calendar days |

---

## 5. Direct Hand-Off to Phase 4 (Daily Dataset Creation)

With Phase 3 complete:
1. We have a verified building list (`1,440` buildings) with zero missing labels and verified floor area.
2. We have gap-free hourly weather across all 19 sites ready for daily weather aggregation (daily mean, min, and max temperature).
3. We have an audited completeness table that defines which building-days meet the $\ge 23$ hours criterion to be aggregated into daily energy consumption ($\text{kWh/day}$).

---

## 6. Academic Evaluation & Viva Guide for Data Preprocessing

### Q1: Why did we screen out 196 buildings from the original 1,636 in metadata?
**Answer:** Machine learning models require clean, complete training data. We applied four objective filters:
1. 58 buildings lacked an electricity meter (`electricity == NaN`).
2. 19 buildings lacked a `primaryspaceusage` label, which is our primary categorical predictor.
3. 119 buildings had $>10\%$ missing hourly readings over the 2-year timeline due to hardware disconnections.
Retaining 1,440 buildings (88.0% of the portfolio) ensures high data quality while preserving sample size and building diversity.

### Q2: What was the Daylight Saving Time (DST) issue in the weather data, and how did you resolve it?
**Answer:** On the first Sunday of November in both 2016 and 2017, the clock shifted back 1 hour from 2:00 AM to 1:00 AM, causing the 1:00 AM timestamp to appear twice in every site's weather series. We resolved this by computing the mean of the duplicate weather measurements for that hour. This maintained physical thermodynamic accuracy while eliminating duplicate index labels.

### Q3: How did you handle incomplete days when preparing for daily aggregation?
**Answer:** We conducted an hourly completeness audit for all 1,052,640 building-days. Days with $\ge 23$ valid hours were classified as usable (with the single missing hour normalized by $24/23$). Days with $<23$ hours were flagged as invalid and excluded. This prevents power outages and sensor dropouts from artificially lowering recorded daily consumption.
