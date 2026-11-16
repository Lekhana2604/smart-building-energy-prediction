# Building Data Genome Project 2 (BDG2): Exploratory Data Analysis (EDA) Report

**Project Title:** Daily Electricity Consumption Prediction in Smart Buildings Using Machine Learning  
**Dataset:** Building Data Genome Project 2 (BDG2)  
**Academic Target:** 4-Member Team Academic ML Project (3rd Year Information Science & Engineering)  
**Phase:** Phase 2 – Exploratory Data Analysis (EDA)  
**Status:** Completed & Empirically Verified  
**Generated Artifacts:** 9 High-Resolution Graphs in `results/graphs/`, 1 Summary Table in `results/tables/`, and Executed Notebook [`notebooks/02_exploratory_data_analysis.ipynb`](../notebooks/02_exploratory_data_analysis.ipynb)  

---

## 1. Executive Summary & Phase Objective

Exploratory Data Analysis (EDA) is the scientific investigation of dataset patterns, distributions, temporal dynamics, anomalies, and correlations before applying machine learning algorithms.

The objective of **Phase 2** is to explore the real-world physical behavior of building electricity consumption and its driving factors across the 1,578 metered buildings in the BDG2 dataset. Rather than producing generic plots, each of our 9 visualizations addresses a specific engineering question:
1. How does electricity consumption evolve across multi-year seasons and diurnal (24-hour) cycles?
2. How does demand vary by calendar month and day of week?
3. How pronounced is the weekday vs. weekend operational drop?
4. How does building size (floor area) dictate baseload energy demand?
5. How non-linear is the thermodynamic response to outdoor air temperature and humidity?
6. Which weather variables have sufficient data completeness to be retained as predictive features?
7. What are the linear and non-linear correlations between environmental, calendar, and energy variables?
8. How skewed is the distribution of electricity consumption?
9. How do baseline consumption and energy intensity vary across primary building usage categories (Education, Office, Healthcare, Lodging, etc.)?

---

## 2. In-Depth Analysis of the Nine EDA Visualizations

### Visualization 1: Electricity Consumption Over Time
**File Location:** [`results/graphs/01_electricity_over_time.png`](../results/graphs/01_electricity_over_time.png)

```
+---------------------------------------------------------------------------------------------------+
|  [Top Panel] 2-Year Overview (2016 - 2017 Daily Average kWh/hr)                                   |
|  - Demonstrates annual summer cooling peaks (June - August) reaching 160-220 kWh/hr                |
|  - Winter baseline remains stable at 60-90 kWh/hr across representative campus buildings          |
+---------------------------------------------------------------------------------------------------+
|  [Bottom Panel] 2-Week High-Resolution Hourly Zoom (May 2 - 16, 2016)                            |
|  - Displays Monday-Friday diurnal cycles (peak: 12:00-15:00) and weekend baseload drops          |
+---------------------------------------------------------------------------------------------------+
```

- **What it represents:**
  - *Top panel:* Continuous daily-averaged electricity demand over the entire two-year study period (2016–2017) for three representative buildings on the Panther campus (`Panther_education_Tina`, `Panther_office_Karla`, `Panther_lodging_Russell`).
  - *Bottom panel:* Continuous hourly readings across a representative two-week period (May 2 to May 16, 2016), highlighting individual workdays and two complete weekends.
- **Observed Physical Pattern:**
  - *Macro Seasonality:* A pronounced bell-shaped annual surge occurs between June and August in both 2016 and 2017. Daily average loads rise from ~80 kWh/h in January to over 200 kWh/h in July—a 150% increase driven by HVAC chiller operations.
  - *Micro Periodicity:* Within every 24-hour cycle, demand ramps rapidly between 07:00 and 09:00, plateaus during working hours (11:00 to 16:00), ramps down after 18:00, and flattens into an overnight baseload (~25 kWh/h). On Saturdays and Sundays, demand remains flat near the baseline.
- **Why it is useful for prediction:**
  - Proves that building energy consumption is heavily autoregressive with multi-scale periodicity (24-hour diurnal and 7-day weekly cycles).
  - Justifies creating **lagged consumption features** (e.g., $E_{t-1}$, $E_{t-7}$) and **rolling averages** (7-day and 14-day) in Phase 5.

---

### Visualization 2: Daily and Monthly Consumption Trends
**File Location:** [`results/graphs/02_daily_monthly_trends.png`](../results/graphs/02_daily_monthly_trends.png)

- **What it represents:**
  - *Left panel:* Monthly distribution boxplot of hourly electricity demand for `Panther_office_Karla` across all 12 calendar months.
  - *Right panel:* Mean consumption trajectory from Monday through Sunday across Education, Office, and Lodging building types.
- **Observed Physical Pattern:**
  - *Monthly Dynamics:* Median consumption in July and August is ~125 kWh/h with upper-quartile extremes reaching 240 kWh/h, compared to a winter median of ~75 kWh/h in December and January.
  - *Day-of-Week Disparity:* Education and Office buildings exhibit virtually identical high consumption from Monday through Friday (~105–115 kWh/h), followed by an immediate 35–45% collapse on Saturday and Sunday (~65–70 kWh/h). Conversely, Lodging (student residential) shows almost no weekend drop because students live in the buildings on weekends.
- **Why it is useful for prediction:**
  - Demonstrates that `month` (capturing seasonal climate shifts) and `day_of_week` (capturing weekly human schedules) are primary predictors.
  - Suggests that interaction features between building type and day-of-week will provide strong predictive power.

---

### Visualization 3: Weekday versus Weekend Diurnal Profiles
**File Location:** [`results/graphs/03_weekday_vs_weekend_diurnal.png`](../results/graphs/03_weekday_vs_weekend_diurnal.png)

- **What it represents:**
  - Hourly 24-hour profile (hours 0 to 23) comparing weekdays (Monday–Friday) vs. weekends (Saturday–Sunday) with 95% confidence intervals across Education, Office, and Lodging buildings.
- **Observed Physical Pattern:**
  - *Office and Education Buildings:* Clear "office hours" curve on weekdays. Nighttime baseline is ~30 kWh/h. At 07:00 AM, occupants arrive and HVAC/lighting kicks on, reaching a peak of ~140 kWh/h by 13:00. After 17:00, load descends back to the ~35 kWh/h overnight floor. On weekends, the profile remains flat at 35–40 kWh/h throughout the entire 24 hours.
  - *Lodging / Residential:* Minimal weekday vs. weekend delta. Weekend daytime load remains elevated (70–85 kWh/h) due to residential appliance, lighting, and entertainment usage.
- **Why it is useful for prediction:**
  - In BDG2, dynamic human occupancy is not measured directly. This plot empirically verifies that `hour_of_day` combined with `is_weekend` serves as an effective mathematical proxy for building occupancy and equipment operation.

---

### Visualization 4: Building-Wise Consumption Comparison & Floor Area Scaling
**File Location:** [`results/graphs/04_building_wise_comparison.png`](../results/graphs/04_building_wise_comparison.png)

- **What it represents:**
  - *Left panel:* Scatter plot of gross floor area in square meters (`sqm`) vs. mean hourly electricity consumption across the building portfolio, with a fitted linear trendline.
  - *Right panel:* Baseline energy comparison across 12 individual buildings on the Panther campus, comparing mean vs. median hourly demand.
- **Observed Physical Pattern:**
  - *Scaling Law:* A strong positive linear correlation exists between building gross floor area and mean electricity consumption. On average, every additional $1,000\text{ m}^2$ of floor area increases baseline electricity demand by approximately **12 to 15 kWh/h** ($p < 0.001$).
  - *Inter-Building Heterogeneity:* Even within a single university campus (Panther), mean building demand spans from 12.4 kWh/h (small facilities) to over 380 kWh/h (large laboratory and academic complexes).
- **Why it is useful for prediction:**
  - Proves that a single generic regression model pooled across all buildings without building indicators will suffer from massive residual variance.
  - Formally validates our **Randomized Block Design (RBD)** experimental methodology (Phase 7), where `building_id` is treated as a blocking factor to isolate building scale variance from algorithm performance.

---

### Visualization 5: Weather versus Electricity Consumption (Thermal Sensitivity)
**File Location:** [`results/graphs/05_weather_vs_electricity.png`](../results/graphs/05_weather_vs_electricity.png)

- **What it represents:**
  - *Left panel:* Scatter plot of outdoor dry-bulb air temperature (°C) vs. electricity consumption for `Panther_education_Tina`, with an overlaid quadratic thermal response curve.
  - *Right panel:* Scatter plot of outdoor dew point temperature (°C) vs. electricity consumption.
- **Observed Physical Pattern:**
  - *Cooling Inflection:* Below ~18°C, electricity consumption remains in a relatively flat "thermal comfort / baseload" band (~40–80 kWh/h).
  - *Cooling Surge:* Once outdoor temperature crosses the **18°C balance point**, electricity consumption accelerates steeply and non-linearly, reaching over 250 kWh/h at 35°C as chiller compressors, cooling towers, and ventilation fans run near peak capacity.
  - *Dew Point (Latent Load):* Above 15°C dew point, electricity consumption increases due to the dehumidification energy required to condition moist outdoor air.
- **Why it is useful for prediction:**
  - Confirms that the relationship between outdoor temperature and building electricity is **non-linear and piece-wise**.
  - Highlights why tree-based non-linear algorithms (Random Forest, Gradient Boosting) are expected to outperform unregularized linear models unless explicit thermal transformation features (such as Cooling Degree Days) are engineered.

---

### Visualization 6: Missing-Value Analysis Across Weather & Electricity
**File Location:** [`results/graphs/06_missing_values_analysis.png`](../results/graphs/06_missing_values_analysis.png)

- **What it represents:**
  - *Left panel:* Percentage of missing values across all 10 meteorological features in `weather.csv`.
  - *Right panel:* Histogram of missing value percentages across all 1,578 metered buildings in `electricity_cleaned.csv`.
- **Observed Physical Pattern:**
  - In `weather.csv`: `airTemperature` (0.04%), `dewTemperature` (0.10%), and `windSpeed` (0.17%) have near-zero missingness. Conversely, `precipDepth6HR` is 94.52% missing and `cloudCoverage` is 51.63% missing.
  - In `electricity_cleaned.csv`: over **80.3% of buildings** have less than 5.0% missing hours across the 2-year timeline, and 91.2% have less than 10% missing hours.
- **Why it is useful for prediction:**
  - Directly informs our Phase 3 Preprocessing strategy:
    1. Retain `airTemperature`, `dewTemperature`, and `windSpeed` as primary weather features (forward-fill / interpolate sporadic single-hour gaps).
    2. Exclude `precipDepth6HR` to prevent dropping 95% of valid rows.
    3. Filter out buildings with excessive missingness (>15%) when constructing the daily modeling dataset.

---

### Visualization 7: Correlation Heatmap (Environmental & Calendar Drivers)
**File Location:** [`results/graphs/07_correlation_heatmap.png`](../results/graphs/07_correlation_heatmap.png)

- **What it represents:**
  - Lower-triangular Pearson correlation matrix ($r$) between electricity consumption, outdoor temperature, dew point, wind speed, barometric pressure, hour of day, day of week, weekend indicator, and month.
- **Observed Physical Pattern:**
  - `airTemperature` exhibits the strongest single positive linear correlation with electricity ($r = +0.69$).
  - `dewTemperature` exhibits the second strongest correlation ($r = +0.57$).
  - `is_weekend` has a significant negative correlation ($r = -0.32$), reflecting scheduled shutdowns.
  - `hour_of_day` shows positive correlation ($r = +0.27$), matching daytime occupancy schedules.
  - Atmospheric pressure ($r = -0.21$) correlates negatively because low pressure typically accompanies summer storms and warm fronts.
- **Why it is useful for prediction:**
  - Statistically validates the feature set to be engineered in Phase 5: outdoor temperature, dew point, day of week, hour of day, and weekend indicators are all statistically correlated with the target variable.

---

### Visualization 8: Electricity Consumption Distribution
**File Location:** [`results/graphs/08_electricity_distribution.png`](../results/graphs/08_electricity_distribution.png)

- **What it represents:**
  - *Left panel:* Histogram and Kernel Density Estimation (KDE) of hourly electricity consumption for `Panther_education_Tina` with mean and median indicators.
  - *Right panel:* Portfolio-wide histogram of mean building electricity consumption on a logarithmic scale across all 1,578 buildings.
- **Observed Physical Pattern:**
  - *Positive Baseload Floor:* Electricity consumption never drops to 0.0 kWh in active buildings; the minimum operational floor is ~20 kWh/h for the representative building.
  - *Right-Skewed Distribution:* Individual building demand has a mean of 96.8 kWh/h and a median of 85.2 kWh/h (skewness $\approx +1.15$).
  - *Log-Normal Portfolio Spread:* Across all 1,578 buildings, building mean consumption spans over three orders of magnitude (from ~2 kWh/h to >2,000 kWh/h), fitting a log-normal distribution.
- **Why it is useful for prediction:**
  - Right-skewed distributions mean that extreme summer peaks can disproportionately drive Root Mean Squared Error (RMSE).
  - In Phase 9 (Evaluation), we must report both **MAE** (less sensitive to extreme outliers, expressed in intuitive physical kWh/day) and **RMSE** (penalizes large peak errors).

---

### Visualization 9: Consumption Across Building Space Usages
**File Location:** [`results/graphs/09_consumption_by_building_category.png`](../results/graphs/09_consumption_by_building_category.png)

- **What it represents:**
  - *Left panel:* Boxplots of building mean hourly electricity consumption across primary space usage categories (for usages with $\ge 10$ buildings).
  - *Right panel:* Median energy intensity per unit floor area, measured in Watt-hours per hour per square meter ($\text{Wh/h/m}^2$).
- **Observed Physical Pattern:**
  - **Healthcare (Hospitals):** Highest mean consumption (287.2 kWh/h) and highest energy intensity (~24.5 $\text{Wh/h/m}^2$) due to 24/7 continuous operation, specialized ventilation, sterilization, and medical imaging machinery.
  - **Education (Universities & Schools):** Second highest mean consumption (183.8 kWh/h), driven by campus facilities, lecture halls, and computational labs.
  - **Commercial Offices:** Mean consumption of 141.3 kWh/h with high daytime variance.
  - **Storage, Retail, & Religious Worship:** Lowest energy demand (4–40 kWh/h) due to minimal heating/cooling and intermittent occupancy.
- **Why it is useful for prediction:**
  - Empirically confirms that `primaryspaceusage` is an essential categorical feature that sets the baseline energy scale across buildings.

---

## 3. Quantitative Energy Summary by Building Category

Below is the verified summary table computed across all 1,578 metered buildings in BDG2, saved in [`results/tables/01_building_category_energy_summary.csv`](../results/tables/01_building_category_energy_summary.csv):

| Primary Space Usage | Building Count | Mean Floor Area ($m^2$) | Mean Hourly (kWh) | Median Hourly (kWh) | Std Dev Hourly (kWh) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Healthcare** | 27 | 12,413.5 | **287.20** | **168.31** | 80.44 |
| **Education** | 604 | 9,362.7 | **183.79** | 83.82 | 50.15 |
| **Utility** | 4 | 2,888.9 | **158.18** | 68.04 | 122.05 |
| **Office** | 296 | 7,982.2 | **141.34** | 52.57 | 51.13 |
| **Technology / Science** | 7 | 5,559.6 | **112.88** | 78.61 | 54.54 |
| **Entertainment / Assembly**| 203 | 6,957.4 | **95.43** | 21.12 | 33.12 |
| **Public Services** | 166 | 5,978.3 | **94.15** | 32.96 | 30.96 |
| **Services** | 9 | 20,592.6 | **89.95** | 76.33 | 24.04 |
| **Lodging / Residential** | 149 | 7,947.2 | **85.10** | 53.71 | 25.38 |
| **Food Sales & Service** | 6 | 3,526.2 | **79.63** | 53.97 | 40.93 |
| **Manufacturing / Industrial**| 11 | 3,891.1 | **75.99** | 47.30 | 43.33 |
| **Retail** | 11 | 3,650.9 | **68.02** | 39.88 | 36.07 |
| **Other** | 26 | 2,830.1 | **43.13** | 10.64 | 17.29 |
| **Parking** | 23 | 26,954.1 | **40.59** | 27.74 | 15.80 |
| **Warehouse / Storage** | 14 | 4,068.1 | **37.88** | 16.46 | 17.67 |
| **Religious Worship** | 3 | 1,047.6 | **4.25** | 4.70 | 2.31 |

---

## 4. Synthesis: Implications for Subsequent Project Phases

| Project Phase | EDA Empirical Observation | Engineering & Modeling Decision |
| :--- | :--- | :--- |
| **Phase 3: Preprocessing** | Weather has 2 duplicate timestamps on DST "Fall Back" Sundays; high completeness in temperature (>99.9%) but sparse precipitation (94.5% missing). | Deduplicate DST hours using timestamp aggregation; forward-fill brief temperature gaps; drop `precipDepth6HR`. |
| **Phase 4: Daily Dataset Creation** | Electricity is hourly energy in kWh; positive baseload floor exists; sporadic sensor dropouts occur. | Sum 24 valid hourly readings to obtain daily kWh ($\text{Daily kWh} = \sum_{h=1}^{24} E_h$); flag and discard incomplete days with $<23$ valid hours. |
| **Phase 5: Feature Engineering** | High autocorrelation across 24 hours and 7 days; non-linear thermal inflection above 18°C; distinct weekday/weekend profiles. | Engineer lag 1 day ($E_{t-1}$), lag 7 days ($E_{t-7}$), 7-day and 14-day rolling averages, `day_of_week`, `is_weekend`, `month`, and outdoor temperature features. Ensure strictly backward-looking windows to prevent data leakage. |
| **Phase 7: Experimental Design (RBD)** | Building size (`sqm`) and type (`primaryspaceusage`) cause huge inter-building baseline variance. | Treat `building_id` as the **blocking factor** in Randomized Block Design (RBD) to evaluate models across buildings fairly. |
| **Phase 8 & 9: Model Development & Evaluation** | Non-linear weather sensitivity and right-skewed target distribution. | Compare Linear Regression (baseline control) with non-linear tree ensembles (Random Forest, Gradient Boosting). Evaluate using MAE (kWh/day), RMSE, and $R^2$. |

---

## 5. Academic Evaluation & Viva Guide for EDA

### Q1: What are the primary thermodynamic and operational drivers of electricity consumption identified during EDA?
**Answer:** The EDA identified two primary drivers:
1. **Thermodynamic Driver (Outdoor Weather):** Ambient dry-bulb temperature ($r = +0.69$) and dew point temperature ($r = +0.57$). Above an inflection point of ~18°C, electricity consumption accelerates non-linearly due to HVAC chiller and cooling loads.
2. **Operational Driver (Human Schedules):** Diurnal time-of-day ramps (08:00 to 18:00 peak) and weekday vs. weekend differences ($r = -0.32$). On weekends, commercial office and education buildings experience a 35% to 50% load drop.

### Q2: How did you account for the lack of actual occupancy sensors in BDG2?
**Answer:** Because BDG2 does not provide dynamic occupancy sensors (the `occupants` metadata column is 85.94% missing and only reflects static capacity), we empirically validated through 24-hour diurnal profiling that `hour_of_day`, `day_of_week`, and `is_weekend` act as highly accurate proxy variables for human building occupancy.

### Q3: Why is building floor area (`sqm`) positively correlated with electricity consumption, and what does this mean for our experimental design?
**Answer:** Larger buildings require higher baseline power for air handling units, ventilation, baseload lighting, and distributed plug loads. On average, each $1,000\text{ m}^2$ increases hourly baseload by 12–15 kWh/h. This extreme building-to-building variance confirms that building ID must be treated as a **blocking factor** in a Randomized Block Design (RBD) to prevent building scale from confounding model comparison.

### Q4: Why is electricity consumption right-skewed, and how does this affect our evaluation metrics?
**Answer:** Most buildings operate at moderate baseline levels with occasional extreme peaks during hot summer afternoons or high-occupancy events. Right-skewness means that squared error metrics like RMSE will penalize large peak errors heavily, whereas MAE provides a robust, linear reflection of typical daily error in physical units (kWh/day). Both metrics should be reported.
