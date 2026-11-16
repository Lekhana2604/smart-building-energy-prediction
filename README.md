<<<<<<< HEAD
# Daily Electricity Consumption Prediction in Smart Buildings Using Machine Learning

An academic machine learning project analyzing and predicting daily electricity consumption in non-residential buildings using the **Building Data Genome Project 2 (BDG2)** dataset.

---

## 1. Project Overview

- **Title:** Daily Electricity Consumption Prediction in Smart Buildings Using Machine Learning
- **Dataset:** Building Data Genome Project 2 (BDG2)
- **Dataset Source:** Zenodo Record [3887306](https://zenodo.org/records/3887306) (Miller et al., 2020)
- **Target Variable:** Daily electricity consumption in **kilowatt-hours per day (kWh/day)**
- **Academic Context:** 4-Member Team Academic ML Project (3rd Year Information Science and Engineering)
- **Experimental Methodology:** Randomized Block Design (RBD) with buildings as blocks, Linear Regression as control, and Random Forest & Gradient Boosting as treatments.

---

## 2. Project Folder Structure

```
smart-building-energy-prediction/
├── data/
│   ├── raw/                               <- Extracted BDG2 Zenodo archive
│   │   └── buds-lab-building-data-genome-project-2-3d0cbaf/
│   │       └── data/
│   │           ├── metadata/              <- metadata.csv (1,636 buildings)
│   │           ├── weather/               <- weather.csv (331,166 hourly rows)
│   │           └── meters/                <- Raw & Cleaned meter CSV files
│   ├── interim/                           <- Cleaned intermediate datasets (Phase 3)
│   └── processed/                         <- Daily aggregated dataset & features (Phases 4-5)
├── notebooks/
│   ├── 01_dataset_understanding.ipynb     <- Phase 1: Dataset Inspection & Understanding (Executed)
│   ├── 02_exploratory_data_analysis.ipynb <- Phase 2: Exploratory Data Analysis (EDA)
│   ├── 03_data_preprocessing.ipynb        <- Phase 3: Data Preprocessing & Cleaning
│   ├── 04_feature_engineering.ipynb       <- Phase 4 & 5: Daily Dataset & Lag Features
│   └── 05_model_training_evaluation.ipynb <- Phases 7-9: RBD Experiment & Model Evaluation
├── src/
│   ├── __init__.py
│   ├── data_loader.py                     <- Reusable modular dataset loader and validator
│   ├── preprocessing.py                   <- Preprocessing pipeline (Phase 3)
│   ├── feature_engineering.py             <- Lag, calendar & weather features (Phase 5)
│   ├── train.py                           <- Model training pipeline (Phase 8)
│   └── evaluate.py                        <- Model evaluation & RBD statistical tests (Phase 9)
├── models/                                <- Serialized trained models (.joblib / .pkl)
├── results/
│   ├── graphs/                            <- High-resolution plots (EDA, predictions, feature importance)
│   └── tables/                            <- Metric comparison tables (MAE, RMSE, R²)
├── docs/
│   ├── dataset_understanding_report.md    <- Comprehensive Phase 1 technical inspection report
│   ├── literature_survey/                 <- Survey papers & citations
│   ├── architecture/                      <- System architecture & workflow diagrams
│   └── report/                            <- IEEE two-column academic conference paper
├── requirements.txt                       <- Pinned Python dependencies with explanations
└── README.md                              <- Project overview & execution guide
```

---

## 3. Installation & Setup

### Prerequisites
- Python 3.10+ (Tested on Python 3.12.10)
- VS Code or Google Antigravity IDE

### 1. Clone or Open the Workspace
Open the project root directory in your IDE:
```bash
# In VS Code / Antigravity terminal:
python --version
```

### 2. Install Dependencies
Install all required libraries using `requirements.txt`:
```bash
pip install -r requirements.txt
```

---

## 4. How to Run the Project Notebooks & Scripts

### Option A: Interactive Jupyter Notebooks
1. **Phase 1 Notebook:** Open [`notebooks/01_dataset_understanding.ipynb`](notebooks/01_dataset_understanding.ipynb) to inspect the raw BDG2 dataset files, schemas, and measurement units.
2. **Phase 2 Notebook:** Open [`notebooks/02_exploratory_data_analysis.ipynb`](notebooks/02_exploratory_data_analysis.ipynb) to run the 9 comprehensive EDA analyses with high-resolution visual plots.
3. **Phase 3 Notebook:** Open [`notebooks/03_data_preprocessing.ipynb`](notebooks/03_data_preprocessing.ipynb) to execute the building screening, weather interpolation, DST resolution, and completeness audit.

### Option B: Command-Line Pipeline Execution
Run the modular pipeline scripts directly from the project root:
```bash
# Test the data loader (Phase 1)
python src/data_loader.py

# Run the end-to-end data preprocessing pipeline (Phase 3)
python src/preprocessing.py
```

---

## 5. Summary of Phase 1 Verified Findings

| Item | Verified Property | Note for Academic Evaluation |
| :--- | :--- | :--- |
| **Dataset Source** | BDG 2.0 (Zenodo 3887306) | 1,636 buildings across 19 sites |
| **Electricity Meter Unit** | **Kilowatt-hours (kWh)** per hour | Cumulative energy, not power (kW) |
| **Temporal Frequency** | Hourly (17,544 rows) | 2 full years (2016-01-01 to 2017-12-31) |
| **Electricity Buildings** | 1,578 buildings | Exactly 1,578 columns in `electricity_cleaned.csv` |
| **Primary Building Types** | Education (617), Office (307) | Dominated by educational/office spaces |
| **Weather Quality** | Temperature & Dew Point | >99.9% complete across 19 sites |
| **Data Limitations** | No dynamic occupancy sensors | Must use calendar/time proxies (day-of-week) |

For complete technical analysis, see [`docs/dataset_understanding_report.md`](docs/dataset_understanding_report.md).
=======
# smart-building-energy-prediction
Smart-building daily electricity consumption prediction using machine learning.
>>>>>>> 82d696c1d91c779780eb2e2da5cd0cf8d310bbb3
