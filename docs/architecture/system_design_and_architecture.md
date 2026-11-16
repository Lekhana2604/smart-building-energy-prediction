# Phase 6 – System Design & Architecture

This document captures the overall architecture of the **Daily Electricity Consumption Prediction** project, the data‑processing pipeline, the model‑training workflow, and the prediction service.  All diagrams are expressed in **Mermaid** so they render directly in the Antigravity UI or any Markdown viewer that supports Mermaid.

---

## 1️⃣ System Architecture Overview

```mermaid
flowchart TD
    %% Data sources
    subgraph RawData [Raw Data (data/raw)]
        ED["Electricity meter CSV files"]
        WD["Weather CSV files"]
        MD["Building metadata CSV files"]
    end
    
    %% Processing layers
    subgraph Processing [Processing Layers]
        direction LR
        D1["01 – Dataset Inspection (notebook)":::notebook]
        D2["02 – Exploratory Data Analysis (notebook)":::notebook]
        D3["03 – Data Preprocessing (src/preprocessing.py)":::script]
        D4["04 – Daily Aggregation (src/daily_aggregator.py)":::script]
        D5["05 – Feature Engineering (src/feature_engineering.py)":::script]
    end
    
    %% Model layer
    subgraph Modeling [Modeling]
        M1["Linear Regression (sklearn)":::model]
        M2["Random Forest Regressor (sklearn)":::model]
        M3["Gradient Boosting Regressor (sklearn)":::model]
        MAll["Trained models (models/*.pkl)":::model]
    end
    
    %% Evaluation & Reporting
    subgraph Evaluation [Evaluation & Reporting]
        E1["Model Evaluation (src/evaluate.py)":::script]
        E2["Results tables & graphs (results/)":::output]
        RPT["Final IEEE Report (docs/report)":::output]
    end
    
    %% Prediction service (optional)
    subgraph Prediction [Prediction Service]
        P1["Load latest features (features_dataset.parquet)":::script]
        P2["Load selected model":::script]
        P3["Predict daily consumption":::script]
        P4["Export predictions (results/predictions.csv)":::output]
    end
    
    %% Connections
    ED --> D3
    WD --> D3
    MD --> D3
    D3 --> D4
    D4 --> D5
    D5 --> M1
    D5 --> M2
    D5 --> M3
    M1 --> E1
    M2 --> E1
    M3 --> E1
    E1 --> E2
    E2 --> RPT
    MAll --> P2
    D5 --> P1
    P1 --> P3
    P2 --> P3
    P3 --> P4
    classDef notebook fill:#e0f7fa,stroke:#006064,stroke-width:2px;
    classDef script fill:#fff3e0,stroke:#e65100,stroke-width:2px;
    classDef model fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px;
    classDef output fill:#f3e5f5,stroke:#4a148c,stroke-width:2px;
```

*Boxes*:
- **Notebook** – interactive Jupyter notebooks used for exploration and documentation.
- **Script** – reusable Python modules under `src/`.
- **Model** – serialized Scikit‑learn models (`*.pkl`).
- **Output** – files stored under `results/` or `docs/`.

---

## 2️⃣ Data‑Processing Flowchart

```mermaid
flowchart LR
    A[Raw CSV files<br/>data/raw/] --> B[interim/ (cleaned hour‑level tables)]
    B --> C[processed/daily_energy_dataset.csv]
    C --> D[processed/features_dataset.csv]
    D --> E[ML models (models/)]
    classDef raw fill:#ffebee,stroke:#b71c1c;
    classDef interim fill:#e3f2fd,stroke:#0d47a1;
    classDef processed fill:#e8f5e9,stroke:#1b5e20;
    class A raw; class B interim; class C,D processed;
```

- **Raw**: original files (electricity, weather, metadata). No modifications.
- **Interim**: cleaned hourly tables (`clean_electricity_hourly.csv`, `clean_weather_hourly.csv`, `clean_metadata.csv`).
- **Processed**: daily aggregation (`daily_energy_dataset.csv`) → feature dataset (`features_dataset.csv`).
- **ML models** consume the feature dataset.

---

## 3️⃣ Model‑Training Workflow

```mermaid
flowchart TD
    subgraph Split [Chronological Train/Val/Test Split]
        T1[Train: 2016‑01‑01 → 2016‑12‑31]
        V1[Validation: 2017‑01‑01 → 2017‑06‑30]
        TE[Test: 2017‑07‑01 → 2017‑12‑31]
    end
    
    T1 --> M1[Linear Regression]
    T1 --> M2[Random Forest]
    T1 --> M3[Gradient Boosting]
    
    M1 --> HV1[Hyper‑parameter tuning (grid search) on Validation]
    M2 --> HV2[Hyper‑parameter tuning (grid search) on Validation]
    M3 --> HV3[Hyper‑parameter tuning (grid search) on Validation]
    
    HV1 --> M1t[Trained Linear Regression]
    HV2 --> M2t[Trained Random Forest]
    HV3 --> M3t[Trained Gradient Boosting]
    
    M1t --> EVAL1[Evaluate on Test]
    M2t --> EVAL2[Evaluate on Test]
    M3t --> EVAL3[Evaluate on Test]
    
    EVAL1 --> RES[Results tables & graphs]
    EVAL2 --> RES
    EVAL3 --> RES
    
    classDef split fill:#fff9c4,stroke:#f57f17;
    classDef model fill:#e8f5e9,stroke:#1b5e20;
    classDef eval fill:#e1f5fe,stroke:#0277bd;
    class Split split; class M1,M2,M3 model; class EVAL1,EVAL2,EVAL3 eval;
```

**Key points**
- **Chronological split** prevents any look‑ahead leakage.
- Hyper‑parameter tuning uses only the validation period.
- The final evaluation is performed on the untouched test period.
- All three models are saved in `models/` for later comparison.

---

## 4️⃣ Prediction (Inference) Workflow

```mermaid
flowchart TD
    P0[New raw hourly data arrives] --> P1[Preprocess (src/preprocessing.py)]
    P1 --> P2[Aggregate to daily (src/daily_aggregator.py)]
    P2 --> P3[Create features (src/feature_engineering.py)]
    P3 --> P4[Load selected model (models/*.pkl)]
    P4 --> P5[Predict daily consumption]
    P5 --> P6[Store predictions (results/predictions_YYYYMMDD.csv)]
    P6 --> P7[Dashboard / reporting]
    classDef process fill:#fff3e0,stroke:#e65100;
    classDef model fill:#e8f5e9,stroke:#1b5e20;
    classDef output fill:#e1f5fe,stroke:#0277bd;
    class P1,P2,P3 process; class P4 model; class P5,P6,P7 output;
```

- The inference pipeline mirrors the training pipeline to guarantee feature‑consistency.
- Predictions can be visualised in a simple dashboard (e.g., Streamlit) or exported for downstream analysis.

---

## 5️⃣ Deliverables Created in This Phase

| File | Location | Purpose |
|------|----------|---------|
| `system_design_and_architecture.md` | `docs/architecture/` | This comprehensive design document (the one you are reading). |
| Mermaid diagrams (embedded above) | Rendered in the markdown file for quick visual reference. |
| Optional: PNG exports of diagrams | Can be generated later with `mmdc` if a static image is required. |

---

### Next Steps
1. **Confirm the architecture** – let me know if any component should be added or modified.
2. When approved, we will move on to **Phase 7 – Experimental Design (Randomized Block Design)** and then to model training (Phase 8).

Feel free to ask for any clarification or additional diagrams!
