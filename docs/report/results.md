# Results

## Regression Performance (Phase 8)
| Model | Test MAE | Test RMSE | Test $R^2$ |
|-------|----------|-----------|------------|
| Decision Tree | 244.38 | 1005.57 | 0.9704 |
| Random Forest | 213.62 | 720.11 | 0.9884 |
| Gradient Boosting | 203.41 | 644.32 | 0.9859 |

## Classification Performance (Phase 9) – Predicting Above‑Median Consumption
| Model | Accuracy | Precision | Recall | F1‑score |
|-------|----------|-----------|--------|----------|
| Decision Tree | 0.9750 | 0.9749 | 0.9752 | 0.9750 |
| Random Forest | 0.9747 | 0.9680 | 0.9818 | 0.9749 |
| Gradient Boosting | 0.9739 | 0.9722 | 0.9757 | 0.9739 |

All three models achieve high $R^2$ (> 0.96) and classification metrics (> 0.97), with ensemble methods marginally outperforming the Decision Tree baseline.
