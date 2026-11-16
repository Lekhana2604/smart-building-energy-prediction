# Conclusion

This project presents a comprehensive end‑to‑end pipeline for forecasting daily electricity consumption in smart buildings using the BDG2 dataset. By transforming raw hourly meter readings into daily aggregates, engineering a rich set of contextual features, and rigorously evaluating three regression models within a Randomized Block Design, we demonstrate that ensemble methods (Random Forest, Gradient Boosting) achieve superior accuracy and robustness compared to a single Decision Tree.

The high regression performance (MAE ≈ 204 kWh, $R^2$ ≈ 0.99) and classification metrics (Accuracy ≈ 97 %) indicate that the models are ready for practical deployment in building energy management systems. Future work will explore incorporating real‑time weather forecasts, occupancy data, and model interpretability techniques to further enhance prediction quality and actionable insights.

Overall, the findings confirm that machine‑learning‑driven consumption forecasting can support demand‑side management, cost optimisation, and sustainability initiatives in smart building environments.