# Discussion

The experimental results demonstrate that ensemble methods—Random Forest and Gradient Boosting—provide superior predictive performance compared to the single Decision Tree baseline. The lower MAE and RMSE of the ensembles indicate a better fit to the underlying consumption patterns, while the higher $R^2$ values reflect a stronger proportion of variance explained.

The classification task, which predicts whether daily consumption exceeds the median value, yields very high accuracy (> 97 %) across all models. This is expected given the relatively balanced class distribution and the rich set of engineered features that capture temporal, weather, and building‑specific effects. Notably, the Random Forest achieves the highest recall (0.982) and a strong precision (0.968), suggesting it is particularly effective at identifying high‑consumption days while maintaining low false‑positive rates.

The Randomized Block Design (RBD) mitigates confounding block effects such as building‑specific characteristics and seasonal trends, ensuring that observed performance differences are attributable primarily to the modeling approach rather than external factors. The chronological train/validation/test split further safeguards against data leakage, providing a realistic assessment of model generalisation.

Limitations of the current study include:
- **Feature Scope:** While a comprehensive set of 37 features was engineered, additional contextual data (e.g., occupancy schedules, HVAC setpoints, or real‑time weather forecasts) could further improve model accuracy.
- **Spatial Generalisation:** Models were trained on the BDG2 dataset, which is limited to specific geographic regions; transferability to other building fleets may require domain adaptation.
- **Model Interpretability:** Ensemble methods, though accurate, are less interpretable than a single Decision Tree; future work could incorporate model‑agnostic explanation techniques (e.g., SHAP) to provide actionable insights to facility managers.

Overall, the findings confirm that data‑driven machine‑learning models can reliably forecast daily electricity consumption and support building energy management strategies.
