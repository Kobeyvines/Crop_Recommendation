# Executive Summary Report: Rhea Crop Recommendation Project

## 1. Executive Summary & Task 1: Data Sourcing

This project was designed to convert a well-structured agronomic benchmark into a practical crop recommendation system for Rhea, with the objective of linking soil fertility and climate conditions to crop suitability in a way that is both interpretable and operationally useful. The foundation dataset was the Crop Recommendation Dataset published on Kaggle by Atharva Ingle, a widely used open-source resource for agricultural machine learning tasks. The dataset contains 2,200 observations distributed evenly across 22 crop classes, with 100 samples per class, which gives the modeling task a balanced and statistically stable starting point. Each observation includes seven numeric agronomic predictors—nitrogen, phosphorus, potassium, temperature, humidity, pH, and rainfall—paired with a crop label, making it suitable for supervised classification and for a recommendation workflow that prioritizes explainable agronomic patterns over opaque prediction alone. The principal limitation is structural rather than methodological: the dataset does not include location telemetry such as region, elevation, or geographic coordinates, so the solution is highly effective for condition-based matching but cannot yet capture the spatial or localized environmental context that would be required for field-level deployment at scale.

> Placeholder figure: ../reports/figures/01_distributions.png

## 2. Task 2: Data Cleaning Protocol

The data cleaning stage confirmed that the dataset was analytically sound before modeling began, which reduced the risk of introducing avoidable noise into the recommendation pipeline. A full null-value audit returned zero missing values across all 2,200 rows and all 8 columns, establishing a clean foundation for subsequent analysis and model training. The target variable was then cast from plain-text strings to a categorical data type so that the crop classes would be treated as discrete, validated labels rather than arbitrary objects, improving consistency in preprocessing and downstream classification operations. The cleaning protocol also deliberately retained per-crop outliers rather than removing them, because the goal of this project was not to force each crop into a narrow statistical center but to preserve the genuine environmental variance that defines real agricultural conditions. That choice is agronomically defensible: a rice field may legitimately show unusually high rainfall, and deleting that observation would erase a meaningful signal rather than correct a data defect.

> Placeholder figure: ../reports/figures/02_missing_values.png

## 3. Task 3: Exploratory Data Analysis Deep-Dive

The exploratory analysis showed that the data contained meaningful class-separating signal across both soil and climate dimensions. Among the numeric predictors, potassium exhibited the largest spread, with a standard deviation of 50.65 ppm, while rainfall followed closely with a standard deviation of 54.96 mm, indicating that nutrient balance and water availability were the strongest sources of variation in the dataset. The pH variable remained comparatively stable at 0.77, suggesting that soil acidity was less variable across crop classes than fertility and moisture conditions. The rainfall analysis was particularly informative because it highlighted a sharp agronomic contrast between crop groups: rice had a median rainfall requirement of approximately 236 mm, whereas muskmelon required only about 22 mm, a difference that materially distinguishes water-intensive crops from drought-tolerant ones. Those patterns were reinforced by the broader visualization set, which showed that crop classes occupied coherent regions of feature space rather than appearing as random mixtures of environmental conditions.

> Placeholder figure: ../reports/figures/03_boxplots.png

> Placeholder figure: ../reports/figures/04_rainfall_by_crop.png

> Placeholder figure: ../reports/figures/05_npk_scatter.png

## 4. Task 4 & 5: Feature Engineering & Unified Pipeline Architecture

To strengthen predictive performance while preserving methodological integrity, the project introduced two engineered features: an NPK Score and a Growing Condition Index. These features were designed to compress agronomically meaningful information into compact representations that the classifier could use without losing the original nutrient and climate measurements. The key innovation was to eliminate data leakage systematically by wrapping both formulas inside a custom Scikit-Learn transformer class, so that the scaling and normalization parameters were learned exclusively from the training partition rather than from the full dataset. This ensured that test observations did not influence the statistics used to create the engineered features, which would otherwise have inflated validation performance artificially. The resulting architecture unified preprocessing, feature engineering, and K-Nearest Neighbors classification into a single pipeline, making the workflow reproducible, auditable, and well aligned with the standards expected of a production-ready decision-support system.

> Placeholder figure: ../reports/figures/06_feature_engineering.png

> Placeholder figure: ../reports/figures/07_pipeline.png

## 5. Performance Evaluation & Deployment Strategy

The final evaluation demonstrated that the recommendation engine was both accurate and stable. On a held-out test set of 440 observations, the model produced 429 correct Top-1 predictions, corresponding to 97.50% accuracy, and 439 correct Top-3 shortlist predictions, corresponding to 99.77% accuracy. Those results are especially meaningful in agricultural decision support because users rarely need a single crop in isolation; they need a short list of agronomically plausible options that can be evaluated against local constraints, labor availability, and market demand. The model also achieved a 5-fold cross-validation score of 97.32% ± 0.81%, indicating that the reported performance was not dependent on a single favorable data split and that the learned decision boundaries generalize well to new samples. For Rhea, the deployment implication is direct: the system is ready to function as a practical recommendation assistant that ranks crop options for farmers and agronomists, while the next phase should focus on operationalization through a lightweight interface that accepts soil and weather inputs, returns ranked recommendations, and preserves the same leakage-safe preprocessing logic that produced the validated results.

> Placeholder figure: ../reports/figures/08_confusion_matrix.png
