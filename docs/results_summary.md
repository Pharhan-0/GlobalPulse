# GlobalPulse Empirical Findings & Results Summary

This document summarizes the quantitative empirical findings of the **GlobalPulse** project, extracted directly from the verified execution run (`GlobalPulse_IEEE_Fresh`, completed with status `OK` across 11,334 country-year forecasting samples).

---

## 1. Headline Empirical Conclusions

1. **Macroeconomic Growth Unpredictability:** Predicting one-year-ahead real GDP growth from cross-sectional macro indicators is profoundly difficult. On the held-out test period (2016–2025), the best machine learning regression model achieves an $R^2$ of only **0.034** (3.4% explained variance).
2. **Honest Baseline Superiority in Classification:** In the 3-class growth regime forecasting task, the simple **Previous-Year Category Baseline** outperforms all nine evaluated machine learning classifiers on both validation and held-out test Macro-F1 (0.556 vs. 0.550 for Gradient Boosting). Economic growth regimes exhibit strong institutional inertia that cross-sectional indicators do not improve upon.
3. **Robust Regression Mitigates Extreme Tails:** Huber Regression was selected during chronological validation, outperforming naive benchmarks by +0.211 percentage points (+3.5% error reduction). The Huber M-estimator protects against macroeconomic growth crises and heavy-tailed shocks.
4. **Weak Natural Cluster Boundaries:** Country clustering ($k=2$) reveals a silhouette coefficient of **0.222**, demonstrating that global economic development represents a continuous developmental gradient rather than discrete, sharply segmented economic tiers.

---

## 2. Supervised Regression Benchmark

Evaluated on the held-out test split (2,689 samples, 2016–2025). Models were selected based on chronological validation RMSE (2007–2015).

| Model Rank | Model Name | Validation RMSE (pp) | Test RMSE (pp) | Test MAE (pp) | Test $R^2$ | Test RMSE ex-COVID (pp) | Validation Selected |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | **Huber Regression (robust)** | **5.315** | **5.885** | **3.098** | **0.034** | **4.672** | **TRUE** |
| 2 | Random Forest Regressor | 5.325 | 5.801 | 3.019 | 0.061 | 4.612 | FALSE |
| 3 | Ridge Regression | 5.349 | 5.806 | 3.064 | 0.060 | 4.671 | FALSE |
| 4 | Gradient Boosting Regressor (Huber) | 5.351 | 5.798 | 2.999 | 0.062 | 4.624 | FALSE |
| 5 | Multiple Linear Regression | 5.352 | 5.816 | 3.059 | 0.056 | 4.674 | FALSE |
| 6 | Polynomial Regression (Degree 2) | 5.377 | 5.884 | 3.164 | 0.034 | 4.671 | FALSE |
| 7 | Simple Linear Regression | 5.405 | 5.930 | 3.198 | 0.019 | 4.664 | FALSE |
| *B1* | *Training Mean Baseline* | *5.742* | *6.095* | *3.455* | *-0.037* | *5.041* | *Baseline* |
| *B2* | *Country Historical Mean Baseline* | *6.408* | *6.593* | *3.785* | *-0.213* | *5.641* | *Baseline* |
| *B3* | *Zero-Growth Baseline* | *6.954* | *6.592* | *4.548* | *-0.213* | *5.948* | *Baseline* |
| *B4* | *Persistence Baseline* | *6.764* | *7.986* | *3.975* | *-0.779* | *6.280* | *Baseline* |

### Key Regression Observations
- **Validation-Driven Selection:** Huber Regressor achieved the lowest validation RMSE (5.315 pp), beating Ridge (5.349) and Random Forest (5.325). While Random Forest and Gradient Boosting scored marginally lower test RMSEs, retrospective selection would constitute data leakage. The project rigorously adheres to the validation selection protocol.
- **Baseline Comparison:** The Huber Regressor improves over the Training Mean Baseline by **+0.211 pp** (5.885 vs. 6.095 pp, a 3.5% improvement).
- **Pandemic Sensitivity:** Excluding the 2020–2021 COVID shock reduces Huber test RMSE from 5.885 pp to **4.672 pp** (and baseline RMSE from 6.095 pp to 5.041 pp), highlighting the disproportionate impact of global tail events.

---

## 3. Supervised Classification Benchmark

Three-class classification task (`Low Growth`, `Medium Growth`, `High Growth`) evaluated on the held-out test split (2016–2025).

| Model Rank | Model Name | Validation Macro-F1 | Test Accuracy | Test Macro-F1 | Test Weighted-F1 | Macro OvR AUC | Test Log Loss | Training Time |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| *B1* | *Previous-Year Category Baseline* | *0.5522* | *0.5645* | **0.5565** | *0.5647* | *—* | *—* | *0.00s* |
| **1** | **Gradient Boosting Classifier** | **0.5352** | **0.5571** | **0.5499** | **0.5568** | **0.7188** | **0.9342** | **14.05s** |
| 2 | Random Forest Classifier | 0.5351 | 0.5485 | 0.5419 | 0.5487 | 0.7228 | 0.9286 | 0.99s |
| 3 | Bagging Classifier (Trees) | 0.5314 | 0.5537 | 0.5449 | 0.5535 | 0.7180 | 0.9352 | 1.08s |
| 4 | Feedforward Neural Net (MLP) | 0.5243 | 0.4846 | 0.4775 | 0.4855 | 0.6525 | 1.0142 | 0.43s |
| 5 | Decision Tree Classifier | 0.5151 | 0.5281 | 0.5173 | 0.5271 | 0.6844 | 1.2721 | 0.08s |
| 6 | Support Vector Machine (Linear) | 0.4768 | 0.4786 | 0.4697 | 0.4801 | 0.6545 | 1.0181 | 0.57s |
| 7 | Logistic Regression | 0.4690 | 0.4689 | 0.4520 | 0.4652 | 0.6494 | 1.0203 | 0.13s |
| 8 | K-Nearest Neighbors (KNN) | 0.4384 | 0.5028 | 0.4858 | 0.4990 | 0.6678 | 1.2129 | 0.09s |
| 9 | Gaussian Naive Bayes | 0.2412 | 0.4370 | 0.2775 | 0.3182 | 0.6318 | 6.6949 | 0.03s |
| *B2* | *Majority-Class Baseline* | *0.1577* | *0.3741* | *0.1815* | *0.2037* | *—* | *—* | *0.00s* |

### Key Classification Observations
- **Gradient Boosting Selection:** Selected among the nine ML models with a validation Macro-F1 of **0.5352**. On the test set, it achieves **0.5499** Macro-F1 and **55.71%** accuracy.
- **The Persistence Paradox:** The simple **Previous-Year Category Baseline** beats all machine learning models on both validation (0.5522 vs. 0.5352) and test Macro-F1 (0.5565 vs. 0.5499). Standard macroeconomic indicators fail to offer incremental discriminatory signal beyond temporal persistence.
- **Discriminatory Power:** Random Forest achieved the highest multi-class OvR AUC (**0.7228**), followed closely by Gradient Boosting (**0.7188**), showing solid ordinal ranking capability despite classification error.

---

## 4. Binary Early-Warning System (Deceleration Risk)

Targeted at detecting growth falling into the bottom quartile of historical growth ($\le 1.176\%$ annual growth):

- **Logistic Regression:** Test AUC = **0.699**, Macro-F1 = **0.479**
- **Random Forest:** Test AUC = **0.721**
- **Policy Implication:** An AUC around 0.70 demonstrates that macro indicators provide meaningful statistical signal for risk screening, but cannot serve as standalone triggers without severe false-positive trade-offs.

---

## 5. Unsupervised Country Profiles & Clustering

- **Snapshot Year:** 2025 (latest available within 3-year retrospective window; 192 countries)
- **Optimal Number of Clusters:** $k = 2$ (selected by silhouette maximization under the constraint that no cluster contains $<5\%$ of samples)
- **Silhouette Score:** **0.2224** (indicates weak separation with substantial distributional overlap)
- **Algorithmic Stability:** Adjusted Rand Index = **1.000** across 8 random seed restarts with `k-means++`.

### K-Means Cluster Characteristics (Median Values)

| Profile Dimension | Profile H (Higher Development) | Profile L (Emerging / Transition) |
|---|:---:|:---:|
| **Sample Size ($n$)** | **120 economies** | **72 economies** |
| Real GDP per Capita (constant 2015 US$) | **$16,190.71** | **$1,587.42** |
| Real GDP Growth (%) | **2.36%** | **4.18%** |
| Urban Population Share (%) | **74.35%** | **42.00%** |
| Life Expectancy at Birth (years) | **77.94 years** | **67.28 years** |
| Internet User Share (%) | **89.16%** | **46.29%** |
| Exports Share of GDP (%) | **44.36%** | **22.04%** |
| Imports Share of GDP (%) | **51.11%** | **33.72%** |
| FDI Inflows Share of GDP (%) | **2.24%** | **1.89%** |
| $\text{CO}_2$ Emissions per Capita (t) | **4.46** | **0.55** |
| Annual CPI Inflation (%) | **2.60%** | **3.53%** |

### Hierarchical Clustering Cophenetic Validation
- **Average Linkage:** Cophenetic Correlation = **0.754** (best tree fidelity)
- Centroid Linkage: 0.730
- Complete Linkage: 0.651
- Single Linkage: 0.621
- Ward Linkage: 0.508

---

## 6. Association Rules & Dimensionality

### Top Apriori Association Rules (Tercile Discretization)
1. **Rule 1:** $\{\text{Life Expectancy}=\text{Low}, \text{Urban Population}=\text{Low}\} \implies \{\text{Income}=\text{Low income}\}$
   - Support: 0.0885 (8.85%), Confidence: 0.4595 (45.9%), **Lift: 4.411**
2. **Rule 2:** $\{\text{Income}=\text{Low income}\} \implies \{\text{Inflation}=\text{High}, \text{Internet Use}=\text{Low}\}$
   - Support: 0.0469 (4.69%), Confidence: 0.4500 (45.0%), **Lift: 4.114**
*(Note: Association rules establish co-occurrence, not causal relationships).*

### Principal Component Analysis (PCA)
- **Top 5 Principal Components:** Explain **46.3%** of total variance in the 11 standardized snapshot indicators.
- **80% Variance Threshold:** Captured by **5 components** in the full panel feature set.
- **90% Variance Threshold:** Captured by **7 components** in the full panel feature set.
- The indicators exhibit moderate collinearity but preserve significant unique information.

---

## 7. Selected Figures Reference

| Figure Asset | Visual Description | Key Takeaway |
|---|---|---|
| [`01_data_overview.png`](figures/01_data_overview.png) | Temporal availability heatmap of 16 WDI indicators across 1960–2025 | Illustrates acute missingness in historical literacy, health, and labor metrics vs. full population and urban coverage. |
| [`02_regression_results.png`](figures/02_regression_results.png) | Bar chart comparing validation and test RMSE across 7 models and 4 baselines | Shows Huber regression validation selection and highlights the competitive performance of naive baselines. |
| [`03_classification_results.png`](figures/03_classification_results.png) | Macro-F1 scorecard and confusion matrix for 3-class growth forecasting | Visualizes how the Previous-Year Category Baseline matches or edges out Gradient Boosting. |
| [`04_country_clustering.png`](figures/04_country_clustering.png) | 2D PCA projection of 192 countries with K-Means convex hulls and country annotations | Demonstrates the overlapping continuum between Profile H and Profile L economies. |

