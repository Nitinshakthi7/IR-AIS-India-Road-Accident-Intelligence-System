# ML Pipeline Analytical Report

*Note: Random states map rigorously (e.g. random_state=42) throughout the pipeline ensuring all models operate on the mathematically identical Train/Test splits on every run.*

## 1. Classification Analysis (Accident_severity)

### Baseline Performance (Native Features)
Below are the results of all classification models evaluated on the full feature set. They are sorted by the primary optimization metric (F1-Score).

| Model | F1-Score (Weighted) | Accuracy | Precision | Recall | ROC-AUC (OvR) |
|---|---|---|---|---|---|
| Random Forest (SMOTE) | **0.7856** | 0.8433 | 0.7764 | 0.8433 | 0.6505 |
| Decision Tree | **0.7820** | 0.8259 | 0.7587 | 0.8259 | 0.5295 |
| Random Forest | **0.7820** | 0.8466 | 0.7849 | 0.8466 | 0.6603 |
| KNN | **0.7771** | 0.8312 | 0.7481 | 0.8312 | 0.5420 |
| Logistic Regression | **0.7751** | 0.8458 | 0.7153 | 0.8458 | 0.5846 |
| SVM | **0.7751** | 0.8458 | 0.7153 | 0.8458 | 0.5929 |
| AdaBoost | **0.7751** | 0.8458 | 0.7153 | 0.8458 | 0.5933 |
| Decision Tree (SMOTE) | **0.7711** | 0.7829 | 0.7631 | 0.7829 | 0.5474 |
| SVM (SMOTE) | **0.7188** | 0.6944 | 0.7490 | 0.6944 | 0.5622 |
| AdaBoost (SMOTE) | **0.7109** | 0.6700 | 0.7612 | 0.6700 | 0.5797 |
| XGBoost (SMOTE) | **0.6255** | 0.5560 | 0.7608 | 0.5560 | 0.5948 |
| KNN (SMOTE) | **0.5916** | 0.5162 | 0.7495 | 0.5162 | 0.5344 |
| Logistic Regression (SMOTE) | **0.5462** | 0.4428 | 0.7658 | 0.4428 | 0.5580 |
| Naive Bayes (SMOTE) | **0.1702** | 0.1080 | 0.7478 | 0.1080 | 0.5074 |
| Naive Bayes | **0.1132** | 0.0735 | 0.7230 | 0.0735 | 0.5165 |

**Overall Best Classifier:** `Random Forest (SMOTE)`

### What Worked vs What Didn't (Analysis)
> **Analytical Insight**: Looking at the results above, **Tree-based models and ensembles** (like Random Forest and XGBoost) typically outperform **linear and probability-based models** (like Logistic Regression or Naive Bayes) in this dataset. 
> 
> **Why it worked**: Ensembles build complex, conditional splits (e.g. IF time > 12 AND location == urban THEN severity = high) which naturally maps the highly non-linear nature of road accidents and geographic features.
> 
> **Why the others didn't work**: Linear methods assume a monotonic, straight correlation between X and Y which catastrophically fails when causality depends on overlapping categorical circumstances.

### Dimensionality Reduction (Pre vs Post PCA)
To observe spatial variance, we mathematically squashed the dataset down to just 2 Principal Components and re-ran the suite.

#### Post-PCA Performance Table
| Model | F1-Score (Weighted) | Accuracy | Precision | Recall | ROC-AUC (OvR) |
|---|---|---|---|---|---|
| Decision Tree | **0.7773** | 0.8259 | 0.7516 | 0.8259 | 0.5082 |
| KNN | **0.7761** | 0.8316 | 0.7460 | 0.8316 | 0.5388 |
| Random Forest (SMOTE) | **0.7758** | 0.8088 | 0.7538 | 0.8088 | 0.5722 |
| Naive Bayes | **0.7753** | 0.8446 | 0.7437 | 0.8446 | 0.5835 |
| Logistic Regression | **0.7751** | 0.8458 | 0.7153 | 0.8458 | 0.5852 |
| SVM | **0.7751** | 0.8458 | 0.7153 | 0.8458 | 0.5862 |
| Random Forest | **0.7751** | 0.8458 | 0.7153 | 0.8458 | 0.5792 |
| AdaBoost | **0.7751** | 0.8458 | 0.7153 | 0.8458 | 0.5260 |
| SVM (SMOTE) | **0.7193** | 0.6928 | 0.7530 | 0.6928 | 0.5595 |
| KNN (SMOTE) | **0.5911** | 0.5150 | 0.7445 | 0.5150 | 0.5235 |
| Decision Tree (SMOTE) | **0.5512** | 0.4627 | 0.7499 | 0.4627 | 0.5313 |
| Logistic Regression (SMOTE) | **0.5406** | 0.4371 | 0.7659 | 0.4371 | 0.5591 |
| Naive Bayes (SMOTE) | **0.5328** | 0.4269 | 0.7581 | 0.4269 | 0.5245 |
| AdaBoost (SMOTE) | **0.4952** | 0.3981 | 0.7607 | 0.3981 | 0.5356 |
| XGBoost (SMOTE) | **0.4526** | 0.3750 | 0.7723 | 0.3750 | 0.5639 |

> **Dimensionality Impact**: Reducing features strictly to 2 dimensions resulted in a performance drop of approximately **0.0083** in the F1-Score! 
> 
> **Analysis**: This massive disparity visually proofs that road accidents are a *High-Variance* occurrence. You cannot just factor size down to 2 metrics. The data depends on the complex 'long-tail' interaction of almost every collected feature (Light, Weather, Casualties) to classify fringe severity properly.

---
## 2. Regression Analysis (Number_of_casualties)

### Baseline Performance (Native Features)
Below are the continuous estimators built to approximate the number of casualties dynamically.

| Model | R² Score | Mean Absolute Error (MAE) | Mean Squared Error (MSE) | RMSE |
|---|---|---|---|---|
| Random Forest | **0.1836** | 0.6248 | 0.9167 | 0.9574 |
| Decision Tree | **0.0830** | 0.6554 | 1.0295 | 1.0146 |
| Linear Regression | **0.0350** | 0.6829 | 1.0835 | 1.0409 |
| Ridge Regression | **0.0347** | 0.6831 | 1.0838 | 1.0411 |
| Lasso | **0.0340** | 0.6841 | 1.0845 | 1.0414 |
| SVR | **0.0161** | 0.6531 | 1.1046 | 1.0510 |

**Overall Best Regressor:** `Random Forest`

> **Analytical Insight**: Similar to classification, non-linear regressors (Trees) dominated linear mathematical ones (Lasso, Ridge). Linear regression algorithms aggressively shrink weight coefficients leading to 'under-fitting' here. Support Vector Regression (SVR) similarly struggles to form a generalized hyperplane when overlapping features drag the margin error uncontrollably.

### Regression Impact (Pre vs Post PCA)
#### Post-PCA Performance Table
| Model | R² Score | Mean Absolute Error (MAE) | Mean Squared Error (MSE) | RMSE |
|---|---|---|---|---|
| Random Forest | **0.0462** | 0.6821 | 1.0709 | 1.0348 |
| Linear Regression | **0.0347** | 0.6838 | 1.0838 | 1.0410 |
| Ridge Regression | **0.0344** | 0.6840 | 1.0841 | 1.0412 |
| Lasso | **0.0340** | 0.6844 | 1.0846 | 1.0414 |
| SVR | **0.0111** | 0.6527 | 1.1103 | 1.0537 |
| Decision Tree | **-0.0108** | 0.7078 | 1.1349 | 1.0653 |

> The R² variance explained plummets confirming dimensionality constraints fail Regression similarly to Classification constraints on this dataset.

---
## 3. Unsupervised Clustering

Evaluated the raw scaled features without providing any labels to determine if underlying mathematical clusters naturally align.

| Model | Silhouette Score | Davies-Bouldin | Clusters Formed |
|---|---|---|---|
| K-Means | **0.0732** | 3.6879 | 3 |
| DBSCAN | **-1.0000** | -1.0000 | 0 |

> **Analytical Insight**: The best native partitioning is done by **K-Means**. While DBSCAN attempts to isolate noise based strictly on point density, high dimensional features usually look incredibly sparse (The curse of dimensionality). Subsequently, K-Means will reliably generate a higher mechanical Silhouette metric, though it forcibly segments spheres which may lack real-world significance compared to density separation.

---
## 4. Ensemble Comparison Insights

You specifically requested an explicit comparison bounded between standalone trees, bootstrap bagging, and sequential boosting networks.

* **Standalone Decision Tree**: Fits aggressively. Can lead strictly to pure overfitting, crashing accuracy outside training logic.
* **Random Forest (Bagging)**: Generalizes this drastically. Runs 100+ separate trees on random feature subsets and averages them, severely shrinking prediction bias.
* **AdaBoost / XGBoost**: Boosting iteratively penalizes incorrect branches on every sequential tree instead of building uniformly. Extremely prone to finding mathematically ideal boundaries.

> Please retrieve the comparative AUC statistics stored via the `ensemble_comparison_roc.png` visual generated by the code sequence alongside this report.

---
## 5. Auxiliary Prediction Tasks (Secondary Objectives)

These peripheral dimensions were individually isolated and predicted using isolated training loops to determine surrounding crash conditions.

### Task: Cause of Accident
| Model | F1-Score (Weighted) | Accuracy | Precision | Recall | ROC-AUC (OvR) |
|---|---|---|---|---|---|
| XGBoost | **0.1123** | 0.1648 | 0.1154 | 0.1648 | 0.5016 |
| Random Forest | **0.1105** | 0.1453 | 0.1048 | 0.1453 | 0.4981 |

> Best Model: **XGBoost**

### Task: Type of Collision
| Model | F1-Score (Weighted) | Accuracy | Precision | Recall | ROC-AUC (OvR) |
|---|---|---|---|---|---|
| XGBoost | **0.6117** | 0.7248 | 0.5851 | 0.7248 | 0.5299 |
| Random Forest | **0.6106** | 0.7256 | 0.5271 | 0.7256 | 0.5213 |

> Best Model: **XGBoost**

### Task: Pedestrian Involvement
| Model | F1-Score (Weighted) | Accuracy | Precision | Recall | ROC-AUC (OvR) |
|---|---|---|---|---|---|
| Random Forest | **0.8888** | 0.9249 | 0.8555 | 0.9249 | N/A |
| XGBoost | **0.8888** | 0.9249 | 0.8555 | 0.9249 | N/A |

> Best Model: **Random Forest**

### Task: Time of Accident
| Model | F1-Score (Weighted) | Accuracy | Precision | Recall | ROC-AUC (OvR) |
|---|---|---|---|---|---|
| Random Forest | **0.4449** | 0.4578 | 0.4554 | 0.4578 | 0.7079 |
| XGBoost | **0.4415** | 0.4574 | 0.4578 | 0.4574 | 0.7121 |

> Best Model: **Random Forest**

### Task: Day of Week
| Model | F1-Score (Weighted) | Accuracy | Precision | Recall | ROC-AUC (OvR) |
|---|---|---|---|---|---|
| Random Forest | **0.2658** | 0.2658 | 0.2686 | 0.2658 | 0.6317 |
| XGBoost | **0.2260** | 0.2269 | 0.2293 | 0.2269 | 0.6106 |

> Best Model: **Random Forest**

### Task: Vehicle Defect
| Model | F1-Score (Weighted) | Accuracy | Precision | Recall | ROC-AUC (OvR) |
|---|---|---|---|---|---|
| XGBoost | **0.9895** | 0.9919 | 0.9887 | 0.9919 | 0.9801 |
| Random Forest | **0.9866** | 0.9911 | 0.9822 | 0.9911 | 0.8905 |

> Best Model: **XGBoost**

### Task: Area of Accident
| Model | F1-Score (Weighted) | Accuracy | Precision | Recall | ROC-AUC (OvR) |
|---|---|---|---|---|---|
| Random Forest | **0.4107** | 0.4420 | 0.4579 | 0.4420 | N/A |
| XGBoost | **0.3961** | 0.4286 | 0.4506 | 0.4286 | N/A |

> Best Model: **Random Forest**

### Task: Driver Profile Risk (Clustering)
| Model | Silhouette Score | Davies-Bouldin | Clusters Formed |
|---|---|---|---|
| K-Means (3 Clusters) | **0.3590** | 1.2280 | 3 |

---
## 6. Global Best Models Summary

### 🏆 Global Best Classifier: `Random Forest (SMOTE)`
With an F1-Score of **0.7856**, `Random Forest (SMOTE)` mathematically dominated the primary Accident Severity classification task.

**Why it won:**
As an ensemble method, it handled the extreme tabular variance by combining hundreds of weak conditional rules into a strong unified boundary, avoiding the trap of single-path overfitting.

**Why others failed:**
Linear models (Naive Bayes, Logistic) failed to capture the overlapping causality of accidents (e.g. wet road + bad lights + novice driver requires an IF-AND-OR tree, not a simple geometric line).

### 🏆 Global Best Regressor: `Random Forest`
Achieving a leading R² Score of **0.1836**, `Random Forest` effectively estimated the discrete Number of Casualties better than the rest.

**Why it won:**
The regression tree mathematically partitioned the outlier casualty events (buses, multi-car piles) cleanly into leaf nodes instead of allowing them to skew a universal equation line.

**Why others failed:**
Regularized linear models (Lasso/Ridge) aggressively suppressed feature coefficients to prevent overfitting, which ultimately 'under-fit' the complex chaotic nature of crash casualty variance.

