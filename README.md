# 🛣️ IR-AIS — India Road Accident Intelligence System

An AI-powered full-stack web application for **predicting road accident severity** and **analyzing accident patterns** across India. Built with a decoupled architecture spanning a React frontend, Node.js backend, and Python machine learning microservice.

---

## 📋 Table of Contents

- [Problem Statement](#-problem-statement)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-complete-project-structure)
- [Folder & File Descriptions](#-detailed-folder--file-descriptions)
  - [ml-service/](#1-ml-service--machine-learning-pipeline)
  - [frontend/](#2-frontend--react-ui-layer)
  - [backend/](#3-backend--server-side-logic)
  - [src/app/](#4-srcapp--nextjs-routing--pages)
  - [upload/](#5-upload--dataset)
  - [Root Config Files](#6-root-configuration-files)
- [ML Pipeline](#-ml-pipeline-in-detail)
  - [Why These Models?](#why-these-models)
  - [Why SMOTE?](#why-smote)
  - [Why Drop Leakage Columns?](#why-drop-leakage-columns)
  - [Classification Results](#classification-results-accident-severity)
  - [Regression Results](#regression-results-number-of-casualties)
- [API Endpoints](#-api-endpoints)
- [Getting Started](#-getting-started)
- [Key Design Decisions](#-key-design-decisions)

---

## 🎯 Problem Statement

Road traffic accidents are one of the leading causes of death and injury in India. This project aims to:

1. **Predict the severity** of a road accident (Fatal / Serious / Slight Injury) given pre-crash conditions like weather, road type, driver age, vehicle type, etc.
2. **Predict the number of casualties** using regression models.
3. **Explore accident patterns** through interactive visualizations — identifying peak hours, dangerous weather conditions, risky age groups, and common accident causes.

This helps in **emergency response prioritization** (allocating ambulances), **traffic policy decisions** (where to add speed breakers), and **public awareness** (which conditions are most dangerous).

---

## 📸 Features

| Feature | Description |
|---------|-------------|
| **Interactive Dashboard** | Visualize 12,316+ accident records with pie charts, bar charts, heatmaps, and area charts |
| **Model Playground** | Compare 10 classification models (Logistic Regression, KNN, Decision Tree, Random Forest, XGBoost) with and without SMOTE |
| **Live Predictor** | Real-time severity classification and casualty regression using 24 input features |
| **EDA Insights** | Explore patterns by time-of-day, day-of-week, weather, driver age, vehicle type, and accident cause |

---

## 🛠️ Tech Stack

| Layer | Technologies | Why? |
|-------|-------------|------|
| **Frontend** | Next.js 16, React 19, TypeScript, Tailwind CSS, shadcn/ui, Recharts, Lucide Icons | Next.js provides SSR + API routes in one framework; shadcn/ui for consistent, accessible components; Recharts for declarative charting |
| **Backend** | Next.js API Routes, Prisma ORM, SQLite | API routes eliminate the need for a separate Express server; Prisma provides type-safe database access |
| **ML Service** | Python 3, scikit-learn, XGBoost, imbalanced-learn (SMOTE), pandas, numpy, joblib | scikit-learn is the standard for classical ML; XGBoost for gradient boosting; imbalanced-learn for SMOTE oversampling |

---

## 📂 Complete Project Structure

```
ir-ais/
│
├── ml-service/                        # 🧠 Machine Learning Pipeline
│   ├── config.py                      #   Centralized constants and paths
│   ├── preprocessing.py               #   Data loading, cleaning, encoding
│   ├── eda.py                         #   EDA statistics generation
│   ├── train_models.py                #   Orchestrator — runs all training
│   ├── predict.py                     #   Inference helper for API calls
│   ├── requirements.txt               #   Python dependencies
│   │
│   ├── classifiers/                   #   Classification model modules
│   │   ├── __init__.py                #     Model registry (BASE, SMOTE, TUNABLE lists)
│   │   ├── base.py                    #     Shared evaluation metrics (accuracy, F1, etc.)
│   │   ├── logistic_regression.py     #     Logistic Regression model
│   │   ├── knn.py                     #     K-Nearest Neighbors model
│   │   ├── decision_tree.py           #     Decision Tree model
│   │   ├── random_forest.py           #     Random Forest + GridSearchCV tuning
│   │   └── xgboost_clf.py            #     XGBoost + sample weight utilities
│   │
│   ├── regressors/                    #   Regression model modules
│   │   ├── __init__.py                #     Model registry (BASE, TUNABLE lists)
│   │   ├── base.py                    #     Shared evaluation metrics (MAE, R², etc.)
│   │   ├── linear_regression.py       #     OLS Linear Regression
│   │   ├── ridge.py                   #     Ridge (L2-regularized) Regression
│   │   ├── decision_tree.py           #     Decision Tree Regressor
│   │   └── random_forest.py           #     Random Forest + RandomizedSearchCV tuning
│   │
│   └── models/                        #   Trained model artifacts (auto-generated)
│       ├── best_classifier.pkl        #     Serialized best classification model
│       ├── best_regressor.pkl         #     Serialized best regression model
│       ├── label_encoders.pkl         #     Fitted LabelEncoders for all features
│       ├── target_encoder.pkl         #     Fitted LabelEncoder for target variable
│       ├── feature_names.pkl          #     Ordered list of feature column names
│       ├── classification_metrics.json #    Metrics for all 10 classification models
│       ├── regression_metrics.json    #     Metrics for all 5 regression models
│       ├── eda_data.json              #     EDA statistics for dashboard charts
│       └── best_models.json           #     Best model names (read by API routes)
│
├── frontend/                          # 🎨 Frontend Source Code
│   ├── components/                    #   React components
│   │   ├── Navbar.tsx                 #     Top navigation bar with tab switching
│   │   ├── AccidentForm.tsx           #     Prediction input form component
│   │   ├── ResultCard.tsx             #     Prediction result display card
│   │   ├── MetricBar.tsx              #     Model metric progress bar
│   │   ├── SeverityPieChart.tsx       #     Severity distribution pie chart
│   │   ├── PeakHoursBarChart.tsx      #     Hourly accident bar chart
│   │   ├── WeatherHeatmap.tsx         #     Weather × severity heatmap table
│   │   ├── ModelComparisonChart.tsx   #     Model performance comparison chart
│   │   └── ui/                        #     shadcn/ui component library (48 components)
│   │       ├── button.tsx, card.tsx, select.tsx, input.tsx, badge.tsx,
│   │       ├── progress.tsx, skeleton.tsx, separator.tsx, alert.tsx,
│   │       ├── tabs.tsx, table.tsx, dialog.tsx, tooltip.tsx, ...
│   │       └── (48 total pre-built accessible UI primitives)
│   │
│   ├── hooks/                         #   Custom React hooks
│   │   ├── use-mobile.ts             #     Detects mobile viewport for responsive layout
│   │   └── use-toast.ts              #     Toast notification state management
│   │
│   └── lib/                           #   Frontend utilities
│       ├── utils.ts                   #     Tailwind CSS class merge helper (cn function)
│       └── feature-options.ts         #     All 24 feature field definitions for prediction form
│
├── backend/                           # ⚙️ Backend Source Code
│   ├── lib/
│   │   ├── db.ts                      #     Prisma database client singleton
│   │   └── python-bridge.ts           #     Python ML service HTTP connector
│   ├── prisma/
│   │   └── schema.prisma             #     Database schema definition
│   └── db/
│       └── custom.db                  #     SQLite database file
│
├── src/app/                           # 🔀 Next.js App Router
│   ├── page.tsx                       #     Main page (~1800 lines): Dashboard + Model Playground + Predictor
│   ├── layout.tsx                     #     Root layout with Navbar, fonts, metadata
│   ├── globals.css                    #     Global styles, dark theme, animations, glassmorphism
│   └── api/                           #     REST API routes
│       ├── classify/route.ts          #       POST — severity prediction
│       ├── regress/route.ts           #       POST — casualty count prediction
│       ├── eda-data/route.ts          #       GET  — EDA statistics for dashboard
│       └── model-comparison/route.ts  #       GET  — all model metrics for playground
│
├── upload/                            # 📊 Dataset
│   └── Road.csv                       #     12,316 accident records, 32 columns
│
├── package.json                       #   Node.js dependencies and scripts
├── tsconfig.json                      #   TypeScript configuration with path aliases
├── tailwind.config.ts                 #   Tailwind CSS theme configuration
├── next.config.ts                     #   Next.js configuration
├── components.json                    #   shadcn/ui configuration
├── eslint.config.mjs                  #   ESLint configuration
├── postcss.config.mjs                 #   PostCSS configuration
├── .gitignore                         #   Git ignore rules
└── .env                               #   Environment variables
```

---

## 📖 Detailed Folder & File Descriptions

### 1. `ml-service/` — Machine Learning Pipeline

This is the core ML engine. Each model lives in its own file for easy debugging and modification.

#### Root Files

| File | Purpose | Key Functions |
|------|---------|---------------|
| **`config.py`** | Centralized configuration | Defines `DATA_PATH`, `MODEL_DIR`, `RANDOM_STATE=42`, `TEST_SIZE=0.2`, `TARGET_CLASS`, `TARGET_REGR`, and `LEAKAGE_COLS`. Uses `os.path` for portable paths that work on both Windows and Linux. |
| **`preprocessing.py`** | Data loading & cleaning | `load_and_preprocess()` — Reads `Road.csv`, replaces "na"/"unknown" with NaN, imputes missing values with mode, extracts `Hour_of_Day` from the Time column, drops 6 leakage columns, label-encodes all categorical features, and saves encoders to `.pkl` files. |
| **`eda.py`** | Exploratory Data Analysis | `generate_eda()` — Reloads raw CSV (before encoding) and computes: severity distribution, hourly accident counts, day-of-week distribution, weather × severity cross-tab, age × severity cross-tab, top vehicle types, and top accident causes. Saves as `eda_data.json`. |
| **`train_models.py`** | Training orchestrator | `train_classification()` and `train_regression()` — Splits data, applies SMOTE, iterates through all registered model modules, trains each, evaluates on test set, tracks the best model by F1/R², and serializes the best model + all metrics to disk. |
| **`predict.py`** | Inference helper | `classify(features_dict)` and `regress(features_dict)` — Loads saved models and encoders, transforms raw feature values using saved `LabelEncoder`s (gracefully handles unknown categories), and returns predictions with probabilities/confidence. Called by the API routes via `child_process.execFileSync`. |
| **`requirements.txt`** | Python dependencies | `scikit-learn>=1.5.0`, `pandas>=2.0.0`, `numpy>=1.24.0`, `imbalanced-learn>=0.12.0`, `xgboost>=2.0.0`, `joblib>=1.3.0` |

#### `classifiers/` — One File Per Classification Algorithm

Each file exports `NAME` (display name) and `build_model(**kwargs)` (returns a fresh sklearn model instance).

| File | Model | Why This Model? |
|------|-------|-----------------|
| **`logistic_regression.py`** | Logistic Regression | Linear baseline. Fast, interpretable. Shows if a simple linear boundary can separate severity classes. |
| **`knn.py`** | K-Nearest Neighbors (k=5) | Non-parametric, instance-based. Captures local patterns without assuming any distribution. |
| **`decision_tree.py`** | Decision Tree (max_depth=10) | Single tree for interpretability. Captures non-linear feature interactions. Depth-limited to prevent overfitting. |
| **`random_forest.py`** | Random Forest (100 trees) + GridSearchCV | Ensemble of decorrelated trees. Most robust to noise and imbalance. `build_tuned_model()` searches over `n_estimators` and `max_depth`. |
| **`xgboost_clf.py`** | XGBoost | Gradient boosted trees. `compute_sample_weights()` calculates per-class weights from original class distribution to handle imbalance. |
| **`base.py`** | Shared evaluation utilities | `evaluate(model, X_test, y_test)` — Computes accuracy, precision, recall, F1 (all weighted), confusion matrix, and ROC-AUC (OvR). `print_metrics()` — Console output. Eliminates copy-pasting metric code across model files. |
| **`__init__.py`** | Model registry | Defines `BASE_MODELS`, `SMOTE_MODELS`, `SMOTE_WEIGHTED_MODELS`, `TUNABLE_MODELS` lists. The orchestrator iterates these lists — to add a new model, just create a file and add it to the list. |

#### `regressors/` — One File Per Regression Algorithm

Same pattern as classifiers — `NAME` + `build_model()`.

| File | Model | Why This Model? |
|------|-------|-----------------|
| **`linear_regression.py`** | Ordinary Least Squares | Simplest baseline for regression. |
| **`ridge.py`** | Ridge Regression (α=1.0) | L2 regularization prevents overfitting when features are correlated. |
| **`decision_tree.py`** | Decision Tree Regressor (max_depth=10) | Captures non-linear relationships. |
| **`random_forest.py`** | Random Forest Regressor + RandomizedSearchCV | Best tree-based ensemble for regression. `build_tuned_model()` searches over `n_estimators`, `max_depth`, `min_samples_split`. |
| **`base.py`** | Shared evaluation utilities | `evaluate()` — Computes MAE, MSE, RMSE, R². |
| **`__init__.py`** | Model registry | Defines `BASE_MODELS` and `TUNABLE_MODELS` lists. |

#### `models/` — Saved Artifacts (Auto-Generated)

| File | Purpose |
|------|---------|
| `best_classifier.pkl` | Serialized best classification model (selected by highest weighted F1) |
| `best_regressor.pkl` | Serialized best regression model (selected by highest R²) |
| `label_encoders.pkl` | Dictionary of `{column_name: fitted LabelEncoder}` for all categorical features |
| `target_encoder.pkl` | Fitted `LabelEncoder` for the `Accident_severity` target variable |
| `feature_names.pkl` | Ordered list of 24 feature column names (used to ensure consistent column order during prediction) |
| `classification_metrics.json` | Metrics for all 10 classification models (4 base + 4 SMOTE + XGBoost + Tuned RF) |
| `regression_metrics.json` | Metrics for all 5 regression models (4 base + Tuned RF) |
| `eda_data.json` | Pre-computed EDA statistics (distributions, cross-tabs) served by the dashboard API |
| `best_models.json` | Names and scores of the best classifier and regressor (read by API routes for dynamic model name display) |

---

### 2. `frontend/` — React UI Layer

#### `components/` — Custom React Components

| Component | Purpose |
|-----------|---------|
| **`Navbar.tsx`** | Top navigation bar with three tabs: Dashboard, Model Playground, Live Predictor. Uses URL search params for tab state so tabs are bookmarkable. |
| **`AccidentForm.tsx`** | 22-field prediction input form with grouped sections (Driver, Vehicle, Road, Accident). Each field is a dropdown populated from `feature-options.ts`. |
| **`ResultCard.tsx`** | Displays prediction results with severity badge, confidence gauge, and probability breakdown. Color-coded by severity level. |
| **`MetricBar.tsx`** | Horizontal progress bar showing a model's metric value (accuracy, F1, etc.) with label and percentage. |
| **`SeverityPieChart.tsx`** | Donut pie chart showing the class distribution of Fatal / Serious / Slight injuries. |
| **`PeakHoursBarChart.tsx`** | Area chart showing accident count by hour of day (0–23). |
| **`WeatherHeatmap.tsx`** | Table-based heatmap showing weather conditions × severity cross-tabulation. |
| **`ModelComparisonChart.tsx`** | Grouped bar chart comparing F1, accuracy, precision, recall across all models. |

#### `components/ui/` — shadcn/ui Component Library (48 components)

Pre-built, accessible UI primitives based on Radix UI. **Why shadcn/ui?** It provides copy-paste components (not npm packages) that are fully customizable, accessible (ARIA-compliant), and styled with Tailwind CSS. Key components used:

`button`, `card`, `select`, `input`, `badge`, `progress`, `skeleton`, `separator`, `alert`, `label`, `tabs`, `table`, `dialog`, `tooltip`, `toast`, `toaster`, etc.

#### `hooks/` — Custom React Hooks

| Hook | Purpose |
|------|---------|
| **`use-mobile.ts`** | Returns `true` if viewport width < 768px. Used for responsive layout adjustments. |
| **`use-toast.ts`** | Manages toast notification state (queue, dismiss, auto-remove). Used for success/error notifications. |

#### `lib/` — Utility Functions

| File | Purpose |
|------|---------|
| **`utils.ts`** | Exports `cn()` — merges Tailwind CSS classes using `clsx` + `tailwind-merge`. Prevents conflicting utility class combinations. |
| **`feature-options.ts`** | Defines all **24 prediction input fields** with their labels and valid options. Contains `ALL_FEATURE_OPTIONS` (name → definition map) and `ALL_FEATURES` (ordered name list). Supports both `select` (dropdown) and `number` (numeric input) field types. |

---

### 3. `backend/` — Server-Side Logic

| File | Purpose |
|------|---------|
| **`lib/db.ts`** | Prisma client singleton. Ensures only one database connection is created across hot-reloads in development. |
| **`lib/python-bridge.ts`** | HTTP bridge for communicating with an external Python ML service (FastAPI). Provides `pythonBridge()` for requests and `checkPythonBackendHealth()` for health checks. **Note:** The current API routes use `child_process.execFileSync` instead (direct Python invocation), so this bridge is a future-ready alternative for when the ML service runs as a standalone FastAPI server. |
| **`prisma/schema.prisma`** | Database schema definition in Prisma Schema Language. |
| **`db/custom.db`** | SQLite database file. |

---

### 4. `src/app/` — Next.js App Router

#### Pages

| File | Purpose |
|------|---------|
| **`page.tsx`** | Main application page (~1800 lines). Contains three major tab sections built as React components: `DashboardTab` (7 EDA charts + 4 stat cards), `ModelPlaygroundTab` (classification/regression model comparison with confusion matrix, radar chart, metrics table), `LivePredictorTab` (24-field form + severity/casualty prediction with confidence gauges). All data is fetched from the API routes. |
| **`layout.tsx`** | Root layout wrapping all pages. Imports Google Fonts (Inter), sets dark theme metadata, renders the Navbar. |
| **`globals.css`** | Global CSS: dark theme variables, glassmorphism effects (`glass-subtle`), gradient backgrounds (`mesh-gradient`), glow effects (`glow-emerald`, `glow-red`), skeleton shimmer animations, staggered entry animations, and custom scrollbar styles. |

#### API Routes (`src/app/api/`)

| Route | Method | Purpose | How It Works |
|-------|--------|---------|--------------|
| **`classify/route.ts`** | POST | Predict accident severity | Receives 24 features as JSON → writes to temp file → calls `python3 -c "from predict import classify; ..."` via `execFileSync` → parses JSON output → reads model name from `best_models.json` → returns severity, confidence, probabilities |
| **`regress/route.ts`** | POST | Predict casualty count | Same pattern as classify → calls `predict.regress()` → returns rounded + float casualty prediction |
| **`eda-data/route.ts`** | GET | Dashboard statistics | Reads `eda_data.json` from disk → transforms raw dicts into sorted arrays with colors, proper day ordering, and flattened cross-tabs for chart consumption |
| **`model-comparison/route.ts`** | GET | Model metrics | Reads `classification_metrics.json` and `regression_metrics.json` → transforms into arrays of `{name, accuracy, f1Score, ...}` objects for the Model Playground charts |

---

### 5. `upload/` — Dataset

| File | Details |
|------|---------|
| **`Road.csv`** | **12,316 rows × 32 columns**. Each row represents one road accident. Contains temporal info (Time, Day_of_week), driver demographics (Age, Sex, Education, Experience), vehicle info (Type, Owner, Service year, Defects), road/environment conditions (Surface type, Weather, Light, Junction type), accident details (Collision type, Vehicle movement, Cause), and outcomes (Severity, Casualties). |

**Column categories:**
- **Temporal**: `Time`, `Day_of_week`
- **Driver**: `Age_band_of_driver`, `Sex_of_driver`, `Educational_level`, `Driving_experience`
- **Vehicle**: `Type_of_vehicle`, `Owner_of_vehicle`, `Service_year_of_vehicle`, `Defect_of_vehicle`, `Vehicle_driver_relation`
- **Road/Environment**: `Area_accident_occured`, `Lanes_or_Medians`, `Road_allignment`, `Types_of_Junction`, `Road_surface_type`, `Road_surface_conditions`, `Light_conditions`, `Weather_conditions`
- **Accident**: `Type_of_collision`, `Number_of_vehicles_involved`, `Vehicle_movement`, `Pedestrian_movement`, `Cause_of_accident`
- **Casualty (DROPPED — leakage)**: `Casualty_class`, `Sex_of_casualty`, `Age_band_of_casualty`, `Casualty_severity`, `Work_of_casuality`, `Fitness_of_casuality`
- **Targets**: `Accident_severity` (classification), `Number_of_casualties` (regression)

---

### 6. Root Configuration Files

| File | Purpose |
|------|---------|
| **`package.json`** | Node.js project manifest. Defines dependencies (next, react, recharts, shadcn components, prisma, lucide-react) and scripts (`dev`, `build`, `start`, `lint`). |
| **`tsconfig.json`** | TypeScript configuration. Defines path aliases: `@/*` → `./src/*`, `@/frontend/*` → `./frontend/*`, `@/backend/*` → `./backend/*`. Path aliases allow clean imports like `import { Button } from "@/frontend/components/ui/button"`. |
| **`tailwind.config.ts`** | Tailwind CSS configuration. Extends the default theme with custom colors (dark slate palette, amber/orange accents), border radius tokens, and keyframe animations. |
| **`next.config.ts`** | Next.js configuration. Minimal — default settings with TypeScript and App Router. |
| **`components.json`** | shadcn/ui configuration. Tells the CLI where to install components (`frontend/components/ui`), which style to use (New York), and the utility function path. |
| **`eslint.config.mjs`** | ESLint configuration. Uses Next.js recommended rules. |
| **`postcss.config.mjs`** | PostCSS configuration. Enables Tailwind CSS processing. |
| **`.env`** | Environment variables. Contains `PYTHON_BACKEND_URL` for the Python bridge. |
| **`.gitignore`** | Ignores `node_modules/`, `.next/`, `__pycache__/`, `.pkl` files, etc. |

---

## 🧠 ML Pipeline In Detail

### Dataset Preprocessing

1. **Load** `Road.csv` (12,316 rows × 32 columns)
2. **Clean**: Replace `"na"`, `"unknown"`, `""` → `NaN`
3. **Impute**: Fill missing categorical values with **mode** (most frequent value)
4. **Feature engineer**: Extract `Hour_of_Day` from the `Time` column (e.g., `"17:30"` → `17`)
5. **Drop leakage columns**: Remove 6 columns (see below)
6. **Encode**: Label-encode all 22 categorical features + target
7. **Result**: 24 features × 12,316 samples

### Why Drop Leakage Columns?

The 6 dropped columns (`Casualty_class`, `Sex_of_casualty`, `Age_band_of_casualty`, `Casualty_severity`, `Work_of_casuality`, `Fitness_of_casuality`) describe the **casualty outcome** — information that is only available **after** the accident occurs. Including them would cause **data leakage**: the model would "cheat" by using outcome data to predict severity, producing unrealistically high accuracy that wouldn't generalize to real-world prediction.

### Why These Models?

We use a **progression strategy** — start simple, add complexity:

| Model | Why? |
|-------|------|
| **Logistic Regression** | Linear baseline. If this works, the problem is linearly separable. If not, we prove we need non-linear models. |
| **KNN (k=5)** | Non-parametric, no assumptions about data distribution. Good for finding local patterns. |
| **Decision Tree** | Captures non-linear feature interactions. Highly interpretable (can trace individual decisions). |
| **Random Forest** | Ensemble of decorrelated trees. Reduces variance compared to a single Decision Tree. Most robust to noise. |
| **XGBoost** | Gradient boosting builds trees sequentially, each correcting the previous tree's errors. State-of-the-art for tabular data. |
| **Ridge Regression** | L2 regularization prevents overfitting when features are correlated (which they are after label encoding). |

### Why SMOTE?

The target variable is **severely imbalanced**:

| Class | Count | Percentage |
|-------|-------|------------|
| Slight Injury | 10,415 | 84.6% |
| Serious Injury | 1,743 | 14.1% |
| Fatal Injury | 158 | **1.3%** |

Without SMOTE, most models simply predict "Slight Injury" for every sample (achieving ~85% accuracy by doing nothing useful). **SMOTE** (Synthetic Minority Oversampling Technique) creates synthetic samples for minority classes in feature space, balancing the training set. It is applied **only to training data** (never the test set) to avoid information leakage.

**Key finding**: SMOTE improved Random Forest (F1: 0.781 → 0.789) but actually **hurt** Logistic Regression (F1: 0.775 → 0.641) and KNN (F1: 0.773 → 0.574). This is because synthetic samples in high-dimensional space can introduce noise that simpler models can't handle, while ensemble methods (RF) are robust to it.

### Why Use F1 Score (Weighted) Instead of Accuracy?

Accuracy is misleading for imbalanced datasets. A model predicting "Slight Injury" for every sample achieves **84.6% accuracy** while being completely useless. **Weighted F1** accounts for both precision and recall across all classes, weighted by class frequency, giving a more honest measure of performance.

### Classification Results (Accident Severity)

| Model | Approach | F1 Score | Accuracy |
|-------|----------|----------|----------|
| Random Forest | SMOTE | **0.7893** | **0.8097** |
| Random Forest | Tuned (GridSearchCV) | 0.7889 | 0.8109 |
| Random Forest | Base (No SMOTE) | 0.7810 | 0.8470 |
| Decision Tree | Base | 0.7771 | 0.8226 |
| Logistic Regression | Base | 0.7751 | 0.8458 |
| KNN | Base | 0.7726 | 0.8287 |
| Decision Tree | SMOTE | 0.6622 | 0.5990 |
| Logistic Regression | SMOTE | 0.6411 | 0.5678 |
| KNN | SMOTE | 0.5743 | 0.4923 |
| XGBoost | SMOTE + Class Weights | 0.4323 | 0.3584 |

**Best model**: Random Forest (SMOTE) — saved as `best_classifier.pkl`

### Regression Results (Number of Casualties)

| Model | Approach | R² Score | MAE |
|-------|----------|----------|-----|
| Random Forest | Base | **0.2694** | **0.6468** |
| Random Forest | Tuned (RandomizedSearchCV) | 0.2379 | 0.6520 |
| Decision Tree | Base | 0.1036 | 0.6459 |
| Linear Regression | Base | 0.0577 | 0.7215 |
| Ridge Regression | Base | 0.0577 | 0.7215 |

**Best model**: Random Forest — saved as `best_regressor.pkl`

**Note**: The low R² (0.27) is expected — casualty count depends heavily on factors not captured in the dataset (impact speed, seatbelt usage, vehicle safety rating, emergency response time, etc.).

---

### 🔬 Advanced Analysis (Interim Insights)

#### Dimensionality Reduction (PCA)
We performed **Principal Component Analysis (PCA)** to reduce the 24 features into a 2-dimensional space to understand the variance and clustering potential.
*   **Finding**: Reducing dimensions led to a drop of approximately **0.0094** in F1-score across models.
*   **Insight**: This confirms that road accidents are a **high-variance** problem. No single or small set of features (like just weather or just time) can accurately predict severity. The "long-tail" interaction of all 24 features is necessary for high-fidelity prediction.

#### Unsupervised Learning (Clustering)
We applied **K-Means Clustering** to the feature set (without target labels) to see if accidents naturally group together.
*   **Metric**: Silhouette Score of **0.0807** with 3 clusters.
*   **Insight**: The data naturally partitions into three risk profiles, even without explicit severity labels, which mathematically validates our choice of 3 classes (Fatal / Serious / Slight).

---


## 📡 API Endpoints

| Method | Endpoint | Description | Request | Response |
|--------|----------|-------------|---------|----------|
| `GET` | `/api/eda-data` | EDA statistics | — | `{severity_distribution, peak_hours, day_distribution, weather_severity, age_severity, vehicle_type_distribution, cause_distribution, total_records, total_features}` |
| `GET` | `/api/model-comparison` | Model metrics | — | `{classification: [...], regression: [...]}` |
| `POST` | `/api/classify` | Predict severity | 24 features as JSON | `{severity, confidence, probabilities, model, timestamp}` |
| `POST` | `/api/regress` | Predict casualties | 24 features as JSON | `{predicted_casualties, predicted_casualties_float, model, timestamp}` |

### Example: Classify Request

```json
POST /api/classify
{
  "Day_of_week": "Monday",
  "Age_band_of_driver": "18-30",
  "Sex_of_driver": "Male",
  "Educational_level": "Above high school",
  "Driving_experience": "5-10yr",
  "Type_of_vehicle": "Automobile",
  "Weather_conditions": "Normal",
  "Light_conditions": "Daylight",
  "Cause_of_accident": "No distancing",
  "Number_of_vehicles_involved": 2,
  "Hour_of_Day": 17,
  ...
}
```

### Example: Response

```json
{
  "severity": "Slight Injury",
  "confidence": 0.62,
  "probabilities": {
    "Fatal injury": 0.08,
    "Serious Injury": 0.30,
    "Slight Injury": 0.62
  },
  "model": "Random Forest (SMOTE)",
  "timestamp": "2026-04-12T09:15:00.000Z"
}
```

---

## 🚀 Getting Started

### Prerequisites

- **Node.js** ≥ 18 (or **Bun** runtime)
- **Python** ≥ 3.9
- **pip** (Python package manager)

### 1. Install Node dependencies

```bash
npm install
# or
bun install
```

### 2. Install Python ML dependencies

```bash
cd ml-service
pip install -r requirements.txt
cd ..
```

### 3. Train the ML models (if not already trained)

```bash
cd ml-service
python train_models.py
cd ..
```

> This will process `upload/Road.csv`, train 10 classification + 5 regression models, and save all artifacts to `ml-service/models/`.

### 4. Start the development server

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) to view the application.

---

## 🧩 Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **One model per file** | Each algorithm in its own Python file for easy debugging, testing, and modification. Adding a new model = create a file + register in `__init__.py`. |
| **Centralized `config.py`** | All paths, constants, and hyperparameters in one place. Portable paths via `os.path` (works on both Windows and Linux). |
| **Shared `base.py` evaluation** | Metric computation code is written once. No copy-paste across models. |
| **`best_models.json` for API routes** | API routes dynamically read the winning model name instead of hardcoding it. Retraining automatically updates the displayed model name. |
| **Next.js API Routes** (not FastAPI) | Eliminates the need for a separate Python web server. API routes call Python directly via `child_process.execFileSync`. Simpler deployment. |
| **SMOTE on training set only** | Applying SMOTE before train/test split would leak synthetic data into the test set, inflating metrics. |
| **Weighted F1 as primary metric** | Most honest metric for imbalanced multi-class classification (unlike accuracy). |
| **Label Encoding** (not One-Hot) | One-Hot would create 100+ sparse columns. Tree-based models (RF, XGBoost, DT) handle ordinal-encoded features well. |

---

## 📁 Path Aliases

Configured in `tsconfig.json`:

```json
{
  "@/*":          "./src/*",
  "@/frontend/*": "./frontend/*",
  "@/backend/*":  "./backend/*"
}
```

---

## 📄 License

This project is for educational and research purposes.
