#!/usr/bin/env python3
"""
IR-AIS ML Training Pipeline — Orchestrator
Runs preprocessing, EDA, dimensionality reduction, clustering, 
classification, and regression training.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
from imblearn.over_sampling import SMOTE
import joblib

from config import RANDOM_STATE, MODEL_DIR, OUTPUT_DIR, TEST_SIZE
from preprocessing import load_and_preprocess
from eda import generate_eda

import classifiers
from classifiers.base import evaluate as clf_evaluate, print_metrics as clf_print
from classifiers import xgboost_clf, random_forest, decision_tree, adaboost

import regressors
from regressors import TUNABLE_MODELS
from regressors.base import evaluate as reg_evaluate, print_metrics as reg_print
from regressors.utils import transform_target, inverse_transform_target

import clustering
from clustering.base import evaluate_clustering, print_metrics as clust_print

from dimensionality import pca

warnings.filterwarnings("ignore")


# ─── Classification Task ─────────────────────────────────────────────────────
def train_classification(X_train, X_test, y_train, y_test, suffix=""):
    print("\n" + "=" * 60)
    print(f"CLASSIFICATION TASK: Accident_severity {suffix}")
    print("=" * 60)

    print(f"\n  Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    print(f"  Class distribution (train): {Counter(y_train)}")

    metrics_all = {}
    best_f1 = -1
    best_clf_model = None
    best_clf_name = ""

    # ── 3a. Base Models (no SMOTE) ──
    print("\n--- 3a. Base Models (No SMOTE) ---")

    for mod in classifiers.BASE_MODELS:
        name = mod.NAME
        print(f"\n  Training {name}...")
        model = mod.build_model(random_state=RANDOM_STATE)
        model.fit(X_train, y_train)
        metrics, f1 = clf_evaluate(model, X_test, y_test, approach="base")
        clf_print(metrics)
        metrics_all[name] = metrics

        if f1 > best_f1:
            best_f1, best_clf_model, best_clf_name = f1, model, name

    # ── 3b. SMOTE Models ──
    print("\n--- 3b. Optimized with SMOTE ---")
    print("  Applying SMOTE to training data...")

    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
    print(f"  After SMOTE - Train size: {X_train_sm.shape[0]}, Class distribution: {Counter(y_train_sm)}")

    for mod in classifiers.SMOTE_MODELS:
        name = f"{mod.NAME} (SMOTE)"
        print(f"\n  Training {name}...")
        model = mod.build_model(random_state=RANDOM_STATE)
        model.fit(X_train_sm, y_train_sm)
        metrics, f1 = clf_evaluate(model, X_test, y_test, approach="smote")
        clf_print(metrics)
        metrics_all[name] = metrics

        if f1 > best_f1:
            best_f1, best_clf_model, best_clf_name = f1, model, name

    # ── XGBoost with sample weights ──
    print(f"\n  Training {xgboost_clf.NAME} (SMOTE + scale_pos_weight)...")
    class_weight_dict = xgboost_clf.compute_sample_weights(y_train)
    sample_weights = xgboost_clf.get_sample_weight_array(y_train_sm, class_weight_dict)

    xgb_model = xgboost_clf.build_model(random_state=RANDOM_STATE)
    xgb_model.fit(X_train_sm, y_train_sm, sample_weight=sample_weights)
    metrics, f1 = clf_evaluate(xgb_model, X_test, y_test, approach="smote_xgboost")
    clf_print(metrics)
    metrics_all["XGBoost (SMOTE)"] = metrics

    if f1 > best_f1:
        best_f1, best_clf_model, best_clf_name = f1, xgb_model, "XGBoost (SMOTE)"

    # ── Save best classifier ──
    print(f"\n  * Best Classifier {suffix}: {best_clf_name} (F1={best_f1:.4f})")
    
    file_name = f"classification_metrics{'_pca' if 'PCA' in suffix else ''}.json"
    joblib.dump(best_clf_model, os.path.join(MODEL_DIR, f"best_classifier{'_pca' if 'PCA' in suffix else ''}.pkl"))

    with open(os.path.join(MODEL_DIR, file_name), "w") as f:
        json.dump(metrics_all, f, indent=2)

    return metrics_all, best_clf_name, best_f1


# ─── Regression Task ──────────────────────────────────────────────────────────
def train_regression(X_train, X_test, y_train, y_test, suffix=""):
    print("\n" + "=" * 60)
    print(f"REGRESSION TASK: Number_of_casualties {suffix}")
    print("=" * 60)

    print(f"\n  Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    print(f"  Target range: {y_train.min()} - {y_train.max()}, Mean: {y_train.mean():.2f}")

    metrics_all = {}
    best_r2 = -float("inf")
    best_reg_model = None
    best_reg_name = ""

    # ── 4a. Base & Tuned Models ──
    print("\n--- 4a. Training & Optimization ---")

    for mod in regressors.BASE_MODELS:
        name = mod.NAME
        print(f"\n  >>> Starting process for: {name}...")
        
        # Apply Log Transform to y for better variance stabilization
        y_train_log = transform_target(y_train)
        
        if mod in regressors.TUNABLE_MODELS:
            print(f"      [TUNING] Optimizing {name} with RandomizedSearchCV. This may take a minute...")
            model, best_params = mod.build_tuned_model(X_train, y_train_log, random_state=RANDOM_STATE)
        else:
            print(f"      [TRAINING] Fitting {name} with default parameters...")
            model = mod.build_model(random_state=RANDOM_STATE)
            model.fit(X_train, y_train_log)
            
        print(f"      [EVALUATING] Computing metrics for {name}...")
            
        # Predict on log scale and transform back for evaluation
        y_pred_log = model.predict(X_test)
        y_pred = inverse_transform_target(y_pred_log)
        
        # Calculate metrics on original scale
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        metrics = {
            "mae": round(float(mae), 4),
            "mse": round(float(mse), 4),
            "rmse": round(float(np.sqrt(mse)), 4),
            "r2": round(float(r2), 4),
            "approach": suffix
        }
        
        reg_print(metrics)
        metrics_all[name] = metrics

        if r2 > best_r2:
            best_r2, best_reg_model, best_reg_name = r2, model, name

    # ── Save best regressor ──
    print(f"\n  * Best Regressor {suffix}: {best_reg_name} (R²={best_r2:.4f})")
    
    file_name = f"regression_metrics{'_pca' if 'PCA' in suffix else ''}.json"
    joblib.dump(best_reg_model, os.path.join(MODEL_DIR, f"best_regressor{'_pca' if 'PCA' in suffix else ''}.pkl"))

    with open(os.path.join(MODEL_DIR, file_name), "w") as f:
        json.dump(metrics_all, f, indent=2)

    return metrics_all, best_reg_name, best_r2


# ─── Clustering Task ──────────────────────────────────────────────────────────
def train_clustering(X):
    print("\n" + "=" * 60)
    print("CLUSTERING TASK: Optimized Cluster Discovery")
    print("=" * 60)
    
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    
    print("\n  Optimizing K (Elbow/Silhouette analysis)...")
    best_sil = -1
    best_k = 2
    
    sample_size = min(2000, len(X))
    rng = np.random.default_rng(RANDOM_STATE)
    sil_idx = rng.choice(len(X), size=sample_size, replace=False)
    X_sil = X[sil_idx]
    
    for k in range(2, 7):
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE)
        lbls = km.fit_predict(X)
        sil = silhouette_score(X_sil, lbls[sil_idx])
        print(f"    K={k}, Silhouette Score: {sil:.4f}")
        if sil > best_sil:
            best_sil = sil
            best_k = k
            
    print(f"  * Selected optimal K={best_k} (Silhouette={best_sil:.4f})")
    
    metrics_all = {}
    
    for mod in clustering.MODELS:
        name = mod.NAME
        print(f"\n  Training {name} with K={best_k}...")
        # Most clustering models in our project support n_clusters
        try:
            model = mod.build_model(n_clusters=best_k, random_state=RANDOM_STATE)
        except:
            model = mod.build_model(random_state=RANDOM_STATE)
            
        labels = model.fit_predict(X)
        metrics = evaluate_clustering(X, labels, approach="optimized")
        clust_print(metrics)
        metrics_all[name] = metrics
        
    with open(os.path.join(MODEL_DIR, "clustering_metrics.json"), "w") as f:
        json.dump(metrics_all, f, indent=2)
        
    return metrics_all


# ─── Auxiliary Tasks (New 8 Prediction Goals) ───────────────────────────────
def train_auxiliary_tasks(X_raw_df, label_encoders):
    print("\n" + "=" * 60)
    print("TRAINING AUXILIARY TASKS (Random Forest & XGBoost)")
    print("=" * 60)
    
    from config import AUXILIARY_TASKS
    
    aux_metrics_all = {}
    
    # These base models will be used to reduce extreme training time
    model_builders = {
        "Random Forest": random_forest.build_model,
        "XGBoost": xgboost_clf.build_model
    }
    
    for task_name, target_col in AUXILIARY_TASKS.items():
        print(f"\n--- {task_name} ---")
        
        y_target = X_raw_df[target_col].copy()
        X_task = X_raw_df.drop(columns=[target_col]).copy()
        
        # Drop logic for leakages specific to these derived tasks
        if target_col == "Pedestrian_involved" and "Pedestrian_movement" in X_task.columns:
            X_task.drop(columns=["Pedestrian_movement"], inplace=True)
        if target_col == "Time_category" and "Hour_of_Day" in X_task.columns:
            X_task.drop(columns=["Hour_of_Day"], inplace=True)
            
        # Scale Data individually for this task
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_task)
        
        y_encoded = y_target.values
        
        try:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_scaled, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_encoded
            )
        except ValueError:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_scaled, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE
            )
            
        task_metrics = {}
        best_f1 = -1
        best_model = None
        best_name = ""
        
        for name, get_model in model_builders.items():
            print(f"  Training {name}...")
            clf = get_model(random_state=RANDOM_STATE)
            
            # XGBoost requires zero-indexed multiclass labels
            if "XGBoost" in name:
                from sklearn.preprocessing import LabelEncoder
                le_xgb = LabelEncoder()
                y_tr_fit = le_xgb.fit_transform(y_tr)
                y_te_fit = le_xgb.transform(y_te)
                clf.fit(X_tr, y_tr_fit)
            else:
                clf.fit(X_tr, y_tr)
                y_tr_fit = y_tr
                y_te_fit = y_te
                
            metrics, f1 = clf_evaluate(clf, X_te, y_te_fit, approach="base")
            clf_print(metrics)
            task_metrics[name] = metrics
            
            if f1 > best_f1:
                best_f1 = f1
                best_model = clf
                best_name = name
                
        print(f"  * Best for {task_name}: {best_name} (F1={best_f1:.4f})")
        
        safe_name = task_name.replace(" ", "_").lower()
        joblib.dump(best_model, os.path.join(MODEL_DIR, f"best_aux_{safe_name}.pkl"))
        joblib.dump(scaler, os.path.join(MODEL_DIR, f"scaler_aux_{safe_name}.pkl"))
        
        aux_metrics_all[task_name] = task_metrics

    # Driver Clustering Task
    print("\n" + "=" * 60)
    print("AUXILIARY TASK: Driver Profile Risk (Clustering)")
    print("=" * 60)
    driver_cols = ['Age_band_of_driver', 'Sex_of_driver', 'Driving_experience']
    
    if all(c in X_raw_df.columns for c in driver_cols):
        X_driver = X_raw_df[driver_cols].copy()
        scaler_dr = StandardScaler()
        X_driver_scaled = scaler_dr.fit_transform(X_driver)
        
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=3, random_state=RANDOM_STATE)
        kmeans.fit(X_driver_scaled)
        
        from sklearn.metrics import silhouette_score, davies_bouldin_score
        labels = kmeans.labels_
        sample_size = min(2000, len(X_driver_scaled))
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(len(X_driver_scaled), size=sample_size, replace=False)
        sil = silhouette_score(X_driver_scaled[idx], labels[idx])
        db = davies_bouldin_score(X_driver_scaled, labels)
        
        print(f"  K-Means Silhouette Score: {sil:.4f}, Davies-Bouldin: {db:.4f}")
        
        joblib.dump(kmeans, os.path.join(MODEL_DIR, "best_aux_driver_cluster.pkl"))
        joblib.dump(scaler_dr, os.path.join(MODEL_DIR, "scaler_aux_driver_cluster.pkl"))
        
        aux_metrics_all["Driver Profile Risk"] = {
            "K-Means (3 Clusters)": {
                "silhouette_score": sil,
                "davies_bouldin": db,
                "n_clusters": 3
            }
        }

    with open(os.path.join(MODEL_DIR, "auxiliary_metrics.json"), "w") as f:
        json.dump(aux_metrics_all, f, indent=2)

# ─── Ensemble Comparison (Day9 Style) ─────────────────────────────────────────
def compare_ensembles(X_train, X_test, y_train, y_test):
    print("\n" + "=" * 60)
    print("ENSEMBLE COMPARISON: DECISION TREE vs RANDOM FOREST vs ALL BOOSTING")
    print("=" * 60)

    models_to_compare = {
        "Decision Tree": decision_tree.build_model(random_state=RANDOM_STATE),
        "Random Forest": random_forest.build_model(random_state=RANDOM_STATE),
        "AdaBoost": adaboost.build_model(random_state=RANDOM_STATE),
        "XGBoost": xgboost_clf.build_model(random_state=RANDOM_STATE)
    }

    plt.figure(figsize=(10, 8))
    
    for name, model in models_to_compare.items():
        print(f"  Evaluating {name} for comparison...")
        model.fit(X_train, y_train)
        
        # We need probabilities for ROC, if target is multiclass, we use OvR conceptually
        # But for ROC plotting typically we do binary. If our target is multi-class, we plot macro-average
        try:
            y_probs = model.predict_proba(X_test)
            
            # Simple check, if binary: probability of class 1. If multi-class, compute macro average AUC
            n_classes = y_probs.shape[1]
            if n_classes == 2:
                y_prob_1 = y_probs[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_prob_1)
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
            else:
                # Approximate macro ROC for multi-class just to have a plot line
                from sklearn.preprocessing import label_binarize
                y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
                fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_probs.ravel())
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f'{name} (Micro-AUC = {roc_auc:.2f})')
                
        except Exception as e:
            print(f"    Skipping ROC for {name}: {str(e)}")

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve: Ensembles vs Trees')
    plt.legend(loc="lower right")
    
    plot_path = os.path.join(OUTPUT_DIR, "ensemble_comparison_roc.png")
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"  Saved comparison plot to {plot_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    X_raw, y_class, y_class_encoded, y_regr, target_encoder, label_encoders, feature_names = load_and_preprocess()

    # EDA
    generate_eda(X_raw)

    print("\n" + "=" * 60)
    print("GLOBAL PREPROCESSING: Scaling Data")
    print("=" * 60)
    
    # Scale Data Universal
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

    # Train/Test Splits
    X_train, X_test, y_class_train, y_class_test = train_test_split(
        X_scaled, y_class_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_class_encoded
    )
    
    X_train_r, X_test_r, y_regr_train, y_regr_test = train_test_split(
        X_scaled, y_regr, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # ── PHASE 1: Baseline Models ──
    cls_metrics, best_clf_name, best_clf_f1 = train_classification(X_train, X_test, y_class_train, y_class_test, "(Native Features)")
    reg_metrics, best_reg_name, best_reg_r2 = train_regression(X_train_r, X_test_r, y_regr_train, y_regr_test, "(Native Features)")

    # ── PHASE 1.5: Auxiliary Classification Targets ──
    train_auxiliary_tasks(X_raw, label_encoders)

    # ── PHASE 2: Dimensionality Reduction ──
    print("\n" + "=" * 60)
    print("DIMENSIONALITY REDUCTION TASK: PCA")
    print("=" * 60)
    # Fit PCA on scaled original data
    X_pca, pca_model = pca.apply_pca(X_scaled, n_components=0.95)
    joblib.dump(pca_model, os.path.join(MODEL_DIR, "pca_model.pkl"))
    
    plot_path = pca.save_pca_plot(X_pca, y_class_encoded, OUTPUT_DIR, target_encoder)
    print(f"  PCA Variance Explained: {sum(pca_model.explained_variance_ratio_):.4f} across {pca_model.n_components_} components")
    print(f"  Saved configuration plot to {plot_path}")

    X_train_pca, X_test_pca, y_class_train_pca, y_class_test_pca = train_test_split(
        X_pca, y_class_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_class_encoded
    )
    
    X_train_r_pca, X_test_r_pca, y_regr_train_pca, y_regr_test_pca = train_test_split(
        X_pca, y_regr, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # ── PHASE 3: PCA Reduced Models ──
    train_classification(X_train_pca, X_test_pca, y_class_train_pca, y_class_test_pca, "(PCA Reduced)")
    train_regression(X_train_r_pca, X_test_r_pca, y_regr_train_pca, y_regr_test_pca, "(PCA Reduced)")

    # ── PHASE 4: Clustering ──
    train_clustering(X_scaled)

    # ── PHASE 5: Ensemble Comparisons ──
    compare_ensembles(X_train, X_test, y_class_train, y_class_test)

    # ── Generative Report Trigger ──
    print("\n" + "=" * 60)
    print("Generating Analytical Post-run Report...")
    print("=" * 60)
    
    try:
        import report_generator
        report_generator.generate_final_report()
        print("  Successfully generated project_analysis_report.md")
    except Exception as e:
        print(f"  Warning: Report generation failed: {str(e)}")

    # Save summary 
    best_models_info = {
        "best_classifier_name": best_clf_name,
        "best_classifier_f1": round(best_clf_f1, 4),
        "best_regressor_name": best_reg_name,
        "best_regressor_r2": round(best_reg_r2, 4),
    }
    with open(os.path.join(MODEL_DIR, "best_models.json"), "w") as f:
        json.dump(best_models_info, f, indent=2)

    print("\nTraining Pipeline Complete!")

if __name__ == "__main__":
    main()
