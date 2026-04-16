"""
IR-AIS Classifiers Package
Registers all classification models for the training pipeline.
"""

from classifiers import (
    logistic_regression,
    knn,
    svm,
    naive_bayes,
    decision_tree,
    random_forest,
    adaboost,
    xgboost_clf,
)

# Models trained without SMOTE (base evaluation)
BASE_MODELS = [logistic_regression, knn, svm, naive_bayes, decision_tree, random_forest, adaboost]

# Models trained on SMOTE-resampled data
SMOTE_MODELS = [logistic_regression, knn, svm, naive_bayes, decision_tree, random_forest, adaboost]

# Models with special SMOTE handling (e.g. sample weights)
SMOTE_WEIGHTED_MODELS = [xgboost_clf]

# Models with hyperparameter tuning support (must have build_tuned_model)
TUNABLE_MODELS = [random_forest]
