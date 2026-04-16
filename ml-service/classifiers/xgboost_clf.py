#!/usr/bin/env python3
"""
IR-AIS Classifier — XGBoost
Gradient boosted trees with class weight support for imbalanced data.
"""

import numpy as np
from collections import Counter
from xgboost import XGBClassifier

NAME = "XGBoost"

# Flag: this model uses sample_weight during .fit()
USES_SAMPLE_WEIGHT = True


def build_model(random_state=42, **kwargs):
    """Return a fresh XGBoost classifier."""
    return XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=random_state,
        use_label_encoder=False,
        eval_metric="mlogloss",
    )


def compute_sample_weights(y_train_original):
    """
    Compute per-class weights from the original (pre-SMOTE) class distribution.
    Minority classes get higher weights.

    Parameters
    ----------
    y_train_original : array-like — original training labels before SMOTE

    Returns
    -------
    class_weight_dict : dict mapping class label → weight
    """
    class_counts = Counter(y_train_original)
    max_count = max(class_counts.values())
    return {cls: max_count / count for cls, count in class_counts.items()}


def get_sample_weight_array(y_train, class_weight_dict):
    """
    Convert per-class weight dict to a per-sample weight array.

    Parameters
    ----------
    y_train : array-like — training labels (possibly SMOTE-resampled)
    class_weight_dict : dict — class label → weight

    Returns
    -------
    np.ndarray of per-sample weights
    """
    return np.array([class_weight_dict[yi] for yi in y_train])
