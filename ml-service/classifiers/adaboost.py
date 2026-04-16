#!/usr/bin/env python3
"""
IR-AIS Classifiers — AdaBoost
AdaBoost Classifier.
"""

from sklearn.ensemble import AdaBoostClassifier

NAME = "AdaBoost"

def build_model(random_state=42, **kwargs):
    """Return a fresh AdaBoost Classifier."""
    return AdaBoostClassifier(random_state=random_state)
