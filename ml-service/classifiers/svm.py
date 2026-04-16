#!/usr/bin/env python3
"""
IR-AIS Classifiers — SVM
Support Vector Classifier.
"""

from sklearn.svm import SVC

NAME = "SVM"

def build_model(random_state=42, **kwargs):
    """Return a fresh Support Vector Classifier."""
    return SVC(kernel='rbf', probability=True, random_state=random_state)
