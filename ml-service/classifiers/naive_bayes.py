#!/usr/bin/env python3
"""
IR-AIS Classifiers — Naive Bayes
Gaussian Naive Bayes.
"""

from sklearn.naive_bayes import GaussianNB

NAME = "Naive Bayes"

def build_model(random_state=42, **kwargs):
    """Return a fresh GaussianNB Classifier."""
    return GaussianNB()
