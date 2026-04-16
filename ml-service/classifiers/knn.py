#!/usr/bin/env python3
"""
IR-AIS Classifier — K-Nearest Neighbors
Non-parametric instance-based classifier.
"""

from sklearn.neighbors import KNeighborsClassifier

NAME = "KNN"


def build_model(n_neighbors=5, **kwargs):
    """Return a fresh KNN model."""
    return KNeighborsClassifier(n_neighbors=n_neighbors)
