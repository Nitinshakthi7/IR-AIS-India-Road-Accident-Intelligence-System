#!/usr/bin/env python3
"""
IR-AIS Clustering — K-Means
"""

from sklearn.cluster import KMeans

NAME = "K-Means"

def build_model(random_state=42, **kwargs):
    """Return a fresh K-Means Modeler."""
    return KMeans(n_clusters=3, random_state=random_state, n_init='auto')
