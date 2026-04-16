#!/usr/bin/env python3
"""
IR-AIS Clustering — DBSCAN
"""

from sklearn.cluster import DBSCAN

NAME = "DBSCAN"

def build_model(random_state=42, **kwargs):
    """Return a fresh DBSCAN Modeler."""
    return DBSCAN(eps=0.5, min_samples=5)
