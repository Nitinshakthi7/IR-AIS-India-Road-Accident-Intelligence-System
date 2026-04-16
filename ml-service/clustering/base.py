#!/usr/bin/env python3
"""
IR-AIS Clustering — Shared Evaluation Utilities
"""

from sklearn.metrics import silhouette_score, davies_bouldin_score
import numpy as np

def evaluate_clustering(X, labels, approach="base", sample_size=2000, random_state=42):
    """
    Evaluate a fitted clustering model.
    """
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    
    if n_clusters < 2:
        return {
            "silhouette_score": -1.0,
            "davies_bouldin": -1.0,
            "n_clusters": n_clusters,
            "approach": approach
        }
    
    n = len(X)
    if n > sample_size:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n, size=sample_size, replace=False)
        X_sil, labels_sil = X[idx], labels[idx]
    else:
        X_sil, labels_sil = X, labels
        
    sil = silhouette_score(X_sil, labels_sil)
    db = davies_bouldin_score(X, labels)
    
    return {
        "silhouette_score": round(float(sil), 4),
        "davies_bouldin": round(float(db), 4),
        "n_clusters": n_clusters,
        "approach": approach
    }

def print_metrics(metrics):
    print(f"    Clusters: {metrics['n_clusters']}, "
          f"Silhouette: {metrics['silhouette_score']:.4f}, "
          f"Davies-Bouldin: {metrics['davies_bouldin']:.4f}")
