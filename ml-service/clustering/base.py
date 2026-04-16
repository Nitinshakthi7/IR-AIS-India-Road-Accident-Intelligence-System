#!/usr/bin/env python3
"""
IR-AIS Clustering — Shared Evaluation Utilities
"""

from sklearn.metrics import silhouette_score, davies_bouldin_score

def evaluate_clustering(X, labels, approach="base"):
    """
    Evaluate a fitted clustering model.
    """
    # Silhouette Score and Davies-Bouldin require more than 1 cluster.
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    
    if n_clusters < 2:
        return {
            "silhouette_score": -1.0,
            "davies_bouldin": -1.0,
            "n_clusters": n_clusters,
            "approach": approach
        }
        
    sil = silhouette_score(X, labels)
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
