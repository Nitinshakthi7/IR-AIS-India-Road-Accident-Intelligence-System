#!/usr/bin/env python3
"""
IR-AIS Dimensionality Reduction — PCA
"""

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
import json

NAME = "PCA"

def apply_pca(X, n_components=0.95):
    """
    Apply PCA to reduce dimensionality. 
    If n_components is between 0 and 1, it selects the number of components 
    such that the amount of variance that needs to be explained is greater 
    than the percentage specified.
    
    Data must be scaled before applying this.
    Returns the transformed data and the fitted model.
    """
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)
    return X_pca, pca

def save_pca_plot(X_pca, y_class, output_dir, target_encoder=None):
    """Save a 2D scatter plot of the PCA reduced data colored by class."""
    plt.figure(figsize=(10, 8))
    
    # If target_encoder is provided, use actual names, otherwise just standard
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_class, cmap='viridis', alpha=0.5, s=15)
    plt.xlabel(f'Principal Component 1')
    plt.ylabel(f'Principal Component 2')
    plt.title('2D PCA Projection of Accident Features')
    plt.colorbar(scatter, label='Class Label')
    plt.grid(True, alpha=0.3)
    
    plot_path = os.path.join(output_dir, "pca_2d_projection.png")
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    return plot_path
