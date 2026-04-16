#!/usr/bin/env python3
"""
IR-AIS Classifier — Decision Tree
Single CART decision tree with depth limiting.
"""

from sklearn.tree import DecisionTreeClassifier

NAME = "Decision Tree"


def build_model(max_depth=10, random_state=42, **kwargs):
    """Return a fresh Decision Tree classifier."""
    return DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
