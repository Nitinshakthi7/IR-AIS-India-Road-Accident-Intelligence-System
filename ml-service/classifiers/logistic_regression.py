#!/usr/bin/env python3
"""
IR-AIS Classifier — Logistic Regression
Multinomial logistic regression with L2 regularization.
"""

from sklearn.linear_model import LogisticRegression

NAME = "Logistic Regression"


def build_model(random_state=42, **kwargs):
    """Return a fresh Logistic Regression model."""
    return LogisticRegression(max_iter=1000, random_state=random_state)
