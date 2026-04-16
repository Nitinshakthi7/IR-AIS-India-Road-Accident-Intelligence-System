#!/usr/bin/env python3
"""
IR-AIS Regressor — Linear Regression
Ordinary Least Squares linear regression.
"""

from sklearn.linear_model import LinearRegression

NAME = "Linear Regression"


def build_model(**kwargs):
    """Return a fresh Linear Regression model."""
    return LinearRegression()

def build_tuned_model(X_train, y_train, random_state=42):
    """Linear Regression doesn't tune, return base model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model, {}
