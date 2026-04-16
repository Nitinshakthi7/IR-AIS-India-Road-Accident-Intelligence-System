#!/usr/bin/env python3
"""
IR-AIS Regressor — Ridge Regression
L2-regularized linear regression.
"""

from sklearn.linear_model import Ridge, RidgeCV

NAME = "Ridge Regression"


def build_model(random_state=42, **kwargs):
    """Return a fresh Ridge Regressor."""
    return Ridge(random_state=random_state)


def build_tuned_model(X_train, y_train, random_state=42):
    """Use RidgeCV for automatic tuning of alpha."""
    # RidgeCV uses Leave-One-Out Cross-Validation by default
    model = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
    model.fit(X_train, y_train)
    print(f"  Best Alpha (Ridge): {model.alpha_}")
    return model, {"alpha": model.alpha_}
