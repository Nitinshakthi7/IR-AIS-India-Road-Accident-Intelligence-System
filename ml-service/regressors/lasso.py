#!/usr/usr/bin/env python3
"""
IR-AIS Regressors — Lasso
L1 Regularized linear regression.
"""

from sklearn.linear_model import Lasso, LassoCV

NAME = "Lasso"

def build_model(random_state=42, **kwargs):
    """Return a fresh Lasso Regressor."""
    return Lasso(random_state=random_state)

def build_tuned_model(X_train, y_train, random_state=42):
    """Use LassoCV for automatic tuning of alpha."""
    # LassoCV automatically finds the best alpha
    model = LassoCV(cv=5, random_state=random_state)
    model.fit(X_train, y_train)
    print(f"  Best Alpha (Lasso): {model.alpha_}")
    return model, {"alpha": model.alpha_}
