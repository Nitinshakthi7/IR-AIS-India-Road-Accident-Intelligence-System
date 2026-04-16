#!/usr/usr/bin/env python3
"""
IR-AIS Regressors — SVR
Support Vector Regression.
"""

from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV

NAME = "SVR"

PARAM_DIST = {
    "C": [0.1, 1, 10],
    "epsilon": [0.01, 0.1, 0.2],
    "kernel": ["rbf", "linear"]
}

def build_model(random_state=42, **kwargs):
    """Return a fresh SVR Regressor."""
    return SVR(kernel='rbf', C=1.0, epsilon=0.1)

def build_tuned_model(X_train, y_train, random_state=42):
    """Run a faster RandomizedSearchCV for SVR."""
    base = SVR()
    search = RandomizedSearchCV(
        base, PARAM_DIST, n_iter=5, cv=3, scoring="r2",
        n_jobs=-1, random_state=random_state, verbose=0
    )
    search.fit(X_train, y_train)
    print(f"  Best params (SVR): {search.best_params_}")
    return search.best_estimator_, search.best_params_
