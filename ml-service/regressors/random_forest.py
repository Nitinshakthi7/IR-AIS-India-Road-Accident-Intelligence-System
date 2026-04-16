#!/usr/bin/env python3
"""
IR-AIS Regressor — Random Forest
Ensemble regression with optional RandomizedSearchCV tuning.
"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

NAME = "Random Forest"

PARAM_DIST = {
    "n_estimators": [50, 100, 200, 300],
    "max_depth": [5, 10, 15, 20, None],
    "min_samples_split": [2, 5, 10],
}


def build_model(n_estimators=100, random_state=42, **kwargs):
    """Return a fresh Random Forest Regressor."""
    return RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)


def build_tuned_model(X_train, y_train, random_state=42):
    """
    Run RandomizedSearchCV to find optimal hyperparameters.

    Returns
    -------
    best_model : fitted RandomForestRegressor with best params
    best_params : dict of best hyperparameters found
    """
    base = RandomForestRegressor(random_state=random_state)
    search = RandomizedSearchCV(
        base, PARAM_DIST, n_iter=10, cv=3, scoring="r2",
        n_jobs=-1, random_state=random_state, verbose=0
    )
    search.fit(X_train, y_train)
    print(f"  Best params: {search.best_params_}")
    return search.best_estimator_, search.best_params_
