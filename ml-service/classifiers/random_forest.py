#!/usr/bin/env python3
"""
IR-AIS Classifier — Random Forest
Ensemble of decision trees with optional GridSearchCV tuning.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

NAME = "Random Forest"

PARAM_GRID = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, 15, None],
}


def build_model(n_estimators=100, random_state=42, **kwargs):
    """Return a fresh Random Forest classifier."""
    return RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)


def build_tuned_model(X_train, y_train, random_state=42):
    """
    Run GridSearchCV to find optimal hyperparameters.

    Returns
    -------
    best_model : fitted RandomForestClassifier with best params
    best_params : dict of best hyperparameters found
    """
    base = RandomForestClassifier(random_state=random_state)
    grid = GridSearchCV(
        base, PARAM_GRID, cv=3, scoring="f1_weighted", n_jobs=-1, verbose=0
    )
    grid.fit(X_train, y_train)
    print(f"  Best params: {grid.best_params_}")
    return grid.best_estimator_, grid.best_params_
