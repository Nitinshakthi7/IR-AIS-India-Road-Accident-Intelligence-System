#!/usr/bin/env python3
"""
IR-AIS Regressor — Decision Tree
Single CART regression tree with depth limiting.
"""

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV

NAME = "Decision Tree"

PARAM_DIST = {
    "max_depth": [3, 5, 10, 20, None],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 2, 4]
}

def build_model(max_depth=10, random_state=42, **kwargs):
    """Return a fresh Decision Tree Regressor."""
    return DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)

def build_tuned_model(X_train, y_train, random_state=42):
    """Run RandomizedSearchCV for Decision Tree."""
    base = DecisionTreeRegressor(random_state=random_state)
    search = RandomizedSearchCV(
        base, PARAM_DIST, n_iter=15, cv=3, scoring="r2",
        n_jobs=-1, random_state=random_state, verbose=0
    )
    search.fit(X_train, y_train)
    print(f"  Best params (Decision Tree): {search.best_params_}")
    return search.best_estimator_, search.best_params_
