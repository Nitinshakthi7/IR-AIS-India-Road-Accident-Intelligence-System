#!/usr/bin/env python3
"""
IR-AIS Regressors — Utility Helpers
Functions for target transformations and scaling.
"""

import numpy as np

def transform_target(y):
    """Logarithmic transformation for skewed targets."""
    return np.log1p(y)

def inverse_transform_target(y_log):
    """Reverse logarithmic transformation."""
    return np.expm1(y_log)
