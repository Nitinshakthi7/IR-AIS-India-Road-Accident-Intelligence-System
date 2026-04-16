#!/usr/bin/env python3
"""
IR-AIS ML Pipeline — Configuration
Centralized constants, paths, and hyperparameters.
"""

import os

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))      # ml-service/
PROJECT_DIR = os.path.dirname(BASE_DIR)                    # project root

DATA_PATH = os.path.join(PROJECT_DIR, "upload", "Road.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")

# ─── Training ─────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.2

# ─── Targets ──────────────────────────────────────────────────────────────────
TARGET_CLASS = "Accident_severity"
TARGET_REGR = "Number_of_casualties"

# ─── Auxiliary Tasks ──────────────────────────────────────────────────────────
AUXILIARY_TASKS = {
    "Cause of Accident": "Cause_of_accident",
    "Type of Collision": "Type_of_collision",
    "Pedestrian Involvement": "Pedestrian_involved",
    "Time of Accident": "Time_category",
    "Day of Week": "Day_of_week",
    "Vehicle Defect": "Defect_of_vehicle",
    "Area of Accident": "Area_accident_occured",
}

# ─── Leakage Columns (dropped before training) ───────────────────────────────
LEAKAGE_COLS = [
    "Casualty_class",
    "Sex_of_casualty",
    "Age_band_of_casualty",
    "Casualty_severity",
    "Work_of_casuality",
    "Fitness_of_casuality",
]

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
