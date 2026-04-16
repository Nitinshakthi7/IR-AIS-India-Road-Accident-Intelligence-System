import pandas as pd
import numpy as np

df = pd.read_csv("upload/Road.csv")
print(df["Defect_of_vehicle"].value_counts())
