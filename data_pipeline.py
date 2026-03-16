import pandas as pd
import numpy as np
import os

data_dir = "./EDAV-Project/data_clean/"

# Load core health datasets
hosp_df = pd.read_csv(os.path.join(data_dir, "hosp_clean.csv"))
er_df = pd.read_csv(os.path.join(data_dir, "er_visits_clean.csv"))
deaths_df = pd.read_csv(os.path.join(data_dir, "deaths_clean.csv"))
heat_df = pd.read_csv(os.path.join(data_dir, "heat_events_clean.csv"))

print("hosp_clean:")
print(hosp_df.columns)
print("heat_clean:")
print(heat_df.columns)
print("deaths_clean:")
print(deaths_df.columns)
