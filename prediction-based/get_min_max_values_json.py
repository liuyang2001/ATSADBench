import pandas as pd
import numpy as np

COLUMNS_TO_PROCESS = ['Variable 1','Variable 2','Variable 3','Variable 4','Variable 5', 'Variable 6', 'Variable 7', 'Variable 8','Variable 9','Variable 10','Variable 11','Variable 12','Variable 13','Variable 14','Variable 15','Variable 16','Variable 17','Variable 18','Variable 19','Variable 20','Variable 21','Variable 22','Variable 23','Variable 24','Variable 25','Variable 26','Variable 27']

file_paths = [r"M_train_data.xlsx"]
all_data = {}
for file_path in file_paths:
    df = pd.read_excel(file_path)
    all_data[file_path] = df

min_max_values = {}
for column in COLUMNS_TO_PROCESS:
    column_min = np.inf
    column_max = -np.inf
    for df in all_data.values():
        if column in df.columns:
            column_min = min(column_min, np.min(df[column]))
            column_max = max(column_max, np.max(df[column]))
    min_max_values[column] = {"min": column_min, "max": column_max}
for column, values in min_max_values.items():
    print(f"{column}: min={values['min']}, max={values['max']}")

with open("dataset/min_max_values.json", "w", encoding='utf-8') as f:
    import json
    json.dump(min_max_values, f, indent=4, ensure_ascii=False)
    print("saved to 'dataset/min_max_values.json'")