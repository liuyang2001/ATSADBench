import pandas as pd
import numpy as np
from config import MULTI_VAR_COLUMNS, COLUMNS_MIN_MAX_VALUES

def minmax_normalize(data, columns):
    normalized_data = data.copy()
    for col in columns:
        min_val = COLUMNS_MIN_MAX_VALUES[col]["min"]
        max_val = COLUMNS_MIN_MAX_VALUES[col]["max"]
        normalized_data[col] = (normalized_data[col] - min_val) / (max_val - min_val)
    return normalized_data
def reverse_scale(data, columns):
    reversed_data = data.copy()
    for col in columns:
        min_val = COLUMNS_MIN_MAX_VALUES[col]["min"]
        max_val = COLUMNS_MIN_MAX_VALUES[col]["max"]
        if max_val > min_val:
            reversed_data[col] = data[col] * (max_val - min_val) + min_val
        else:
            reversed_data[col] = data[col]  
    return reversed_data
if __name__ == "__main__":
    file_path = "dataset/M_train_data.xlsx"
    df = pd.read_excel(file_path, usecols=MULTI_VAR_COLUMNS+["Segment_Boundary"])
    normalized_df = minmax_normalize(df, MULTI_VAR_COLUMNS)

    output_path = "dataset/min_max_M_train_data.xlsx"
    normalized_df.to_excel(output_path, index=False)