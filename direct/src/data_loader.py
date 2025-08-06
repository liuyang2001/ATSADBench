import pandas as pd
import numpy as np
from config import DATA_PATH, TEST_DATA_PATTERN, NORMAL_DATA_FIRST_SIX, NORMAL_DATA_SEVENTH_NINE, MULTI_VAR_COLUMNS, SINGLE_VAR_COLUMN, WINDOW_SIZE,STEP_SIZE,COLUMNS_MIN_MAX_VALUES

def scale_data(data, columns_to_scale, min_values_dict):
    scaled_data = data.copy()
    if columns_to_scale == ['Data']:
        # "U" tasks
        # Determine whether to process the first 2000 rows or the last 2000 rows
        mid_point = 2000
        # Process in segments
        first_half = data.iloc[:mid_point]
        second_half = data.iloc[mid_point:]
        # Use the minimum value of Variable 9 for the first 2000 rows.
        if 'Variable 9' in min_values_dict:
            first_half.loc[:, 'Data'] = first_half['Data'] - min_values_dict['Variable 9']['min']
        # Use the minimum value of Variable 16 for the last 2000 rows.
        if 'Variable 16' in min_values_dict:
            second_half.loc[:, 'Data'] = second_half['Data'] - min_values_dict['Variable 16']['min']
        # Merge back into scaled_data
        scaled_data.iloc[:mid_point] = first_half
        scaled_data.iloc[mid_point:] = second_half
    else:
        # "M" tasks
        # Process the other columns as usual
        for col in columns_to_scale:
            scaled_data[col] = scaled_data[col] - min_values_dict.get(col)['min']
    return scaled_data
def quantize_data(data, columns_to_quantize):
    """Multiply by 10^8 and round to the nearest integer, discarding the decimal part."""
    scale_factor = 10**8
    quantized_data = data.copy()
    for col in columns_to_quantize:
        quantized_data[col] = np.round(quantized_data[col] * scale_factor).astype(np.int64)
    return quantized_data

def preprocess_data(data, columns_to_process):
    scaled = scale_data(data, columns_to_process, COLUMNS_MIN_MAX_VALUES)
    quantized = quantize_data(scaled, columns_to_process)
    result = data.copy()
    for idx, col in enumerate(columns_to_process):
        result[col] = quantized[col]
    return result

def load_data(task_name, is_multi_var):
    test_path = f"{DATA_PATH}{task_name}{TEST_DATA_PATTERN}"
    if is_multi_var:
        df = pd.read_excel(test_path, usecols=MULTI_VAR_COLUMNS + ["Segment_Boundary", "Label"])
        columns_to_process = MULTI_VAR_COLUMNS
    else:
        df = pd.read_excel(test_path, usecols=SINGLE_VAR_COLUMN + ["Segment_Boundary", "Label"])
        columns_to_process = SINGLE_VAR_COLUMN
    old_df=df.copy()
    preprocessed_df = preprocess_data(df, columns_to_process)
    return old_df,preprocessed_df

def get_rolling_windows(data, is_multi_var, step):
    windows = []
    true_labels = []
    preprocessed_data = data 
    columns = MULTI_VAR_COLUMNS if is_multi_var else SINGLE_VAR_COLUMN
    data_values = preprocessed_data[columns].values
    i = 0
    while i < len(data) - WINDOW_SIZE + 1:
        window = data_values[i:i + WINDOW_SIZE]
        labels = data["Label"].values[i:i + WINDOW_SIZE]
        segment_boundary = data["Segment_Boundary"].values[i:i + WINDOW_SIZE]
        
        # Check whether there is any Segment_Boundary = 1 in the entire window
        if 1 not in segment_boundary:
            window_df = pd.DataFrame(window, columns=columns)
            windows.append(window_df)
            true_labels.append(labels)
            i += step
        else:
            # Find the positions where Segment_Boundary equals 1
            boundary_idx = np.where(segment_boundary == 1)[0]
            if boundary_idx[-1] == WINDOW_SIZE - 1:  # If it is the last time step
                window_df = pd.DataFrame(window, columns=columns)
                windows.append(window_df)
                true_labels.append(labels)
                i += WINDOW_SIZE 
            else:
                i += boundary_idx[0] + 1
    return windows, true_labels