import pandas as pd
import numpy as np
from config import DATA_PATH, TEST_DATA_PATTERN, NORMAL_DATA_FIRST_SIX, NORMAL_DATA_SEVENTH_NINE, MULTI_VAR_COLUMNS, SINGLE_VAR_COLUMN, WINDOW_SIZE,STEP_SIZE,COLUMNS_MIN_MAX_VALUES,HORIZON
import math
def scale_data(data, columns_to_scale, min_values_dict):
    scaled_data = data.copy()
    if columns_to_scale == ['Data']:
        # "U" tasks
        # Determine whether to process the first 2000 rows or the last 2000 rows
        mid_point = 2000+WINDOW_SIZE
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
def dequantize_data(data, columns_to_dequantize):
    scale_factor = 10**8
    dequantized_data = data.copy()
    for col in columns_to_dequantize:
        dequantized_data[col] = dequantized_data[col].astype(float) / scale_factor
    return dequantized_data
def descale_data(data, columns_to_descale, min_values_dict):
    descaled_data = data.copy()
    if columns_to_descale == ['Data']:
        mid_point = 2000
        first_half = descaled_data.iloc[:mid_point].copy()
        second_half = descaled_data.iloc[mid_point:].copy()
        if 'Variable 9' in min_values_dict:
            first_half.loc[:, 'Data'] = first_half['Data'] + min_values_dict['Variable 9']['min']
        if 'Variable 16' in min_values_dict:
            second_half.loc[:, 'Data'] = second_half['Data'] + min_values_dict['Variable 16']['min']
        descaled_data.iloc[:mid_point] = first_half
        descaled_data.iloc[mid_point:] = second_half
    else:
        for col in columns_to_descale:
            if col in min_values_dict and 'min' in min_values_dict[col]:
                descaled_data[col] = descaled_data[col] + min_values_dict[col]['min']
            else:
                print(f"Warning: No minimum value found for column {col} in min_values_dict")
    return descaled_data
def deprocess_data(data, columns_to_process):
    dequantized = dequantize_data(data, columns_to_process)
    descaled = descale_data(dequantized, columns_to_process, COLUMNS_MIN_MAX_VALUES)
    result = data.copy()
    for col in columns_to_process:
        result[col] = descaled[col]
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

def get_rolling_windows(data, is_multi_var,step):
    windows = []
    true_labels = []
    preprocessed_data = data
    columns = MULTI_VAR_COLUMNS if is_multi_var else SINGLE_VAR_COLUMN
    data_values = preprocessed_data[MULTI_VAR_COLUMNS if is_multi_var else SINGLE_VAR_COLUMN].values
    i = 0
    boundary = math.ceil((2000+WINDOW_SIZE - WINDOW_SIZE) / STEP_SIZE)
    while i < 2000+WINDOW_SIZE+(boundary-1)*STEP_SIZE+1:
        window = data_values[i:i + WINDOW_SIZE]
        horizon= data_values[i + WINDOW_SIZE:i + WINDOW_SIZE+HORIZON]
        if i+WINDOW_SIZE+HORIZON>=4000+2*WINDOW_SIZE:
            segment_boundary = data["Segment_Boundary"].values[i + WINDOW_SIZE:4000+2*WINDOW_SIZE]
        else:
            segment_boundary = data["Segment_Boundary"].values[i + WINDOW_SIZE:i + WINDOW_SIZE+HORIZON]
        
        # Check whether there is any Segment_Boundary = 1 in the entire window
        if 1 not in segment_boundary:
            window_dict={
                "window":pd.DataFrame(window,columns=columns),
                "current_window_index":[j for j in range(i,i + WINDOW_SIZE)],
                "prediction_index":[j for j in range(i+WINDOW_SIZE,min((i + WINDOW_SIZE+HORIZON),4000+2*WINDOW_SIZE))]
            }
            windows.append(window_dict)
            i += step
        else:
            # Find the positions where Segment_Boundary equals 1
            boundary_idx = np.where(segment_boundary == 1)[0]
            if boundary_idx[-1] == HORIZON - 1:  # If it is the last time step
                window_dict={
                "window":pd.DataFrame(window,columns=columns),
                "current_window_index":[j for j in range(i,i + WINDOW_SIZE)],
                "prediction_index":[j for j in range(i+WINDOW_SIZE,(i+WINDOW_SIZE+HORIZON))]
            }
                windows.append(window_dict)
                i += WINDOW_SIZE+HORIZON  
            else:
                window_dict={
                "window":pd.DataFrame(window,columns=columns),
                "current_window_index":[j for j in range(i,i + WINDOW_SIZE)],
                "prediction_index":[j for j in range(i+WINDOW_SIZE,(i+WINDOW_SIZE+boundary_idx[-1]+1))]
            }
                windows.append(window_dict)
                i += boundary_idx[-1]+1+WINDOW_SIZE
    return windows, true_labels