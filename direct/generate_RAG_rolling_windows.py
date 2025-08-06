import pandas as pd
import numpy as np
from config import MULTI_VAR_COLUMNS  

def get_RAG_rolling_windows(window_size,task_name):
    file_path = "dataset/train.xlsx"
    df = pd.read_excel(file_path, usecols=MULTI_VAR_COLUMNS + ["Segment_Boundary"])
    is_multi_var = task_name.startswith("Multi-")
    if is_multi_var:
        data_values = df[MULTI_VAR_COLUMNS].values  
        i = 0
        windows = []
        while i < len(df) - window_size + 1:
            window = data_values[i:i + window_size]
            segment_boundary = df["Segment_Boundary"].values[i:i + window_size]

            if 1 not in segment_boundary:
                windows.append(window)
                i += 1  
            else:
                boundary_idx = np.where(segment_boundary == 1)[0]
                if boundary_idx[-1] == window_size - 1:  
                    windows.append(window)
                    i += window_size  
                else:
                    i += boundary_idx[0] + 1
        return windows
    else:
        windows_0 = []
        windows_1 = []
        i = 0
        while i < len(df) - window_size + 1:
            q0_window = df["Variable 9"].values[i:i + window_size] if "Variable 9" in df.columns else np.zeros(window_size)
            q1_window = df["Variable 17"].values[i:i + window_size] if "Variable 17" in df.columns else np.zeros(window_size)
            segment_boundary = df["Segment_Boundary"].values[i:i + window_size]
            
            if 1 not in segment_boundary:
                windows_0.append(q0_window)
                windows_1.append(q1_window)
                i += 1 
            else:
                boundary_idx = np.where(segment_boundary == 1)[0]
                if boundary_idx[-1] == window_size - 1:  
                    windows_0.append(q0_window)
                    windows_1.append(q1_window)
                    i += window_size  
                else:
                    i += boundary_idx[0] + 1
        return [windows_0, windows_1]  