import pandas as pd

def get_positive_data(task, column_names, WINDOW_SIZE, is_first_half):

    task_mapping = {
        "M-IL-FVA": {1: 0, 0: 3},  
        "M-IL-CDA": {1: 1, 0: 4},  
        "M-IL-TVDA": {1: 2, 0: 5},  
        "M-OL-FVA": {1: 6, 0: 9},   
        "M-OL-CDA": {1: 7, 0: 10},  
        "M-OL-TVDA": {1: 8, 0: 11},   
        "U-FVA": {1: 6, 0: 9},    
        "U-CDA": {1: 7, 0: 10},   
        "U-TVDA": {1: 8, 0: 11}   
    }
    
    # 读取保存的Excel文件
    with pd.ExcelFile("dataset/positive_samples.xlsx") as xls:
        if task in task_mapping:
            dataset_idx = task_mapping[task][is_first_half]
            data = pd.read_excel(xls, sheet_name=f"Dataset_{dataset_idx + 1}")
            
            if column_names == ["Data"]:
                if 6 <= dataset_idx <= 8:  
                    column_names = ["variable 9"]
                elif 9 <= dataset_idx <= 11:  
                    column_names = ["variable 16"]
            
            missing_columns = [col for col in column_names if col not in data.columns]
            if missing_columns:
                return f"Error: Column {missing_columns} do not exist"
            if len(data) < WINDOW_SIZE:
                return f"Error: Data length is less than {WINDOW_SIZE}"
            return data[column_names].iloc[-WINDOW_SIZE:],column_names
        else:
            return f"Error: Unknown task '{task}'"