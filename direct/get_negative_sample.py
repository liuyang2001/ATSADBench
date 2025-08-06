import pandas as pd

def get_negative_sample(task, column_names, WINDOW_SIZE, is_first_half, negative_sample_number):
    with pd.ExcelFile("dataset/negative_samples.xlsx") as xls:
        if negative_sample_number == 1:
            is_multi_var = task.startswith("M-")
            if is_multi_var:
                task_mapping = {
                    "M-IL-FVA": {True: 3, False: 0},  
                    "M-IL-CDA": {True: 4, False: 1},  
                    "M-IL-TVDA": {True: 5, False: 2}, 
                    "M-OL-FVA": {True: 9, False: 6},   
                    "M-OL-CDA": {True: 10, False: 7},  
                    "M-OL-TVDA": {True: 11, False: 8}  
                }
                dataset_idx = task_mapping[task][is_first_half]
                data = pd.read_excel(xls, sheet_name=f"Negative_Dataset_{dataset_idx + 1}")
                half_window = WINDOW_SIZE // 2
                start_idx = 250  
                if WINDOW_SIZE%2==0:
                    sample = data[column_names].iloc[start_idx - half_window:start_idx + half_window]
                else:
                    sample = data[column_names].iloc[start_idx - half_window:start_idx + half_window+1]
                # generate negative_analysis_process and negative_final_answer
                if dataset_idx + 1 in [1, 4, 7, 10]:  
                    if dataset_idx + 1 == 1:
                        variable_sequence = "1st, 2nd, 3rd, and 4th variable sequences"
                    elif dataset_idx + 1 == 4:
                        variable_sequence = "13th variable sequence"
                    elif dataset_idx + 1 == 7:
                        variable_sequence = "9th, 10th, 11th and 12th variable sequences"
                    elif dataset_idx + 1 == 10:
                        variable_sequence = "16th variable sequence"
                    negative_analysis_process = f"In the {variable_sequence}, data from timestep index {int(WINDOW_SIZE/2)} to index {WINDOW_SIZE-1} remains constant and violates the relationship with other variables, determined as a constant value fault."
                elif dataset_idx + 1 in [2, 5, 8, 11]:  
                    if dataset_idx + 1 == 2:
                        variable_sequence = "1st"
                    elif dataset_idx + 1 == 5:
                        variable_sequence = "13th"
                    elif dataset_idx + 1 == 8:
                        variable_sequence = "9th"
                    elif dataset_idx + 1 == 11:
                        variable_sequence = "16th"
                    negative_analysis_process = f"In the {variable_sequence} variable sequence, data from timestep index {int(WINDOW_SIZE/2)} to index {WINDOW_SIZE-1} shows a mutation compared to previous timesteps and violates the relationship with other variables, determined as a mutation fault."
                elif dataset_idx + 1 in [3, 9]:  
                    if dataset_idx + 1 == 3:
                        variable_sequence = "1st, 2nd, 3rd, and 4th variable sequences"
                    elif dataset_idx + 1 == 9:
                        variable_sequence = "9th, 10th, 11th and 12th variable sequences"
                    negative_analysis_process = f"In the {variable_sequence}, data from timestep index {int(WINDOW_SIZE/2)} to index {WINDOW_SIZE-1} shows an overall offset compared to previous timesteps, indicating a deviation between the sensor output and the true value, and violates the relationship with other sensor outputs, determined as a bias fault."
                elif dataset_idx + 1 in [6,12]:  
                    if dataset_idx + 1 == 6:
                        variable_sequence = "13th"
                    elif dataset_idx + 1 == 12:
                        variable_sequence = "16th"
                    negative_analysis_process = f"In the {variable_sequence} variable sequence, data from timestep index {int(WINDOW_SIZE/2)} to index {WINDOW_SIZE-1} has added random noise compared to previous timesteps, indicating a deviation between the sensor output and the true value, and violates the relationship with other sensor outputs, determined as a bias fault."
                negative_final_answer = '[' + ','.join(str(i) for i in range(int(WINDOW_SIZE/2), WINDOW_SIZE)) + ']'
                return sample, negative_analysis_process, negative_final_answer
            else:
                task_mapping = {
                    "Uni-CVAD": {True: 9, False: 6},  
                    "Uni-CDAD": {True: 10, False: 7}, 
                    "Uni-TVDAD": {True: 11, False: 8} 
                }
                dataset_idx = task_mapping[task][is_first_half]
                data = pd.read_excel(xls, sheet_name=f"Negative_Dataset_{dataset_idx + 1}")
                adjusted_columns = ['variable 16'] if is_first_half else ['variable 9']
                half_window = WINDOW_SIZE // 2
                start_idx = 250  
                if WINDOW_SIZE%2==0:
                    sample = data[adjusted_columns].iloc[start_idx - half_window:start_idx + half_window]
                else:
                    sample = data[adjusted_columns].iloc[start_idx - half_window:start_idx + half_window+1]
                # generate negative_analysis_process and negative_final_answer
                if dataset_idx + 1 in [7, 10]:  
                    negative_analysis_process = f"In the sequence, data from timestep index {int(WINDOW_SIZE/2)} to index {WINDOW_SIZE-1} remains constant, determined as a constant value fault."
                elif dataset_idx + 1 in [8, 11]:  
                    negative_analysis_process = f"In the sequence, data from timestep index {int(WINDOW_SIZE/2)} to index {WINDOW_SIZE-1} shows a mutation compared to previous timesteps, determined as a mutation fault."
                elif dataset_idx + 1 in [9]:  
                    negative_analysis_process = f"In the sequence, data from timestep index {int(WINDOW_SIZE/2)} to index {WINDOW_SIZE-1} shows an overall offset compared to previous timesteps, indicating a deviation between the sensor output and the true value, determined as a bias fault."
                elif dataset_idx + 1 in [12]: 
                    negative_analysis_process = f"In the sequence, data from timestep index {int(WINDOW_SIZE/2)} to index {WINDOW_SIZE-1} has added random noise compared to previous timesteps, indicating a deviation between the sensor output and the true value, determined as a bias fault."
                negative_final_answer = '[' + ','.join(str(i) for i in range(int(WINDOW_SIZE/2), WINDOW_SIZE)) + ']'
                return sample, negative_analysis_process, negative_final_answer
        elif negative_sample_number == 2:
            is_multi_var = task.startswith("M-")
            if is_multi_var:
                task_mapping = {
                    "M-IL-FVA": {True: [1, 2], False: [4, 5]},  
                    "M-IL-CDA": {True: [0, 2], False: [3, 5]},  
                    "M-IL-TVDA": {True: [0, 1], False: [3, 4]}, 
                    "M-OL-FVA": {True: [7, 8], False: [10, 11]},   
                    "M-OL-CDA": {True: [6, 8], False: [9, 11]},  
                    "M-OL-TVDA": {True: [6, 7], False: [9, 10]}  
                }
                dataset_indices = task_mapping[task][is_first_half]
                data1 = pd.read_excel(xls, sheet_name=f"Negative_Dataset_{dataset_indices[0] + 1}")
                data2 = pd.read_excel(xls, sheet_name=f"Negative_Dataset_{dataset_indices[1] + 1}")
                half_window = WINDOW_SIZE // 2
                start_idx = 250  
                if WINDOW_SIZE%2==0:
                    sample1 = data1[column_names].iloc[start_idx - half_window:start_idx + half_window]
                    sample2 = data2[column_names].iloc[start_idx - half_window:start_idx + half_window]
                else:
                    sample1 = data1[column_names].iloc[start_idx - half_window:start_idx + half_window+1]
                    sample2 = data2[column_names].iloc[start_idx - half_window:start_idx + half_window+1]
                if dataset_indices[0] + 1 in [1, 4, 7, 10]:  
                    if dataset_indices[0] + 1 == 1:
                        variable_sequence = "1st, 2nd, 3rd, and 4th variable sequences"
                    elif dataset_indices[0] + 1 == 4:
                        variable_sequence = "13th variable sequence"
                    elif dataset_indices[0] + 1 == 7:
                        variable_sequence = "9th, 10th, 11th and 12th variable sequences"
                    elif dataset_indices[0] + 1 == 10:
                        variable_sequence = "16th variable sequence"
                    negative_analysis_process1 = f"In the {variable_sequence}, data from timestep index {int(WINDOW_SIZE/2)} to index {WINDOW_SIZE-1} remains constant and violates the relationship with other variables, determined as a constant value fault."
                elif dataset_indices[0] + 1 in [2, 5, 8, 11]:  
                    if dataset_indices[0] + 1 == 2:
                        variable_sequence = "1st"
                    elif dataset_indices[0] + 1 == 5:
                        variable_sequence = "13th"
                    elif dataset_indices[0] + 1 == 8:
                        variable_sequence = "9th"
                    elif dataset_indices[0] + 1 == 11:
                        variable_sequence = "16th"
                    negative_analysis_process1 = f"In the {variable_sequence} variable sequence, data from timestep index {int(WINDOW_SIZE/2)} to index {WINDOW_SIZE-1} shows a mutation compared to previous timesteps and violates the relationship with other variables, determined as a mutation fault."
                elif dataset_indices[0] + 1 in [3, 9]:  
                    if dataset_indices[0] + 1 == 3:
                        variable_sequence = "1st, 2nd, 3rd, and 4th variable sequences"
                    elif dataset_indices[0] + 1 == 9:
                        variable_sequence = "9th, 10th, 11th and 12th variable sequences"
                    negative_analysis_process1 = f"In the {variable_sequence}, data from timestep index {int(WINDOW_SIZE/2)} to index {WINDOW_SIZE-1} shows an overall offset compared to previous timesteps, indicating a deviation between the sensor output and the true value, and violates the relationship with other sensor outputs, determined as a bias fault."
                elif dataset_indices[0] + 1 in [6,12]: 
                    if dataset_indices[0] + 1 == 6:
                        variable_sequence = "13th"
                    elif dataset_indices[0] + 1 == 12:
                        variable_sequence = "16th"
                    negative_analysis_process1 = f"In the {variable_sequence} variable sequence, data from timestep index {int(WINDOW_SIZE/2)} to index {WINDOW_SIZE-1} has added random noise compared to previous timesteps, indicating a deviation between the sensor output and the true value, and violates the relationship with other sensor outputs, determined as a bias fault."
                # ------------------------------------------------------------------------------------
                if dataset_indices[1] + 1 in [1, 4, 7, 10]:  
                    if dataset_indices[1] + 1 == 1:
                        variable_sequence = "1st, 2nd, 3rd, and 4th variable sequences"
                    elif dataset_indices[1] + 1 == 4:
                        variable_sequence = "13th variable sequence"
                    elif dataset_indices[1] + 1 == 7:
                        variable_sequence = "9th, 10th, 11th and 12th variable sequences"
                    elif dataset_indices[1] + 1 == 10:
                        variable_sequence = "16th variable sequence"
                    negative_analysis_process2 = f"In the {variable_sequence}, data from timestep index {int(WINDOW_SIZE/2)} to index {WINDOW_SIZE-1} remains constant and violates the relationship with other variables, determined as a constant value fault."
                elif dataset_indices[1] + 1 in [2, 5, 8, 11]: 
                    if dataset_indices[1] + 1 == 2:
                        variable_sequence = "1st"
                    elif dataset_indices[1] + 1 == 5:
                        variable_sequence = "13th"
                    elif dataset_indices[1] + 1 == 8:
                        variable_sequence = "9th"
                    elif dataset_indices[1] + 1 == 11:
                        variable_sequence = "16th"
                    negative_analysis_process2 = f"In the {variable_sequence} variable sequence, data from timestep index {int(WINDOW_SIZE/2)} to index {WINDOW_SIZE-1} shows a mutation compared to previous timesteps and violates the relationship with other variables, determined as a mutation fault."
                elif dataset_indices[1] + 1 in [3, 9]: 
                    if dataset_indices[1] + 1 == 3:
                        variable_sequence = "1st, 2nd, 3rd, and 4th variable sequences"
                    elif dataset_indices[1] + 1 == 9:
                        variable_sequence = "9th, 10th, 11th and 12th variable sequences"
                    negative_analysis_process2 = f"In the {variable_sequence}, data from timestep index {int(WINDOW_SIZE/2)} to index {WINDOW_SIZE-1} shows an overall offset compared to previous timesteps, indicating a deviation between the sensor output and the true value, and violates the relationship with other sensor outputs, determined as a bias fault."
                elif dataset_indices[1] + 1 in [6,12]:  
                    if dataset_indices[1] + 1 == 6:
                        variable_sequence = "13th"
                    elif dataset_indices[1] + 1 == 12:
                        variable_sequence = "16th"
                    negative_analysis_process2 = f"In the {variable_sequence} variable sequence, data from timestep index {int(WINDOW_SIZE/2)} to index {WINDOW_SIZE-1} has added random noise compared to previous timesteps, indicating a deviation between the sensor output and the true value, and violates the relationship with other sensor outputs, determined as a bias fault."
                # -------------------------------------------------------------------------------------
                negative_final_answer = '[' + ','.join(str(i) for i in range(int(WINDOW_SIZE/2), WINDOW_SIZE)) + ']'
                return [sample1, sample2], negative_analysis_process1, negative_analysis_process2, negative_final_answer
            else:
                task_mapping = {
                    "U-FVA": {True: [7, 8], False: [10, 11]},  
                    "U-CDA": {True: [6, 8], False: [9, 11]},  
                    "U-TVDA": {True: [6, 7], False: [9, 10]}   
                }
                dataset_indices = task_mapping[task][is_first_half]
                data1 = pd.read_excel(xls, sheet_name=f"Negative_Dataset_{dataset_indices[0] + 1}")
                data2 = pd.read_excel(xls, sheet_name=f"Negative_Dataset_{dataset_indices[1] + 1}")
                adjusted_columns = ['variable 9'] if is_first_half else ['variable 16']
                half_window = WINDOW_SIZE // 2
                start_idx = 250  
                if WINDOW_SIZE%2==0:
                    sample1 = data1[adjusted_columns].iloc[start_idx - half_window:start_idx + half_window]
                    sample2 = data2[adjusted_columns].iloc[start_idx - half_window:start_idx + half_window]
                else:
                    sample1 = data1[adjusted_columns].iloc[start_idx - half_window:start_idx + half_window+1]
                    sample2 = data2[adjusted_columns].iloc[start_idx - half_window:start_idx + half_window+1]
                # generate negative_analysis_process and negative_final_answer
                if dataset_indices[0] + 1 in [7, 10]:  
                    negative_analysis_process1 = f"In the sequence, data from timestep index {int(WINDOW_SIZE/2)} to index {WINDOW_SIZE-1} remains constant, determined as a constant value fault."
                elif dataset_indices[0] + 1 in [8, 11]:  
                    negative_analysis_process1 = f"In the sequence, data from timestep index {int(WINDOW_SIZE/2)} to index {WINDOW_SIZE-1} shows a mutation compared to previous timesteps, determined as a mutation fault."
                elif dataset_indices[0] + 1 in [9]:  
                    negative_analysis_process1 = f"In the sequence, data from timestep index {int(WINDOW_SIZE/2)} to index {WINDOW_SIZE-1} shows an overall offset compared to previous timesteps, indicating a deviation between the sensor output and the true value, determined as a bias fault."
                elif dataset_indices[0] + 1 in [12]: 
                    negative_analysis_process1 = f"In the sequence, data from timestep index {int(WINDOW_SIZE/2)} to index {WINDOW_SIZE-1} has added random noise compared to previous timesteps, indicating a deviation between the sensor output and the true value, determined as a bias fault."
                # ----------------------------------------------------------------------------------
                if dataset_indices[1] + 1 in [7, 10]:  
                    negative_analysis_process2 = f"In the sequence, data from timestep index {int(WINDOW_SIZE/2)} to index {WINDOW_SIZE-1} remains constant, determined as a constant value fault."
                elif dataset_indices[1] + 1 in [8, 11]:  
                    negative_analysis_process2 = f"In the sequence, data from timestep index {int(WINDOW_SIZE/2)} to index {WINDOW_SIZE-1} shows a mutation compared to previous timesteps, determined as a mutation fault."
                elif dataset_indices[1] + 1 in [9]:  
                    negative_analysis_process2 = f"In the sequence, data from timestep index {int(WINDOW_SIZE/2)} to index {WINDOW_SIZE-1} shows an overall offset compared to previous timesteps, indicating a deviation between the sensor output and the true value, determined as a bias fault."
                elif dataset_indices[1] + 1 in [12]: 
                    negative_analysis_process2 = f"In the sequence, data from timestep index {int(WINDOW_SIZE/2)} to index {WINDOW_SIZE-1} has added random noise compared to previous timesteps, indicating a deviation between the sensor output and the true value, determined as a bias fault."
                negative_final_answer = '[' + ','.join(str(i) for i in range(int(WINDOW_SIZE/2), WINDOW_SIZE)) + ']'
                return [sample1, sample2], negative_analysis_process1, negative_analysis_process2, negative_final_answer
        else:
            return f"error：no negative_strategy '{negative_sample_number}'"
