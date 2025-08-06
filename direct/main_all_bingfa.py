import argparse
from src.data_loader import load_data, get_rolling_windows, preprocess_data
from src.prompter import generate_prompt, detect_anomalies
from src.model_handler import get_model_handler
from src.utils import calculate_metrics, save_results
from config import WINDOW_SIZE, STEP_SIZE, BETA, MULTI_VAR_COLUMNS, ALPHA, num_samples, stage, SINGLE_VAR_COLUMN, NEGATIVE_SAMPLE_NUMBER, RAG_NUMBER, POSITIVE_SAMPLE_NUMBER
import numpy as np
import os
import pandas as pd
import time
import json
import re
from get_positive_data import get_positive_data  
from get_negative_sample import get_negative_sample
from generate_RAG_rolling_windows import get_RAG_rolling_windows
from get_min_max_values import minmax_normalize, reverse_scale
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_window(window_idx, window, labels, task_name, model_name, model_handler, test_data_length, is_multi_var, windows, origin_windows, boundary, RAG_NUMBER, rag_windows, POSITIVE_SAMPLE_NUMBER, NEGATIVE_SAMPLE_NUMBER):
    """
    processing a single window
    """
    positive_analysis_process=None
    positive_final_answer=None
    negative_analysis_process=None
    negative_final_answer=None
    start_idx = window_idx * STEP_SIZE if window_idx < boundary else 2000 + (window_idx - boundary) * STEP_SIZE
    end_idx = min(start_idx + WINDOW_SIZE, test_data_length) - 1
    print(f"Processing window {window_idx}: timestep {start_idx} to {end_idx}")

    # get positive samples
    column_names = MULTI_VAR_COLUMNS if is_multi_var else SINGLE_VAR_COLUMN
    is_first_half = window_idx < boundary
    positive_samples, cols = get_positive_data(task_name, column_names, WINDOW_SIZE, is_first_half)
    scaled_positive_samples = None
    scaled_negative_samples = None
    preprocessed_rag_window = None

    if POSITIVE_SAMPLE_NUMBER == 1:
        scaled_positive_samples = preprocess_data(positive_samples, cols)
        scaled_positive_samples = scaled_positive_samples[cols].values
        positive_analysis_process = (
            "The trends of all 27 variable sequences are normal, and the relationships between variables are not violated, with no anomalies detected."
            if is_multi_var else
            "The sequence trend is normal, with no anomalies detected."
        )
        positive_final_answer = "None"

    if NEGATIVE_SAMPLE_NUMBER==3:
        negative_samples0,negative_analysis_process0, negative_final_answer0 = get_negative_sample(task_name, column_names, WINDOW_SIZE, is_first_half, negative_sample_number=1)  
        if is_multi_var:
            scaled_negative_samples0 = preprocess_data(negative_samples0, cols)
            scaled_negative_samples0 = scaled_negative_samples0[cols].values
        else:
            scaled_negative_samples0 = preprocess_data(negative_samples0, [negative_samples0.columns[0]])
            scaled_negative_samples0 = scaled_negative_samples0[[negative_samples0.columns[0]]].values
        negative_samples,negative_analysis_process1, negative_analysis_process2, negative_final_answer = get_negative_sample(task_name, column_names, WINDOW_SIZE, is_first_half, negative_sample_number=2)  
        negative_analysis_process=[negative_analysis_process1,negative_analysis_process2]
        scaled_negative_samples = [preprocess_data(item, cols) for item in negative_samples]
        scaled_negative_samples = [item[cols].values for item in scaled_negative_samples]
        scaled_negative_samples = [scaled_negative_samples0] + scaled_negative_samples
        negative_analysis_process = [negative_analysis_process0] + negative_analysis_process
    if RAG_NUMBER == 1:
        origin_test_window = origin_windows[window_idx].copy()
        origin_test_window.columns = cols
        scaled_test_window = minmax_normalize(origin_test_window, cols).values
        best_rag_window = None
        min_distance = float('inf')
        rag_windows_subset = rag_windows[0] if not is_multi_var and is_first_half else rag_windows[1] if not is_multi_var else rag_windows
        for rag_window in rag_windows_subset:
            distance = np.sqrt(np.sum((scaled_test_window - rag_window) ** 2))
            if distance < min_distance:
                min_distance = distance
                best_rag_window = rag_window
        best_rag_df = pd.DataFrame(best_rag_window, columns=cols)
        reversed_rag_window = reverse_scale(best_rag_df, cols)
        preprocessed_rag_window = preprocess_data(reversed_rag_window, cols)
        preprocessed_rag_window = preprocessed_rag_window[cols].values

    prompt = generate_prompt(
        window.values, model_name, is_multi_var, scaled_positive_samples, scaled_negative_samples, preprocessed_rag_window,
        positive_analysis_process=positive_analysis_process, positive_final_answer=positive_final_answer,
        negative_analysis_process=negative_analysis_process, negative_final_answer=negative_final_answer
    )
    detect_anomalies(window_idx, prompt, model_handler, window, task_name, model_name, start_idx, end_idx)

def main(task_name, model_name):
    print(f"task:{task_name}")
    print(f"model:{model_name}")
    print(f"WINDOW_SIZE:{WINDOW_SIZE}")
    print(f"STEP_SIZE:{STEP_SIZE}")
    print(f"stage:{stage}")
    print(f"POSITIVE_SAMPLE_NUMBER:{POSITIVE_SAMPLE_NUMBER}")
    print(f"NEGATIVE_SAMPLE_NUMBER:{NEGATIVE_SAMPLE_NUMBER}")
    print(f"RAG_NUMBER:{RAG_NUMBER}")
    is_multi_var = task_name.startswith("M-")
    task_idx = ["M-IL-FVA" "M-IL-CDA" "M-IL-TVDA" "M-OL-FVA" "M-OL-CDA" "M-OL-TVDA" "U-FVA" "U-CDA" "U-TVDA"].index(task_name)
    origin_df, df = load_data(task_name, is_multi_var)
    test_data_length = len(df)
    # Generate rolling windows
    windows, true_labels = get_rolling_windows(df, is_multi_var, step=STEP_SIZE)
    origin_windows, _ = get_rolling_windows(origin_df, is_multi_var, step=STEP_SIZE)
    if stage == "generate":
        # Initialize the model
        model_handler = get_model_handler(model_name)

        # Record the start time
        start_time = time.time()
        print(f"start_time: {start_time}")
        rag_windows=None
        if RAG_NUMBER == 1:
            rag_windows = get_RAG_rolling_windows(WINDOW_SIZE, task_name)
        boundary = int((2000 - WINDOW_SIZE) / STEP_SIZE + 1)
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    process_window, window_idx, window, labels, task_name, model_name, model_handler,
                    test_data_length, is_multi_var, windows, origin_windows, boundary, RAG_NUMBER, rag_windows,
                    POSITIVE_SAMPLE_NUMBER, NEGATIVE_SAMPLE_NUMBER
                )
                for window_idx, (window, labels) in enumerate(zip(windows, true_labels))
            ]
            for future in as_completed(futures):
                try:
                    future.result() 
                except Exception as e:
                    print(f"Window processing failed with error: {e}")

        # Record the end time
        end_time = time.time()
        print(f"end_time: {end_time}")
        total_time = end_time - start_time
        print(f"total_time: {total_time}")
    elif stage == "process":
        all_query_results = []
        output_dir = f"{task_name}_{model_name}_{WINDOW_SIZE}_origin_result"
        os.makedirs(output_dir, exist_ok=True)
        for window_idx in range(len(windows)):
            json_path = f"{output_dir}/{task_name}_{WINDOW_SIZE}_query_results_{window_idx}.json"
            if os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    query_results = json.load(f)
                    all_query_results.extend(query_results)
            else:
                print(f"Warning: JSON file for window {window_idx} not found, skipping")
        window_labels_from_json = [[] for _ in range(len(windows))]
        for result in all_query_results:
            window_idx = result["window_idx"]
            anomalies = result["final_answer"]
            if anomalies:  
                anomalies = list(dict.fromkeys([float(x) for x in anomalies if 0 <= float(x) < WINDOW_SIZE]))
                window_labels_from_json[window_idx].extend(anomalies)
        for window_idx in range(len(windows)):
            anomalies_list = window_labels_from_json[window_idx]
            if anomalies_list:
                anomaly_counts = np.bincount([int(x) for x in anomalies_list], minlength=WINDOW_SIZE)
                threshold_alpha = num_samples * ALPHA
                window_labels = np.zeros(WINDOW_SIZE, dtype=int)
                for i in range(WINDOW_SIZE):
                    if anomaly_counts[i] >= threshold_alpha:
                        window_labels[i] = 1
                window_labels_from_json[window_idx] = window_labels.tolist()
            else:
                window_labels_from_json[window_idx] = [0] * WINDOW_SIZE
        output_dir = f"{task_name}_results"
        os.makedirs(output_dir, exist_ok=True)
        file_path = f"{output_dir}/{task_name}_{model_name}_{WINDOW_SIZE}_window_labels.txt"
        with open(file_path, 'w') as f:
            for idx, labels in enumerate(window_labels_from_json):
                f.write(f"Window {idx}: {labels}\n")

        all_true_anomalies = df["Label"].values  
        all_predicted_anomalies = [[] for _ in range(test_data_length)]  

        for window_idx, labels in enumerate(window_labels_from_json):
            boundary = int((2000 - WINDOW_SIZE) / STEP_SIZE + 1)
            if window_idx < boundary:
                start_idx = window_idx * STEP_SIZE
            else:
                start_idx = 2000 + (window_idx - boundary) * STEP_SIZE
            end_idx = min(start_idx + WINDOW_SIZE, test_data_length)
            for i, label in enumerate(labels[:end_idx - start_idx]):
                all_predicted_anomalies[start_idx + i].append(label)

        final_anomalies = np.zeros(test_data_length, dtype=int)
        for i in range(test_data_length):
            if all_predicted_anomalies[i]: 
                anomaly_count = sum(1 for label in all_predicted_anomalies[i] if label == 1)
                total_windows = len(all_predicted_anomalies[i]) 
                threshold_beta = total_windows * BETA
                if anomaly_count >= threshold_beta:
                    final_anomalies[i] = 1

        output_dir = f"{task_name}_results"
        data = {
            "True_Anomalies": all_true_anomalies,
            "Final_Anomalies": final_anomalies,
            "Predicted_Anomalies": [str(lst) for lst in all_predicted_anomalies] 
        }
        df_results = pd.DataFrame(data)
        df_results.to_excel(f"{output_dir}/{task_name}_{model_name}_{WINDOW_SIZE}_results.xlsx", index=False)
        log_file_path = f"{task_name}_logs/{task_name}_{model_name}_{WINDOW_SIZE}.log"
        total_time = 0
        if total_time is None:
            raise ValueError("No total_time found in log file")
        print(f"Extracted total_time from log: {total_time}")
        save_results(task_name, model_name, all_true_anomalies.tolist(), final_anomalies.tolist(), total_time)
    else:
        raise ValueError(f"Invalid stage: {stage}. Must be 'generate' or 'process'.")

if __name__ == "__main__":
    TASKS = ["M-IL-FVA" "M-IL-CDA" "M-IL-TVDA" "M-OL-FVA" "M-OL-CDA" "M-OL-TVDA" "U-FVA" "U-CDA" "U-TVDA"]
    parser = argparse.ArgumentParser(description="Run direct for a specific task and model")
    parser.add_argument("--task", type=str, default="M-IL-FVA", choices=TASKS, help="Task name")
    parser.add_argument("--model", type=str, default="deepseek-chat", choices=["deepseek-chat","qwen3-235b-a22b"], help="Model name")
    args = parser.parse_args()
    main(args.task, args.model)