import argparse
from src.data_loader import load_data, get_rolling_windows,preprocess_data,deprocess_data
from src.prompter import generate_prompt, detect_anomalies
from src.model_handler import get_model_handler
from src.utils import save_results,evaluate_predictions
from config import WINDOW_SIZE, STEP_SIZE, BETA, MULTI_VAR_COLUMNS, ALPHA,num_samples,stage,SINGLE_VAR_COLUMN,RAG_NUMBER,POSITIVE_SAMPLE_NUMBER,HORIZON,DATA_PATH,TEST_DATA_PATTERN
import numpy as np
import os
import pandas as pd
import time
import json
import re
from get_positive_data import get_positive_data  
from generate_RAG_rolling_windows import get_RAG_rolling_windows
from get_min_max_first_to_sixth_shared_train_data import minmax_normalize,reverse_scale
import math
from concurrent.futures import ThreadPoolExecutor

def process_window(window_idx, window_dict, model_handler, task_name, model_name, is_multi_var, rag_windows,origin_windows):
    window = window_dict["window"]
    current_window_index = window_dict["current_window_index"]
    prediction_index = window_dict["prediction_index"]
    print(f"the number of window_idx is {window_idx}")
    print(f"the number of current window timestep is {current_window_index[0]}, end timestep is {current_window_index[-1]}")
    print(f"prediction starts:{prediction_index[0]},ends:{prediction_index[-1]}")
    output_dir=f"{task_name}_{model_name}_{WINDOW_SIZE}_origin_result"
    os.makedirs(output_dir,exist_ok='True')
    json_path = f"{output_dir}/{task_name}_{WINDOW_SIZE}_query_results_{window_idx}.json"
    if os.path.exists(json_path):
        print(f"{json_path} already exists , step!")
        return
    # Obtain positive samples
    column_names = MULTI_VAR_COLUMNS if is_multi_var else SINGLE_VAR_COLUMN
    boundary = math.ceil((2000 + WINDOW_SIZE - WINDOW_SIZE) / STEP_SIZE)
    is_first_half = window_idx < boundary
    positive_samples, cols = get_positive_data(task_name, column_names, WINDOW_SIZE, is_first_half)
    preprocessed_rag_window = None
    scaled_positive_samples = None
    positive_final_answer = None
    positive_analysis_process = None
    if POSITIVE_SAMPLE_NUMBER == 1:
        origin_scaled_positive_samples = preprocess_data(positive_samples, cols)
        origin_scaled_positive_samples = origin_scaled_positive_samples[cols].values
        scaled_positive_samples = origin_scaled_positive_samples[:WINDOW_SIZE]
        positive_final_answer = origin_scaled_positive_samples[-HORIZON:]
        if is_multi_var:
            positive_analysis_process = f"Through analyzing the trends and inter-variable relationships of the sequences, predict the data for the next {HORIZON} time steps."
        else:
            positive_analysis_process = f"Through analyzing the trends of the sequence, predict the data for the next {HORIZON} time steps."
    if RAG_NUMBER == 1:
        origin_test_window_dict = origin_windows[window_idx].copy()
        origin_test_window = origin_test_window_dict["window"]
        origin_test_window.columns = cols
        scaled_test_window = minmax_normalize(origin_test_window, cols).values
        best_rag_window = None
        min_distance = float('inf')
        if not is_multi_var:
            rag_windows = rag_windows[0] if is_first_half else rag_windows[1]
        best_rag_window_idx=None
        for rag_window_idx,rag_window in enumerate(rag_windows):
            if is_multi_var:
                distance = np.sqrt(np.sum((scaled_test_window - rag_window) ** 2))
            else:
                distance = np.sqrt(np.sum((scaled_test_window[:, 0] - rag_window) ** 2))
            if distance < min_distance:
                min_distance = distance
                best_rag_window = rag_window
                best_rag_window_idx=rag_window_idx
        best_rag_df = pd.DataFrame(best_rag_window, columns=cols)
        print(f"best_rag_window_idx:{best_rag_window_idx}")
        reversed_rag_window = reverse_scale(best_rag_df, cols)
        preprocessed_rag_window = preprocess_data(reversed_rag_window, cols)
        preprocessed_rag_window = preprocessed_rag_window[cols].values
    prompt = generate_prompt(window.values, model_name, is_multi_var, scaled_positive_samples, preprocessed_rag_window, positive_analysis_process=positive_analysis_process, positive_final_answer=positive_final_answer)
    detect_anomalies(window_idx, prompt, model_handler, window_dict, task_name, model_name, is_multi_var)

def main(task_name, model_name):
    is_multi_var = task_name.startswith("M-")
    task_idx = ["M-IL-FVA", "M-IL-CDA", "M-IL-TVDA", "M-OL-FVA", "M-OL-CDA", "M-OL-TVDA", "U-FVA", "U-CDA", "U-TVDA"].index(task_name)
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
        rag_windows = None
        if RAG_NUMBER == 1:
            rag_windows = get_RAG_rolling_windows(WINDOW_SIZE, task_name)
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    process_window,
                    window_idx,
                    window_dict,
                    model_handler,
                    task_name,
                    model_name,
                    is_multi_var,
                    rag_windows,
                    origin_windows
                )
                for window_idx, window_dict in enumerate(windows)
            ]
            for future in futures:
                future.result()  
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
                    query_result = json.load(f)
                    all_query_results = all_query_results + query_result
            else:
                print(f"Warning: JSON file for window {window_idx} not found, skipping")
        if is_multi_var:
            final_prediction = [[[] for _ in range(test_data_length)] for _ in range(len(MULTI_VAR_COLUMNS))]
        else:
            final_prediction = [[] for _ in range(test_data_length)]
        for query_result in all_query_results:
            prediction_index = query_result["prediction_index"]
            final_answer = query_result["final_answer"]
            window_idx=query_result["window_idx"]
            if is_multi_var:
                if len(final_answer) != len(MULTI_VAR_COLUMNS):
                    print(f"Warning: final_answer has {len(final_answer)} variables,window_idx is {window_idx}")
                    continue
                if  any(len(var_seq) < len(prediction_index) for var_seq in final_answer):
                    print(f"Warning: var_seq{var_seq} has {len(var_seq)} length < {len(prediction_index)}")
                    continue
                for var_idx, var_seq in enumerate(final_answer):
                    for idx, pred_idx in enumerate(prediction_index):
                        final_prediction[var_idx][pred_idx].append(var_seq[idx])
            else:
                if len(final_answer) >= len(prediction_index):
                    for idx, pred_idx in enumerate(prediction_index):
                        final_prediction[pred_idx].append(final_answer[idx])
                else:
                    print(f"Warning: final_answer length {len(final_answer)}  < prediction_index length {len(prediction_index)} for window {query_result['window_idx']}")
        if is_multi_var:
            final_result = []
            for var_idx in range(len(MULTI_VAR_COLUMNS)):
                var_result = []
                for i in range(test_data_length):
                    if final_prediction[var_idx][i]:
                        median = np.median(final_prediction[var_idx][i])
                        var_result.append(median)
                    else:
                        print(f"Warning: No predictions for variable {var_idx}, index {i}")
                        var_result.append(None)
                final_result.append(var_result)
        else:
            final_result = []
            for i in range(test_data_length):
                if final_prediction[i]:
                    median = np.median(final_prediction[i])
                    final_result.append(median)
                else:
                    print(f"Warning: No predictions for index {i}")
                    final_result.append(None)
        if is_multi_var:
            pred_df = pd.DataFrame({MULTI_VAR_COLUMNS[i]: final_result[i] for i in range(len(MULTI_VAR_COLUMNS))})
        else:
            pred_df = pd.DataFrame({SINGLE_VAR_COLUMN[0]: final_result})
        old_pred_df = pred_df.copy()
        new_pred_df = deprocess_data(pred_df, MULTI_VAR_COLUMNS if is_multi_var else SINGLE_VAR_COLUMN)
        sheet_names = ["Deprocessed_Predictions", "Original_Predictions"]
        output_dir = f"{task_name}_results"
        os.makedirs(output_dir, exist_ok=True)
        pred_excel_path = f"{output_dir}/{task_name}_{model_name}_{WINDOW_SIZE}_pred.xlsx"
        with pd.ExcelWriter(pred_excel_path, engine='openpyxl') as writer:
            new_pred_df.to_excel(writer, sheet_name=sheet_names[0], index=False)
            old_pred_df.to_excel(writer, sheet_name=sheet_names[1], index=False)
        test_path = f"{DATA_PATH}{task_name}{TEST_DATA_PATTERN}"
        if is_multi_var:
            gt_df = pd.read_excel(test_path, usecols=MULTI_VAR_COLUMNS + ["Segment_Boundary", "Label"])
        else:
            gt_df = pd.read_excel(test_path, usecols=SINGLE_VAR_COLUMN + ["Segment_Boundary", "Label"])
        result_df, true_labels, pred_labels_array = evaluate_predictions(gt_df, new_pred_df, is_multi_var, output_dir, f"{task_name}_{model_name}_{WINDOW_SIZE}_pred_errors.xlsx")
        total_time = 0
        save_results(task_name, model_name, true_labels, pred_labels_array, total_time)
    else:
        raise ValueError(f"Invalid stage: {stage}. Must be 'generate' or 'process'.")

if __name__ == "__main__":
    TASKS = ["M-IL-FVA", "M-IL-CDA", "M-IL-TVDA", "M-OL-FVA", "M-OL-CDA", "M-OL-TVDA", "U-FVA", "U-CDA", "U-TVDA"]
    parser = argparse.ArgumentParser(description="Run prediction-based for a specific task and model")
    parser.add_argument("--task", type=str, default="M-IL-FVA", choices=TASKS, help="Task name")
    parser.add_argument("--model", type=str, default="deepseek-chat", choices=["deepseek-chat","qwen3-235b-a22b"], help="Model name")
    args = parser.parse_args()
    main(args.task, args.model)