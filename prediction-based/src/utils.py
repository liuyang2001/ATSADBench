import numpy as np
import pandas as pd
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from config import WINDOW_SIZE,MULTI_VAR_COLUMNS,SINGLE_VAR_COLUMN
def evaluate_predictions(gt_df, pred_df, is_multi_var, output_dir, output_filename):

    columns_to_process = MULTI_VAR_COLUMNS if is_multi_var else [SINGLE_VAR_COLUMN[0]]  

    errors = pd.Series(np.nan, index=gt_df.index)

    if is_multi_var:
        gt_normalized = gt_df[columns_to_process].copy()
        pred_normalized = pred_df[columns_to_process].copy()

        # Min-Max normalization
        for col in columns_to_process:
            min_val = min(gt_df[col].min(),pred_df[col].min())
            max_val = max(gt_df[col].max(),pred_df[col].max())
            if max_val > min_val:  
                print(f"min:{min_val},max:{max_val}")
                gt_normalized[col] = (gt_df[col] - min_val) / (max_val - min_val)
                pred_normalized[col] = (pred_df[col] - min_val) / (max_val - min_val)
            else:
                print(f"min=max:{min_val}")


        diff = np.abs(gt_normalized - pred_normalized)
        errors = diff.sum(axis=1,min_count=1)  
    else:
        errors = np.abs(gt_df[SINGLE_VAR_COLUMN[0]] - pred_df[SINGLE_VAR_COLUMN[0]])  

    non_null_mask = ~errors.isna()
    if non_null_mask.sum().item() > 0:
        anomaly_rate = gt_df.loc[non_null_mask, "Label"].mean()
        print(f"anomaly_rate:{anomaly_rate}")
    else:
        anomaly_rate = 0.0
        print("Warning: No non-null error values found, setting anomaly_rate to 0")

    if anomaly_rate > 0 and non_null_mask.sum().item() > 0:  
        sorted_errors = errors[non_null_mask].sort_values(ascending=False)
        threshold_idx = int(np.ceil(anomaly_rate * len(sorted_errors))) - 1
        threshold = sorted_errors.iloc[threshold_idx]
    else:
        threshold = np.inf  

    pred_labels = pd.Series(np.nan, index=gt_df.index)
    pred_labels[non_null_mask] = (errors[non_null_mask] > threshold).astype(int)

    result_df = pd.DataFrame({
        "Error": errors,
        "Label": gt_df["Label"],
        "Predicted_Label": pred_labels,
        "Threshold": pd.Series(threshold, index=[0]).reindex(gt_df.index)
    })

    import os
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    result_df.to_excel(output_path, index=False)

    if non_null_mask.sum().item() > 0:
        true_labels = gt_df.loc[non_null_mask, "Label"].tolist()
        pred_labels_array = pred_labels[non_null_mask].tolist()
    else:
        true_labels = np.array([])
        pred_labels_array = np.array([])
        print("Warning: No valid non-null errors found, returning empty label arrays")

    return result_df, true_labels, pred_labels_array
def calculate_metrics(true_anomalies, predicted_anomalies):
    gt = np.array(true_anomalies)
    pred = np.array(predicted_anomalies)
    
    true_positives = np.sum((gt == 1) & (pred == 1))
    false_positives = np.sum((gt == 0) & (pred == 1))
    false_negatives = np.sum((gt == 1) & (pred == 0))
    true_negatives = np.sum((gt == 0) & (pred == 0))
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives) if (true_positives + true_negatives + false_positives + false_negatives) > 0 else 0
    
    # point adjustment
    anomaly_state = False
    pred_pa = pred.copy()  
    for i in range(len(gt)):
        if gt[i] == 1 and pred_pa[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred_pa[j] == 0:
                        pred_pa[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred_pa[j] == 0:
                        pred_pa[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred_pa[i] = 1

    pred_pa = np.array(pred_pa)
    gt = np.array(gt)
    print("pred (after PA): ", pred_pa.shape)
    print("gt (after PA):   ", gt.shape)

    accuracy_pa = accuracy_score(gt, pred_pa)
    precision_pa, recall_pa, f1_pa, _ = precision_recall_fscore_support(gt, pred_pa, average='binary')

    return precision, recall, f1, accuracy, precision_pa, recall_pa, f1_pa, accuracy_pa, pred_pa

def save_results(task_name, model_name, true_anomalies, predicted_anomalies, total_time):
    output_dir = f"{task_name}_results"
    os.makedirs(output_dir, exist_ok=True)
    precision, recall, f1, accuracy, precision_pa, recall_pa, f1_pa, accuracy_pa, pred_pa = calculate_metrics(true_anomalies, predicted_anomalies)
    
    file_path = f"{output_dir}/{task_name}_{model_name}_{WINDOW_SIZE}_metrics.txt"
    with open(file_path, 'w') as f:
        f.write(f"Task: {task_name}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Precision: {precision:.6f}\n")
        f.write(f"Recall: {recall:.6f}\n")
        f.write(f"F1_Score: {f1:.6f}\n")
        f.write(f"Accuracy: {accuracy:.6f}\n")
        f.write(f"Precision_pa: {precision_pa:.6f}\n")
        f.write(f"Recall_pa: {recall_pa:.6f}\n")
        f.write(f"F1_Score_pa: {f1_pa:.6f}\n")
        f.write(f"Accuracy_pa: {accuracy_pa:.6f}\n")
        f.write(f"Total Execution Time: {total_time:.6f} seconds\n")
    
    excel_file_path = f"{output_dir}/{task_name}_{model_name}_{WINDOW_SIZE}_results.xlsx"

    data = {
        "True_Anomalies": true_anomalies,
        "Predicted_Anomalies": predicted_anomalies,
        "Point_Adjusted_Anomalies": pred_pa
    }
    df_labels = pd.DataFrame(data)
    df_labels.to_excel(excel_file_path, index=False)