import numpy as np
import pandas as pd
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from config import WINDOW_SIZE
def calculate_metrics(true_anomalies, predicted_anomalies):
    gt = np.array(true_anomalies)
    pred = np.array(predicted_anomalies)
    
    # calculate TP, FP, FN, TN
    true_positives = np.sum((gt == 1) & (pred == 1))
    false_positives = np.sum((gt == 0) & (pred == 1))
    false_negatives = np.sum((gt == 1) & (pred == 0))
    true_negatives = np.sum((gt == 0) & (pred == 0))
    
    # calculate precision recall F1
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives) if (true_positives + true_negatives + false_positives + false_negatives) > 0 else 0
    
    # point-adjustment
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
    if os.path.exists(excel_file_path):
        existing_df = pd.read_excel(excel_file_path)
        print(f"{output_dir}/{task_name}_{model_name}_{WINDOW_SIZE}_results.xlsx exists!")
        new_data = {
            "True_Anomalies": true_anomalies,
            "Predicted_Anomalies": predicted_anomalies,
            "Point_Adjusted_Anomalies": pred_pa
        }
        new_df = pd.DataFrame(new_data)
        combined_df = pd.concat([existing_df, new_df], axis=1) if not existing_df.empty else new_df
        combined_df.to_excel(excel_file_path, index=False)
    else:
        print(f"{output_dir}/{task_name}_{model_name}_{WINDOW_SIZE}_results.xlsx does not exist!")
        data = {
            "True_Anomalies": true_anomalies,
            "Predicted_Anomalies": predicted_anomalies,
            "Point_Adjusted_Anomalies": pred_pa
        }
        df_labels = pd.DataFrame(data)
        df_labels.to_excel(excel_file_path, index=False)