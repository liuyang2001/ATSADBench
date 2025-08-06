import os
import json
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Define function to calculate metrics and process files
def process_files(task_dirs, file_type, is_multi=True):
    task_results = []
    total_precision = []
    total_recall = []
    total_f1 = []
    total_accuracy = []
    total_latency = []
    total_contiguity = []

    # Loop through all tasks
    for task in task_dirs:
        if is_multi:
            task_dir = os.path.join(task + "_deepseek-chat_10_origin_result")
        else:
            task_dir = os.path.join(task + "_deepseek-chat_500_origin_result")
        files = [f for f in os.listdir(task_dir) if f.endswith('.json')]
        
        window_data = []
        true_labels = []
        labels=[]
        anomaly_counts = []
        anomaly_windows = []
        
        # Handle specific case for each task
        if is_multi:
            true_labels = [0]*100 + [1]*100 + [0]*100 + [1]*100  # For 50% anomaly rate
        else:
            true_labels = [0]*6 + [1]*10 + [0]*6 + [1]*10  # For 62.5% anomaly rate

        # Process each file in the directory and collect anomaly counts
        for i, file in enumerate(files):
            with open(os.path.join(task_dir, file), encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    data = data[0]
                anomaly_count = len(data.get("final_answer", []))
                anomaly_counts.append(anomaly_count)
                
                # Check if this window is anomalous (anomaly count >= threshold)
                anomaly_windows.append(anomaly_count)

        # Set the threshold based on task type (multi or uni)
        if is_multi:
            threshold_value = sorted(anomaly_counts)[len(anomaly_counts) // 2]  # 50% threshold
        else:
            threshold_value = sorted(anomaly_counts)[int(len(anomaly_counts) * 5 / 8)]  # 62.5% threshold

        # Re-process the files, classifying based on the threshold
        for i, file in enumerate(files):
            anomaly_count = anomaly_counts[i]
            label = 1 if anomaly_count >= threshold_value else 0
            window_data.append([anomaly_count, label, true_labels[i], threshold_value])
            labels.append(label)

        # Calculate latency and contiguity for each task
        # Find the indices of the anomaly windows
        anomaly_indices = [i for i, label in enumerate(labels) if label==1]
        
        # Calculate latency
        latency_values = []
        if is_multi:  # Multi-variable task
            # For multi-variable task, find the first value in anomaly_indices that is > 99
            first_latency = next((i for i in anomaly_indices if i > 99), None)
            if first_latency is not None:
                latency_values.append(first_latency - 99)

            # For the second value, find the first value > 299
            second_latency = next((i for i in anomaly_indices if i > 299), None)
            if second_latency is not None:
                latency_values.append(second_latency - 299)

        else:  # Uni-variable task
            # For uni-variable task, find the first value in anomaly_indices that is > 5
            first_latency = next((i for i in anomaly_indices if i > 5), None)
            if first_latency is not None:
                latency_values.append(first_latency - 5)

            # For the second value, find the first value > 21
            second_latency = next((i for i in anomaly_indices if i > 21), None)
            if second_latency is not None:
                latency_values.append(second_latency - 21)

        avg_latency = sum(latency_values) / len(latency_values) if latency_values else 0

        # Calculate contiguity
        # contiguity_values = []
        # Function to find longest consecutive segment in a range
        def longest_consecutive_segment(indices):
            max_count = 0
            current_count = 1
            for i in range(1, len(indices)):
                if indices[i] == indices[i-1] + 1:
                    current_count += 1
                else:
                    max_count = max(max_count, current_count)
                    current_count = 1
            return max(max_count, current_count)
        if is_multi:  # Multi-variable task
            # For multi-variable tasks, find the longest consecutive segment in anomaly_indices between 100-199
            first_segment = [i for i in anomaly_indices if 100 <= i <= 199]
            second_segment = [i for i in anomaly_indices if 300 <= i <= 399]

            

            # Find the longest consecutive segment in both the first and second segment
            first_contiguity = longest_consecutive_segment(first_segment)
            second_contiguity = longest_consecutive_segment(second_segment)

            # Take the average and divide by 100
            contiguity = (first_contiguity + second_contiguity) / 2 / 100

        else:  # Uni-variable task
            # For uni-variable tasks, find the longest consecutive segment in anomaly_indices between 6-15
            first_segment = [i for i in anomaly_indices if 6 <= i <= 15]
            second_segment = [i for i in anomaly_indices if 22 <= i <= 31]

            # Find the longest consecutive segment in both the first and second segment
            first_contiguity = longest_consecutive_segment(first_segment)
            second_contiguity = longest_consecutive_segment(second_segment)

            # Take the average and divide by 10
            contiguity = (first_contiguity + second_contiguity) / 2 / 10
        # Save data to DataFrame and Excel
        df = pd.DataFrame(window_data, columns=["Anomaly Count", "Predicted Label", "True Label", "Threshold"])
        df.to_excel(f"{task}_results.xlsx", index=False)

        # Calculate precision, recall, f1, accuracy
        y_true = df['True Label']
        y_pred = df['Predicted Label']
        
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        
        total_precision.append(precision)
        total_recall.append(recall)
        total_f1.append(f1)
        total_accuracy.append(accuracy)
        total_latency.append(avg_latency)
        total_contiguity.append(contiguity)
    # Output all metrics to a text file and Excel
    metrics = {
        "Task": task_dirs,
        "Precision": total_precision,
        "Recall": total_recall,
        "F1 Score": total_f1,
        "Accuracy": total_accuracy,
        "Average Latency": total_latency,
        "Average Contiguity": total_contiguity,
    }

    # Convert metrics to DataFrame for Excel output
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_excel(f"task_metrics_{is_multi}.xlsx", index=False)
    # Output all metrics to a text file
    with open("task_metrics.txt", "a") as f:
        for i, task in enumerate(task_dirs):
            f.write(f"{task} - Precision: {total_precision[i]}, Recall: {total_recall[i]}, F1: {total_f1[i]}, Accuracy: {total_accuracy[i]}, Average Latency: {total_latency[i]}, Average Contiguity: {total_contiguity[i]}\n")

# List of tasks
multi_tasks = ["M-IL-FVA", "M-IL-CDA", "M-IL-TVDA", "M-OL-FVA", "M-OL-CDA", "M-OL-TVDA"]
uni_tasks = ["U-FVA", "U-CDA", "U-TVDA"]

# Process multi tasks (50% threshold)
process_files(multi_tasks, file_type='json', is_multi=True)

# Process uni tasks (62.5% threshold)
process_files(uni_tasks, file_type='json', is_multi=False)
