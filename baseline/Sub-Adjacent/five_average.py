import sys
import os
import numpy as np

if len(sys.argv) != 7:
    print("Usage: python five_average.py <input_file1> <input_file2> <input_file3> <input_file4> <input_file5> <output_file>")
    sys.exit(1)
file_paths = sys.argv[1:6]
output_file_path = sys.argv[6]

metrics = [
    "epoch", 
    "Precision", "Recall", "F1-Score", "Precision_pa", "Recall_pa",
    "F1-Score_pa", "ROC_score", "PRC_score"
]


metrics_values = {metric: [] for metric in metrics}

all_runs_values = {metric: [] for metric in metrics}

for file_path in file_paths:
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    current_block = {}
    in_average_section = False
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("Average Metrics (Mean ± Std):"):
            in_average_section = True
            continue
        if line.startswith("----------------------------------------------"):
            if in_average_section:
                in_average_section = False
                continue
            if all(metric in current_block for metric in metrics):
                for metric in metrics:
                    all_runs_values[metric].append(float(current_block[metric]))
            current_block = {}
            continue
        if in_average_section:
            continue
        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if key in metrics:
                current_block[key] = value

    if not in_average_section and all(metric in current_block for metric in metrics):
        for metric in metrics:
            all_runs_values[metric].append(float(current_block[metric]))

    in_average_section = False
    current_metrics = {}
    for line in lines:
        line = line.strip()
        if line.startswith("Average Metrics (Mean ± Std):"):
            in_average_section = True
            continue
        if line.startswith("----------------------------------------------") and in_average_section:
            break
        if in_average_section and ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            mean_value = float(value.split("±")[0].strip())
            if key in metrics:
                current_metrics[key] = mean_value

    if all(metric in current_block for metric in metrics):
        for metric in metrics:
            metrics_values[metric].append(current_metrics[metric])
    else:
        print(f"Warning: Average Metrics section incomplete in {file_path}")

average_metrics = {}
for metric, values in metrics_values.items():
    if len(values) == len(file_paths):
        mean = np.mean(values)
        std = np.std(values)
        average_metrics[metric] = (mean, std)
    else:
        print(f"Warning: Not enough values for {metric} (found {len(values)} values, expected {len(file_paths)})")

all_runs_metrics = {}
for metric, values in all_runs_values.items():
    if len(values) == 50:  
        mean = np.mean(values)
        std = np.std(values)
        all_runs_metrics[metric] = (mean, std)
    else:
        print(f"Warning: Not enough runs for {metric} (found {len(values)} runs, expected 50)")

with open(output_file_path, "w", encoding="utf-8") as file:
    file.write("Final Average Metrics (Across 5 Configs, Mean ± Std of Averages):\n")
    for metric, (mean, std) in average_metrics.items():
        file.write(f"{metric}: {mean:.6f} ± {std:.6f}\n")
    file.write("\nFinal Metrics (Across 50 Runs, Mean ± Std):\n")
    for metric, (mean, std) in all_runs_metrics.items():
        file.write(f"{metric}: {mean:.6f} ± {std:.6f}\n")
    file.write("----------------------------------------------\n")

print("Final Average Metrics (Across 5 Configs, Mean ± Std of Averages):")
for metric, (mean, std) in average_metrics.items():
    print(f"{metric}: {mean:.6f} ± {std:.6f}")

print("\nFinal Metrics (Across 50 Runs, Mean ± Std):")
for metric, (mean, std) in all_runs_metrics.items():
    print(f"{metric}: {mean:.6f} ± {std:.6f}")