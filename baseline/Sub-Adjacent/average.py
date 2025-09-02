import sys
import numpy as np

if len(sys.argv) != 2:
    print("Usage: python average.py <output_file>")
    sys.exit(1)
file_path = sys.argv[1]
# file_path=r'sub_adjacent_transformer\aaa_my_outputs\U-TVDA_output1.txt'
metrics = [
    "epoch",  
    "Precision", "Recall", "F1-Score", "Precision_pa", "Recall_pa",
    "F1-Score_pa", "ROC_score", "PRC_score"
]


metrics_values = {metric: [] for metric in metrics}

with open(file_path, "r", encoding="utf-8") as file:
    lines = file.readlines()

current_block = {}
for line in lines:
    line = line.strip()
    if not line:
        continue
    if line.startswith("----------------------------------------------"):
        if all(metric in current_block for metric in metrics):
            for metric in metrics:
                metrics_values[metric].append(float(current_block[metric]))
        current_block = {}
        continue
    if ":" in line:
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        current_block[key] = value

if all(metric in current_block for metric in metrics):
    for metric in metrics:
        metrics_values[metric].append(float(current_block[metric]))

average_metrics = {}
for metric, values in metrics_values.items():
    if values:
        mean = np.mean(values)
        std = np.std(values)
        average_metrics[metric] = (mean, std)
    else:
        print(f"Warning: No values found for {metric}")

with open(file_path, "a", encoding="utf-8") as file:
    file.write("\n----------------------------------------------\n")
    file.write("Average Metrics (Mean ± Std):\n")
    for metric, (mean, std) in average_metrics.items():
        file.write(f"{metric}: {mean:.6f} ± {std:.6f}\n")
    file.write("----------------------------------------------\n")

print("Average Metrics (Mean ± Std):")
for metric, (mean, std) in average_metrics.items():
    print(f"{metric}: {mean:.6f} ± {std:.6f}")