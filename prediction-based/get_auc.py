import pandas as pd
import os
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

def process_task(directory, task):
    if task in ["M-IL-FVA", "M-IL-CDA", "M-IL-TVDA", "M-OL-FVA", "M-OL-CDA", "M-OL-TVDA"]:
        file_path = os.path.join(directory, f"{task}_results/{task}_deepseek-chat_20_pred_errors.xlsx")
    else:
        file_path = os.path.join(directory, f"{task}_results/{task}_deepseek-chat_1000_pred_errors.xlsx")
    df = pd.read_excel(file_path)
    df = df[df['Error'].notna()]
    total_rows = len(df)
    label_1_count = df['Label'].eq(1).sum()
    label_0_count = df['Label'].eq(0).sum()
    print(f"{task}: Total rows {total_rows}, Label 1: {label_1_count}, Label 0: {label_0_count}")
    
    roc_score = roc_auc_score(df['Label'], df['Error'])
    precision, recall, _ = precision_recall_curve(df['Label'], df['Error'])
    prc_score = auc(recall, precision)
    return task, roc_score, prc_score

def main(directory, output_file):
    tasks = ["M-IL-FVA", "M-IL-CDA", "M-IL-TVDA", "M-OL-FVA", "M-OL-CDA", "M-OL-TVDA", "U-FVA", "U-CDA", "U-TVDA"]
    results = []
    for task in tasks:
        task, roc, prc = process_task(directory, task)
        results.append({'Task': task, 'ROC_score': roc, 'PRC_score': prc})
    
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_file, index=False)
    print(df_results)

if __name__ == "__main__":
    main("DeepSeek-V3", "DeepSeek-V3/metrics_results_ROC_20_1000.csv")