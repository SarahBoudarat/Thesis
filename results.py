import pandas as pd
import os

# paths
metric_paths = {
    "1M_Top5": {
        "zero-shot": r"C:\Users\sarah.boudarat\PycharmProjects\Thesis\.venv\experiment_logs\movielens1m\zero-shot\user_level_metrics_4.csv",
        "few-shot": r"C:\Users\sarah.boudarat\PycharmProjects\Thesis\.venv\experiment_logs\movielens1m\few-shot\user_level_metrics_4.csv",
        "chain-of-thought": r"C:\Users\sarah.boudarat\PycharmProjects\Thesis\.venv\experiment_logs\movielens1m\chain_of_thought\user_level_metrics_4.csv"
    },
    "1M_Top3": {
        "zero-shot": r"C:\Users\sarah.boudarat\PycharmProjects\Thesis\.venv\experiment_logs\movielens1m\zero-shot-top3\user_level_metrics_4_top3.csv",
        "few-shot": r"C:\Users\sarah.boudarat\PycharmProjects\Thesis\.venv\experiment_logs\movielens1m\few-shot-top3\user_level_metrics_4_top3.csv",
        "chain-of-thought": r"C:\Users\sarah.boudarat\PycharmProjects\Thesis\.venv\experiment_logs\movielens1m\chain_of_thought-top3\user_level_metrics_4_top3.csv"
    },
    "100K_Top5": {
        "zero-shot": r"C:\Users\sarah.boudarat\PycharmProjects\Thesis\.venv\experiment_logs\scaleup\zero-shot\user_level_metrics_4.csv",
        "few-shot": r"C:\Users\sarah.boudarat\PycharmProjects\Thesis\.venv\experiment_logs\scaleup\few-shot\user_level_metrics_100.csv",
        "chain-of-thought": r"C:\Users\sarah.boudarat\PycharmProjects\Thesis\.venv\experiment_logs\scaleup\chain_of_thought\user_level_metrics_100.csv"
    },
    "100K_Top3": {
        "zero-shot": r"C:\Users\sarah.boudarat\PycharmProjects\Thesis\.venv\experiment_logs\scaleup\zero-shot-top3\user_level_metrics_4_top3.csv",
        "few-shot": r"C:\Users\sarah.boudarat\PycharmProjects\Thesis\.venv\experiment_logs\scaleup\few-shot-top3\user_level_metrics_4_top3.csv",
        "chain-of-thought": r"C:\Users\sarah.boudarat\PycharmProjects\Thesis\.venv\experiment_logs\scaleup\chain_of_thought-top3\user_level_metrics_4_top3.csv"
    }
}

# Metrics to average
metrics = ["Hit@5", "Precision@5", "Recall@5", "NDCG@5", "Hit@3", "Precision@3", "Recall@3", "NDCG@3"]

# Output Excel file
output_file = r"C:\Users\sarah.boudarat\PycharmProjects\Thesis\summary_metrics_only.xlsx"

# Collect summary
summary_tables = {}

for setting, paths in metric_paths.items():
    rows = []
    for prompt, path in paths.items():
        row = {"Prompting": prompt}
        if os.path.exists(path):
            df = pd.read_csv(path)
            df.columns = df.columns.str.strip()
            for metric in metrics:
                row[metric] = round(df[metric].mean(), 4) if metric in df.columns else "N/A"
        else:
            for metric in metrics:
                row[metric] = "N/A"
        rows.append(row)

    #  Set DataFrame once
    summary_tables[setting] = pd.DataFrame(rows).set_index("Prompting")

# Write to Excel
with pd.ExcelWriter(output_file) as writer:
    for sheet_name, df in summary_tables.items():
        df.to_excel(writer, sheet_name=sheet_name)

print(f"\n Summary metrics saved to Excel:\n{output_file}")
