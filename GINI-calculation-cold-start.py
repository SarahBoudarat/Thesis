import pandas as pd
import os
import numpy as np
from collections import Counter
import datetime
import re

# ----------------------------
# Logging setup
# ----------------------------
log_lines = []
def log(msg):
    print(msg)
    log_lines.append(msg)

# ----------------------------
# Gini calculation
# ----------------------------
def gini(array):
    """Compute the Gini coefficient of a numpy array."""
    array = np.array(array, dtype=np.float64)
    if np.amin(array) < 0:
        array = array - np.amin(array)
    array = array + 1e-10  # avoid division by zero
    array = np.sort(array)
    n = array.shape[0]
    index = np.arange(1, n + 1)
    return (np.sum((2 * index - n - 1) * array)) / (n * np.sum(array))

def item_coverage(recommended_items, total_items):
    """Compute the item coverage (ItemCV)."""
    return len(set(recommended_items)) / total_items
# ----------------------------
# File paths
# ----------------------------
raw_reco_paths = {
    "1M_Top5": {
        "zero-shot": r"C:\Users\sarah.boudarat\PycharmProjects\Thesis\.venv\experiment_logs\movielens1m\cold_start_zero_shot\raw_recommendations_4.csv",
        "few-shot": r"C:\Users\sarah.boudarat\PycharmProjects\Thesis\.venv\experiment_logs\movielens1m\cold_start_few_shot\raw_recommendations_4.csv",
        "chain-of-thought": r"C:\Users\sarah.boudarat\PycharmProjects\Thesis\.venv\experiment_logs\movielens1m\cold_start_chain_of_thought_top5\raw_recommendations_4_top5.csv"
    },
    "1M_Top3": {
        "zero-shot": r"C:\Users\sarah.boudarat\PycharmProjects\Thesis\.venv\experiment_logs\movielens1m\cold_start_zero_shot_top3\raw_recommendations_4_top3.csv",
        "few-shot": r"C:\Users\sarah.boudarat\PycharmProjects\Thesis\.venv\experiment_logs\movielens1m\cold_start_few_shot_top3\raw_recommendations_4_top3.csv",
        "chain-of-thought": r"C:\Users\sarah.boudarat\PycharmProjects\Thesis\.venv\experiment_logs\movielens1m\cold_start_chain_of_thought_top3\raw_recommendations_4_top3.csv"
    },
    "100K_Top5": {
        "zero-shot": r"C:\Users\sarah.boudarat\PycharmProjects\Thesis\.venv\experiment_logs\scaleup\cold_start_zero_shot\raw_recommendations_4.csv",
        "few-shot": r"C:\Users\sarah.boudarat\PycharmProjects\Thesis\.venv\experiment_logs\scaleup\cold_start_few_shot\raw_recommendations_4.csv",
        "chain-of-thought": r"C:\Users\sarah.boudarat\PycharmProjects\Thesis\.venv\experiment_logs\scaleup\cold_start_chain_of_thought\raw_recommendations_100.csv"
    },
    "100K_Top3": {
        "zero-shot": r"C:\Users\sarah.boudarat\PycharmProjects\Thesis\.venv\experiment_logs\scaleup\cold_start_zero_shot_top3\raw_recommendations_4_top3.csv",
        "few-shot": r"C:\Users\sarah.boudarat\PycharmProjects\Thesis\.venv\experiment_logs\scaleup\cold_start_few_shot_top3\raw_recommendations_4_top3.csv",
        "chain-of-thought": r"C:\Users\sarah.boudarat\PycharmProjects\Thesis\.venv\experiment_logs\scaleup\cold_start_chain_of_thought_top3\raw_recommendations_4_top3.csv"
    }
}

# ----------------------------
# Output paths
# ----------------------------
total_items = 3706
output_file = r"C:\Users\sarah.boudarat\PycharmProjects\Thesis\cold-start-diversity_summary.xlsx"
log_file = r"C:\Users\sarah.boudarat\PycharmProjects\Thesis\diversity_log_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".txt"

# ----------------------------
# Calculate and log results
# ----------------------------
diversity_tables = {}

for setting, prompts in raw_reco_paths.items():
    rows = []

    for prompt_type, path in prompts.items():
        row = {"Prompting": prompt_type}

        if os.path.exists(path):
            df = pd.read_csv(path)
            reco_col = next((col for col in df.columns if "recommend" in col.lower()), None)

            if reco_col:
                all_items = []
                for r in df[reco_col]:
                    if isinstance(r, str):
                        all_items.extend(re.split(r",\s*", r))
                item_counts = Counter(all_items)
                gini_score = round(gini(list(item_counts.values())), 4)
                itemcv = round(len(set(all_items)) / total_items, 4)
                row["Gini"] = gini_score
                row["ItemCV"] = itemcv
                log(f" {setting} | {prompt_type}: Gini = {gini_score}, ItemCV = {itemcv}")
            else:
                row["Gini"] = "N/A"
                row["ItemCV"] = "N/A"
                log(f"No recommendation column found in: {path}")
        else:
            row["Gini"] = "N/A"
            row["ItemCV"] = "N/A"
            log(f"File not found: {path}")

        rows.append(row)

    df_setting = pd.DataFrame(rows).set_index("Prompting")
    diversity_tables[setting] = df_setting

# ----------------------------
# Export to Excel
# ----------------------------
with pd.ExcelWriter(output_file) as writer:
    for sheet_name, df in diversity_tables.items():
        df.to_excel(writer, sheet_name=sheet_name)
log(f"\n Diversity metrics saved to Excel:\n{output_file}")

# ----------------------------
# Save logs
# ----------------------------
with open(log_file, "w", encoding="utf-8") as f:
    for line in log_lines:
        f.write(line + "\n")
log(f" Log file saved to:\n{log_file}")

