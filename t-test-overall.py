import pandas as pd
import os
from scipy.stats import ttest_rel

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

metrics = ["Hit@5", "Precision@5", "Recall@5", "NDCG@5", "Hit@3", "Precision@3", "Recall@3", "NDCG@3"]

output_path = r"C:\Users\sarah.boudarat\PycharmProjects\Thesis\t_test_results.xlsx"

# Actual comparison
def compute_tests(merged_df, m1, m2, metrics):
    results = []
    for metric in metrics:
        col1 = f"{metric}_{m1}"
        col2 = f"{metric}_{m2}"
        if col1 in merged_df.columns and col2 in merged_df.columns:
            stat, pval = ttest_rel(merged_df[col1], merged_df[col2])
            results.append({
                "Metric": metric,
                "Comparison": f"{m1} vs {m2}",
                "t-statistic": round(stat, 4),
                "p-value": round(pval, 4)
            })
    return results

all_results = {}

for setting, paths in metric_paths.items():
    try:
        dfs = {}
        for name, path in paths.items():
            df = pd.read_csv(path)
            df.columns = df.columns.str.strip()
            df = df.rename(columns={col: f"{col}_{name}" for col in df.columns if col != "user_id"})
            df = df.rename(columns={f"user_id_{name}": "user_id"})
            dfs[name] = df

        # Merge all on user_id
        df_merged = dfs["chain-of-thought"].merge(dfs["few-shot"], on="user_id").merge(dfs["zero-shot"], on="user_id")

        # Compute comparisons
        results = []
        results += compute_tests(df_merged, "chain-of-thought", "few-shot", metrics)
        results += compute_tests(df_merged, "chain-of-thought", "zero-shot", metrics)
        results += compute_tests(df_merged, "few-shot", "zero-shot", metrics)

        all_results[setting] = pd.DataFrame(results)

    except Exception as e:
        print(f"⚠️ Skipping {setting} due to error: {e}")

# Write Excel
with pd.ExcelWriter(output_path) as writer:
    for setting, df in all_results.items():
        df.to_excel(writer, sheet_name=setting, index=False)

print(f" All paired t-test results saved to:\n{output_path}")
