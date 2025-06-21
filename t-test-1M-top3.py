import pandas as pd
from scipy.stats import ttest_rel

# -----------------------------
# Load data for 1M Top-3
# -----------------------------
zs_df = pd.read_csv(r"C:\Users\sarah.boudarat\PycharmProjects\Thesis\.venv\experiment_logs\movielens1m\zero-shot-top3\user_level_metrics_4_top3.csv")
fs_df = pd.read_csv(r"C:\Users\sarah.boudarat\PycharmProjects\Thesis\.venv\experiment_logs\movielens1m\few-shot-top3\user_level_metrics_4_top3.csv")
cot_df = pd.read_csv(r"C:\Users\sarah.boudarat\PycharmProjects\Thesis\.venv\experiment_logs\movielens1m\chain_of_thought-top3\user_level_metrics_4_top3.csv")

# Add suffixes
zs_df = zs_df.add_suffix("_zs")
fs_df = fs_df.add_suffix("_fs")
cot_df = cot_df.add_suffix("_cot")

# Restore user_id
zs_df = zs_df.rename(columns={"user_id_zs": "user_id"})
fs_df = fs_df.rename(columns={"user_id_fs": "user_id"})
cot_df = cot_df.rename(columns={"user_id_cot": "user_id"})

# Merge on user_id
df = cot_df.merge(fs_df, on="user_id").merge(zs_df, on="user_id")

# -----------------------------
# Metrics to test
# -----------------------------
metrics = ["Hit@3", "Precision@3", "Recall@3", "NDCG@3"]

# -----------------------------
# Perform paired t-tests
# -----------------------------
results = []

for metric in metrics:
    t1 = ttest_rel(df[f"{metric}_cot"], df[f"{metric}_fs"])
    t2 = ttest_rel(df[f"{metric}_cot"], df[f"{metric}_zs"])
    t3 = ttest_rel(df[f"{metric}_fs"],  df[f"{metric}_zs"])

    results.extend([
        {"Setting": "1M_Top3", "Metric": metric, "Comparison": "cot vs few",  "t-statistic": t1.statistic, "p-value": t1.pvalue},
        {"Setting": "1M_Top3", "Metric": metric, "Comparison": "cot vs zero", "t-statistic": t2.statistic, "p-value": t2.pvalue},
        {"Setting": "1M_Top3", "Metric": metric, "Comparison": "few vs zero", "t-statistic": t3.statistic, "p-value": t3.pvalue},
    ])

# -----------------------------
# Save to Excel
# -----------------------------
result_df = pd.DataFrame(results)
output_path = r"C:\Users\sarah.boudarat\PycharmProjects\Thesis\.venv\experiment_logs\movielens1m\t_test_results_1M_Top3.xlsx"
result_df.to_excel(output_path, index=False)

print(f"All paired t-test results saved to:\n{output_path}")
