import pandas as pd
from scipy.stats import ttest_rel

# -----------------------------
# Load data
# -----------------------------
cot_df = pd.read_csv("experiment_logs/scaleup/chain_of_thought/user_level_metrics_100.csv")
fs_df = pd.read_csv("experiment_logs/scaleup/few-shot/user_level_metrics_100.csv")
zs_df = pd.read_csv("experiment_logs/scaleup/zero-shot/user_level_metrics_100.csv")

# Add suffixes for disambiguation
cot_df = cot_df.add_suffix("_cot")
fs_df = fs_df.add_suffix("_fs")
zs_df = zs_df.add_suffix("_zs")

# Restore 'user_id' before merging
cot_df = cot_df.rename(columns={"user_id_cot": "user_id"})
fs_df = fs_df.rename(columns={"user_id_fs": "user_id"})
zs_df = zs_df.rename(columns={"user_id_zs": "user_id"})

# Merge on user_id
df = cot_df.merge(fs_df, on="user_id").merge(zs_df, on="user_id")

# -----------------------------
# Metrics to test
# -----------------------------
metrics = ["Hit@5", "Precision@5", "Recall@5", "NDCG@5"]

# -----------------------------
# Perform paired t-tests
# -----------------------------
results = []

for metric in metrics:
    t1 = ttest_rel(df[f"{metric}_cot"], df[f"{metric}_fs"])
    t2 = ttest_rel(df[f"{metric}_cot"], df[f"{metric}_zs"])
    t3 = ttest_rel(df[f"{metric}_fs"],  df[f"{metric}_zs"])

    results.extend([
        {"Setting": "100K_Top5", "Metric": metric, "Comparison": "cot vs few",  "t-statistic": t1.statistic, "p-value": t1.pvalue},
        {"Setting": "100K_Top5", "Metric": metric, "Comparison": "cot vs zero", "t-statistic": t2.statistic, "p-value": t2.pvalue},
        {"Setting": "100K_Top5", "Metric": metric, "Comparison": "few vs zero", "t-statistic": t3.statistic, "p-value": t3.pvalue},
    ])

# Save results to Excel
result_df = pd.DataFrame(results)
output_path = "experiment_logs/scaleup/t_test_results_100K_Top5.xlsx"
result_df.to_excel(output_path, index=False)

print(f"All paired t-test results saved to:\n{output_path}")


