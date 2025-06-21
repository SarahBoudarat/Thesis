import pandas as pd
from scipy.stats import ttest_rel

# Load data
cot_df = pd.read_csv(r"C:\Users\sarah.boudarat\PycharmProjects\Thesis\.venv\experiment_logs\scaleup\cold_start_chain_of_thought\user_level_metrics_100.csv")
fs_df = pd.read_csv(r"C:\Users\sarah.boudarat\PycharmProjects\Thesis\.venv\experiment_logs\scaleup\cold_start_few_shot\user_level_metrics_4.csv")
zs_df = pd.read_csv(r"C:\Users\sarah.boudarat\PycharmProjects\Thesis\.venv\experiment_logs\scaleup\cold_start_zero_shot\user_level_metrics_4.csv")

# Add suffixes before merging
cot_df = cot_df.add_suffix("_cot")
fs_df = fs_df.add_suffix("_fs")
zs_df = zs_df.add_suffix("_zs")

# Restore user_id column name before merging
cot_df = cot_df.rename(columns={"user_id_cot": "user_id"})
fs_df = fs_df.rename(columns={"user_id_fs": "user_id"})
zs_df = zs_df.rename(columns={"user_id_zs": "user_id"})

# Merge all on user_id
df = cot_df.merge(fs_df, on="user_id").merge(zs_df, on="user_id")

# Metrics to test
metrics = ["Hit@5", "Precision@5", "Recall@5", "NDCG@5"]

# Store results
results = []

for metric in metrics:
    print(f"\nT-test for {metric}:")
    t1 = ttest_rel(df[f"{metric}_cot"], df[f"{metric}_fs"])
    t2 = ttest_rel(df[f"{metric}_cot"], df[f"{metric}_zs"])
    t3 = ttest_rel(df[f"{metric}_fs"],  df[f"{metric}_zs"])

    print(f"  CoT vs Few-Shot:       t = {t1.statistic:.4f}, p = {t1.pvalue:.4f}")
    print(f"  CoT vs Zero-Shot:      t = {t2.statistic:.4f}, p = {t2.pvalue:.4f}")
    print(f"  Few-Shot vs Zero-Shot: t = {t3.statistic:.4f}, p = {t3.pvalue:.4f}")

    results.extend([
        {"Metric": metric, "Comparison": "CoT vs Few-Shot",       "t-statistic": t1.statistic, "p-value": t1.pvalue},
        {"Metric": metric, "Comparison": "CoT vs Zero-Shot",      "t-statistic": t2.statistic, "p-value": t2.pvalue},
        {"Metric": metric, "Comparison": "Few-Shot vs Zero-Shot", "t-statistic": t3.statistic, "p-value": t3.pvalue},
    ])

# Save to Excel
result_df = pd.DataFrame(results)
output_path = r"C:\Users\sarah.boudarat\PycharmProjects\Thesis\.venv\experiment_logs\scaleup\cold_start_t_test_results_100K_Top5.xlsx"
result_df.to_excel(output_path, index=False)
print(f"\nAll paired t-test results saved to:\n{output_path}")
