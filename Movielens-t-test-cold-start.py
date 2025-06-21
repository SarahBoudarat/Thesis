import pandas as pd
from scipy.stats import ttest_rel

# Load CSVs
zero = pd.read_csv("experiment_logs/movielens1m/cold_start_zero_shot/user_level_metrics_4.csv")
few = pd.read_csv("experiment_logs/movielens1m/cold_start_few_shot/user_level_metrics_4.csv")
cot = pd.read_csv("experiment_logs/movielens1m/cold_start_chain_of_thought_top5/user_level_metrics_4_top5.csv")

# Rename columns to track origin
zero = zero.rename(columns={col: f"zero_{col}" for col in zero.columns if col != "user_id"})
few = few.rename(columns={col: f"few_{col}" for col in few.columns if col != "user_id"})
cot = cot.rename(columns={col: f"cot_{col}" for col in cot.columns if col != "user_id"})

# Merge all by user_id
merged = zero.merge(few, on="user_id").merge(cot, on="user_id")

# Metrics to test
metrics = ["Hit@5", "Precision@5", "Recall@5", "NDCG@5"]

# Prepare results container
results = []

# Run all comparisons
comparisons = [
    ("cot", "zero", "CoT vs Zero-Shot"),
    ("cot", "few", "CoT vs Few-Shot"),
    ("few", "zero", "Few-Shot vs Zero-Shot"),
]

for group1, group2, label in comparisons:
    for metric in metrics:
        col1 = f"{group1}_{metric}"
        col2 = f"{group2}_{metric}"
        t_stat, p_val = ttest_rel(merged[col1], merged[col2])
        results.append({
            "Metric": metric,
            "Comparison": label,
            "t-statistic": round(t_stat, 4),
            "p-value": round(p_val, 4)
        })

# Convert to DataFrame
results_df = pd.DataFrame(results)

# Print in console
print("=== Paired t-test Results (Cold Start – 1M, Top-5) ===\n")
print(results_df)

# Export to Excel
results_df.to_excel("cold_start_1m_top5_ttest_results.xlsx", index=False)
print("\nResults saved to cold_start_1m_top5_ttest_results.xlsx")


