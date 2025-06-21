import pandas as pd
from scipy.stats import ttest_rel

# Update these paths to match your system
zero_shot_path = r"C:\Users\sarah.boudarat\PycharmProjects\Thesis\.venv\experiment_logs\scaleup\cold_start_zero_shot_top3\user_level_metrics_4_top3.csv"
few_shot_path = r"C:\Users\sarah.boudarat\PycharmProjects\Thesis\.venv\experiment_logs\scaleup\cold_start_few_shot_top3\user_level_metrics_4_top3.csv"
cot_path = r"C:\Users\sarah.boudarat\PycharmProjects\Thesis\.venv\experiment_logs\scaleup\cold_start_chain_of_thought_top3\user_level_metrics_4_top3.csv"

# Load CSVs
zero_shot = pd.read_csv(zero_shot_path).rename(columns={
    "Hit@3": "Hit@3_zero", "Precision@3": "Precision@3_zero",
    "Recall@3": "Recall@3_zero", "NDCG@3": "NDCG@3_zero"
})
few_shot = pd.read_csv(few_shot_path).rename(columns={
    "Hit@3": "Hit@3_few", "Precision@3": "Precision@3_few",
    "Recall@3": "Recall@3_few", "NDCG@3": "NDCG@3_few"
})
cot = pd.read_csv(cot_path).rename(columns={
    "Hit@3": "Hit@3_cot", "Precision@3": "Precision@3_cot",
    "Recall@3": "Recall@3_cot", "NDCG@3": "NDCG@3_cot"
})

# Merge on user_id
merged = zero_shot.merge(few_shot, on="user_id").merge(cot, on="user_id")

# Paired t-tests
metrics = ["Hit@3", "Precision@3", "Recall@3", "NDCG@3"]
comparisons = [("cot", "few"), ("cot", "zero"), ("few", "zero")]

print("T-Tests for Cold-Start Top-3:")
for metric in metrics:
    for a, b in comparisons:
        t, p = ttest_rel(merged[f"{metric}_{a}"], merged[f"{metric}_{b}"])
        print(f"{metric}: {a} vs {b} → t = {t:.4f}, p = {p:.4f}")

results = []

for metric in metrics:
    for a, b in comparisons:
        t, p = ttest_rel(merged[f"{metric}_{a}"], merged[f"{metric}_{b}"])
        print(f"{metric}: {a} vs {b} → t = {t:.4f}, p = {p:.4f}")
        results.append({
            "Metric": metric,
            "Comparison": f"{a} vs {b}",
            "t-statistic": t,
            "p-value": p
        })

# Save results to Excel
result_df = pd.DataFrame(results)
output_path = r"C:\Users\sarah.boudarat\PycharmProjects\Thesis\.venv\experiment_logs\scaleup\cold_start_t_test_results_100K_Top3.xlsx"
result_df.to_excel(output_path, index=False)
print(f"\nAll paired t-test results saved to:\n{output_path}")
