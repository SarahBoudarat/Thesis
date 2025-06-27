import pandas as pd
from scipy.stats import ttest_rel

# -----------------------------
# File Paths 
# -----------------------------
zero_shot_path = r"C:\Users\sarah.boudarat\PycharmProjects\Thesis\.venv\experiment_logs\movielens1m\cold_start_zero_shot_top3\user_level_metrics_4_top3.csv"
few_shot_path = r"C:\Users\sarah.boudarat\PycharmProjects\Thesis\.venv\experiment_logs\movielens1m\cold_start_few_shot_top3\user_level_metrics_4_top3.csv"
cot_path = r"C:\Users\sarah.boudarat\PycharmProjects\Thesis\.venv\experiment_logs\movielens1m\cold_start_chain_of_thought_top3\user_level_metrics_4_top3.csv"

# -----------------------------
# Load and Rename Columns
# -----------------------------
zero = pd.read_csv(zero_shot_path).rename(columns={
    "Hit@3": "Hit@3_zero", "Precision@3": "Precision@3_zero",
    "Recall@3": "Recall@3_zero", "NDCG@3": "NDCG@3_zero"
})
few = pd.read_csv(few_shot_path).rename(columns={
    "Hit@3": "Hit@3_few", "Precision@3": "Precision@3_few",
    "Recall@3": "Recall@3_few", "NDCG@3": "NDCG@3_few"
})
cot = pd.read_csv(cot_path).rename(columns={
    "Hit@3": "Hit@3_cot", "Precision@3": "Precision@3_cot",
    "Recall@3": "Recall@3_cot", "NDCG@3": "NDCG@3_cot"
})

# -----------------------------
# Merge by User ID
# -----------------------------
df = zero.merge(few, on="user_id").merge(cot, on="user_id")

# -----------------------------
# Run Paired T-Tests
# -----------------------------
metrics = ["Hit@3", "Precision@3", "Recall@3", "NDCG@3"]
comparisons = [("cot", "few"), ("cot", "zero"), ("few", "zero")]

print("=== T-Tests for Cold-Start Top-3 (MovieLens 1M) ===\n")
for metric in metrics:
    for a, b in comparisons:
        t, p = ttest_rel(df[f"{metric}_{a}"], df[f"{metric}_{b}"])
        print(f"{metric}: {a} vs {b} → t = {t:.4f}, p = {p:.4f}")
    print()
