import pandas as pd
from scipy.stats import ttest_rel

# === File paths ===
path_zero = "C:/Users/sarah.boudarat/PycharmProjects/Thesis/.venv/experiment_logs/movielens1m/zero-shot-top3/user_level_metrics_4_top3.csv"
path_few = "C:/Users/sarah.boudarat/PycharmProjects/Thesis/.venv/experiment_logs/movielens1m/few-shot-top3/user_level_metrics_4_top3.csv"
path_cot = "C:/Users/sarah.boudarat/PycharmProjects/Thesis/.venv/experiment_logs/movielens1m/chain_of_thought-top3/user_level_metrics_4_top3.csv"

# === Load data ===
df_zero = pd.read_csv(path_zero)
df_few = pd.read_csv(path_few)
df_cot = pd.read_csv(path_cot)

# === Rename columns to track versions ===
df_zero = df_zero.rename(columns={col: f"{col}_zero" for col in df_zero.columns if col != "user_id"})
df_few = df_few.rename(columns={col: f"{col}_few" for col in df_few.columns if col != "user_id"})
df_cot = df_cot.rename(columns={col: f"{col}_cot" for col in df_cot.columns if col != "user_id"})

# === Merge all three on user_id ===
df = df_zero.merge(df_few, on="user_id").merge(df_cot, on="user_id")

# === Metrics to test ===
metrics = ["Hit@3", "Precision@3", "Recall@3", "NDCG@3", "InDataset@3"]

# === Paired t-tests ===
print(" Paired t-test results (Top-3, MovieLens 1M):\n")
for metric in metrics:
    z = df[f"{metric}_zero"]
    f = df[f"{metric}_few"]
    c = df[f"{metric}_cot"]

    t_zf, p_zf = ttest_rel(z, f)
    t_zc, p_zc = ttest_rel(z, c)
    t_fc, p_fc = ttest_rel(f, c)

    print(f"{metric}")
    print(f"   Zero vs Few  →  t = {t_zf:.4f}, p = {p_zf:.4f}")
    print(f"   Zero vs CoT  →  t = {t_zc:.4f}, p = {p_zc:.4f}")
    print(f"   Few  vs CoT  →  t = {t_fc:.4f}, p = {p_fc:.4f}")
    print("-" * 50)

results = []

for metric in metrics:
    z = df[f"{metric}_zero"]
    f = df[f"{metric}_few"]
    c = df[f"{metric}_cot"]

    t_zf, p_zf = ttest_rel(z, f)
    t_zc, p_zc = ttest_rel(z, c)
    t_fc, p_fc = ttest_rel(f, c)

    results.extend([
        {"Metric": metric, "Comparison": "Zero vs Few", "t-statistic": t_zf, "p-value": p_zf},
        {"Metric": metric, "Comparison": "Zero vs CoT", "t-statistic": t_zc, "p-value": p_zc},
        {"Metric": metric, "Comparison": "Few vs CoT",  "t-statistic": t_fc, "p-value": p_fc},
    ])
    print(f"{metric}")
    print(f"   Zero vs Few  →  t = {t_zf:.4f}, p = {p_zf:.4f}")
    print(f"   Zero vs CoT  →  t = {t_zc:.4f}, p = {p_zc:.4f}")
    print(f"   Few  vs CoT  →  t = {t_fc:.4f}, p = {p_fc:.4f}")
    print("-" * 50)

# Save results to Excel
result_df = pd.DataFrame(results)
output_path = "C:/Users/sarah.boudarat/PycharmProjects/Thesis/.venv/experiment_logs/movielens1m/cold_t_test_results_ML1M_Top3.xlsx"
result_df.to_excel(output_path, index=False)
print(f"\nAll paired t-test results saved to:\n{output_path}")
