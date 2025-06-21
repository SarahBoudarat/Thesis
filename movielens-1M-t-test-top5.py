import pandas as pd
from scipy.stats import ttest_rel

# Paths to your user-level metrics
path_zero = "experiment_logs/movielens1m/zero-shot/user_level_metrics_4.csv"
path_few = "experiment_logs/movielens1m/few-shot/user_level_metrics_4.csv"
path_cot = "experiment_logs/movielens1m/chain_of_thought/user_level_metrics_4.csv"

# Load all results
df_zero = pd.read_csv(path_zero)
df_few = pd.read_csv(path_few)
df_cot = pd.read_csv(path_cot)

# Rename metrics columns to track source
df_zero = df_zero.rename(columns={col: f"{col}_zero" for col in df_zero.columns if col != "user_id"})
df_few = df_few.rename(columns={col: f"{col}_few" for col in df_few.columns if col != "user_id"})
df_cot = df_cot.rename(columns={col: f"{col}_cot" for col in df_cot.columns if col != "user_id"})

# Merge all on user_id
df = df_zero.merge(df_few, on="user_id").merge(df_cot, on="user_id")

# Define metrics to test
metrics = ["Hit@5", "Precision@5", "Recall@5", "NDCG@5", "InDataset@5"]

# Run paired t-tests and print results
print("Paired t-test results (MovieLens 1M, Top-5, Warm Start):\n")
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

