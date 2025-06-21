import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_rel, wilcoxon, t
import warnings

warnings.filterwarnings('ignore')

# --------- 1. LOAD DATA -----------
paths = {
    "zero_shot": "experiment_logs/scaleup/zero-shot/user_level_metrics_100.csv",
    "few_shot": "experiment_logs/scaleup/few-shot/user_level_metrics_100.csv",
    "cot": "experiment_logs/scaleup/chain_of_thought/user_level_metrics_100.csv"
}
dfs = {}
for k, v in paths.items():
    df = pd.read_csv(v)
    df['prompt_type'] = k
    dfs[k] = df
results = pd.concat(dfs.values(), ignore_index=True)
metrics = ['Hit@5', 'Precision@5', 'Recall@5', 'NDCG@5', 'InDataset@5']
methods = ['zero_shot', 'few_shot', 'cot']

log_lines = []

# --------- 2. SUMMARY TABLE & LOG -----------
summary = results.groupby('prompt_type')[metrics].agg(['mean', 'std'])
summary_csv_path = "experiment_logs/metrics_summary_table.csv"
summary.to_csv(summary_csv_path)
log_lines.append("SUMMARY TABLE (mean, std):\n" + str(summary) + "\n")

# --------- 3. CONFIDENCE INTERVALS & LOG -----------
def conf_interval(series, conf=0.95):
    n = len(series)
    m = np.mean(series)
    se = np.std(series, ddof=1) / np.sqrt(n)
    h = se * t.ppf((1 + conf) / 2., n-1)
    return (m, m-h, m+h)

log_lines.append("Mean (95% CI) for each method & metric:")
ci_records = []
for m in methods:
    df = results[results['prompt_type'] == m]
    log_lines.append(f"\n{m}:")
    for met in metrics:
        mean, low, high = conf_interval(df[met])
        line = f"{met}: {mean:.3f} [{low:.3f} - {high:.3f}]"
        log_lines.append("  " + line)
        ci_records.append([m, met, mean, low, high])
pd.DataFrame(ci_records, columns=["method", "metric", "mean", "lower", "upper"]).to_csv("experiment_logs/metrics_ci_table.csv", index=False)

# --------- 4. STATISTICAL SIGNIFICANCE & LOG ----------
def cohens_d(a, b):
    return (np.mean(a) - np.mean(b)) / np.sqrt((np.std(a, ddof=1) ** 2 + np.std(b, ddof=1) ** 2) / 2)

test_results = []
log_lines.append("\nStatistical Test Results (paired t-test, Wilcoxon, Cohen's d):")
for metric in metrics:
    log_lines.append(f"\n{metric}:")
    # 1. Zero-Shot vs Few-Shot
    a = results[results['prompt_type']=='zero_shot'][metric].values
    b = results[results['prompt_type']=='few_shot'][metric].values
    tval, pval = ttest_rel(a, b)
    try:
        wval, wp = wilcoxon(a, b)
    except:
        wval, wp = (np.nan, np.nan)
    d = cohens_d(a, b)
    res = f"  Zero-Shot vs Few-Shot: t={tval:.3f}, p={pval:.4f}; Wilcoxon p={wp:.4f}; Cohen's d={d:.3f}"
    log_lines.append(res)
    test_results.append(["zero_shot", "few_shot", metric, tval, pval, wp, d])

    # 2. Few-Shot vs CoT
    a = results[results['prompt_type']=='few_shot'][metric].values
    b = results[results['prompt_type']=='cot'][metric].values
    tval, pval = ttest_rel(a, b)
    try:
        wval, wp = wilcoxon(a, b)
    except:
        wval, wp = (np.nan, np.nan)
    d = cohens_d(a, b)
    res = f"  Few-Shot vs CoT: t={tval:.3f}, p={pval:.4f}; Wilcoxon p={wp:.4f}; Cohen's d={d:.3f}"
    log_lines.append(res)
    test_results.append(["few_shot", "cot", metric, tval, pval, wp, d])

    # 3. Zero-Shot vs CoT
    a = results[results['prompt_type']=='zero_shot'][metric].values
    b = results[results['prompt_type']=='cot'][metric].values
    tval, pval = ttest_rel(a, b)
    try:
        wval, wp = wilcoxon(a, b)
    except:
        wval, wp = (np.nan, np.nan)
    d = cohens_d(a, b)
    res = f"  Zero-Shot vs CoT: t={tval:.3f}, p={pval:.4f}; Wilcoxon p={wp:.4f}; Cohen's d={d:.3f}"
    log_lines.append(res)
    test_results.append(["zero_shot", "cot", metric, tval, pval, wp, d])
pd.DataFrame(test_results, columns=["method_a", "method_b", "metric", "ttest", "ttest_p", "wilcoxon_p", "cohens_d"]).to_csv("experiment_logs/statistical_significance_table.csv", index=False)

# --------- 5. PLOTS (printed as before) ----------
means = results.groupby('prompt_type')[metrics].mean()
means.plot(kind='bar', figsize=(10, 6))
plt.title("Mean Metrics by Prompt Type")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

for metric in metrics:
    results.boxplot(column=metric, by='prompt_type', grid=False, figsize=(7,4))
    plt.title(f"Distribution of {metric} by Prompt Type")
    plt.suptitle("")
    plt.xlabel("")
    plt.show()

# --------- 6. WRITE TO LOG FILE ----------
with open("experiment_logs/stats_analysis_log.txt", "w", encoding="utf-8") as f:
    f.write('\n'.join(log_lines))

print("All plots displayed and statistical results logged to experiment_logs/stats_analysis_log.txt")



