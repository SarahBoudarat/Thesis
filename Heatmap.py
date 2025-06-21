

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

standard_paths = {
    "1M_Top5": {
        "Zero-Shot": r"C:\Users\sarah.boudarat\PycharmProjects\Thesis\.venv\experiment_logs\movielens1m\zero-shot\user_level_metrics_4.csv",
        "Few-Shot": r"C:\Users\sarah.boudarat\PycharmProjects\Thesis\.venv\experiment_logs\movielens1m\few-shot\user_level_metrics_4.csv",
        "Chain-of-Thought": r"C:\Users\sarah.boudarat\PycharmProjects\Thesis\.venv\experiment_logs\movielens1m\chain_of_thought\user_level_metrics_4.csv"
    },
    "1M_Top3": {
        "Zero-Shot": r"C:\Users\sarah.boudarat\PycharmProjects\Thesis\.venv\experiment_logs\movielens1m\zero-shot-top3\user_level_metrics_4_top3.csv",
        "Few-Shot": r"C:\Users\sarah.boudarat\PycharmProjects\Thesis\.venv\experiment_logs\movielens1m\few-shot-top3\user_level_metrics_4_top3.csv",
        "Chain-of-Thought": r"C:\Users\sarah.boudarat\PycharmProjects\Thesis\.venv\experiment_logs\movielens1m\chain_of_thought-top3\user_level_metrics_4_top3.csv"
    },
    "100K_Top5": {
        "Zero-Shot": r"C:\Users\sarah.boudarat\PycharmProjects\Thesis\.venv\experiment_logs\scaleup\zero-shot\user_level_metrics_4.csv",
        "Few-Shot": r"C:\Users\sarah.boudarat\PycharmProjects\Thesis\.venv\experiment_logs\scaleup\few-shot\user_level_metrics_100.csv",
        "Chain-of-Thought": r"C:\Users\sarah.boudarat\PycharmProjects\Thesis\.venv\experiment_logs\scaleup\chain_of_thought\user_level_metrics_100.csv"
    },
    "100K_Top3": {
        "Zero-Shot": r"C:\Users\sarah.boudarat\PycharmProjects\Thesis\.venv\experiment_logs\scaleup\zero-shot-top3\user_level_metrics_4_top3.csv",
        "Few-Shot": r"C:\Users\sarah.boudarat\PycharmProjects\Thesis\.venv\experiment_logs\scaleup\few-shot-top3\user_level_metrics_4_top3.csv",
        "Chain-of-Thought": r"C:\Users\sarah.boudarat\PycharmProjects\Thesis\.venv\experiment_logs\scaleup\chain_of_thought-top3\user_level_metrics_4_top3.csv"
    }
}

cold_start_paths = {
    "1M_Top5": {
        "Zero-Shot": r"C:\Users\sarah.boudarat\PycharmProjects\Thesis\.venv\experiment_logs\movielens1m\cold_start_zero_shot\user_level_metrics_4.csv",
        "Few-Shot": r"C:\Users\sarah.boudarat\PycharmProjects\Thesis\.venv\experiment_logs\movielens1m\cold_start_few_shot\user_level_metrics_4.csv",
        "Chain-of-Thought": r"C:\Users\sarah.boudarat\PycharmProjects\Thesis\.venv\experiment_logs\movielens1m\cold_start_chain_of_thought_top5\user_level_metrics_4_top5.csv"
    },
    "1M_Top3": {
        "Zero-Shot": r"C:\Users\sarah.boudarat\PycharmProjects\Thesis\.venv\experiment_logs\movielens1m\cold_start_zero_shot_top3\user_level_metrics_4_top3.csv",
        "Few-Shot": r"C:\Users\sarah.boudarat\PycharmProjects\Thesis\.venv\experiment_logs\movielens1m\cold_start_few_shot_top3\user_level_metrics_4_top3.csv",
        "Chain-of-Thought": r"C:\Users\sarah.boudarat\PycharmProjects\Thesis\.venv\experiment_logs\movielens1m\cold_start_chain_of_thought_top3\user_level_metrics_4_top3.csv"
    },
    "100K_Top5": {
        "Zero-Shot": r"C:\Users\sarah.boudarat\PycharmProjects\Thesis\.venv\experiment_logs\scaleup\cold_start_zero_shot\user_level_metrics_4.csv",
        "Few-Shot": r"C:\Users\sarah.boudarat\PycharmProjects\Thesis\.venv\experiment_logs\scaleup\cold_start_few_shot\user_level_metrics_4.csv",
        "Chain-of-Thought": r"C:\Users\sarah.boudarat\PycharmProjects\Thesis\.venv\experiment_logs\scaleup\cold_start_chain_of_thought\user_level_metrics_100.csv"
    },
    "100K_Top3": {
        "Zero-Shot": r"C:\Users\sarah.boudarat\PycharmProjects\Thesis\.venv\experiment_logs\scaleup\cold_start_zero_shot_top3\user_level_metrics_4_top3.csv",
        "Few-Shot": r"C:\Users\sarah.boudarat\PycharmProjects\Thesis\.venv\experiment_logs\scaleup\cold_start_few_shot_top3\user_level_metrics_4_top3.csv",
        "Chain-of-Thought": r"C:\Users\sarah.boudarat\PycharmProjects\Thesis\.venv\experiment_logs\scaleup\cold_start_chain_of_thought_top3\user_level_metrics_4_top3.csv"
    }
}

def get_metric_columns(setup_key):
    return [f"{metric}@3" if "Top3" in setup_key else f"{metric}@5"
            for metric in ["Hit", "Precision", "Recall", "NDCG"]]

setup_labels = {
    "100K_Top5": "100K Top-5",
    "100K_Top3": "100K Top-3",
    "1M_Top5": "1M Top-5",
    "1M_Top3": "1M Top-3"
}
plot_order = ["100K_Top5", "100K_Top3", "1M_Top5", "1M_Top3"]

def collect_diffs(compare_A, compare_B):
    """Collect metric diffs (B − A) for all setups and scenarios."""
    all_results = []
    for scenario in ['Standard', 'Cold-Start']:
        path_dict = standard_paths if scenario == 'Standard' else cold_start_paths
        for setup in plot_order:
            metrics = get_metric_columns(setup)
            row_label = f"{setup_labels[setup]} - {scenario}"
            try:
                df_A = pd.read_csv(path_dict[setup][compare_A])
                df_B = pd.read_csv(path_dict[setup][compare_B])
                mean_A = df_A[metrics].mean()
                mean_B = df_B[metrics].mean()
                diff = mean_B - mean_A
                all_results.append([row_label] + list(diff.values))
            except Exception as e:
                print(f"Skipping {row_label}: {e}")
    if not all_results:
        raise ValueError("No results found for this comparison.")
    return pd.DataFrame(
        all_results,
        columns=["Setup"] + get_metric_columns("Top5")
    ).set_index("Setup")

# --- Run for each comparison you care about ---

comparisons = [
    ("Zero-Shot", "Few-Shot", "Few-Shot minus Zero-Shot"),
    ("Few-Shot", "Chain-of-Thought", "Chain-of-Thought minus Few-Shot"),
    ("Zero-Shot", "Chain-of-Thought", "Chain-of-Thought minus Zero-Shot")
]

for A, B, comp_label in comparisons:
    print(f"\n==== {comp_label} ====")
    diff_df = collect_diffs(A, B)

    # --- Plot Top-5 ---
    top5_rows = [idx for idx in diff_df.index if "Top-5" in idx]
    top5_cols = [col for col in diff_df.columns if "@5" in col]
    df_5 = diff_df.loc[top5_rows, top5_cols]
    if not df_5.empty:
        plt.figure(figsize=(7, 2.6))
        sns.heatmap(
            df_5, annot=True, fmt=".3f", cmap="RdBu_r", center=0,
            linewidths=0.5, cbar_kws={}, annot_kws={"fontsize":9}
        )
        plt.title(f"{comp_label} (Top-5, All Scenarios)", fontsize=7, fontweight="bold", pad=9)
        plt.ylabel("")
        plt.xlabel("")
        plt.yticks(fontsize=10)
        plt.xticks(fontsize=11, rotation=30, ha='right')
        plt.tight_layout()
        plt.show()
    else:
        print(f"No Top-5 data for {comp_label}")

    # --- Plot Top-3 ---
    top3_rows = [idx for idx in diff_df.index if "Top-3" in idx]
    top3_cols = [col for col in diff_df.columns if "@3" in col]
    df_3 = diff_df.loc[top3_rows, top3_cols]
    if not df_3.empty:
        plt.figure(figsize=(7, 2.6))
        sns.heatmap(
            df_3, annot=True, fmt=".3f", cmap="RdBu_r", center=0,
            linewidths=0.5, cbar_kws={}, annot_kws={"fontsize":9}
        )
        plt.title(f"{comp_label} (Top-3, All Scenarios)", fontsize=7, fontweight="bold", pad=9)
        plt.ylabel("")
        plt.xlabel("")
        plt.yticks(fontsize=10)
        plt.xticks(fontsize=11, rotation=30, ha='right')
        plt.tight_layout()
        plt.show()
    else:
        print(f"No Top-3 data for {comp_label}")
