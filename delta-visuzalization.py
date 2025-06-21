

import os
import pandas as pd
import matplotlib.pyplot as plt

def get_metric_columns(setup_key):
    return [f"{metric}@3" if "Top3" in setup_key else f"{metric}@5"
            for metric in ["Hit", "Precision", "Recall", "NDCG"]]

prompt_styles = ["Zero-Shot", "Few-Shot", "Chain-of-Thought"]

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

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
setup_labels = {
    "100K_Top5": "100K Top-5",
    "100K_Top3": "100K Top-3",
    "1M_Top5": "1M Top-5",
    "1M_Top3": "1M Top-3"
}
plot_order = ["100K_Top5", "100K_Top3", "1M_Top5", "1M_Top3"]

for ax, setup in zip(axes.flat, plot_order):
    metrics = get_metric_columns(setup)
    deltas = {}

    for prompt in prompt_styles:
        std_path = standard_paths[setup][prompt]
        cold_path = cold_start_paths[setup][prompt]
        if not os.path.exists(std_path) or not os.path.exists(cold_path):
            ax.set_title(f"{setup_labels[setup]} (Missing File)")
            ax.axis('off')
            continue
        df_std = pd.read_csv(std_path)
        df_cold = pd.read_csv(cold_path)
        std_avg = df_std[metrics].mean()
        cold_avg = df_cold[metrics].mean()
        deltas[prompt] = (cold_avg - std_avg).values
    for prompt in prompt_styles:
        if prompt in deltas:
            ax.plot(metrics, deltas[prompt], marker='o', label=prompt)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_title(f"{setup_labels[setup]}", fontweight="bold", fontsize=11)
    ax.set_ylabel("Δ Score (Cold-Start − Standard)", fontsize=9)
    ax.set_xlabel("Metric")
    ax.grid(True)

handles, labels = axes[0][0].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc='upper left',
    bbox_to_anchor=(0.01, 0.99),  # X, Y in figure coordinates (0,0 is bottom left)
    ncol=1,
    title="Prompting Strategy",
    fontsize=11,
    frameon=False,
    borderaxespad=0.
)

plt.suptitle("Delta Performance (Cold-Start – Standard) by Prompting Strategys",
             fontsize=15, fontweight="bold")

plt.tight_layout(rect=[0, 0, 1, 0.95])  # leaves space at top for the legend
plt.show()


