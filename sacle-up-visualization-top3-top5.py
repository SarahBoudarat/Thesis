import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_comparison(zs_path, fs_path, cot_path, metrics, plot_title, plot_path):
    # Load and tag
    zs = pd.read_csv(zs_path)
    zs["Prompting Strategy"] = "Zero-Shot"
    fs = pd.read_csv(fs_path)
    fs["Prompting Strategy"] = "Few-Shot"
    cot = pd.read_csv(cot_path)
    cot["Prompting Strategy"] = "Chain-of-Thought"

    # Combine
    all_data = pd.concat([zs, fs, cot], ignore_index=True)
    # Melt
    melted = pd.melt(
        all_data,
        id_vars="Prompting Strategy",
        value_vars=metrics,
        var_name="Evaluation Metric",
        value_name="Score"
    )
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=melted, x="Evaluation Metric", y="Score", hue="Prompting Strategy", ax=ax)
    for p in ax.patches:
        height = p.get_height()
        if not pd.isna(height):
            ax.annotate(f"{height:.2f}", (p.get_x() + p.get_width() / 2, height), ha='center', va='bottom', fontsize=9)
    ax.set_title(plot_title, fontsize=14)
    ax.set_ylabel("Average Score")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Prompting Strategy")
    plt.tight_layout()
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=300)
    plt.show()
    print(f"Plot saved to {plot_path}")

# ------------ RUN BOTH PLOTS ---------------

# Top-5 Plot
plot_comparison(
    zs_path="experiment_logs/scaleup/zero-shot/user_level_metrics_4.csv",
    fs_path="experiment_logs/scaleup/few-shot/user_level_metrics_4.csv",
    cot_path="experiment_logs/scaleup/chain_of_thought/user_level_metrics_4.csv",
    metrics=["Hit@5", "Precision@5", "Recall@5", "NDCG@5", "InDataset@5"],
    plot_title="LLM Recommendation Performance by Prompting Strategy (Top-5)",
    plot_path="experiment_logs/scaleup/plot_top5.png"
)

# Top-3 Plot
plot_comparison(
    zs_path="experiment_logs/scaleup/zero-shot-top3/user_level_metrics_4_top3.csv",
    fs_path="experiment_logs/scaleup/few-shot-top3/user_level_metrics_4_top3.csv",
    cot_path="experiment_logs/scaleup/chain_of_thought-top3/user_level_metrics_4_top3.csv",
    metrics=["Hit@3", "Precision@3", "Recall@3", "NDCG@3", "InDataset@3"],
    plot_title="LLM Recommendation Performance by Prompting Strategy (Top-3)",
    plot_path="experiment_logs/scaleup/plot_top3.png"
)
