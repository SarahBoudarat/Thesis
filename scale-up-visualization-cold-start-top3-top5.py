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
    sns.barplot(
        data=melted,
        x="Evaluation Metric",
        y="Score",
        hue="Prompting Strategy",
        ax=ax,
        errorbar="sd",  # Shows std deviation as error bars (for seaborn >= 0.12)
        capsize=0.1
    )

    for p in ax.patches:
        height = p.get_height()
        if not pd.isna(height):
            ax.annotate(f"{height:.2f}", (p.get_x() + p.get_width() / 2, height), ha='center', va='bottom', fontsize=9)
    ax.set_title(plot_title, fontsize=14)
    ax.set_ylabel("Average Score")
    # Compute maximum bar height including error bars
    bar_max = melted.groupby(['Prompting Strategy', 'Evaluation Metric'])['Score'].mean().max()
    bar_std = melted.groupby(['Prompting Strategy', 'Evaluation Metric'])['Score'].std().max()
    ymax = max(bar_max + bar_std, 1.0) * 1.15  # 15% extra padding

    ax.set_ylim(0, ymax)

    ax.legend(title="Prompting Strategy")
    plt.tight_layout()
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=300)
    plt.show()
    print(f"Plot saved to {plot_path}")

# ------------ COLD-START TOP-5 ---------------
plot_comparison(
    zs_path="experiment_logs/scaleup/cold_start_zero_shot/user_level_metrics_4.csv",
    fs_path="experiment_logs/scaleup/cold_start_few_shot/user_level_metrics_4.csv",
    cot_path="experiment_logs/scaleup/cold_start_chain_of_thought/user_level_metrics_100.csv",
    metrics=["Hit@5", "Precision@5", "Recall@5", "NDCG@5", "InDataset@5"],
    plot_title="LLM Recommendation Performance by Prompting Strategy (Cold-Start, Top-5)",
    plot_path="experiment_logs/scaleup/cold_start_plot_top5.png"
)

# ------------ COLD-START TOP-3 ---------------
plot_comparison(
    zs_path=r"C:\Users\sarah.boudarat\PycharmProjects\Thesis\.venv\experiment_logs\scaleup\cold_start_zero_shot_top3\user_level_metrics_4_top3.csv",
    fs_path=r"C:\Users\sarah.boudarat\PycharmProjects\Thesis\.venv\experiment_logs\scaleup\cold_start_few_shot_top3\user_level_metrics_4_top3.csv",
    cot_path=r"C:\Users\sarah.boudarat\PycharmProjects\Thesis\.venv\experiment_logs\scaleup\cold_start_chain_of_thought_top3\user_level_metrics_4_top3.csv",
    metrics=["Hit@3", "Precision@3", "Recall@3", "NDCG@3", "InDataset@3"],
    plot_title="LLM Recommendation Performance by Prompting Strategy (Cold-Start, Top-3)",
    plot_path="experiment_logs/scaleup/cold_start_plot_top3.png"
)
