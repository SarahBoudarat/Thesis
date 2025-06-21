import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- File Paths ---

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


def plot_userlevel_boxplot(zs_path, fs_path, cot_path, metrics, plot_title, plot_path):
    # Load and tag
    zs = pd.read_csv(zs_path)
    zs["Prompting Strategy"] = "Zero-Shot"
    fs = pd.read_csv(fs_path)
    fs["Prompting Strategy"] = "Few-Shot"
    cot = pd.read_csv(cot_path)
    cot["Prompting Strategy"] = "Chain-of-Thought"
    all_data = pd.concat([zs, fs, cot], ignore_index=True)
    # Melt to long format for seaborn
    melted = pd.melt(
        all_data,
        id_vars="Prompting Strategy",
        value_vars=metrics,
        var_name="Evaluation Metric",
        value_name="Score"
    )
    plt.figure(figsize=(12, 6))
    sns.boxplot(
        data=melted,
        x="Evaluation Metric",
        y="Score",
        hue="Prompting Strategy",
        showfliers=True,
        width=0.7,
        notch=True
    )
    plt.title(plot_title, fontsize=15)
    plt.ylabel("User Score", fontsize=13)
    plt.ylim(0, 1.05)
    plt.legend(title="Prompting Strategy")
    plt.tight_layout()
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=300)
    plt.show()
    print(f"Boxplot saved to {plot_path}")

# --- Settings to iterate over ---
all_settings = [
    ("100K_Top5", standard_paths, "Standard"),
    ("100K_Top5", cold_start_paths, "Cold-Start"),
    ("1M_Top5", standard_paths, "Standard"),
    ("1M_Top5", cold_start_paths, "Cold-Start"),
    ("100K_Top3", standard_paths, "Standard"),
    ("100K_Top3", cold_start_paths, "Cold-Start"),
    ("1M_Top3", standard_paths, "Standard"),
    ("1M_Top3", cold_start_paths, "Cold-Start"),
]

metrics_top5 = ["Hit@5", "Precision@5", "Recall@5", "NDCG@5"]
metrics_top3 = ["Hit@3", "Precision@3", "Recall@3", "NDCG@3"]

for key, path_dict, scenario in all_settings:
    if "Top5" in key:
        metrics = metrics_top5
    else:
        metrics = metrics_top3
    plot_title = f"User-level Distribution by Prompting Strategy ({key.replace('_', ', ')} - {scenario})"
    plot_path = f"plots/boxplot_{key}_{scenario}.png"
    plot_userlevel_boxplot(
        zs_path=path_dict[key]["Zero-Shot"],
        fs_path=path_dict[key]["Few-Shot"],
        cot_path=path_dict[key]["Chain-of-Thought"],
        metrics=metrics,
        plot_title=plot_title,
        plot_path=plot_path
    )
