import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


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


def plot_radar(metrics, scores_dict, plot_title, plot_path):
    N = len(metrics)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    metrics_with_pad = ["\n" + m for m in metrics]

    fig, ax = plt.subplots(figsize=(8.5, 8.5), subplot_kw=dict(polar=True))
    ax.grid(color="lightgrey", linestyle='--', linewidth=1, alpha=0.7)

    # Plot each strategy
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for (i, (method, values)) in enumerate(scores_dict.items()):
        data = list(values) + [values[0]]
        ax.plot(angles, data, label=method, linewidth=3, marker='o', markersize=9, color=colors[i])
        ax.fill(angles, data, alpha=0.13, color=colors[i])
        # Label only the maximum value (cleanest)
        max_idx = np.argmax(values)
        angle = angles[max_idx]
        value = values[max_idx]
        if value > 0.01:
            offset_angle = angle + 0.10  # slight offset from axis
            ax.text(
                offset_angle, value + 0.02, f"{value:.2f}",
                ha='center', va='center', fontsize=13, color=colors[i], weight='bold'
            )

    ax.set_thetagrids(np.degrees(angles[:-1]), metrics_with_pad, fontsize=14)
    ax.set_ylim(0, 0.3)
    ax.set_title(plot_title, y=1.13, fontsize=15, weight='bold')
    # Legend is now always visible and not cropped:
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=3, fontsize=13, frameon=False)
    plt.subplots_adjust(top=0.80, bottom=0.18)
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Radar plot saved to {plot_path}")


# ---- CALL THE FUNCTION IN A LOOP (THIS IS WHAT WAS MISSING!) ----

metrics_top5 = ["Hit@5", "Precision@5", "Recall@5", "NDCG@5"]
radar_settings = [
    ("100K_Top5", standard_paths, "Standard"),
    ("100K_Top5", cold_start_paths, "Cold-Start"),
    ("1M_Top5", standard_paths, "Standard"),
    ("1M_Top5", cold_start_paths, "Cold-Start"),
]

for key, path_dict, scenario in radar_settings:
    scores_dict = {}
    for prompt in ["Zero-Shot", "Few-Shot", "Chain-of-Thought"]:
        path = path_dict[key][prompt]
        df = pd.read_csv(path)
        scores_dict[prompt] = df[metrics_top5].mean().values
    plot_title = (
        f"Comparative Metric Profile of Prompting Strategies\n"
        f"({key.replace('_', ', ')}, {scenario} Scenario)"
    )
    plot_path = f"plots/radar_{key}_{scenario}.png"
    plot_radar(metrics_top5, scores_dict, plot_title, plot_path)