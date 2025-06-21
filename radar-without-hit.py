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
    "100K_Top5": {
        "Zero-Shot": r"C:\Users\sarah.boudarat\PycharmProjects\Thesis\.venv\experiment_logs\scaleup\zero-shot\user_level_metrics_4.csv",
        "Few-Shot": r"C:\Users\sarah.boudarat\PycharmProjects\Thesis\.venv\experiment_logs\scaleup\few-shot\user_level_metrics_100.csv",
        "Chain-of-Thought": r"C:\Users\sarah.boudarat\PycharmProjects\Thesis\.venv\experiment_logs\scaleup\chain_of_thought\user_level_metrics_100.csv"
    }
}
cold_start_paths = {
    "1M_Top5": {
        "Zero-Shot": r"C:\Users\sarah.boudarat\PycharmProjects\Thesis\.venv\experiment_logs\movielens1m\cold_start_zero_shot\user_level_metrics_4.csv",
        "Few-Shot": r"C:\Users\sarah.boudarat\PycharmProjects\Thesis\.venv\experiment_logs\movielens1m\cold_start_few_shot\user_level_metrics_4.csv",
        "Chain-of-Thought": r"C:\Users\sarah.boudarat\PycharmProjects\Thesis\.venv\experiment_logs\movielens1m\cold_start_chain_of_thought_top5\user_level_metrics_4_top5.csv"
    },
    "100K_Top5": {
        "Zero-Shot": r"C:\Users\sarah.boudarat\PycharmProjects\Thesis\.venv\experiment_logs\scaleup\cold_start_zero_shot\user_level_metrics_4.csv",
        "Few-Shot": r"C:\Users\sarah.boudarat\PycharmProjects\Thesis\.venv\experiment_logs\scaleup\cold_start_few_shot\user_level_metrics_4.csv",
        "Chain-of-Thought": r"C:\Users\sarah.boudarat\PycharmProjects\Thesis\.venv\experiment_logs\scaleup\cold_start_chain_of_thought\user_level_metrics_100.csv"
    }
}

def plot_radar(metrics, scores_dict, plot_title, plot_path):
    N = len(metrics)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    metrics_with_pad = ["\n" + m for m in metrics]

    fig, ax = plt.subplots(figsize=(8.5, 8.5), subplot_kw=dict(polar=True))
    ax.grid(color="lightgrey", linestyle='--', linewidth=1, alpha=0.7)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for (i, (method, values)) in enumerate(scores_dict.items()):
        data = list(values) + [values[0]]
        ax.plot(angles, data, label=method, linewidth=3, marker='o', markersize=9, color=colors[i])
        ax.fill(angles, data, alpha=0.13, color=colors[i])
        # Label all values at each axis (optional; comment if too busy)
        for angle, value in zip(angles, data):
            if value > 0.01:
                ax.text(angle, value + 0.013, f"{value:.2f}", ha='center', va='center', fontsize=13, color=colors[i], weight='bold')

    ax.set_thetagrids(np.degrees(angles[:-1]), metrics_with_pad, fontsize=14)
    ax.set_ylim(0, 0.3)
    ax.set_title(plot_title, y=1.13, fontsize=15, weight='bold')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=3, fontsize=13, frameon=False)
    plt.subplots_adjust(top=0.80, bottom=0.18)
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Radar plot saved to {plot_path}")

# --- Use only the 3 metrics for radar ---
metrics_radar = ["Precision@5", "Recall@5", "NDCG@5"]
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
        scores_dict[prompt] = df[metrics_radar].mean().values
    plot_title = (
        f"Prompting Strategies Metric Profile (Radar Plot)\n"
        f"{key.replace('_', ', ')} — {scenario} Scenario"
    )
    plot_path = f"plots/radar_{key}_{scenario}_noHit.png"
    plot_radar(metrics_radar, scores_dict, plot_title, plot_path)
