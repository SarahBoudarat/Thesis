import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Load results from CSVs
# -----------------------------
# Zero-shot
zero_shot = pd.read_csv("experiments/cold_start_zero_shot/user_level_metrics_4.csv")
zero_shot["Prompting Strategy"] = "Zero-Shot"

# Few-shot
few_shot = pd.read_csv("experiments/cold_start_few_shot/user_level_metrics_4.csv")
few_shot["Prompting Strategy"] = "Few-Shot"

# Chain-of-Thought
cot = pd.read_csv("experiments/cold_start_chain_of_thought/user_level_metrics_4.csv")
cot["Prompting Strategy"] = "Chain-of-Thought"

# -----------------------------
# Combine All
# -----------------------------
all_data = pd.concat([zero_shot, few_shot, cot], ignore_index=True)

# Melt for plotting
melted = pd.melt(
    all_data,
    id_vars="Prompting Strategy",
    value_vars=["Hit@5", "Precision@5", "Recall@5", "NDCG@5", "InDataset@5"],
    var_name="Evaluation Metric",
    value_name="Score"
)

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(12, 6))
sns.barplot(
    data=melted,
    x="Evaluation Metric",
    y="Score",
    hue="Prompting Strategy"
)

# Annotate values
for p in plt.gca().patches:
    height = p.get_height()
    if not pd.isna(height):
        plt.gca().annotate(f"{height:.2f}", (p.get_x() + p.get_width() / 2, height),
                           ha='center', va='bottom', fontsize=9)

plt.title("LLM Recommendation Performance by Prompting Strategy (Cold Start Setting)", fontsize=14)
plt.ylabel("Average Score")
plt.ylim(0, 1.05)
plt.legend(title="Prompting Strategy")
plt.tight_layout()
plt.savefig("experiment_logs/cold_start_pilot_plot.png", dpi=300)
plt.show()
