import pickle
import numpy as np
import pandas as pd

def load_eval(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["evaluation_results"]

methods = {
    "Zero-Shot": "cold_start_zero_shot_data.pkl",
    "Few-Shot": "cold_start_few_shot_data.pkl",
    "Chain-of-Thought": "cold_start_cot_data.pkl"
}

summary = []

for method, path in methods.items():
    results = load_eval(path)
    row = {"Method": method}
    for metric in ["Hit@5", "Precision@5", "Recall@5", "NDCG@5", "InDataset@5"]:
        values = [results[u][metric] for u in results]
        row[metric] = np.mean(values)
    summary.append(row)

df = pd.DataFrame(summary)
print(df.round(3))


import matplotlib.pyplot as plt
import pickle
import numpy as np

# Load data
files = {
    "Zero-Shot": "cold_start_zero_shot_data.pkl",
    "Few-Shot": "cold_start_few_shot_data.pkl",
    "Chain-of-Thought": "cold_start_cot_data.pkl"
}

metrics = ["Hit@5", "Precision@5", "Recall@5", "NDCG@5", "InDataset@5"]
results = {}

for name, file in files.items():
    with open(file, "rb") as f:
        data = pickle.load(f)
    eval_data = data["evaluation_results"]
    avg = {metric: np.mean([v[metric] for v in eval_data.values()]) for metric in metrics}
    results[name] = avg

# Create DataFrame
df = pd.DataFrame(results).T[metrics]

# Plot
df.plot(kind="bar", figsize=(10, 6))
plt.title("Cold-Start Recommendation: Prompting Strategies Comparison")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.legend(title="Metrics")
plt.tight_layout()
plt.show()

