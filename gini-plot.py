import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load Excel with Gini and ItemCV
diversity_summary_path = r"C:\Users\sarah.boudarat\PycharmProjects\Thesis\diversity_summary.xlsx"
diversity_tables = pd.read_excel(diversity_summary_path, sheet_name=None)  # dict of dfs per sheet

# Prepare a combined DataFrame for plotting
# Prepare a combined DataFrame for plotting
records = []
for scenario, df in diversity_tables.items():
    for prompt in df.index:
        records.append({
            "Scenario": scenario,
            "Prompting Strategy": prompt,  # this might be numeric or codes
            "Gini": df.loc[prompt, "Gini"],
            "ItemCV": df.loc[prompt, "ItemCV"]
        })
df_all = pd.DataFrame(records)

# Map numeric codes or bad labels to descriptive strings
label_map = {0: "Zero-Shot", 1: "Few-Shot", 2: "Chain-of-Thought"}
df_all["Prompting Strategy"] = df_all["Prompting Strategy"].map(label_map)

# Now continue with plotting...


# Plot line plot function
def plot_line(df, metric, title, save_path):
    plt.figure(figsize=(12,6))
    for prompt in df["Prompting Strategy"].unique():
        subset = df[df["Prompting Strategy"] == prompt]
        plt.plot(subset["Scenario"], subset[metric], marker="o", label=prompt)
    plt.title(title)
    plt.ylabel(metric)
    plt.xlabel("Scenario")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Prompting Strategy")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.show()

# Plot heatmap function
def plot_heatmap(df, metric, title, save_path):
    pivot = df.pivot(index="Prompting Strategy", columns="Scenario", values=metric)
    plt.figure(figsize=(10, 4))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="coolwarm", cbar_kws={'label': metric})
    plt.title(title)
    plt.ylabel("Prompting Strategy")
    plt.xlabel("Scenario")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.show()

# Use the functions:

plot_line(
    df_all,
    "Gini",
    "Gini Coefficient Across Prompting Strategies and Scenarios",
    r"C:\Users\sarah.boudarat\PycharmProjects\Thesis\plots\gini_lineplot.png"
)

plot_line(
    df_all,
    "ItemCV",
    "Item Coverage Across Prompting Strategies and Scenarios",
    r"C:\Users\sarah.boudarat\PycharmProjects\Thesis\plots\itemcv_lineplot.png"
)

plot_heatmap(
    df_all,
    "Gini",
    "Heatmap of Gini Coefficient by Prompting Strategy and Scenario",
    r"C:\Users\sarah.boudarat\PycharmProjects\Thesis\plots\gini_heatmap.png"
)

plot_heatmap(
    df_all,
    "ItemCV",
    "Heatmap of Item Coverage by Prompting Strategy and Scenario",
    r"C:\Users\sarah.boudarat\PycharmProjects\Thesis\plots\itemcv_heatmap.png"
)
