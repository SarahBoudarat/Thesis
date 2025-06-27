import pandas as pd
import matplotlib.pyplot as plt
import os

# File paths 
standard_path = r"C:\Users\sarah.boudarat\PycharmProjects\Thesis\diversity_summary.xlsx"
coldstart_path = r"C:\Users\sarah.boudarat\PycharmProjects\Thesis\cold-start-diversity_summary.xlsx"

#  Load both Excel sheets 
standard_tables = pd.read_excel(standard_path, sheet_name=None)
coldstart_tables = pd.read_excel(coldstart_path, sheet_name=None)

# Extract function 
def extract_records(source_dict, scenario_tag):
    records = []
    for sheet, df in source_dict.items():
        for prompt in df.index:
            records.append({
                "Scenario": sheet,
                "Prompting Strategy": prompt,
                "Gini": df.loc[prompt, "Gini"],
                "ItemCV": df.loc[prompt, "ItemCV"],
                "Scenario Type": scenario_tag  # "Standard" or "Cold-Start"
            })
    return records

#  Combine all data 
records = extract_records(standard_tables, "Standard") + extract_records(coldstart_tables, "Cold-Start")
df_all = pd.DataFrame(records)

#  Map numeric prompt codes to strings 
label_map = {0: "Zero-Shot", 1: "Few-Shot", 2: "Chain-of-Thought"}
df_all["Prompting Strategy"] = df_all["Prompting Strategy"].map(label_map)
df_all = df_all.dropna(subset=["Prompting Strategy"])  # Ensure clean labels

#  Plotting function 
def plot_line(df, metric, title, save_path):
    plt.figure(figsize=(12, 6))
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

# === Filter by scenario type
df_standard = df_all[df_all["Scenario Type"] == "Standard"]
df_cold = df_all[df_all["Scenario Type"] == "Cold-Start"]

# === Standard scenario plots
plot_line(
    df_standard,
    "Gini",
    "Gini Coefficient — Standard Scenario",
    r"C:\Users\sarah.boudarat\PycharmProjects\Thesis\plots\standard_gini_lineplot.png"
)

plot_line(
    df_standard,
    "ItemCV",
    "Item Coverage — Standard Scenario",
    r"C:\Users\sarah.boudarat\PycharmProjects\Thesis\plots\standard_itemcv_lineplot.png"
)

# === Cold-start scenario plots
plot_line(
    df_cold,
    "Gini",
    "Gini Coefficient — Cold-Start Scenario",
    r"C:\Users\sarah.boudarat\PycharmProjects\Thesis\plots\coldstart_gini_lineplot.png"
)

plot_line(
    df_cold,
    "ItemCV",
    "Item Coverage — Cold-Start Scenario",
    r"C:\Users\sarah.boudarat\PycharmProjects\Thesis\plots\coldstart_itemcv_lineplot.png"
)
