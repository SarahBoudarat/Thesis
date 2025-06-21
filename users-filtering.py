import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime

# -----------------------------
# SETUP
# -----------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_lines = []

def log(message):
    log_entry = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(log_entry)
    log_lines.append(log_entry)

# -----------------------------
# LOAD DATA
# -----------------------------
log("Loading datasets...")
ratings = pd.read_csv("ratings.csv", encoding="utf-8")
movies = pd.read_csv("movies.csv", encoding="latin1")
ratings_merged = pd.merge(ratings, movies, on="movieId")

# -----------------------------
# FILTER ACTIVE USERS
# -----------------------------
log("Filtering users with at least 50 ratings...")
user_activity = ratings_merged.groupby("userId").size()
active_users = user_activity[user_activity >= 50].index
filtered_ratings = ratings_merged[ratings_merged["userId"].isin(active_users)]

# -----------------------------
# SAMPLE USERS
# -----------------------------
log("Sampling 100 active users...")
sampled_users_100 = np.random.choice(active_users, size=100, replace=False)

log("Sampling 10 users from the 100 for the pilot experiment...")
sampled_users_10 = np.random.choice(sampled_users_100, size=10, replace=False)

# -----------------------------
# CREATE SPLITS FOR BOTH SETS
# -----------------------------
def split_data(user_list, source_ratings):
    subset = source_ratings[source_ratings["userId"].isin(user_list)]
    subset_sorted = subset.sort_values(by=["userId", "timestamp"])
    train_rows, test_rows = [], []
    for user_id, group in subset_sorted.groupby("userId"):
        group_sorted = group.sort_values("timestamp")
        train_size = int(len(group_sorted) * 0.8)
        train_rows.append(group_sorted.iloc[:train_size])
        test_rows.append(group_sorted.iloc[train_size:])
    return pd.concat(train_rows).reset_index(drop=True), pd.concat(test_rows).reset_index(drop=True)

log("Splitting 100 users into train/test...")
train_data_100, test_data_100 = split_data(sampled_users_100, filtered_ratings)

log("Splitting 10 users into train/test...")
train_data_10, test_data_10 = split_data(sampled_users_10, filtered_ratings)

# -----------------------------
# SAVE FILES
# -----------------------------
output_dir = f"experiment_logs/{timestamp}"
os.makedirs(output_dir, exist_ok=True)

log("Saving data for 100 users...")
with open(f"{output_dir}/data_for_100_users.pkl", "wb") as f:
    pickle.dump({
        "sampled_users_100": sampled_users_100,
        "train_data_100": train_data_100,
        "test_data_100": test_data_100,
        "movies": movies,
        "random_seed": RANDOM_SEED
    }, f)

log("Saving data for 10 users...")
with open(f"{output_dir}/data_for_10_users.pkl", "wb") as f:
    pickle.dump({
        "sampled_users_10": sampled_users_10,
        "train_data_10": train_data_10,
        "test_data_10": test_data_10,
        "movies": movies,
        "random_seed": RANDOM_SEED
    }, f)

# CSVs
log("Exporting CSV files for visibility...")
pd.DataFrame({"userId": sampled_users_100}).to_csv(f"{output_dir}/sampled_user_ids_100.csv", index=False)
pd.DataFrame({"userId": sampled_users_10}).to_csv(f"{output_dir}/sampled_user_ids_10.csv", index=False)
train_data_100.to_csv(f"{output_dir}/train_data_100.csv", index=False)
test_data_100.to_csv(f"{output_dir}/test_data_100.csv", index=False)
train_data_10.to_csv(f"{output_dir}/train_data_10.csv", index=False)
test_data_10.to_csv(f"{output_dir}/test_data_10.csv", index=False)

# Log file
with open(f"{output_dir}/README_experiment_log.txt", "w") as f:
    f.write("\n".join(log_lines))

log("All files saved.")
