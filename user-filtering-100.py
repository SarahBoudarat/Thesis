import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime


# set up

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

output_dir = "experiment_logs/scaleup"
os.makedirs(output_dir, exist_ok=True)

log_lines = []
def log(message):
    log_entry = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(log_entry)
    log_lines.append(log_entry)

log("EXPERIMENT SCALE-UP: This run logs the main experiment with 100 users (≥50 ratings each), not a pilot sample.")


# load the data

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

log(f"Total users with ≥50 ratings: {len(active_users)}")

# -----------------------------
# SAMPLE USERS
# -----------------------------
log("Sampling 100 active users for main experiment (fixed seed)...")
sampled_users_100 = np.random.choice(active_users, size=100, replace=False)

# Show stats for sampled users (for reporting)
user_counts = user_activity.loc[sampled_users_100].sort_values(ascending=False)
log("Sampled users: stats on number of ratings per user")
log(f"Min ratings: {user_counts.min()}")
log(f"Max ratings: {user_counts.max()}")
log(f"Mean ratings: {user_counts.mean():.2f}")
log(f"Median ratings: {user_counts.median()}")
log("First 10 users and their rating counts:")
for uid, cnt in user_counts.head(10).items():
    log(f"User {uid}: {cnt} ratings")

# -----------------------------
# CREATE CHRONOLOGICAL SPLITS
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

log("Splitting 100 users into train/test sets (chronological 80/20 split)...")
train_data_100, test_data_100 = split_data(sampled_users_100, filtered_ratings)

# Log train/test size statistics
train_sizes = train_data_100.groupby("userId").size()
test_sizes = test_data_100.groupby("userId").size()
log(f"Train size per user: min={train_sizes.min()}, max={train_sizes.max()}, mean={train_sizes.mean():.2f}")
log(f"Test size per user: min={test_sizes.min()}, max={test_sizes.max()}, mean={test_sizes.mean():.2f}")

# -----------------------------
# SAVE FILES & LOGS
# -----------------------------
log("Saving experiment data (pickles and CSVs)...")
with open(f"{output_dir}/data_for_100_users.pkl", "wb") as f:
    pickle.dump({
        "sampled_users_100": sampled_users_100,
        "train_data_100": train_data_100,
        "test_data_100": test_data_100,
        "movies": movies,
        "random_seed": RANDOM_SEED
    }, f)

pd.DataFrame({"userId": sampled_users_100}).to_csv(f"{output_dir}/sampled_user_ids_100.csv", index=False)
train_data_100.to_csv(f"{output_dir}/train_data_100.csv", index=False)
test_data_100.to_csv(f"{output_dir}/test_data_100.csv", index=False)

# Add rating stats per sampled user for documentation
user_counts.to_csv(f"{output_dir}/sampled_user_rating_counts_100.csv", header=["num_ratings"])

# Log file
with open(f"{output_dir}/README_experiment_log.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines))


log("All files and logs saved successfully.")

