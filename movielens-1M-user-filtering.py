import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime

# Set up
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

output_dir = "experiment_logs/movielens1m"
os.makedirs(output_dir, exist_ok=True)

log_lines = []
def log(message):
    log_entry = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(log_entry)
    log_lines.append(log_entry)

log("EXPERIMENT (MOVIELENS 1M): Running full experiment for users with ≥100 ratings.")

# Load data using your absolute paths
log("Loading and merging ratings with movies...")
ratings = pd.read_csv("C:/Users/sarah.boudarat/PycharmProjects/Thesis/ratings.dat", sep="::", engine="python",
                      names=["userId", "movieId", "rating", "timestamp"])
movies = pd.read_csv("C:/Users/sarah.boudarat/PycharmProjects/Thesis/movies.dat",
                     sep="::", engine="python",
                     names=["movieId", "title", "genres"],
                     encoding="latin1")

df = pd.merge(ratings, movies, on="movieId")

# Filter users with ≥100 ratings
log("Filtering users with at least 100 ratings...")
user_activity = df.groupby("userId").size()
active_users = user_activity[user_activity >= 100].index
filtered_df = df[df["userId"].isin(active_users)]

log(f"Total users with ≥100 ratings: {len(active_users)}")

# Sample 100 users
log("Sampling 100 users...")
sampled_users_100 = np.random.choice(active_users, size=100, replace=False)

# Show stats
user_counts = user_activity.loc[sampled_users_100].sort_values(ascending=False)
log(f"Sampled users: stats on number of ratings per user")
log(f"Min ratings: {user_counts.min()}")
log(f"Max ratings: {user_counts.max()}")
log(f"Mean ratings: {user_counts.mean():.2f}")
log(f"Median ratings: {user_counts.median()}")
log("First 10 users and their rating counts:")
for uid, cnt in user_counts.head(10).items():
    log(f"User {uid}: {cnt} ratings")

# Split train/test chronologically (80/20)
def split_data(user_list, source_df):
    subset = source_df[source_df["userId"].isin(user_list)]
    subset_sorted = subset.sort_values(by=["userId", "timestamp"])
    train_rows, test_rows = [], []
    for user_id, group in subset_sorted.groupby("userId"):
        group_sorted = group.sort_values("timestamp")
        train_size = int(len(group_sorted) * 0.8)
        train_rows.append(group_sorted.iloc[:train_size])
        test_rows.append(group_sorted.iloc[train_size:])
    return pd.concat(train_rows).reset_index(drop=True), pd.concat(test_rows).reset_index(drop=True)

log("Splitting data for sampled users (chronological 80/20)...")
train_data_100, test_data_100 = split_data(sampled_users_100, filtered_df)

train_sizes = train_data_100.groupby("userId").size()
test_sizes = test_data_100.groupby("userId").size()
log(f"Train size per user: min={train_sizes.min()}, max={train_sizes.max()}, mean={train_sizes.mean():.2f}")
log(f"Test size per user: min={test_sizes.min()}, max={test_sizes.max()}, mean={test_sizes.mean():.2f}")

# Save data
log("Saving data and logs...")
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
user_counts.to_csv(f"{output_dir}/sampled_user_rating_counts_100.csv", header=["num_ratings"])

with open(f"{output_dir}/README_experiment_log.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines))

log("Experiment setup complete. All files saved.")
