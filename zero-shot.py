import pandas as pd
import numpy as np
import openai
from rapidfuzz import process

# Load ratings and movies datasets
ratings = pd.read_csv("ratings.csv")  # columns: userId, movieId, rating, timestamp
movies = pd.read_csv("movies.csv")    # columns: movieId, title, genres

# Quick look
print(ratings.head())
print(movies.head())

# Merge ratings with movie titles
ratings_merged = pd.merge(ratings, movies, on="movieId")
print(ratings_merged.head())


# Count number of ratings per user
user_activity = ratings_merged.groupby('userId').size()

# Keep users with >= 30 interactions
active_users = user_activity[user_activity >= 50].index

# Filter dataset
filtered_ratings_50 = ratings_merged[ratings_merged['userId'].isin(active_users)]

# Randomly sample 100 users (for testing ChatGPT later)

np.random.seed(42)
sampled_users_100 = np.random.choice(active_users, size=100, replace=False)

# Final working set
sampled_data = filtered_ratings_50[filtered_ratings_50['userId'].isin(sampled_users_100)]

# Confirm
print(f"Number of users in sample: {sampled_data['userId'].nunique()}")


# Now sample 10 users from the previously sampled 100
np.random.seed(42)
sampled_users_10 = np.random.choice(sampled_users_100, size=10, replace=False)

# Filter the data to just these 10 users
sampled_data_10 = sampled_data[sampled_data['userId'].isin(sampled_users_10)]

# Confirm
print(f"Number of users in sample: {sampled_data_10['userId'].nunique()}")
print("User IDs:", sampled_users_10)

"""
total_users = ratings['userId'].nunique()
print(f"Total users: {total_users}")

user_rating_counts = ratings.groupby('userId').size()
active_users = user_rating_counts[user_rating_counts >= 100]
num_active_users = active_users.count()
print(f"Users with ≥30 ratings: {num_active_users}")

proportion_active = num_active_users / total_users
print(f"Proportion of users with ≥30 ratings: {proportion_active:.2%}")


# Sort by user and timestamp
sampled_data_sorted = sampled_data.sort_values(by=['userId', 'timestamp'])

train_rows = []
test_rows = []

# 80/20 split per user
for user_id, group in sampled_data_sorted.groupby('userId'):
    group_sorted = group.sort_values('timestamp')
    n = len(group_sorted)
    train_size = int(n * 0.8)

    train_rows.append(group_sorted.iloc[:train_size])
    test_rows.append(group_sorted.iloc[train_size:])

# Combine into final DataFrames
train_data = pd.concat(train_rows).reset_index(drop=True)
test_data = pd.concat(test_rows).reset_index(drop=True)

# Final sanity check
print(f"Train shape: {train_data.shape}")
print(f"Test shape: {test_data.shape}")
print(f"Train users: {train_data['userId'].nunique()} | Test users: {test_data['userId'].nunique()}")


# Step 1: Create an empty dictionary to store prompts
zero_shot_prompts = {}

# Step 2: Loop through each user and build a prompt
for user_id, group in train_data.groupby("userId"):
    # Get all movie titles the user rated in their training set
    movie_titles = group["title"].tolist()

    # Join the titles into a single comma-separated string
    movies_str = ', '.join(movie_titles)

    # Build the zero-shot prompt
    prompt = (
        f"I’ve enjoyed the following movies: {movies_str}.\n"
        "Based on my preferences, what 30 movies would you recommend I watch next?"
    )

    # Store the prompt
    zero_shot_prompts[user_id] = prompt

# Step 3: Preview a few prompts
for uid, prompt in list(zero_shot_prompts.items())[:3]:
    print(f"\nUser {uid} Prompt:\n{prompt}\n{'-' * 60}")
    
"""

# Sort by user and timestamp
sampled_data_10_sorted = sampled_data_10.sort_values(by=['userId', 'timestamp'])

# Prepare lists to hold train and test rows
train_rows = []
test_rows = []

# 80/20 split per user (for these 10 users)
for user_id, group in sampled_data_10_sorted.groupby('userId'):
    group_sorted = group.sort_values('timestamp')
    n = len(group_sorted)
    train_size = int(n * 0.8)

    train_rows.append(group_sorted.iloc[:train_size])
    test_rows.append(group_sorted.iloc[train_size:])

# Combine into final DataFrames
train_data_10 = pd.concat(train_rows).reset_index(drop=True)
test_data_10 = pd.concat(test_rows).reset_index(drop=True)

# Sanity check
print(f"Train shape: {train_data_10.shape}")
print(f"Test shape: {test_data_10.shape}")
print(f"Train users: {train_data_10['userId'].nunique()} | Test users: {test_data_10['userId'].nunique()}")


client = openai.OpenAI(api_key="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")

# Dictionary to store recommendations
recommendations_with_history = {}

# Request recommendations for each user
for user_id in sampled_users_10:
    user_movies = train_data_10[train_data_10['userId'] == user_id]['title'].tolist()
    movie_history = ', '.join(user_movies[-10:])

    prompt = (
        f"I’ve watched and liked the following movies: {movie_history}.\n"
        "Based on this, give 5 movie recommendations. Just list the movie titles."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=200
        )

        # Split into lines and clean titles
        raw_lines = response.choices[0].message.content.strip().split('\n')
        cleaned_titles = []

        for line in raw_lines:
            line = line.strip()
            if line and line[0].isdigit():
                line = line.lstrip('0123456789. )-').strip()
            if line:
                cleaned_titles.append(line)

        recommendations_with_history[user_id] = cleaned_titles
        print(f"User {user_id} → Recommended:\n{cleaned_titles}\n")

    except Exception as e:
        print(f"Error for user {user_id}: {e}")

# Function for fuzzy matching
def get_best_match(title, movie_list, threshold=90):
    match = process.extractOne(title, movie_list, score_cutoff=threshold)
    return match[0] if match else None

# Prepare for evaluation
all_titles = movies['title'].tolist()

hit_count = 0
total_precision = 0
total_recall = 0
not_in_dataset_count = 0

for user_id, recommended_titles in recommendations_with_history.items():
    user_test_titles = test_data_10[test_data_10['userId'] == user_id]['title'].tolist()
    user_test_set_size = len(user_test_titles)
    true_positives = 0

    for rec in recommended_titles:
        matched_title = get_best_match(rec, all_titles)

        if matched_title is None:
            not_in_dataset_count += 1
            print(f"User {user_id} → GPT: {rec} → Not in dataset → SKIPPED")
            continue

        if matched_title in user_test_titles:
            true_positives += 1
            print(f"User {user_id} → GPT: {rec} → Matched: {matched_title} → HIT")
        else:
            print(f"User {user_id} → GPT: {rec} → Matched: {matched_title} → MISS")

    # HIT@5 (user-level)
    if true_positives > 0:
        hit_count += 1

    # Precision@5 and Recall@5
    precision = true_positives / 5
    recall = true_positives / user_test_set_size if user_test_set_size > 0 else 0
    total_precision += precision
    total_recall += recall

    print(f"User {user_id} → Precision@5: {precision:.2f} | Recall@5: {recall:.2f}\n")

# Final metrics
hit_at_5 = hit_count / len(sampled_users_10)
avg_precision_at_5 = total_precision / len(sampled_users_10)
avg_recall_at_5 = total_recall / len(sampled_users_10)

print(f"\nHit@5: {hit_at_5:.2f}")
print(f"Average Precision@5: {avg_precision_at_5:.2f}")
print(f"Average Recall@5: {avg_recall_at_5:.2f}")
print(f"Total GPT recommendations not in dataset: {not_in_dataset_count}")
