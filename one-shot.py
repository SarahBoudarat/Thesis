import pandas as pd
import numpy as np
import openai
from rapidfuzz import process
import pickle

# -----------------------------
# Data Preparation Section
# -----------------------------

# Load ratings and movies datasets
ratings = pd.read_csv("ratings.csv")
movies = pd.read_csv("movies.csv")

# Merge ratings with movie titles
ratings_merged = pd.merge(ratings, movies, on="movieId")

# Filter users with >= 50 ratings
user_activity = ratings_merged.groupby('userId').size()
active_users = user_activity[user_activity >= 50].index
filtered_ratings_50 = ratings_merged[ratings_merged['userId'].isin(active_users)]

# Sample 100 users, then 10 from them
np.random.seed(42)
sampled_users_100 = np.random.choice(active_users, size=100, replace=False)
sampled_users_10 = np.random.choice(sampled_users_100, size=10, replace=False)

sampled_data = filtered_ratings_50[filtered_ratings_50['userId'].isin(sampled_users_10)]
sampled_data_sorted = sampled_data.sort_values(by=['userId', 'timestamp'])

# 80/20 split per user
train_rows = []
test_rows = []
for user_id, group in sampled_data_sorted.groupby('userId'):
    group_sorted = group.sort_values('timestamp')
    train_size = int(len(group_sorted) * 0.8)
    train_rows.append(group_sorted.iloc[:train_size])
    test_rows.append(group_sorted.iloc[train_size:])

train_data_10 = pd.concat(train_rows).reset_index(drop=True)
test_data_10 = pd.concat(test_rows).reset_index(drop=True)

# Save data for reuse
with open("data_for_10_users.pkl", "wb") as f:
    pickle.dump({
        "sampled_users_10": sampled_users_10,
        "train_data_10": train_data_10,
        "test_data_10": test_data_10,
        "movies": movies
    }, f)

# Save user IDs
with open("sampled_user_ids.txt", "w") as f:
    for uid in sampled_users_10:
        f.write(f"{uid}\n")

# -----------------------------
# GPT Recommendation Section
# -----------------------------

# Load one-shot example
one_shot_example = (
    "Example:\n"
    "I’ve watched and liked the following movies: The Matrix, Inception, Fight Club.\n"
    "Based on this, recommend 5 movies I might enjoy next. Just give the movie titles.\n"
    "→ The Dark Knight, Memento, Interstellar, Se7en, The Prestige\n\n"
)

# Reload saved data
with open("data_for_10_users.pkl", "rb") as f:
    data = pickle.load(f)

sampled_users_10 = data["sampled_users_10"]
train_data_10 = data["train_data_10"]
test_data_10 = data["test_data_10"]
movies = data["movies"]

client = openai.OpenAI(api_key="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")
recommendations_one_shot = {}

for user_id in sampled_users_10:
    user_movies = train_data_10[train_data_10['userId'] == user_id]['title'].tolist()
    movie_history = ', '.join(user_movies[-10:])

    prompt = (
        f"{one_shot_example}"
        f"Now for me:\n"
        f"I’ve watched and liked the following movies: {movie_history}.\n"
        "Based on this, recommend 5 movies I might enjoy next. Just give the movie titles.\n"
        "→"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=100
        )

        raw_response = response.choices[0].message.content.strip()

        # Try splitting on commas if it's all in one line
        if "," in raw_response and "\n" not in raw_response:
            recs = raw_response.split(",")
        else:
            recs = raw_response.split("\n")

        cleaned = [r.lstrip("→-•1234567890). ").strip() for r in recs if r.strip()]

        recommendations_one_shot[user_id] = cleaned
        print(f"User {user_id} → Recommended: {cleaned}")

    except Exception as e:
        print(f"Error for user {user_id}: {e}")

with open("recommendations_one_shot.pkl", "wb") as f:
    pickle.dump(recommendations_one_shot, f)

# -----------------------------
# Evaluation Section
# -----------------------------

with open("recommendations_one_shot.pkl", "rb") as f:
    recommendations = pickle.load(f)

all_titles = movies['title'].tolist()

def get_best_match(title, movie_list, threshold=90):
    match = process.extractOne(title, movie_list, score_cutoff=threshold)
    return match[0] if match else None

hit_count = 0
total_precision = 0
total_recall = 0
not_in_dataset_count = 0

for user_id in sampled_users_10:
    user_test_titles = test_data_10[test_data_10['userId'] == user_id]['title'].tolist()
    user_test_set_size = len(user_test_titles)
    recommended_titles = recommendations.get(user_id, [])  # should be a list of 5 clean strings
    matched_titles = [get_best_match(title, all_titles) for title in recommended_titles]

    valid_matches = [title for title in matched_titles if title and title in user_test_titles]

    print(f"\nUser {user_id} Test Set: {user_test_titles}")
    print(f"Matched Titles: {matched_titles}")
    print(f"Valid Matches in Test Set: {valid_matches}")

    if valid_matches:
        hit_count += 1
        precision = len(valid_matches) / len(recommended_titles)
        recall = len(valid_matches) / user_test_set_size if user_test_set_size > 0 else 0
    else:
        precision = 0
        recall = 0

    total_precision += precision
    total_recall += recall
    not_in_dataset_count += sum(1 for match in matched_titles if match is None)

hit_at_5 = hit_count / len(sampled_users_10)
avg_precision = total_precision / len(sampled_users_10)
avg_recall = total_recall / len(sampled_users_10)

print("\nFinal Evaluation Results")
print(f"Hit@5: {hit_at_5:.2f}")
print(f"Average Precision@5: {avg_precision:.2f}")
print(f"Average Recall@5: {avg_recall:.2f}")
print(f"Total GPT recommendations not in dataset: {not_in_dataset_count}")
