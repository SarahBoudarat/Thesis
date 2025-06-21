import pandas as pd
import numpy as np
import openai
from rapidfuzz import process
import pickle
from collections import defaultdict
from rapidfuzz import fuzz



# -----------------------------
# Data Preparation Section
# -----------------------------

# Load ratings and movies datasets
ratings = pd.read_csv("../ratings.csv")
movies = pd.read_csv("../movies.csv")

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
with open("../data_for_10_users.pkl", "wb") as f:
    pickle.dump({
        "sampled_users_10": sampled_users_10,
        "train_data_10": train_data_10,
        "test_data_10": test_data_10,
        "movies": movies
    }, f)

# Save user IDs
with open("../sampled_user_ids.txt", "w") as f:
    for uid in sampled_users_10:
        f.write(f"{uid}\n")


# Reload saved data
with open("../data_for_10_users.pkl", "rb") as f:
    data = pickle.load(f)

sampled_users_10 = data["sampled_users_10"]
train_data_10 = data["train_data_10"]
test_data_10 = data["test_data_10"]
movies = data["movies"]

# -----------------------------
# Fixed Chain-of-Thought Prompt Test
# -----------------------------

# Set OpenAI API key
client = openai.OpenAI(api_key="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")

user_recommendations = defaultdict(dict)

for user_id in sampled_users_10:
    # Get user's training ratings
    user_ratings = train_data_10[train_data_10['userId'] == user_id]

    # Get top 5 rated movies for history
    top_movies = user_ratings.sort_values(by='rating', ascending=False).head(10)['title'].tolist()

    if len(top_movies) < 2:
        print(f"User {user_id} has fewer than 2 top-rated movies. Skipping.")
        continue

    # Format movie history into bullet points
    history_str = "\n".join([f"- {title}" for title in top_movies])

    # Construct CoT prompt asking for Top-5 recommendations
    prompt_text = f"""
Instruction: Based on the user's movie history, analyze their preferences and recommend 5 movies they are likely to enjoy. Do not explain the output — return movie titles only.

User's Movie History:
{history_str}

Chain of Thought:
To recommend suitable movies, I will analyze the themes, genres, and tones of the user's previous movies. Based on these inferred preferences, I will recommend 5 movies that align with these patterns.

Top-5 Recommendations (movie titles only, ranked from 1 to 5):
"""

    # Call OpenAI API
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt_text}
            ],
            temperature=0.7,
            max_tokens=200
        )

        full_response = response.choices[0].message.content.strip()

        # Clean and extract up to 5 movie titles
        movie_lines = [
            line.strip().lstrip("12345.").strip()
            for line in full_response.split("\n")
            if line.strip()
        ]
        top_5_movies = movie_lines[:5]

        user_recommendations[user_id]['history'] = top_movies
        user_recommendations[user_id]['recommendations'] = top_5_movies

        print(f"\nUser {user_id} — History:\n{top_movies}\nTop-5 Recommendations:\n{top_5_movies}\n")

    except Exception as e:
        print(f"Failed for user {user_id}: {e}")


FUZZY_THRESHOLD = 70

def fuzzy_match_verbose(movie_title, test_titles, threshold=FUZZY_THRESHOLD):
    for test_title in test_titles:
        score = fuzz.token_sort_ratio(movie_title, test_title)
        print(f"  Comparing to test title: '{test_title}' → Score: {score}")
        if score >= threshold:
            return True
    return False

evaluation_results = defaultdict(dict)

print("\nDEBUGGING EVALUATION")
print("--------------------")

for user_id in sampled_users_10:
    if user_id not in user_recommendations or 'recommendations' not in user_recommendations[user_id]:
        continue

    recommended = user_recommendations[user_id]['recommendations']
    test_movies = test_data_10[test_data_10['userId'] == user_id]['title'].tolist()

    print(f"\nUser {user_id}")
    print(f"Test set movies ({len(test_movies)}): {test_movies}")
    print(f"Recommendations ({len(recommended)}): {recommended}")

    hits = []
    for title in recommended:
        print(f"\nChecking match for: '{title}'")
        if fuzzy_match_verbose(title.lower(), [t.lower() for t in test_movies]):
            print("Match found!")
            hits.append(title)
        else:
            print("No match.")

    hit_at_5 = int(len(hits) > 0)
    precision_at_5 = len(hits) / len(recommended) if recommended else 0
    recall_at_5 = len(hits) / len(test_movies) if test_movies else 0
    dcg = sum([1 / np.log2(idx + 2) for idx, movie in enumerate(recommended) if movie in hits])
    ideal_dcg = sum([1 / np.log2(i + 2) for i in range(min(len(test_movies), 5))])
    ndcg_at_5 = dcg / ideal_dcg if ideal_dcg > 0 else 0

    evaluation_results[user_id] = {
        "Hit@5": hit_at_5,
        "Precision@5": precision_at_5,
        "Recall@5": recall_at_5,
        "NDCG@5": ndcg_at_5
    }

metrics = ["Hit@5", "Precision@5", "Recall@5", "NDCG@5"]
average_scores = {m: np.mean([evaluation_results[u][m] for u in evaluation_results]) for m in metrics}

print("\nAverage Evaluation Metrics for Chain-of-Thought Top-5 (Debug Version):")
for metric, score in average_scores.items():
    print(f"{metric}: {score:.3f}")