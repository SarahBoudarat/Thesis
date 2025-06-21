import pandas as pd
import numpy as np
import openai
from rapidfuzz import process
import pickle
from collections import defaultdict
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer, util

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# -----------------------------
# Data Preparation
# -----------------------------
ratings = pd.read_csv("../ratings.csv")
movies = pd.read_csv("../movies.csv")
ratings_merged = pd.merge(ratings, movies, on="movieId")

# Filter active users
user_activity = ratings_merged.groupby('userId').size()
active_users = user_activity[user_activity >= 50].index
filtered_ratings = ratings_merged[ratings_merged['userId'].isin(active_users)]

# Sample 100, then 10 users
np.random.seed(42)
sampled_users_100 = np.random.choice(active_users, size=100, replace=False)
sampled_users_10 = np.random.choice(sampled_users_100, size=10, replace=False)

sampled_data = filtered_ratings[filtered_ratings['userId'].isin(sampled_users_10)]
sampled_data_sorted = sampled_data.sort_values(by=['userId', 'timestamp'])

# Train/test split
train_rows, test_rows = [], []
for user_id, group in sampled_data_sorted.groupby('userId'):
    group_sorted = group.sort_values('timestamp')
    split = int(len(group_sorted) * 0.8)
    train_rows.append(group_sorted.iloc[:split])
    test_rows.append(group_sorted.iloc[split:])

train_data_10 = pd.concat(train_rows).reset_index(drop=True)
test_data_10 = pd.concat(test_rows).reset_index(drop=True)

# -----------------------------
# Chain-of-Thought Prompting
# -----------------------------
client = openai.OpenAI(api_key="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")
user_recommendations = defaultdict(dict)

for user_id in sampled_users_10:
    user_ratings = train_data_10[train_data_10['userId'] == user_id]
    top_movies = user_ratings.sort_values(by='rating', ascending=False).head(5)['title'].tolist()

    if len(top_movies) < 2:
        print(f"User {user_id} skipped (not enough rated movies).")
        continue

    movie1, movie2 = top_movies[0], top_movies[1]
    history = ", ".join(top_movies)

    prompt = f"""
Instruction: Recommend 5 movies based on the user's preferences.

User’s Preferences: The user enjoys films with strong storytelling and emotional depth. They have previously enjoyed: {history}.

Chain of Thought: To recommend suitable movies, I will consider the user's taste for impactful narratives, strong character development, and emotional or thought-provoking themes. Their enjoyment of “{movie1}” and “{movie2}” suggests a preference for compelling stories and immersive experiences.

1. Focus on genres and tones similar to {movie1} and {movie2}.
2. Prioritize emotionally resonant, well-received films.
3. Ensure recommendations match their pattern of immersive storytelling.

Recommendations (just 5 movie titles, ranked 1 to 5):
"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=200
        )
        result = response.choices[0].message.content.strip()
        lines = [line.strip().lstrip("12345.→-• ").strip() for line in result.split("\n") if line.strip()]
        top_5 = lines[:5]

        user_recommendations[user_id]['history'] = top_movies
        user_recommendations[user_id]['recommendations'] = top_5

        print(f"\nUser {user_id} — Based on: {movie1}, {movie2}\nTop-5 Recommendations:\n{top_5}\n")

    except Exception as e:
        print(f"Failed for user {user_id}: {e}")

# -----------------------------
# Semantic Evaluation
# -----------------------------
evaluation_results = defaultdict(dict)
SIMILARITY_THRESHOLD = 0.7

print("\nSemantic Evaluation Results (Refined CoT)")
print("------------------------------------------")

for user_id in sampled_users_10:
    if user_id not in user_recommendations:
        continue

    recs = user_recommendations[user_id].get('recommendations', [])
    test_titles = test_data_10[test_data_10['userId'] == user_id]['title'].tolist()

    if not recs or not test_titles:
        continue

    rec_embeds = model.encode(recs, convert_to_tensor=True)
    test_embeds = model.encode(test_titles, convert_to_tensor=True)
    cosine_scores = util.cos_sim(rec_embeds, test_embeds)

    hits = []
    for i, rec in enumerate(recs):
        max_sim = max(cosine_scores[i])
        if max_sim >= SIMILARITY_THRESHOLD:
            hits.append(rec)

    hit_at_5 = int(len(hits) > 0)
    precision = len(hits) / len(recs)
    recall = len(hits) / len(test_titles)
    dcg = sum([1 / np.log2(idx + 2) for idx, movie in enumerate(recs) if movie in hits])
    ideal_dcg = sum([1 / np.log2(i + 2) for i in range(min(len(test_titles), 5))])
    ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0

    evaluation_results[user_id] = {
        "Hit@5": hit_at_5,
        "Precision@5": precision,
        "Recall@5": recall,
        "NDCG@5": ndcg
    }

# Print average metrics
metrics = ["Hit@5", "Precision@5", "Recall@5", "NDCG@5"]
average_scores = {m: np.mean([evaluation_results[u][m] for u in evaluation_results]) for m in metrics}

print("\nAverage Evaluation Metrics for Refined CoT (Top-5, Semantic Similarity):")
for metric, score in average_scores.items():
    print(f"{metric}: {score:.3f}")

