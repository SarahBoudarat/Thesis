import pandas as pd
import numpy as np
import pickle
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util
import openai

# -----------------------------
# SETUP
# -----------------------------
client = openai.OpenAI(api_key="")
model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# STEP 1: Load and Sample Data
# -----------------------------
ratings = pd.read_csv("ratings.csv")
movies = pd.read_csv("movies.csv")
ratings_merged = pd.merge(ratings, movies, on="movieId")

user_activity = ratings_merged.groupby("userId").size()
active_users = user_activity[user_activity >= 50].index
filtered_ratings = ratings_merged[ratings_merged["userId"].isin(active_users)]

np.random.seed(42)
sampled_users_100 = np.random.choice(active_users, size=100, replace=False)
sampled_users_10 = np.random.choice(sampled_users_100, size=10, replace=False)

sampled_data = filtered_ratings[filtered_ratings["userId"].isin(sampled_users_10)]
sampled_data_sorted = sampled_data.sort_values(by=["userId", "timestamp"])

train_rows, test_rows = [], []
for user_id, group in sampled_data_sorted.groupby("userId"):
    group_sorted = group.sort_values("timestamp")
    train_size = int(len(group_sorted) * 0.8)
    train_rows.append(group_sorted.iloc[:train_size])
    test_rows.append(group_sorted.iloc[train_size:])

train_data_10 = pd.concat(train_rows).reset_index(drop=True)
test_data_10 = pd.concat(test_rows).reset_index(drop=True)

# Save for consistency
with open("data_for_10_users.pkl", "wb") as f:
    pickle.dump({
        "sampled_users_10": sampled_users_10,
        "train_data_10": train_data_10,
        "test_data_10": test_data_10,
        "movies": movies
    }, f)

with open("cold_start_train_data.pkl", "rb") as f:
    cold_data = pickle.load(f)

cold_start_train_data = cold_data["cold_start_train_data"]


# Simulate cold-start: keep only top-3 rated movies per user in train set
cold_start_train_rows = []

for user_id in sampled_users_10:
    user_ratings = train_data_10[train_data_10["userId"] == user_id]
    top_rated = user_ratings.sort_values(by="rating", ascending=False).head(3)
    cold_start_train_rows.append(top_rated)

cold_start_train_data = pd.concat(cold_start_train_rows).reset_index(drop=True)

# Save for reuse
with open("cold_start_train_data.pkl", "wb") as f:
    pickle.dump({
        "sampled_users_10": sampled_users_10,
        "cold_start_train_data": cold_start_train_data,
        "test_data_10": test_data_10,
        "movies": movies
    }, f)

print("Cold-start training data created and saved.")


one_shot_example = (
    "Example:\n"
    "I’ve watched and liked the following movies: The Matrix, Inception, Fight Club.\n"
    "Based on this, recommend 5 movies I might enjoy next. Just list the movie titles.\n"
    "→ The Dark Knight, Memento, Interstellar, Se7en, The Prestige\n\n"
)

recommendations = {}

for user_id in sampled_users_10:
    user_movies = cold_start_train_data[cold_start_train_data["userId"] == user_id]["title"].tolist()
    history = ', '.join(user_movies[-10:])

    prompt = (
        f"{one_shot_example}"
        f"Now for me:\n"
        f"I’ve watched and liked the following movies: {history}.\n"
        "Based on this, recommend 5 movies I might enjoy next. Just list the movie titles."
    )



    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=200
        )

        text = response.choices[0].message.content.strip()
        recs = [line.strip("→•- ").rstrip("1234567890. )").strip() for line in text.replace(",", "\n").split("\n") if
                line.strip()]
        recommendations[user_id] = recs[:5]

        print(f"\nUser {user_id} → Recommendations: {recs[:5]}")

    except Exception as e:
        print(f"Error for user {user_id}: {e}")

# -----------------------------
# STEP 3: Semantic Evaluation
# -----------------------------
SIMILARITY_THRESHOLD = 0.7
evaluation_results = defaultdict(dict)

print("\nSemantic Evaluation Results")
print("----------------------------")

for user_id in sampled_users_10:
    recs = recommendations.get(user_id, [])
    test_titles = test_data_10[test_data_10["userId"] == user_id]["title"].tolist()

    if not recs or not test_titles:
        continue

    rec_embeds = model.encode(recs, convert_to_tensor=True)
    test_embeds = model.encode(test_titles, convert_to_tensor=True)
    cosine_scores = util.cos_sim(rec_embeds, test_embeds)

    hits = []
    for i, rec_title in enumerate(recs):
        max_sim = max(cosine_scores[i])
        if max_sim >= SIMILARITY_THRESHOLD:
            hits.append(rec_title)

    hit_at_5 = int(len(hits) > 0)
    precision = len(hits) / 5
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

# -----------------------------
# STEP 4: Summary
# -----------------------------
if evaluation_results:
    print("\nAverage Evaluation Metrics for Few-Shot (Semantic Similarity):")
    for metric in ["Hit@5", "Precision@5", "Recall@5", "NDCG@5"]:
        scores = [evaluation_results[u][metric] for u in evaluation_results]
        avg = np.mean(scores)
        print(f"{metric}: {avg:.3f}")
else:
    print("\n No evaluation results were generated. Check GPT output or semantic similarity threshold.")
