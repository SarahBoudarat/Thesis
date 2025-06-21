# zero_shot_cold_start.py
import pandas as pd
import numpy as np
import pickle
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util
import openai
from difflib import get_close_matches

# -----------------------------
# SETUP
# -----------------------------
client = openai.OpenAI(api_key="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")
model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# FUNCTION: Load Data
# -----------------------------
def load_data(pkl_path="data_for_10_users.pkl"):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    data["movies"]["genres"] = data["movies"]["genres"].str.split("|")
    return data

# -----------------------------
# FUNCTION: Build User Profiles
# -----------------------------
def build_user_profiles(train_data, movies):
    train_data = train_data.merge(movies[["movieId", "genres"]], on="movieId", how="left")
    train_data["genres"] = train_data["genres_y"]  # Fix merged columns
    liked = train_data[train_data["rating"] >= 4].explode("genres")
    user_genres = liked.groupby(["userId", "genres"]).size().reset_index(name="count")
    top_genres = user_genres.sort_values(["userId", "count"], ascending=[True, False])
    top_2_genres = top_genres.groupby("userId").head(2)
    user_profiles = top_2_genres.groupby("userId")["genres"].apply(list).reset_index()
    user_profiles["age"] = np.random.randint(18, 60, size=len(user_profiles))
    return user_profiles

# -----------------------------
# FUNCTION: Get Recommendations
# -----------------------------
def get_gpt_recommendations(user_profiles):
    recommendations = {}
    for _, row in user_profiles.iterrows():
        user_id = row["userId"]
        genres = ", ".join(row["genres"])
        age = row["age"]

        prompt = (
            f"I’m {age} years old and I enjoy {genres} movies. "
            "I just joined this platform. Recommend 5 movies I might like. "
            "Only list the movie titles, one per line, with no description or extra text."
        )

        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=200
            )
            text = response.choices[0].message.content.strip()
            recs = [line.strip("→•- ").lstrip("1234567890. )").strip()
                    for line in text.replace(",", "\n").split("\n")
                    if line.strip() and not line.lower().startswith("of course")]
            recommendations[user_id] = recs[:5]
        except Exception as e:
            print(f"Error for user {user_id}: {e}")
    return recommendations

# -----------------------------
# FUNCTION: Fuzzy Match Helper
# -----------------------------
def is_in_dataset(title, known_titles, cutoff=0.6):
    return bool(get_close_matches(title, known_titles, n=1, cutoff=cutoff))

# -----------------------------
# FUNCTION: Evaluate Recommendations
# -----------------------------
def evaluate(recommendations, test_data, model, movies):
    SIMILARITY_THRESHOLD = 0.7
    evaluation_results = defaultdict(dict)
    known_titles = set(movies["title"])

    for user_id, recs in recommendations.items():
        test_titles = test_data[test_data["userId"] == user_id]["title"].tolist()
        if not recs or not test_titles:
            continue

        rec_embeds = model.encode(recs, convert_to_tensor=True)
        test_embeds = model.encode(test_titles, convert_to_tensor=True)
        cosine_scores = util.cos_sim(rec_embeds, test_embeds)

        hits = [recs[i] for i in range(len(recs)) if max(cosine_scores[i]) >= SIMILARITY_THRESHOLD]

        in_dataset_count = 0
        for title in recs:
            if is_in_dataset(title, known_titles):
                in_dataset_count += 1
            else:
                print(f"[Hallucinated] Not in dataset: {title}")

        evaluation_results[user_id] = {
            "Hit@5": int(len(hits) > 0),
            "Precision@5": len(hits) / 5,
            "Recall@5": len(hits) / len(test_titles),
            "NDCG@5": (
                sum([1 / np.log2(idx + 2) for idx, title in enumerate(recs) if title in hits]) /
                sum([1 / np.log2(i + 2) for i in range(min(len(test_titles), 5))])
                if test_titles else 0
            ),
            "InDataset@5": in_dataset_count / 5
        }

    return evaluation_results

# -----------------------------
# MAIN SCRIPT
# -----------------------------
data = load_data()
user_profiles = build_user_profiles(data["train_data_10"], data["movies"])
recommendations = get_gpt_recommendations(user_profiles)
evaluation_results = evaluate(recommendations, data["test_data_10"], model, data["movies"])

# Save results
with open("cold_start_zero_shot_data.pkl", "wb") as f:
    pickle.dump({
        "user_profiles": user_profiles,
        "recommendations": recommendations,
        "evaluation_results": dict(evaluation_results),
    }, f)

# Print summary
if evaluation_results:
    print("\nAverage Evaluation Metrics for Zero-Shot Cold Start:")
    for metric in ["Hit@5", "Precision@5", "Recall@5", "NDCG@5", "InDataset@5"]:
        scores = [evaluation_results[u][metric] for u in evaluation_results]
        print(f"{metric}: {np.mean(scores):.3f}")
else:
    print("\nNo evaluation results were generated.")

