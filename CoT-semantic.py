import pandas as pd
import numpy as np
import openai
from rapidfuzz import process, fuzz
import pickle
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util

# -----------------------------
# SETUP
# -----------------------------
client = openai.OpenAI(api_key="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")
model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# STEP 1: Load Preprocessed Data
# -----------------------------
with open("data_for_10_users.pkl", "rb") as f:
    data = pickle.load(f)

sampled_users_10 = data["sampled_users_10"]
train_data_10 = data["train_data_10"]
test_data_10 = data["test_data_10"]
movies = data["movies"]

# Clean movie titles
movies["clean_title"] = movies["title"].str.replace(r"\(\d{4}\)", "", regex=True).str.strip()
known_titles = set(movies["clean_title"])

def clean_for_match(title):
    return title.strip().replace("(", "").replace(")", "").strip()

def is_in_dataset(title):
    cleaned = clean_for_match(title)
    match = process.extractOne(cleaned, known_titles, scorer=fuzz.token_sort_ratio)
    if match and match[1] >= 70:
        print(f"[Matched] {title} → {match[0]}")
        return True
    else:
        print(f"[Hallucinated] Not in dataset: {title}")
        return False

# -----------------------------
# STEP 2: Chain-of-Thought Prompting
# -----------------------------
user_recommendations = defaultdict(dict)

for user_id in sampled_users_10:
    user_ratings = train_data_10[train_data_10["userId"] == user_id]
    top_movies = user_ratings.sort_values(by="rating", ascending=False).head(10)["title"].tolist()

    if len(top_movies) < 2:
        print(f"User {user_id} has fewer than 2 top-rated movies. Skipping.")
        continue

    history_str = "\n".join([f"- {title}" for title in top_movies])

    few_shot_example = (
        "Example:\n"
        "User's Movie History:\n"
        "- The Matrix\n"
        "- Inception\n"
        "- Fight Club\n\n"
        "Chain of Thought:\n"
        "Let's think step by step. The user enjoys cerebral science fiction and psychological thrillers that blend action with complex narratives.\n\n"
        "Top-5 Recommendations:\n"
        "The Dark Knight, Memento, Interstellar, Se7en, The Prestige\n\n"
        "Now for me:\n"
    )

    prompt_text = (
        f"{few_shot_example}"
        f"User's Movie History:\n{history_str}\n\n"
        f"Chain of Thought:\n"
        f"Let's think step by step. To recommend suitable movies, I will analyze the themes, genres, and tones of the user's previous movies. "
        f"Based on these inferred preferences, I will recommend 5 movies that align with these patterns.\n\n"
        f"Based on this, recommend 5 movies I might enjoy next. Just list the movie titles."

    )
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt_text}],
            temperature=0.7,
            max_tokens=200
        )

        text = response.choices[0].message.content.strip()
        movie_lines = [line.strip("→-• ").rstrip("1234567890. )").strip() for line in text.replace(",", "\n").split("\n") if line.strip()]
        top_5 = movie_lines[:5]

        user_recommendations[user_id]["history"] = top_movies
        user_recommendations[user_id]["recommendations"] = top_5

        print(f"\nUser {user_id} → Recommendations: {top_5}")

    except Exception as e:
        print(f"Error for user {user_id}: {e}")

# -----------------------------
# STEP 3: Semantic Evaluation
# -----------------------------
evaluation_results = defaultdict(dict)
SIMILARITY_THRESHOLD = 0.5

print("\nSemantic Evaluation Results")
print("----------------------------")

for user_id in sampled_users_10:
    if user_id not in user_recommendations or "recommendations" not in user_recommendations[user_id]:
        continue

    recs = user_recommendations[user_id]["recommendations"]
    test_titles = test_data_10[test_data_10["userId"] == user_id]["title"].tolist()

    if not recs or not test_titles:
        continue

    rec_embeds = model.encode(recs, convert_to_tensor=True)
    test_embeds = model.encode(test_titles, convert_to_tensor=True)
    cosine_scores = util.cos_sim(rec_embeds, test_embeds)

    hits = [recs[i] for i in range(len(recs)) if max(cosine_scores[i]) >= SIMILARITY_THRESHOLD]
    in_dataset_count = sum([1 for title in recs if is_in_dataset(title)])

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
        "NDCG@5": ndcg,
        "InDataset@5": in_dataset_count / 5
    }

# -----------------------------
# STEP 4: Summary & Save
# -----------------------------
metrics = ["Hit@5", "Precision@5", "Recall@5", "NDCG@5", "InDataset@5"]
avg_scores = {m: np.mean([evaluation_results[u][m] for u in evaluation_results]) for m in metrics}

print("\nAverage Evaluation Metrics for Chain-of-Thought Warm-Start:")
for metric in metrics:
    print(f"{metric}: {avg_scores[metric]:.3f}")

with open("warm_start_cot_results.pkl", "wb") as f:
    pickle.dump({
        "recommendations": dict(user_recommendations),
        "evaluation_results": dict(evaluation_results)
    }, f)

