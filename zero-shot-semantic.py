# warm_start_zero_shot.py
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
# STEP 1: Load Pre-Filtered Data (Consistent Across All Experiments)
# -----------------------------
with open("data_for_10_users.pkl", "rb") as f:
    data = pickle.load(f)

sampled_users_10 = data["sampled_users_10"]
train_data_10 = data["train_data_10"]
test_data_10 = data["test_data_10"]
movies = data["movies"]

with open("data_for_10_users.pkl", "wb") as f:
    pickle.dump({
        "sampled_users_10": sampled_users_10,
        "train_data_10": train_data_10,
        "test_data_10": test_data_10,
        "movies": movies
    }, f)

# -----------------------------
# STEP 2: Zero-Shot Prompting
# -----------------------------
recommendations = {}

for user_id in sampled_users_10:
    user_movies = train_data_10[train_data_10["userId"] == user_id]["title"].tolist()
    history = ', '.join(user_movies[-10:])

    prompt = (
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
        recs = [line.strip("→•- ").rstrip("1234567890. )").strip() for line in text.replace(",", "\n").split("\n") if line.strip()]
        recommendations[user_id] = recs[:5]

        print(f"\nUser {user_id} → Recommendations: {recs[:5]}")

    except Exception as e:
        print(f"Error for user {user_id}: {e}")

# -----------------------------
# STEP 3: Semantic Evaluation + InDataset Check
# -----------------------------
SIMILARITY_THRESHOLD = 0.5
evaluation_results = defaultdict(dict)
movies["clean_title"] = movies["title"].str.replace(r"\(\d{4}\)", "", regex=True).str.strip()
known_titles = set(movies["clean_title"])

def clean_for_match(title):
    return title.strip().replace("(", "").replace(")", "").strip()


def is_in_dataset(title):
    cleaned = clean_for_match(title)
    matches = get_close_matches(cleaned, known_titles, n=1, cutoff=0.5)
    if matches:
        print(f"[Matched] {title} → {matches[0]}")
        return True
    else:
        print(f"[Hallucinated] Not in dataset: {title}")
        return False


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
# STEP 4: Save and Report
# -----------------------------
with open("warm_start_zero_shot_results.pkl", "wb") as f:
    pickle.dump({
        "recommendations": recommendations,
        "evaluation_results": dict(evaluation_results)
    }, f)

if evaluation_results:
    print("\nAverage Evaluation Metrics for Warm-Start Zero-Shot:")
    for metric in ["Hit@5", "Precision@5", "Recall@5", "NDCG@5", "InDataset@5"]:
        scores = [evaluation_results[u][metric] for u in evaluation_results]
        avg = np.mean(scores)
        print(f"{metric}: {avg:.3f}")
else:
    print("\nNo evaluation results were generated. Check GPT output or similarity threshold.")


