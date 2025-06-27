import pandas as pd
import numpy as np
import pickle
import os
import re
from collections import defaultdict
import openai

# -----------------------------
# SETUP
# -----------------------------
client = openai.OpenAI(api_key="")
model_name = "gpt-4-turbo"
prompt_type = "cold_start_few_shot_top3"
output_dir = os.path.join("experiment_logs", "movielens1m", prompt_type)
os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# LOAD DATA
# -----------------------------
with open("experiment_logs/movielens1m/data_for_100_users.pkl", "rb") as f:
    data = pickle.load(f)

sampled_users_X = data["sampled_users_100"]
train_data_X = data["train_data_100"]
test_data_X = data["test_data_100"]
movies = data["movies"]

# -----------------------------
# NORMALIZE FUNCTION
# -----------------------------
def normalize(title):
    title = re.sub(r'\(\d{4}\)', '', title)
    title = title.lower()
    title = re.sub(r'\b(the|a|an)\b', '', title)
    title = re.sub(r'[^a-z0-9]', '', title)
    return title.strip()

movies["clean_title"] = movies["title"].apply(normalize)
known_titles = set(movies["clean_title"])

# -----------------------------
# FEW-SHOT EXAMPLES
# -----------------------------
few_shot_example = (
    "You are an expert movie recommender system.\n"
    "Only recommend movies from the MovieLens 1M dataset. Output the movie title exactly as it appears.\n"
    "\n"
    "Example 1:\n"
    "I have watched and liked the following movies: The Matrix, Inception, Fight Club.\n"
    "Based on this, recommend 3 movies I might enjoy next. Just list the movie titles.\n"
    "→ The Dark Knight, Memento, Interstellar\n\n"
    "Example 2:\n"
    "I have watched and liked the following movies: Titanic, The Notebook, La La Land.\n"
    "Based on this, recommend 3 movies I might enjoy next. Just list the movie titles.\n"
    "→ A Walk to Remember, Me Before You, The Fault in Our Stars\n\n"
)
# -----------------------------
# COLD START: TOP-3 HIGH-RATED MOVIES
# -----------------------------
top_k = 3
high_rated = train_data_X[train_data_X["rating"] >= 4]
topk_per_user = high_rated.sort_values(by="rating", ascending=False).groupby("userId").head(top_k)

recommendations = {}
prompt_logs = []

for user_id in sampled_users_X:
    user_movies = topk_per_user[topk_per_user["userId"] == user_id]["title"].tolist()
    if not user_movies:
        continue
    history = ', '.join(user_movies)
    prompt = (
        f"{few_shot_example}"
        f"Now for me:\n"
        f"I have watched and liked the following movies: {history}.\n"
        f"Recommend 3 movies I might enjoy. Just list the movie titles."
    )
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=300
        )
        text = response.choices[0].message.content.strip()
        recs = []
        for line in text.splitlines():
            line = line.strip()
            # If a line contains several movies, split by comma
            if ',' in line and not line.lower().startswith("movie title:"):
                for part in line.split(','):
                    part = part.strip()
                    if not part:
                        continue
                    if part.lower().startswith("movie title:"):
                        rec = part.split(":", 1)[-1].strip()
                    elif re.match(r"^\d+\.\s*", part):
                        rec = part.split('.', 1)[-1].strip()
                    elif re.match(r"^- ", part):
                        rec = part[2:].strip()
                    else:
                        rec = part
                    rec = re.sub(r'^[\W\d]+', '', rec).replace('""', '"').strip('"').strip()
                    recs.append(rec)
            else:
                if line.lower().startswith("movie title:"):
                    rec = line.split(":", 1)[-1].strip()
                elif re.match(r"^\d+\.\s*", line):
                    rec = line.split('.', 1)[-1].strip()
                elif re.match(r"^- ", line):
                    rec = line[2:].strip()
                elif line:
                    rec = line
                else:
                    continue
                rec = re.sub(r'^[\W\d]+', '', rec).replace('""', '"').strip('"').strip()
                recs.append(rec)
        recommendations[user_id] = recs[:3]
        prompt_logs.append({
            "user_id": user_id,
            "prompt": prompt,
            "response": text,
            "prompt_type": prompt_type,
            "model": model_name
        })
    except Exception as e:
        print(f"Error for user {user_id}: {e}")

# -----------------------------
# SAVE PROMPT LOGS & RAW RECS
# -----------------------------
pd.DataFrame(prompt_logs).to_csv(f"{output_dir}/prompt_log_4_top3.csv", index=False)
pd.DataFrame([
    {"user_id": user_id, "rank": i + 1, "recommended_title": title}
    for user_id, recs in recommendations.items()
    for i, title in enumerate(recs)
]).to_csv(f"{output_dir}/raw_recommendations_4_top3.csv", index=False)

# -----------------------------
# EVALUATION
# -----------------------------
evaluation_results = defaultdict(dict)
rows_for_csv = []

for user_id, recs in recommendations.items():
    test_titles = test_data_X[test_data_X["userId"] == user_id]["title"].tolist()
    if not recs or not test_titles:
        continue
    test_titles_clean = [normalize(title) for title in test_titles]
    recs_clean = [normalize(title) for title in recs]
    hits = [r for r, rc in zip(recs, recs_clean) if rc in test_titles_clean]
    in_dataset_count = sum([1 for title in recs_clean if title in known_titles])

    dcg = sum([1 / np.log2(i + 2) for i, rc in enumerate(recs_clean) if rc in test_titles_clean])
    idcg = sum([1 / np.log2(i + 2) for i in range(min(len(test_titles), 3))])
    ndcg = dcg / idcg if idcg > 0 else 0

    evaluation_results[user_id] = {
        "Hit@3": int(len(hits) > 0),
        "Precision@3": len(hits) / 3,
        "Recall@3": len(hits) / len(test_titles),
        "NDCG@3": ndcg,
        "InDataset@3": in_dataset_count / 3
    }

    for i, raw_title in enumerate(recs):
        norm_title = recs_clean[i]
        label = (
            "match" if norm_title in test_titles_clean else
            "in_dataset" if norm_title in known_titles else
            "not_in_dataset"
        )
        rows_for_csv.append({
            "user_id": user_id,
            "rank": i + 1,
            "original_title": raw_title,
            "normalized_title": norm_title,
            "label": label,
            "in_test_set": norm_title in test_titles_clean,
            "in_movie_lens": norm_title in known_titles,
            "test_titles_clean": "; ".join(test_titles_clean)
        })

# Save evaluation results
pd.DataFrame.from_dict(evaluation_results, orient="index").reset_index().rename(columns={"index": "user_id"}).to_csv(f"{output_dir}/user_level_metrics_4_top3.csv", index=False)
pd.DataFrame(rows_for_csv).to_csv(f"{output_dir}/recommendation_analysis_4_top3.csv", index=False)

# Print average metrics
print("\nAverage Evaluation Metrics (Cold-Start Few-Shot - Top-3):")
for metric in ["Hit@3", "Precision@3", "Recall@3", "NDCG@3", "InDataset@3"]:
    scores = [res[metric] for res in evaluation_results.values()]
    print(f"{metric}: {np.mean(scores):.3f}")
