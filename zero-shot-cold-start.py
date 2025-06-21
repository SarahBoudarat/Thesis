import pandas as pd
import numpy as np
import openai
import pickle
import os
import re
import unicodedata
from collections import defaultdict

# -----------------------------
# SETUP
# -----------------------------
prompt_type = "cold_start_zero_shot"
model_name = "gpt-4"
output_dir = os.path.join("experiments", prompt_type)
os.makedirs(output_dir, exist_ok=True)

client = openai.OpenAI(api_key="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")

# -----------------------------
# NORMALIZATION
# -----------------------------
def normalize(title):
    title = re.sub(r"\(\d{4}\)", '', title)
    title = unicodedata.normalize('NFKD', title).encode('ASCII', 'ignore').decode('utf-8')
    title = title.lower()
    title = re.sub(r'aka.*', '', title)
    title = re.sub(r'\b(the|a|an|le|la|el|los|las|les)\b', '', title)
    title = re.sub(r'[^\w\s]', '', title)
    title = re.sub(r'\s+', ' ', title)
    return title.strip()

# -----------------------------
# LOAD DATA
# -----------------------------
with open("data_for_10_users.pkl", "rb") as f:
    data = pickle.load(f)

sampled_users_10 = data["sampled_users_10"]
train_data_10 = data["train_data_10"]
test_data_10 = data["test_data_10"]
movies = data["movies"]
movies["clean_title"] = movies["title"].apply(normalize)
known_titles = set(movies["clean_title"])

# -----------------------------
# COLD START: TOP 4 HIGH-RATED MOVIES
# -----------------------------
high_rated = train_data_10[train_data_10["rating"] >= 4]
top_k = 4
topk_per_user = high_rated.sort_values(by="rating", ascending=False).groupby("userId").head(top_k)

# -----------------------------
# GPT PROMPTING USING TITLE HISTORY
# -----------------------------
recommendations = {}
prompt_logs = []

for user_id in sampled_users_10:
    top_movies = topk_per_user[topk_per_user["userId"] == user_id]["title"].tolist()
    if not top_movies:
        continue

    history_str = "\n".join([f"- {title}" for title in top_movies])
    prompt = (
    f"I just signed up on this movie platform. "
    f"To get started, here are {len(top_movies)} movies I really liked:{history_str}"
        "Can you recommend 5 movies I might enjoy next?\n"
        "Please list only the movie titles, one per line. No descriptions."
    )

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=200
        )
        text = response.choices[0].message.content.strip()
        recs = [re.sub(r"^[\d\W]+", "", line).strip() for line in text.replace(",", "\n").split("\n") if line.strip()]
        recommendations[user_id] = recs[:5]
        prompt_logs.append({
            "user_id": user_id,
            "prompt": prompt,
            "response": text,
            "prompt_type": prompt_type,
            "model": model_name
        })
    except Exception as e:
        print(f"Error for user {user_id}: {e}")

# Save logs
pd.DataFrame(prompt_logs).to_csv(f"{output_dir}/prompt_log_4.csv", index=False)
pd.DataFrame([
    {"user_id": uid, "rank": i+1, "recommended_title": title}
    for uid, recs in recommendations.items()
    for i, title in enumerate(recs)
]).to_csv(f"{output_dir}/raw_recommendations_4.csv", index=False)

# -----------------------------
# EVALUATION
# -----------------------------
evaluation_results = defaultdict(dict)
rows_for_csv = []

for user_id, recs in recommendations.items():
    test_titles = test_data_10[test_data_10["userId"] == user_id]["title"].tolist()
    test_titles_clean = [normalize(t) for t in test_titles]
    recs_clean = [normalize(r) for r in recs]
    hits = [r for r, rc in zip(recs, recs_clean) if rc in test_titles_clean]
    in_dataset_count = sum([1 for rc in recs_clean if rc in known_titles])

    dcg = sum([1 / np.log2(i + 2) for i, rc in enumerate(recs_clean) if rc in test_titles_clean])
    idcg = sum([1 / np.log2(i + 2) for i in range(min(len(test_titles), 5))])
    ndcg = dcg / idcg if idcg > 0 else 0

    evaluation_results[user_id] = {
        "Hit@5": int(len(hits) > 0),
        "Precision@5": len(hits) / 5,
        "Recall@5": len(hits) / len(test_titles) if test_titles else 0,
        "NDCG@5": ndcg,
        "InDataset@5": in_dataset_count / 5
    }

    print(f"\nUser {user_id} test titles:")
    print(test_titles)
    print(f"Recommendations: {recs}")

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
pd.DataFrame.from_dict(evaluation_results, orient="index").reset_index().rename(columns={"index": "user_id"}).to_csv(f"{output_dir}/user_level_metrics_4.csv", index=False)
pd.DataFrame(rows_for_csv).to_csv(f"{output_dir}/recommendation_analysis_4.csv", index=False)

# Print summary
print("\nAverage Evaluation Metrics (Cold Start Zero-Shot):")
for metric in ["Hit@5", "Precision@5", "Recall@5", "NDCG@5", "InDataset@5"]:
    scores = [res[metric] for res in evaluation_results.values()]
    print(f"{metric}: {np.mean(scores):.3f}")

