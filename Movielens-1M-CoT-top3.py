# Chain-of-Thought Prompting (Top-3 Evaluation)

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
prompt_type = "chain_of_thought-top3"
model_name = "gpt-4-turbo"
output_dir = os.path.join("experiment_logs", "movielens1m", prompt_type)
os.makedirs(output_dir, exist_ok=True)

client = openai.OpenAI(api_key="")

log_lines = []
def log(message):
    print(message)
    log_lines.append(message)

log("EXPERIMENT MovieLens 1M: Chain-of-Thought prompting (Top-3) for 100 users (≥100 ratings each).")

# -----------------------------
# STEP 1: Load Data
# -----------------------------
with open("experiment_logs/movielens1m/data_for_100_users.pkl", "rb") as f:
    data = pickle.load(f)

sampled_users_100 = data["sampled_users_100"]
train_data_100 = data["train_data_100"]
test_data_100 = data["test_data_100"]
movies = data["movies"]

log(f"Loaded {len(sampled_users_100)} users.")

# -----------------------------
# STEP 2: CLEANING & NORMALIZATION
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

movies["clean_title"] = movies["title"].str.replace(r"\(\d{4}\)", "", regex=True).str.strip()
known_titles = set(movies["clean_title"])

# -----------------------------
# STEP 3: PROMPTING
# -----------------------------
def parse_recommendations(text):
    return [
        line[len("Movie Title:"):].strip()
        for line in text.splitlines()
        if line.lower().startswith("movie title:")
    ][:3]


prompt_logs = []
recommendations = {}


for user_id in sampled_users_100:
    top_movies = (
        train_data_100[train_data_100["userId"] == user_id]
        .sort_values(by="timestamp")
        ["title"].tolist()[-50:]
    )

    if len(top_movies) < 2:
        continue

    history_str = "\n".join([f"- {title}" for title in top_movies])

    prompt_text = (
        f"You are an expert movie recommender system.\n"
        f"Only recommend movies from the MovieLens 1M dataset. Output each title exactly as it appears.\n\n"
        f"User's Movie History:\n{history_str}\n\n"
        f"Chain of Thought or Reasoning:\n"
        f"Let's think step by step. Analyze the user's preferences based on genre, tone, and themes.\n"
        f"Then recommend 3 suitable movies.\n\n"
        f"Recommendation Output Format:\n"
        f"Movie Title: <Title 1>\n"
        f"Movie Title: <Title 2>\n"
        f"Movie Title: <Title 3>"
    )

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt_text}],
            temperature=0.7,
            max_tokens=500
        )
        text = response.choices[0].message.content.strip()
        recs = parse_recommendations(text)
        recommendations[user_id] = recs[:3]
        prompt_logs.append({
            "user_id": user_id,
            "prompt": prompt_text,
            "response": text,
            "prompt_type": prompt_type,
            "model": model_name
        })
    except Exception as e:
        log(f"Error for user {user_id}: {e}")

# Save prompts & raw recommendations
pd.DataFrame(prompt_logs).to_csv(os.path.join(output_dir, "prompt_log_4_top3.csv"), index=False)
log("Saved prompt logs.")

pd.DataFrame([
    {"user_id": uid, "rank": i+1, "recommended_title": title}
    for uid, recs in recommendations.items()
    for i, title in enumerate(recs)
]).to_csv(os.path.join(output_dir, "raw_recommendations_4_top3.csv"), index=False)
log("Saved raw recommendations.")

# -----------------------------
# STEP 4: EVALUATION
# -----------------------------
evaluation_results = defaultdict(dict)
rows_for_csv = []

for user_id, recs in recommendations.items():
    test_titles = test_data_100[test_data_100["userId"] == user_id]["title"].tolist()
    test_titles_clean = [normalize(t) for t in test_titles]
    recs_clean = [normalize(r) for r in recs]
    hits = [r for r, rc in zip(recs, recs_clean) if rc in test_titles_clean]
    in_dataset_count = sum([1 for rc in recs_clean if rc in [normalize(t) for t in known_titles]])

    dcg = sum([1 / np.log2(i + 2) for i, rc in enumerate(recs_clean) if rc in test_titles_clean])
    idcg = sum([1 / np.log2(i + 2) for i in range(min(len(test_titles), 3))])
    ndcg = dcg / idcg if idcg > 0 else 0

    evaluation_results[user_id] = {
        "Hit@3": int(len(hits) > 0),
        "Precision@3": len(hits) / 3,
        "Recall@3": len(hits) / len(test_titles) if test_titles else 0,
        "NDCG@3": ndcg,
        "InDataset@3": in_dataset_count / 3
    }

    for i, raw_title in enumerate(recs):
        norm_title = recs_clean[i]
        label = (
            "match" if norm_title in test_titles_clean else
            "in_dataset" if norm_title in [normalize(t) for t in known_titles] else
            "not_in_dataset"
        )
        rows_for_csv.append({
            "user_id": user_id,
            "rank": i + 1,
            "original_title": raw_title,
            "normalized_title": norm_title,
            "label": label,
            "in_test_set": norm_title in test_titles_clean,
            "in_movie_lens": norm_title in [normalize(t) for t in known_titles],
            "test_titles_clean": "; ".join(test_titles_clean)
        })

# Save evaluation results
pd.DataFrame.from_dict(evaluation_results, orient="index").reset_index().rename(columns={"index": "user_id"}).to_csv(os.path.join(output_dir, "user_level_metrics_4_top3.csv"), index=False)
pd.DataFrame(rows_for_csv).to_csv(os.path.join(output_dir, "recommendation_analysis_4_top3.csv"), index=False)
log("Saved evaluation metrics and analysis.")

# Print overall summary
print("\nAverage Evaluation Metrics (Chain-of-Thought - Top-3):")
for metric in ["Hit@3", "Precision@3", "Recall@3", "NDCG@3", "InDataset@3"]:
    scores = [res[metric] for res in evaluation_results.values()]
    print(f"{metric}: {np.mean(scores):.3f}")

# Save experiment log
with open(os.path.join(output_dir, "README_experiment_log.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines))
log("Saved experiment log.")
