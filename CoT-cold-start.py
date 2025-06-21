# cot_cold_start.py
import pandas as pd
import numpy as np
import pickle
import os
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util
import openai
import re

# -----------------------------
# SETUP
# -----------------------------
client = openai.OpenAI(api_key="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")
model = SentenceTransformer("all-MiniLM-L6-v2")
output_dir = os.path.join("experiments", "cold_start_chain_of_thought")
os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# FUNCTION: Load Data
# -----------------------------
def load_data(pkl_path="data_for_10_users.pkl"):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data

# -----------------------------
# FUNCTION: Get Top Liked Movies
# -----------------------------
def get_top_liked_movies(train_data, top_n=3):
    top_movies = (
        train_data[train_data["rating"] >= 4]
        .sort_values(by=["userId", "rating"], ascending=[True, False])
        .groupby("userId").head(top_n)
    )
    return top_movies.groupby("userId")["title"].apply(list).reset_index()

# -----------------------------
# FUNCTION: Generate CoT Prompt
# -----------------------------
def construct_cot_prompt(user_movies):
    exemplar = (
        "User A liked:\n"
        "1. The Matrix\n"
        "2. Inception\n"
        "3. Blade Runner\n\n"
        "They enjoy mind-bending sci-fi with action and strong visual style.\n"
        "Recommended movies:\n"
        "1. Interstellar\n"
        "2. Minority Report\n"
        "3. Source Code\n"
        "4. The Prestige\n"
        "5. Looper\n\n"
        "User B liked:\n"
        "1. Notting Hill\n"
        "2. Love Actually\n"
        "3. Crazy Rich Asians\n\n"
        "They enjoy romantic comedies with emotional warmth and humor.\n"
        "Recommended movies:\n"
        "1. About Time\n"
        "2. The Proposal\n"
        "3. The Holiday\n"
        "4. La La Land\n"
        "5. The Big Sick\n\n"
    )
    liked_lines = "\n".join(f"{i+1}. {title}" for i, title in enumerate(user_movies))
    user_prompt = (
        f"User C liked:\n"
        f"{liked_lines}\n\n"
        "Please recommend 5 movies they might enjoy. Only list the movie titles."
    )
    return exemplar + user_prompt

# -----------------------------
# FUNCTION: Get GPT Recommendations (CoT-style)
# -----------------------------
def get_cot_recommendations(user_examples):
    recommendations = {}
    prompt_logs = []
    for _, row in user_examples.iterrows():
        user_id = row["userId"]
        prompt = construct_cot_prompt(row["title"])

        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=200
            )
            text = response.choices[0].message.content.strip()
            recs = [re.sub(r"^[\W\d]+", "", line).strip() for line in text.replace(",", "\n").split("\n") if line.strip() and not line.lower().startswith("they")]
            recommendations[user_id] = recs[:5]
            prompt_logs.append({
                "user_id": user_id,
                "prompt": prompt,
                "response": text,
                "prompt_type": "chain_of_thought",
                "model": "gpt-3.5-turbo"
            })
        except Exception as e:
            print(f"Error for user {user_id}: {e}")
    return recommendations, prompt_logs

# -----------------------------
# MAIN SCRIPT
# -----------------------------
data = load_data()
top_liked = get_top_liked_movies(data["train_data_10"], top_n=3)
recommendations, prompt_logs = get_cot_recommendations(top_liked)

# Save prompt logs
pd.DataFrame(prompt_logs).to_csv(f"{output_dir}/prompt_log_4.csv", index=False)

# Save raw recommendations
pd.DataFrame([
    {"user_id": user_id, "rank": i+1, "recommended_title": title}
    for user_id, recs in recommendations.items()
    for i, title in enumerate(recs)
]).to_csv(f"{output_dir}/raw_recommendations_4.csv", index=False)

# -----------------------------
# EVALUATION
# -----------------------------
def normalize(title):
    title = re.sub(r'\(\d{4}\)', '', title)
    title = title.lower()
    title = re.sub(r'\b(the|a|an)\b', '', title)
    title = re.sub(r'[^a-z0-9]', '', title)
    return title.strip()

movies = data["movies"]
movies["clean_title"] = movies["title"].str.replace(r"\s\(\d{4}\)", "", regex=True)
known_titles = set(movies["clean_title"].apply(normalize))

recommendation_analysis = []
evaluation_results = defaultdict(dict)

for user_id, recs in recommendations.items():
    test_titles = data["test_data_10"][data["test_data_10"]["userId"] == user_id]["title"].tolist()
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

    for i, raw_title in enumerate(recs):
        norm_title = recs_clean[i]
        label = (
            "match" if norm_title in test_titles_clean else
            "in_dataset" if norm_title in known_titles else
            "not_in_dataset"
        )
        recommendation_analysis.append({
            "user_id": user_id,
            "rank": i + 1,
            "original_title": raw_title,
            "normalized_title": norm_title,
            "label": label,
            "in_test_set": norm_title in test_titles_clean,
            "in_movie_lens": norm_title in known_titles,
            "test_titles_clean": "; ".join(test_titles_clean)
        })

# Save evaluation logs
pd.DataFrame.from_dict(evaluation_results, orient="index").reset_index().rename(columns={"index": "user_id"}).to_csv(f"{output_dir}/user_level_metrics_4.csv", index=False)
pd.DataFrame(recommendation_analysis).to_csv(f"{output_dir}/recommendation_analysis_4.csv", index=False)

# Summary
print("\nAverage Evaluation Metrics (CoT Cold Start):")
for metric in ["Hit@5", "Precision@5", "Recall@5", "NDCG@5", "InDataset@5"]:
    scores = [res[metric] for res in evaluation_results.values()]
    print(f"{metric}: {np.mean(scores):.3f}")
