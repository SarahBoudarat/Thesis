# cot_cold_start_scaleup.py

import pandas as pd
import numpy as np
import pickle
import os
import re
from collections import defaultdict
import openai


# setup

client = openai.OpenAI(api_key=)
model_name = "gpt-4-turbo"
prompt_type = "cold_start_chain_of_thought"
output_dir = os.path.join("experiment_logs", "scaleup", prompt_type)
os.makedirs(output_dir, exist_ok=True)

# data load

with open("experiment_logs/scaleup/data_for_100_users.pkl", "rb") as f:
    data = pickle.load(f)

sampled_users_100 = data["sampled_users_100"]
train_data_100 = data["train_data_100"]
test_data_100 = data["test_data_100"]
movies = data["movies"]


# normalization function

def normalize(title):
    title = re.sub(r'\(\d{4}\)', '', title)
    title = title.lower()
    title = re.sub(r'\b(the|a|an)\b', '', title)
    title = re.sub(r'[^a-z0-9]', '', title)
    return title.strip()

movies["clean_title"] = movies["title"].str.replace(r"\s\(\d{4}\)", "", regex=True)
known_titles = set(movies["clean_title"].apply(normalize))


# SELECT TOP 3 HIGHLY RATED MOVIES PER USER (COLD START)

top_k = 3
high_rated = train_data_100[train_data_100["rating"] >= 4]
topk_per_user = high_rated.sort_values(by=["userId", "rating"], ascending=[True, False]).groupby("userId").head(top_k)
top_movies_per_user = topk_per_user.groupby("userId")["title"].apply(list).reset_index()


# BUILD CHAIN-OF-THOUGHT PROMPT

def construct_cot_prompt(user_movies, user_id):
    liked_lines = "\n".join(f"- {title}" for title in user_movies)
    prompt = (
        f"You are an expert movie recommender system.\n"
        f"Only recommend movies from the MovieLens 100K dataset. Output each title exactly as it appears.\n\n"
        f"User's Movie History:\n{liked_lines}\n\n"
        f"Chain of Thought or Reasoning:\n"
        f"Let's think step by step. Analyze the user's preferences based on genre, tone, and themes.\n"
        f"Then recommend 5 suitable movies.\n\n"
        f"Recommendation Output Format:\n"
        f"Movie Title: <Title 1>\n"
        f"Movie Title: <Title 2>\n"
        f"..."
    )
    import tiktoken
    enc = tiktoken.encoding_for_model(model_name)
    print(f"Token count for user {user_id}: {len(enc.encode(prompt))}")

    return prompt



# gpt 
recommendations = {}
prompt_logs = []

for _, row in top_movies_per_user.iterrows():
    user_id = row["userId"]
    user_movies = row["title"]
    prompt = construct_cot_prompt(user_movies, user_id)
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500
        )
        text = response.choices[0].message.content.strip()
        recs = [
                   line[len("Movie Title:"):].strip()
                   for line in text.splitlines()
                   if line.lower().startswith("movie title:")
               ][:5]
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

# save prompt logs
pd.DataFrame(prompt_logs).to_csv(f"{output_dir}/prompt_log_100.csv", index=False)

# Save raw recommendations
pd.DataFrame([
    {"user_id": user_id, "rank": i+1, "recommended_title": title}
    for user_id, recs in recommendations.items()
    for i, title in enumerate(recs)
]).to_csv(f"{output_dir}/raw_recommendations_100.csv", index=False)

# Evaluate

recommendation_analysis = []
evaluation_results = defaultdict(dict)

for user_id, recs in recommendations.items():
    test_titles = test_data_100[test_data_100["userId"] == user_id]["title"].tolist()
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

# save evaluation logs/ I have to check again the logs
pd.DataFrame.from_dict(evaluation_results, orient="index").reset_index().rename(columns={"index": "user_id"}).to_csv(f"{output_dir}/user_level_metrics_100.csv", index=False)
pd.DataFrame(recommendation_analysis).to_csv(f"{output_dir}/recommendation_analysis_100.csv", index=False)

# print summary
print("\nAverage Evaluation Metrics (CoT Cold Start - Scale Up):")
for metric in ["Hit@5", "Precision@5", "Recall@5", "NDCG@5", "InDataset@5"]:
    scores = [res[metric] for res in evaluation_results.values()]
    print(f"{metric}: {np.mean(scores):.3f}")

print("Saved all logs and experiment results.")
