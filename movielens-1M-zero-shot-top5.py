import pandas as pd
import numpy as np
import pickle
import os
from collections import defaultdict
import re
import openai

# -----------------------------
# SETUP
# -----------------------------
client = openai.OpenAI(api_key="")
model_name = "gpt-4-turbo"
prompt_type = "zero-shot"

output_dir = os.path.join("experiment_logs", "movielens1m", prompt_type)
os.makedirs(output_dir, exist_ok=True)

log_lines = []
def log(message):
    log_entry = f"{message}"
    print(log_entry)
    log_lines.append(log_entry)

log("EXPERIMENT Movielens 1M: Zero-Shot prompting for 100 users (≥100 ratings each).")

# -----------------------------
# STEP 1: Load 100-User Data
# -----------------------------
log("Loading scale-up user data...")
with open("experiment_logs/movielens1m/data_for_100_users.pkl", "rb") as f:
    data = pickle.load(f)

sampled_users_100 = data["sampled_users_100"]
train_data_100 = data["train_data_100"]
test_data_100 = data["test_data_100"]
movies = data["movies"]

log(f"Loaded {len(sampled_users_100)} users.")

# -----------------------------
# STEP 2: Zero-Shot Prompting (100 users)
# -----------------------------
recommendations = {}
prompt_logs = []

for user_id in sampled_users_100:
    user_movies = train_data_100[train_data_100["userId"] == user_id]["title"].tolist()
    history = ', '.join(user_movies[-50:])

    prompt = (
        "You are an expert movie recommender system.\n"
        "Only recommend movies from the MovieLens 1M dataset. Output the movie title exactly as it appears.\n"
        f"I’ve watched and liked the following movies: {history}.\n"
        "Based on this, recommend 5 movies I might enjoy next. Just list the movie titles."
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
            if line.lower().startswith("movie title:"):
                rec = line.split(":", 1)[-1].strip()
            elif re.match(r"^\d+\.\s", line):
                rec = line.split('.', 1)[-1].strip()
            elif line:
                rec = line
            else:
                continue
            rec = re.sub(r'^[\W\d]+', '', rec).replace('""', '"').strip('"').strip()
            recs.append(rec)

        recommendations[user_id] = recs[:5]
        prompt_logs.append({
            "user_id": user_id,
            "prompt": prompt,
            "response": text,
            "prompt_type": prompt_type,
            "model": model_name
        })
    except Exception as e:
        log(f"Error for user {user_id}: {e}")

# Save prompt logs
df_prompt_log = pd.DataFrame(prompt_logs)
df_prompt_log.to_csv(os.path.join(output_dir, "prompt_log_4.csv"), index=False)
log("Saved prompt logs.")

# Save raw recommendations
df_recs = pd.DataFrame([
    {"user_id": user_id, "rank": i+1, "recommended_title": title}
    for user_id, recs in recommendations.items()
    for i, title in enumerate(recs)
])
df_recs.to_csv(os.path.join(output_dir, "raw_recommendations_4.csv"), index=False)
log("Saved raw recommendations.")

# -----------------------------
# STEP 3: Normalization
# -----------------------------
def normalize(title):
    title = re.sub(r'\(\d{4}\)', '', title)  # remove year
    title = title.lower()
    title = re.sub(r'\b(the|a|an)\b', '', title)
    title = re.sub(r'[^a-z0-9]', '', title)
    return title.strip()

# -----------------------------
# STEP 4: Evaluation
# -----------------------------
def evaluate_exact(recommendations, test_data, movies, verbose=True):
    evaluation_results = defaultdict(dict)
    rows_for_csv = []

    movies["clean_title"] = movies["title"].str.replace(r"\s\(\d{4}\)", "", regex=True)
    known_titles = set(normalize(title) for title in movies["clean_title"])

    for user_id, recs in recommendations.items():
        test_titles = test_data[test_data["userId"] == user_id]["title"].tolist()
        if not recs or not test_titles:
            continue

        test_titles_clean = [normalize(title) for title in test_titles]
        recs_clean = [normalize(title) for title in recs]

        hits = [recs[i] for i in range(len(recs_clean)) if recs_clean[i] in test_titles_clean]
        in_dataset_count = sum([1 for title in recs_clean if title in known_titles])

        evaluation_results[user_id] = {
            "Hit@5": int(len(hits) > 0),
            "Precision@5": len(hits) / 5,
            "Recall@5": len(hits) / len(test_titles),
            "NDCG@5": (
                sum([1 / np.log2(idx + 2) for idx, title in enumerate(recs_clean) if title in test_titles_clean]) /
                sum([1 / np.log2(i + 2) for i in range(min(len(test_titles), 5))])
            ) if min(len(test_titles), 5) > 0 else 0.0,
            "InDataset@5": in_dataset_count / 5
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

        if verbose:
            log(
                f"User {user_id} Evaluation: "
                f"Hit@5: {evaluation_results[user_id]['Hit@5']:.3f}, "
                f"Precision@5: {evaluation_results[user_id]['Precision@5']:.3f}, "
                f"Recall@5: {evaluation_results[user_id]['Recall@5']:.3f}, "
                f"NDCG@5: {evaluation_results[user_id]['NDCG@5']:.3f}, "
                f"InDataset@5: {evaluation_results[user_id]['InDataset@5']:.3f}"
            )

    df_eval = pd.DataFrame(rows_for_csv)
    df_eval.to_csv(os.path.join(output_dir, "recommendation_analysis_4.csv"), index=False)
    log("Saved recommendation analysis.")
    return evaluation_results

# -----------------------------
# STEP 5: Run Evaluation & Save Summary
# -----------------------------
evaluation_results = evaluate_exact(recommendations, test_data_100, movies)

if evaluation_results:
    log("\nAverage Evaluation Metrics (Exact Match - Warm Start):")
    summary = {}
    for metric in ["Hit@5", "Precision@5", "Recall@5", "NDCG@5", "InDataset@5"]:
        scores = [evaluation_results[u][metric] for u in evaluation_results]
        avg = np.mean(scores)
        summary[metric] = avg
        log(f"{metric}: {avg:.3f}")

    df_user_summary = pd.DataFrame.from_dict(evaluation_results, orient="index")
    df_user_summary.index.name = "user_id"
    df_user_summary.reset_index(inplace=True)
    df_user_summary.to_csv(os.path.join(output_dir, "user_level_metrics_4.csv"), index=False)
    log("Saved user-level metrics.")
else:
    log("No evaluation results were generated.")

# -----------------------------
# STEP 6: Save Experiment Log
# -----------------------------
with open(os.path.join(output_dir, "README_experiment_log.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines))
log("Saved experiment log.")



