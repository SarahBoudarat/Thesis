# Few-Shot Prompting with Static Examples (Top-3 Evaluation)

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
client = openai.OpenAI(api_key="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")
model_name = "gpt-4-turbo"
prompt_type = "few-shot-top3"

output_dir = os.path.join("experiment_logs", "scaleup", prompt_type)
os.makedirs(output_dir, exist_ok=True)

log_lines = []
def log(message):
    print(message)
    log_lines.append(message)

log("EXPERIMENT SCALE-UP: Few-Shot prompting with static examples for 100 users (≥50 ratings each).")

# -----------------------------
# STEP 1: Load 100-User Data
# -----------------------------
with open("experiment_logs/scaleup/data_for_100_users.pkl", "rb") as f:
    data = pickle.load(f)

sampled_users_100 = data["sampled_users_100"]
train_data_100 = data["train_data_100"]
test_data_100 = data["test_data_100"]
movies = data["movies"]

log(f"Loaded {len(sampled_users_100)} users.")

# -----------------------------
# STEP 2: Static Few-Shot Prompting
# -----------------------------
recommendations = {}
prompt_logs = []

few_shot_example = (
    "You are an expert movie recommender system.\n"
    "Only recommend movies from the MovieLens 100K dataset. Output the movie title exactly as it appears.\n"
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

for user_id in sampled_users_100:
    user_movies = train_data_100[train_data_100["userId"] == user_id].sort_values("timestamp", ascending=False)["title"].tolist()
    history = ', '.join(user_movies[:50])

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
        log(f"Error for user {user_id}: {e}")

# Save prompt logs
pd.DataFrame(prompt_logs).to_csv(os.path.join(output_dir, "prompt_log_4_top3.csv"), index=False)
log("Saved prompt logs.")

# Save raw recommendations
pd.DataFrame([
    {"user_id": user_id, "rank": i+1, "recommended_title": title}
    for user_id, recs in recommendations.items()
    for i, title in enumerate(recs)
]).to_csv(os.path.join(output_dir, "raw_recommendations_4_top3.csv"), index=False)
log("Saved raw recommendations.")

# -----------------------------
# STEP 3: Normalization
# -----------------------------
def normalize(title):
    title = re.sub(r'\(\d{4}\)', '', title)
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
            "Hit@3": int(len(hits) > 0),
            "Precision@3": len(hits) / 3,
            "Recall@3": len(hits) / len(test_titles),
            "NDCG@3": (
                sum([1 / np.log2(idx + 2) for idx, title in enumerate(recs_clean) if title in test_titles_clean]) /
                sum([1 / np.log2(i + 2) for i in range(min(len(test_titles), 3))])
            ) if min(len(test_titles), 3) > 0 else 0.0,
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

        if verbose:
            log(
                f"User {user_id} Evaluation: "
                f"Hit@3: {evaluation_results[user_id]['Hit@3']:.3f}, "
                f"Precision@3: {evaluation_results[user_id]['Precision@3']:.3f}, "
                f"Recall@3: {evaluation_results[user_id]['Recall@3']:.3f}, "
                f"NDCG@3: {evaluation_results[user_id]['NDCG@3']:.3f}, "
                f"InDataset@3: {evaluation_results[user_id]['InDataset@3']:.3f}"
            )

    df_eval = pd.DataFrame(rows_for_csv)
    df_eval.to_csv(os.path.join(output_dir, "recommendation_analysis_4_top3.csv"), index=False)
    log("Saved recommendation analysis.")
    return evaluation_results

# -----------------------------
# STEP 5: Run Evaluation & Save Summary
# -----------------------------
evaluation_results = evaluate_exact(recommendations, test_data_100, movies)

if evaluation_results:
    log("\nAverage Evaluation Metrics (Few-Shot - Static - Top-3):")
    summary = {}
    for metric in ["Hit@3", "Precision@3", "Recall@3", "NDCG@3", "InDataset@3"]:
        scores = [evaluation_results[u][metric] for u in evaluation_results]
        avg = np.mean(scores)
        summary[metric] = avg
        log(f"{metric}: {avg:.3f}")

    df_user_summary = pd.DataFrame.from_dict(evaluation_results, orient="index")
    df_user_summary.index.name = "user_id"
    df_user_summary.reset_index(inplace=True)
    df_user_summary.to_csv(os.path.join(output_dir, "user_level_metrics_4_top3.csv"), index=False)
    log("Saved user-level metrics.")
else:
    log("No evaluation results were generated.")

# -----------------------------
# STEP 6: Save Experiment Log
# -----------------------------
with open(os.path.join(output_dir, "README_experiment_log.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines))
log("Saved experiment log.")