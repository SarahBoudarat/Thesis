import pandas as pd
import numpy as np
import pickle
import os
from collections import defaultdict
import re
import openai
from sentence_transformers import SentenceTransformer

# -----------------------------
# SETUP
# -----------------------------
client = openai.OpenAI(api_key="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")
model = SentenceTransformer("all-MiniLM-L6-v2")
model_name = "gpt-4"
prompt_type = "few_shot"
output_dir = os.path.join("experiments", prompt_type)
os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# STEP 1: Load Data
# -----------------------------
with open("data_for_10_users.pkl", "rb") as f:
    data = pickle.load(f)

sampled_users_10 = data["sampled_users_10"]
train_data_10 = data["train_data_10"]
test_data_10 = data["test_data_10"]
movies = data["movies"]

# -----------------------------
# STEP 2: Few-Shot Prompting
# -----------------------------
recommendations = {}
prompt_logs = []

few_shot_example = (
    "Example:\n"
    "I have watched and liked the following movies: The Matrix, Inception, Fight Club.\n"
    "Based on this, recommend 5 movies I might enjoy next. Just list the movie titles.\n"
    "→ The Dark Knight, Memento, Interstellar, Se7en, The Prestige\n\n"
    "Example 2:\n"
    "I have watched and liked the following movies: Titanic, The Notebook, La La Land.\n"
    "Based on this, recommend 5 movies I might enjoy next. Just list the movie titles.\n"
    "→ A Walk to Remember, Me Before You, The Fault in Our Stars, About Time, Eternal Sunshine of the Spotless Mind\n\n"
)

for user_id in sampled_users_10:
    user_movies = train_data_10[train_data_10["userId"] == user_id]["title"].tolist()
    history = ', '.join(user_movies[-10:])

    prompt = (
        f"{few_shot_example}"
        f"Now for me:\n"
        f"I have watched and liked the following movies: {history}.\n"
        "Based on this, recommend 5 movies I might enjoy next. Just list the movie titles."
    )

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=200
        )

        text = response.choices[0].message.content.strip()
        print(f"\n==== USER {user_id} ====")
        print("Prompt:\n", prompt)
        print("\nLLM Output:\n", text)
        recs = [
            re.sub(r'^[\W\d]+', '', line).replace('""', '"').strip('"').strip()
            for line in text.replace(",", "\n").split("\n") if line.strip()
        ]
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

# Save raw recommendations
df_recs = pd.DataFrame([
    {"user_id": user_id, "rank": i+1, "recommended_title": title}
    for user_id, recs in recommendations.items()
    for i, title in enumerate(recs)
])
df_recs.to_csv(f"{output_dir}/raw_recommendations_4.csv", index=False)

# Save prompt logs
df_prompt_log = pd.DataFrame(prompt_logs)
df_prompt_log.to_csv(f"{output_dir}/prompt_log_4.csv", index=False)

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
            "Hit@5": int(len(hits) > 0),
            "Precision@5": len(hits) / 5,
            "Recall@5": len(hits) / len(test_titles),
            "NDCG@5": (
                sum([1 / np.log2(idx + 2) for idx, title in enumerate(recs_clean) if title in test_titles_clean]) /
                sum([1 / np.log2(i + 2) for i in range(min(len(test_titles), 5))])
            ),
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
            print(f"\nUser {user_id} Evaluation:")
            for k, v in evaluation_results[user_id].items():
                print(f"  {k}: {v:.3f}")

    df_eval = pd.DataFrame(rows_for_csv)
    df_eval.to_csv(f"{output_dir}/recommendation_analysis_4.csv", index=False)

    return evaluation_results

# -----------------------------
# STEP 5: Run Evaluation & Save Summary
# -----------------------------
evaluation_results = evaluate_exact(recommendations, test_data_10, movies)

if evaluation_results:
    print("\nAverage Evaluation Metrics (Few-Shot - Exact Match):")
    summary = {}
    for metric in ["Hit@5", "Precision@5", "Recall@5", "NDCG@5", "InDataset@5"]:
        scores = [evaluation_results[u][metric] for u in evaluation_results]
        avg = np.mean(scores)
        summary[metric] = avg
        print(f"{metric}: {avg:.3f}")

    df_user_summary = pd.DataFrame.from_dict(evaluation_results, orient="index")
    df_user_summary.index.name = "user_id"
    df_user_summary.reset_index(inplace=True)
    df_user_summary.to_csv(f"{output_dir}/user_level_metrics_4.csv", index=False)
else:
    print("\nNo evaluation results were generated.")
