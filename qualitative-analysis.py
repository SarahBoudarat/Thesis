import pandas as pd

# === Step 1: Set file paths ===
cot_path = r"C:\Users\sarah.boudarat\PycharmProjects\Thesis\.venv\experiment_logs\scaleup\chain_of_thought\prompt_log_4.csv"
few_path = r"C:\Users\sarah.boudarat\PycharmProjects\Thesis\.venv\experiment_logs\scaleup\few-shot\prompt_log_4.csv"
zero_path = r"C:\Users\sarah.boudarat\PycharmProjects\Thesis\.venv\experiment_logs\scaleup\zero-shot\prompt_log_4.csv"

# === Step 2: Load all prompt logs ===
try:
    df_cot = pd.read_csv(cot_path)
    df_few = pd.read_csv(few_path)
    df_zero = pd.read_csv(zero_path)
except FileNotFoundError as e:
    print(f" File not found: {e}")
    exit()


# === Step 3: Define function to get prompt and response for a user ===
def extract_prompt_response(df, user_id):
    row = df[df['user_id'] == user_id]
    if row.empty:
        return ("Not found", "Not found")

    prompt = row.iloc[0]['prompt']
    response = row.iloc[0].get('response') or row.iloc[0].get('output', 'Not found')
    return (prompt, response)


# === Step 4: Choose user and extract data ===
user_id = 424

cot_prompt, cot_response = extract_prompt_response(df_cot, user_id)
few_prompt, few_response = extract_prompt_response(df_few, user_id)
zero_prompt, zero_response = extract_prompt_response(df_zero, user_id)

# === Step 5: Display all three outputs ===
print("=" * 100)
print(f" User {user_id} — Zero-Shot Prompt\n\n{zero_prompt}\n")
print(f" Zero-Shot Response:\n{zero_response}")
print("=" * 100)
print(f" User {user_id} — Few-Shot Prompt\n\n{few_prompt}\n")
print(f" Few-Shot Response:\n{few_response}")
print("=" * 100)
print(f" User {user_id} — Chain-of-Thought Prompt\n\n{cot_prompt}\n")
print(f" CoT Response:\n{cot_response}")
print("=" * 100)
