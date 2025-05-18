import re
import pandas as pd
import os
from ollama import Client
from tqdm import tqdm

# === Config ===
CSV_PATH = 'vqa_output.csv'
OUTPUT_CSV = 'vqa_qa_cleaned.csv'
DESCRIPTION_COLUMN = 'vqa_description'
MODEL_NAME = 'llama3.2'
START_INDEX = 16907
MAX_SAMPLES = 28977 - START_INDEX

MAX_RETRIES = 3  # Number of retries if QA generation fails

qa_columns = ['q1', 'a1', 'q2', 'a2', 'q3', 'a3', 'q4', 'a4']

# === Initialize Client ===
ollamaClient = Client("http://localhost:11434")

# === QA Generation Function ===
def generate_qa_from_description(description):
    prompt = (
        f"Based on the following description of a product image, generate 2 to 4 full-statement questions with one-word answers. "
        f"The questions should be based on visual elements that can be asked looking at the image. "
        f"The output should be in this format:\n"
        f"Q1: <question> | A1: <answer>\nQ2: ... | A2: ...\n\n"
        f"Description:\n{description}"
    )
    try:
        response = ollamaClient.chat(
            model=MODEL_NAME,
            messages=[{'role': 'user', 'content': prompt}]
        )
        return response['message']['content'].strip()
    except Exception as e:
        print(f"❌ Error: {e}")
        return ""

# === Cleaning & Extraction ===
def clean_and_parse_qa(text):
    text = re.sub(r"(?i)^.*?(Q1:|1\.|•)", r"Q1:", text, flags=re.DOTALL)
    text = re.sub(r"\s+Q", "\nQ", text)
    text = re.sub(r"\n{2,}", "\n", text).strip()

    pattern = r"Q\d+:\s*(.*?)\s*\|\s*A\d+:\s*(\w+)"
    matches = re.findall(pattern, text)

    flat = []
    for q, a in matches[:4]:
        flat.extend([q.strip(), a.strip()])
    while len(flat) < 8:
        flat.extend(["", ""])
    return flat

def is_valid_qa(cleaned):
    return any(q.strip() != "" and a.strip() != "" for q, a in zip(cleaned[::2], cleaned[1::2]))

# === Load CSVs ===
df = pd.read_csv(CSV_PATH)
end_index = START_INDEX + MAX_SAMPLES
df_subset = df.iloc[START_INDEX:end_index]

# === Check already processed rows ===
processed_indices = set()
if os.path.exists(OUTPUT_CSV):
    df_out = pd.read_csv(OUTPUT_CSV)
    processed_indices = set(df_out.index)

# === Open output file in append mode ===
header_written = os.path.exists(OUTPUT_CSV)
with open(OUTPUT_CSV, 'a', encoding='utf-8', newline='') as f_out:
    if not header_written:
        header = list(df.columns) + qa_columns
        f_out.write(",".join(header) + "\n")

    for global_idx, row in tqdm(df_subset.iterrows(), total=len(df_subset), desc="Generating QA"):
        if global_idx in processed_indices:
            continue

        desc = row[DESCRIPTION_COLUMN]
        if pd.isna(desc) or desc.strip() == "":
            output_row = list(row.values) + [""] * 8
        else:
            cleaned = [""] * 8
            for attempt in range(1, MAX_RETRIES + 1):
                raw_qa = generate_qa_from_description(desc)
                cleaned = clean_and_parse_qa(raw_qa)

                if is_valid_qa(cleaned):
                    break
                else:
                    print(f"⚠️ Retry {attempt} failed for index {global_idx}")

            if not is_valid_qa(cleaned):
                print(f"❌ Failed to generate valid QA after {MAX_RETRIES} retries for index {global_idx}")

            output_row = list(row.values) + cleaned

        # Escape commas
        safe_row = [str(x).replace(',', ' ') for x in output_row]
        f_out.write(",".join(safe_row) + "\n")

print(f"\n✅ QA pairs with retries saved to: {OUTPUT_CSV}")
print(f"🗂️  Processed rows: {START_INDEX} to {end_index - 1}")
