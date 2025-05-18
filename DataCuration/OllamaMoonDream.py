import os
import pandas as pd
from ollama import Client
from tqdm import tqdm

# === Config ===
CSV_PATH = 'sampled_balanced_output.csv'
IMAGE_COLUMN = 'path'
TYPE_COLUMN = 'product_type'
OUTPUT_CSV = 'vqa_output.csv'
BASE_IMAGE_DIR = './Dataset/images/small/'
MODEL_NAME = 'moondream'
START_INDEX = 16909
MAX_IMAGES = 53087-START_INDEX  # Adjusted to match the range of your dataset

# === Load Data ===
df = pd.read_csv(CSV_PATH)
end_index = START_INDEX + MAX_IMAGES
df_subset = df.iloc[START_INDEX:end_index].copy()

# === Existing Output Check ===
processed_indices = set()
if os.path.exists(OUTPUT_CSV):
    df_out = pd.read_csv(OUTPUT_CSV)
    processed_indices = set(df_out.index)
    header_written = True
else:
    header_written = False

# === Initialize Client ===
ollamaClient = Client("http://localhost:11434")

# === VQA Function ===
def get_image_description(img_path, product_type):
    full_path = os.path.join(BASE_IMAGE_DIR, img_path)
    if not os.path.isfile(full_path):
        print(f"⚠️  Missing image: {full_path}")
        return None

    prompt = f"The image contains a product of category '{product_type}'. Describe it in as much detail as possible."

    try:
        messages = [{
            'role': 'user',
            'content': prompt,
            'images': [full_path]
        }]
        response = ollamaClient.chat(model=MODEL_NAME, messages=messages)
        raw_text = response['message']['content']
        return raw_text.strip().replace('\n', ' ').lower()
    except Exception as e:
        print(f"❌ Error processing {img_path}: {e}")
        return None

# === Process and Save Incrementally ===
with open(OUTPUT_CSV, 'a', encoding='utf-8', newline='') as f_out:
    if not header_written:
        header = list(df_subset.columns) + ['vqa_description']
        f_out.write(",".join(header) + "\n")

    for idx, row in tqdm(df_subset.iterrows(), total=len(df_subset), desc="Generating VQA"):
        global_idx = START_INDEX + idx
        if global_idx in processed_indices:
            continue

        image_path = row[IMAGE_COLUMN]
        product_type = row[TYPE_COLUMN]
        description = get_image_description(image_path, product_type)

        row_data = list(row.values) + [description if description else ""]
        safe_row = [str(x).replace(",", " ") for x in row_data]  # escape commas

        f_out.write(",".join(safe_row) + "\n")

print(f"\n✅ VQA responses saved to: {OUTPUT_CSV}")
print(f"🗂️  Range processed: {START_INDEX} to {end_index - 1}")
