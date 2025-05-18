import argparse
from PIL import Image
from tqdm import tqdm
import requests
import os
import zipfile

import torch
import pandas as pd
from peft import PeftModel, PeftConfig
from transformers import BlipProcessor, BlipForQuestionAnswering


def download_and_unzip_google_drive(link):

    # Extract file ID from the Google Drive link
    file_id = link.split("/d/")[1].split("/")[0]
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    
    zip_path = "downloaded.zip"
    
    # Download the file
    with requests.get(download_url, stream=True) as response:
        if response.status_code == 200:
            with open(zip_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
        else:
            raise Exception("Failed to download file. Check the link or access permissions.")
    
    extracted_folder = "checkpoint/"
    os.makedirs(extracted_folder, exist_ok=True)

    # Unzip the file
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_folder)

    # Remove the ZIP file after extraction
    os.remove(zip_path)

    extracted_contents = os.listdir(extracted_folder)
    extracted_folder_name = next((item for item in extracted_contents if os.path.isdir(os.path.join(extracted_folder, item))), None)

    return extracted_folder + (extracted_folder_name if extracted_folder_name else extracted_folder)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True, help='Path to image folder')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to image-metadata CSV')
    args = parser.parse_args()

    # Load metadata CSV
    df = pd.read_csv(args.csv_path)

    # Load model and processor, move model to GPU if available
    model_path = download_and_unzip_google_drive('https://drive.google.com/file/d/1EmjaKnZp42PPXgRadr26NCkCeM_RZyj6/view')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = PeftConfig.from_pretrained(model_path)
    processor = BlipProcessor.from_pretrained(config.base_model_name_or_path,use_fast=True)
    base_model = BlipForQuestionAnswering.from_pretrained(config.base_model_name_or_path)
    model = PeftModel.from_pretrained(base_model, model_path).to(device)
    model.eval()

    generated_answers = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):

        image_path = f"{args.image_dir}/{row['image_name']}"
        question = str(row['question'])

        try:
            with torch.inference_mode():
                image = Image.open(image_path).convert("RGB")

                inputs = processor(images=image, text=question, return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}

                generated_ids = model.generate(**inputs)
                prediction = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip().lower()
                
        except Exception as e:
            prediction = "error"
        prediction = str(prediction).split()[0].lower()
        generated_answers.append(prediction)

    df["generated_answer"] = generated_answers
    df.to_csv("results.csv", index=False)


if __name__ == "__main__":
    main()