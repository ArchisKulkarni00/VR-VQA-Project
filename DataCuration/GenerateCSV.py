import os
import json
import csv

# Folder where your .json files are located
json_folder = './listings/metadata'  # Change this to your folder path

# Output CSV file
output_csv = './Dataset/images/metadata/filtered_products.csv'

# Fields to extract
required_keys = ['main_image_id', 'product_type']

# Container for all extracted data
all_rows = []

# Loop through all files in the folder
for filename in os.listdir(json_folder):
    if filename.endswith('.json'):
        file_path = os.path.join(json_folder, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                try:
                    data = json.loads(line)
                    # Only include rows that have both keys
                    if all(key in data for key in required_keys):
                        row = {}
                        row['main_image_id'] = data['main_image_id']
                        # Extract 'value' from first dict in 'product_type' list
                        if isinstance(data['product_type'], list) and data['product_type']:
                            row['product_type'] = data['product_type'][0].get('value', '')
                        else:
                            row['product_type'] = ''
                        all_rows.append(row)
                except json.JSONDecodeError:
                    print(f"Skipped invalid JSON line in {filename}")
    print("Wrote:", filename)

# Write to CSV
with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=required_keys)
    writer.writeheader()
    writer.writerows(all_rows)

print(f"✅ Extracted {len(all_rows)} rows to {output_csv}")
