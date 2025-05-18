import pandas as pd
import os

# Paths to the two CSVs
csv1_path = './Dataset/images/metadata/images.csv'        # contains 'image_id' and 'path'
csv2_path = './Dataset/images/metadata/filtered_products.csv'    # contains 'main_image_id' and 'product_type'
output_csv = 'merged_output.csv'

# Base directory where images are stored
image_base_path = './Dataset/images/small/'  # Adjust this if your actual images are elsewhere

# Load CSV files
df_images = pd.read_csv(csv1_path)
df_products = pd.read_csv(csv2_path)

# Merge on matching IDs
merged_df = pd.merge(
    df_images,
    df_products,
    left_on='image_id',
    right_on='main_image_id',
    how='inner'
)

# Keep only relevant columns
merged_df = merged_df[['image_id', 'path', 'product_type']]

# Remove duplicate image IDs
merged_df = merged_df.drop_duplicates(subset='image_id')

# Prepend the base path and check file existence
print("🔍 Checking image file existence...")

# Create full path column
merged_df['full_path'] = image_base_path + merged_df['path'].astype(str)

# Keep only rows where image exists
merged_df = merged_df[merged_df['full_path'].apply(os.path.exists)]

# Drop 'full_path' column if not needed
merged_df = merged_df.drop(columns=['full_path'])

# Save cleaned and filtered result
merged_df.to_csv(output_csv, index=False)

print(f"✅ Final cleaned CSV saved to: {output_csv}")
