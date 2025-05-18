import pandas as pd

# Load the merged CSV
df = pd.read_csv('merged_output.csv')

print("📌 Dataset Overview\n")
print(f"Total rows: {len(df)}")
print(f"Total columns: {df.shape[1]}")
print(f"Columns: {list(df.columns)}\n")

# Check for missing values
print("🔍 Missing Values:")
print(df.isnull().sum(), "\n")

# Unique classes in product_type
print("🧩 Unique Classes in 'product_type':")
print(df['product_type'].value_counts(), "\n")

# Class distribution as percentage
print("📈 Class Distribution (%):")
print(df['product_type'].value_counts(normalize=True).mul(100).round(2), "\n")

# Basic stats about file paths (optional)
print("📂 Sample image paths:")
print(df['path'].sample(5).to_string(index=False), "\n")

# Check for duplicate image IDs (if needed)
duplicate_count = df.duplicated(subset='image_id').sum()
print(f"🔁 Duplicate image_ids: {duplicate_count}\n")

# Summary statistics of all columns (especially if numeric ones get added later)
print("📊 General Info:")
print(df.describe(include='all'))
