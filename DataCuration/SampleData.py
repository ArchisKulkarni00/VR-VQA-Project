import pandas as pd

# Inputs
INPUT_CSV = "merged_output.csv"
OUTPUT_CSV = "sampled_balanced_output.csv"
MAX_SAMPLES_PER_CLASS = 2000
TOTAL_TARGET_SAMPLES = 100000

# Load dataset
df = pd.read_csv(INPUT_CSV)

# Group and sample with class cap
sampled_dfs = []
grouped = df.groupby('product_type')

for class_name, group in grouped:
    count = len(group)
    if count <= MAX_SAMPLES_PER_CLASS:
        sampled = group
    else:
        sampled = group.sample(n=MAX_SAMPLES_PER_CLASS, random_state=42)
    sampled_dfs.append(sampled)

# Combine all class-limited samples
pooled_df = pd.concat(sampled_dfs).reset_index(drop=True)

print(f"📦 Pooled dataset size after class-capped sampling: {len(pooled_df)}")

# Enforce total sample target
if len(pooled_df) > TOTAL_TARGET_SAMPLES:
    final_df = pooled_df.sample(n=TOTAL_TARGET_SAMPLES, random_state=42)
    print(f"🎯 Targeting total {TOTAL_TARGET_SAMPLES} samples. Randomly downsampled the pool.")
else:
    final_df = pooled_df
    print(f"⚠️ Only {len(pooled_df)} samples available, below target of {TOTAL_TARGET_SAMPLES}.")

# Shuffle and save
final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
final_df.to_csv(OUTPUT_CSV, index=False)

# Summary stats
print(f"\n✅ Final sampled dataset saved: {OUTPUT_CSV}")
print(f"📊 Total samples selected: {len(final_df)}")
print(f"🎯 Max per class: {MAX_SAMPLES_PER_CLASS} | Desired total: {TOTAL_TARGET_SAMPLES}")
print("\n📈 Class Distribution (top 20 classes):")
print(final_df['product_type'].value_counts(normalize=True).mul(100).round(2).head(20))
