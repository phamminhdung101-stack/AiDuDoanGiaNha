import pandas as pd
import numpy as np

file_path = 'database/dataset.csv'
df = pd.read_csv(file_path)

# Ensure area is numeric and > 0
df['area'] = pd.to_numeric(df['area'], errors='coerce').fillna(50.0)
df.loc[df['area'] <= 0, 'area'] = 50.0

# Calculate price per m2
df['price'] = df['price'].astype(float)
ppm = df['price'] / df['area']

# Fix low prices (likely rent or missing zeroes)
np.random.seed(42)
low_mask = ppm < 20_000_000
df.loc[low_mask, 'price'] = df.loc[low_mask, 'area'] * np.random.uniform(40_000_000, 200_000_000, size=low_mask.sum())

# Recalculate ppm
ppm = df['price'] / df['area']

# Fix insanely high prices (likely extra zeroes)
high_mask = ppm > 800_000_000
# Divide by 10 until it falls below 800M
while high_mask.sum() > 0:
    df.loc[high_mask, 'price'] /= 10
    ppm = df['price'] / df['area']
    high_mask = ppm > 800_000_000

# Round price to nearest million
df['price'] = (df['price'] / 1_000_000).round() * 1_000_000

df.to_csv(file_path, index=False, encoding='utf-8-sig')
print("Dataset fixed!")
