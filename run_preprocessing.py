import pandas as pd
import numpy as np
import os
import shutil

def run():
    # 1. Backup
    src = 'database/dataset.csv'
    bak = 'database/dataset_backup.csv'
    if not os.path.exists(bak):
        shutil.copy(src, bak)
        print(f"Created backup at {bak}")

    # 2. Analyze before
    df = pd.read_csv(src)
    print(f"\n--- BEFORE ---")
    print(f"Shape: {df.shape}")
    print("Missing values:")
    print(df.isnull().sum())
    print("Duplicates:", df.duplicated().sum())

    # 3. Cleaning
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Ensure numeric
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['area'] = pd.to_numeric(df['area'], errors='coerce')
    df['rooms'] = pd.to_numeric(df['rooms'], errors='coerce')
    df['toilets'] = pd.to_numeric(df['toilets'], errors='coerce')
    
    # Drop rows with NaN in critical cols
    df = df.dropna(subset=['price', 'area'])
    
    # Fill missing in rooms/toilets with median
    df['rooms'] = df['rooms'].fillna(df['rooms'].median())
    df['toilets'] = df['toilets'].fillna(df['toilets'].median())
    
    # Fill missing categorical
    df['location'] = df['location'].fillna('unknown')
    df['ward'] = df['ward'].fillna('unknown')
    
    # Filter valid
    df = df[(df['price'] > 0) & (df['area'] > 0)]
    
    # 4. Feature Engineering
    df['price_per_m2'] = df['price'] / df['area']
    
    # 5. Outlier removal using IQR on price_per_m2 and area
    Q1_ppm = df['price_per_m2'].quantile(0.25)
    Q3_ppm = df['price_per_m2'].quantile(0.75)
    IQR_ppm = Q3_ppm - Q1_ppm
    lower_ppm = Q1_ppm - 1.5 * IQR_ppm
    upper_ppm = Q3_ppm + 1.5 * IQR_ppm
    
    Q1_area = df['area'].quantile(0.25)
    Q3_area = df['area'].quantile(0.75)
    IQR_area = Q3_area - Q1_area
    lower_area = Q1_area - 1.5 * IQR_area
    upper_area = Q3_area + 1.5 * IQR_area
    
    outliers_ppm = df[(df['price_per_m2'] < lower_ppm) | (df['price_per_m2'] > upper_ppm)].shape[0]
    
    df_clean_ppm = df[(df['price_per_m2'] >= lower_ppm) & (df['price_per_m2'] <= upper_ppm)]
    
    outliers_area = df_clean_ppm[(df_clean_ppm['area'] < lower_area) | (df_clean_ppm['area'] > upper_area)].shape[0]
    
    df_clean = df_clean_ppm[(df_clean_ppm['area'] >= lower_area) & (df_clean_ppm['area'] <= upper_area)]
    
    print(f"\n--- PROCESSING ---")
    print(f"Removed duplicates.")
    print(f"Filled missing rooms/toilets with median.")
    print(f"Added feature 'price_per_m2'.")
    print(f"Detected and removed {outliers_ppm} price_per_m2 outliers.")
    print(f"Detected and removed {outliers_area} area outliers.")

    # 6. Analyze after
    print(f"\n--- AFTER ---")
    print(f"Shape: {df_clean.shape}")
    print("Missing values:")
    print(df_clean.isnull().sum())
    
    # Save back to dataset.csv
    df_clean.to_csv(src, index=False, encoding='utf-8-sig')
    print("\nPreprocessed dataset saved to", src)

if __name__ == '__main__':
    run()
