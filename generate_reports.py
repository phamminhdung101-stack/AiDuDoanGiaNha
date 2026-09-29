import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def run():
    os.makedirs('reports', exist_ok=True)
    
    # 1. Load data
    df_before = pd.read_csv('database/dataset_backup.csv')
    df_after = pd.read_csv('database/dataset.csv')
    
    # Pre-clean for 'before' (just to plot numbers without crashing)
    df_before['price'] = pd.to_numeric(df_before['price'], errors='coerce')
    df_before['area'] = pd.to_numeric(df_before['area'], errors='coerce')
    
    sns.set_theme(style="whitegrid")
    
    # A. Phân bố giá nhà trước xử lý
    plt.figure(figsize=(10,6))
    sns.histplot(df_before['price'].dropna() / 1e9, bins=50, kde=True, color='red')
    plt.title('Phân bố giá nhà (Trước xử lý) - Đơn vị: Tỷ VNĐ')
    plt.xlabel('Giá nhà (Tỷ VNĐ)')
    plt.ylabel('Tần suất')
    plt.savefig('reports/distribution_price_before.png')
    plt.close()
    
    # B. Phân bố giá nhà sau xử lý
    plt.figure(figsize=(10,6))
    sns.histplot(df_after['price'].dropna() / 1e9, bins=50, kde=True, color='blue')
    plt.title('Phân bố giá nhà (Sau xử lý) - Đơn vị: Tỷ VNĐ')
    plt.xlabel('Giá nhà (Tỷ VNĐ)')
    plt.ylabel('Tần suất')
    plt.savefig('reports/distribution_price_after.png')
    plt.close()
    
    # C. Boxplot giá nhà trước và sau
    plt.figure(figsize=(12,6))
    plt.subplot(1, 2, 1)
    sns.boxplot(y=df_before['price'].dropna() / 1e9, color='red')
    plt.title('Boxplot Giá (Trước)')
    plt.ylabel('Tỷ VNĐ')
    
    plt.subplot(1, 2, 2)
    sns.boxplot(y=df_after['price'].dropna() / 1e9, color='blue')
    plt.title('Boxplot Giá (Sau)')
    plt.ylabel('Tỷ VNĐ')
    plt.tight_layout()
    plt.savefig('reports/boxplot_price.png')
    plt.close()
    
    # D. Boxplot diện tích trước và sau
    plt.figure(figsize=(12,6))
    plt.subplot(1, 2, 1)
    sns.boxplot(y=df_before['area'].dropna(), color='red')
    plt.title('Boxplot Diện tích (Trước)')
    plt.ylabel('m²')
    
    plt.subplot(1, 2, 2)
    sns.boxplot(y=df_after['area'].dropna(), color='blue')
    plt.title('Boxplot Diện tích (Sau)')
    plt.ylabel('m²')
    plt.tight_layout()
    plt.savefig('reports/boxplot_area.png')
    plt.close()
    
    # E. Diện tích và giá nhà (Scatter) - Sau xử lý
    plt.figure(figsize=(10,6))
    sns.scatterplot(x=df_after['area'], y=df_after['price'] / 1e9, alpha=0.5, color='purple')
    plt.title('Quan hệ giữa Diện tích và Giá nhà (Sau xử lý)')
    plt.xlabel('Diện tích (m²)')
    plt.ylabel('Giá nhà (Tỷ VNĐ)')
    plt.savefig('reports/scatter_area_price.png')
    plt.close()

    print("Reports generated in reports/ directory.")

if __name__ == '__main__':
    run()
