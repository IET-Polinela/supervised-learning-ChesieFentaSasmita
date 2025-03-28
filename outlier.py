
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset hasil encoding dari preprocessing
df = pd.read_csv("processed_data.csv")

# Menampilkan dan menyimpan boxplot untuk semua fitur numerik
plt.figure(figsize=(15, 8))
sns.boxplot(data=df)
plt.xticks(rotation=90)
plt.title("Boxplot untuk Deteksi Outlier")

# Simpan gambar sebagai file PNG
plt.savefig("boxplot_outlier.png", dpi=300)
plt.show()

# Metode IQR untuk deteksi outlier
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

# Menentukan batas outlier
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Menandai outlier
outlier_mask = (df < lower_bound) | (df > upper_bound)

# Dataset dengan outlier (tanpa perubahan)
df_with_outliers = df.copy()
df_with_outliers.to_csv("dataset_with_outliers.csv", index=False)

# Menghapus outlier dari dataset
df_without_outliers = df[~outlier_mask.any(axis=1)]
df_without_outliers.to_csv("dataset_without_outliers.csv", index=False)

# Menampilkan jumlah data sebelum & sesudah menghapus outlier
print(f"Jumlah data awal: {df.shape[0]}")
print(f"Jumlah data setelah menghapus outlier: {df_without_outliers.shape[0]}")
