
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Load dataset tanpa outlier
df = pd.read_csv("dataset_without_outliers.csv")

# Pisahkan fitur numerik
numerical_cols = df.select_dtypes(include=["number"]).columns

# Lakukan scaling menggunakan StandardScaler dan MinMaxScaler
scaler_standard = StandardScaler()
scaler_minmax = MinMaxScaler()

df_standard_scaled = df.copy()
df_minmax_scaled = df.copy()

df_standard_scaled[numerical_cols] = scaler_standard.fit_transform(df[numerical_cols])
df_minmax_scaled[numerical_cols] = scaler_minmax.fit_transform(df[numerical_cols])

# Simpan dataset yang telah discaling
final_scaled_dataset = df_minmax_scaled  # Pilih metode scaling yang akan digunakan
final_scaled_dataset.to_csv("dataset_scaled_final.csv", index=False)

# Visualisasi perbandingan sebelum dan sesudah scaling
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

sns.histplot(df[numerical_cols].values.flatten(), bins=50, kde=True, ax=axes[0])
axes[0].set_title("Distribusi Data Asli")

sns.histplot(df_standard_scaled[numerical_cols].values.flatten(), bins=50, kde=True, ax=axes[1])
axes[1].set_title("Distribusi Data setelah Standard Scaling")

sns.histplot(df_minmax_scaled[numerical_cols].values.flatten(), bins=50, kde=True, ax=axes[2])
axes[2].set_title("Distribusi Data setelah MinMax Scaling")

plt.tight_layout()
plt.savefig("histogram_scaling_comparison.png")
plt.show() 
