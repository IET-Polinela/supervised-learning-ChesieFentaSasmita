
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Pastikan nama file dataset sudah benar
file_path_outlier = "dataset_with_outliers.csv"
file_path_cleaned = "dataset_scaled_final.csv"

# Load dataset dengan outlier dan dataset yang telah diproses
df_outlier = pd.read_csv(file_path_outlier)
df_cleaned = pd.read_csv(file_path_cleaned)

# Cek dan tangani missing values
df_outlier = df_outlier.dropna()
df_cleaned = df_cleaned.dropna()

# Asumsi kolom terakhir adalah target
X_outlier = df_outlier.iloc[:, :-1]
y_outlier = df_outlier.iloc[:, -1]

X_cleaned = df_cleaned.iloc[:, :-1]
y_cleaned = df_cleaned.iloc[:, -1]

# Split data menjadi train dan test
X_train_out, X_test_out, y_train_out, y_test_out = train_test_split(X_outlier, y_outlier, test_size=0.2, random_state=42)
X_train_clean, X_test_clean, y_train_clean, y_test_clean = train_test_split(X_cleaned, y_cleaned, test_size=0.2, random_state=42)

# Train model Linear Regression untuk kedua dataset
model_outlier = LinearRegression()
model_cleaned = LinearRegression()

model_outlier.fit(X_train_out, y_train_out)
model_cleaned.fit(X_train_clean, y_train_clean)

# Prediksi
y_pred_out = model_outlier.predict(X_test_out)
y_pred_clean = model_cleaned.predict(X_test_clean)

# Hitung MSE dan R² Score
mse_out = mean_squared_error(y_test_out, y_pred_out)
r2_out = r2_score(y_test_out, y_pred_out)

mse_clean = mean_squared_error(y_test_clean, y_pred_clean)
r2_clean = r2_score(y_test_clean, y_pred_clean)

# Buat visualisasi untuk dataset dengan outlier
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Scatter plot antara hasil prediksi dan nilai aktual
sns.scatterplot(x=y_test_out, y=y_pred_out, ax=axes[0])
axes[0].set_title("Scatter Plot: Prediksi vs Aktual (Outlier)")
axes[0].set_xlabel("Nilai Aktual")
axes[0].set_ylabel("Prediksi")

# Residual plot
residuals_out = y_test_out - y_pred_out
sns.scatterplot(x=y_pred_out, y=residuals_out, ax=axes[1])
axes[1].axhline(y=0, color='r', linestyle='--')
axes[1].set_title("Residual Plot (Outlier)")
axes[1].set_xlabel("Prediksi")
axes[1].set_ylabel("Residual")

# Distribusi residual
sns.histplot(residuals_out, kde=True, ax=axes[2])
axes[2].set_title("Distribusi Residual (Outlier)")

plt.tight_layout()
plt.savefig("visualisasi_outlier.png")
plt.close()

# Buat visualisasi untuk dataset tanpa outlier dan sudah diskalakan
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Scatter plot antara hasil prediksi dan nilai aktual
sns.scatterplot(x=y_test_clean, y=y_pred_clean, ax=axes[0])
axes[0].set_title("Scatter Plot: Prediksi vs Aktual (Cleaned)")
axes[0].set_xlabel("Nilai Aktual")
axes[0].set_ylabel("Prediksi")

# Residual plot
residuals_clean = y_test_clean - y_pred_clean
sns.scatterplot(x=y_pred_clean, y=residuals_clean, ax=axes[1])
axes[1].axhline(y=0, color='r', linestyle='--')
axes[1].set_title("Residual Plot (Cleaned)")
axes[1].set_xlabel("Prediksi")
axes[1].set_ylabel("Residual")

# Distribusi residual
sns.histplot(residuals_clean, kde=True, ax=axes[2])
axes[2].set_title("Distribusi Residual (Cleaned)")

plt.tight_layout()
plt.savefig("visualisasi_cleaned.png")
plt.close()

print("MSE (Outlier):", mse_out)
print("R2 Score (Outlier):", r2_out)
print("MSE (Cleaned):", mse_clean)
print("R2 Score (Cleaned):", r2_clean)
