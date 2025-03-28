
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Load dataset
file_path = "dataset_scaled_final.csv"
df = pd.read_csv(file_path)

# Pastikan tidak ada missing values sebelum pemrosesan
df = df.dropna()

X = df.drop(columns=["SalePrice"])  # Fitur
y = df["SalePrice"]  # Target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function for Polynomial Regression
def polynomial_regression(degree):
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    # Pastikan tidak ada NaN setelah transformasi
    X_train_poly = np.nan_to_num(X_train_poly)
    X_test_poly = np.nan_to_num(X_test_poly)
    
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    y_pred = model.predict(X_test_poly)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return y_pred, mse, r2

# Run Polynomial Regression for degree 2 & 3
y_pred_2, mse_2, r2_2 = polynomial_regression(2)
y_pred_3, mse_3, r2_3 = polynomial_regression(3)

# Scatter Plot
plt.figure(figsize=(18, 5))
plt.subplot(1, 3, 1)
plt.scatter(y_test, y_pred_2, label="Degree 2", alpha=0.6)
plt.scatter(y_test, y_pred_3, label="Degree 3", alpha=0.6)
plt.xlabel("Nilai Aktual")
plt.ylabel("Prediksi")
plt.title("Scatter Plot: Prediksi vs Aktual")
plt.legend()

# Residual Plot
plt.subplot(1, 3, 2)
residuals_2 = y_test - y_pred_2
residuals_3 = y_test - y_pred_3
plt.scatter(y_pred_2, residuals_2, label="Degree 2", alpha=0.6)
plt.scatter(y_pred_3, residuals_3, label="Degree 3", alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Prediksi")
plt.ylabel("Residual")
plt.title("Residual Plot")
plt.legend()

# Distribusi Residual
plt.subplot(1, 3, 3)
plt.hist(residuals_2, bins=10, alpha=0.6, label="Degree 2", density=True)
plt.hist(residuals_3, bins=10, alpha=0.6, label="Degree 3", density=True)
plt.xlabel("SalePrice")
plt.ylabel("Count")
plt.title("Distribusi Residual")
plt.legend()

# Save figure
plt.tight_layout()
plt.savefig("polynomial_visualization.png")
plt.show()

# Print MSE & R2 Scores
print(f"Degree 2 -> MSE: {mse_2:.4f}, R2 Score: {r2_2:.4f}")
print(f"Degree 3 -> MSE: {mse_3:.4f}, R2 Score: {r2_3:.4f}")
