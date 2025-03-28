
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("dataset_scaled_final.csv")

# Hapus missing values
df = df.dropna()

X = df.drop(columns=["SalePrice"])  
y = df["SalePrice"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standarisasi data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Evaluasi model dengan berbagai nilai K
knn_results = {}
for k in [3, 5, 7]:
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)

    mse_knn = mean_squared_error(y_test, y_pred_knn)
    r2_knn = r2_score(y_test, y_pred_knn)

    knn_results[f"K={k}"] = {'MSE': mse_knn, 'R2': r2_knn}

# Output hasil evaluasi
for k, res in knn_results.items():
    print(f"{k} -> MSE: {res['MSE']:.4f}, R²: {res['R2']:.4f}")
