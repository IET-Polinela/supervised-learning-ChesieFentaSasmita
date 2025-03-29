
import matplotlib.pyplot as plt
import numpy as np

# Contoh data hasil prediksi (gantilah dengan hasil dari modelmu)
y_test = np.array([100, 150, 200, 250, 300, 350, 400])  # Nilai aktual dari dataset uji
y_pred_lr = np.array([110, 145, 190, 260, 295, 340, 410])  # Linear Regression
y_pred_poly2 = np.array([105, 148, 205, 245, 310, 345, 390])  # Polynomial Regression (degree=2)
y_pred_poly3 = np.array([102, 152, 198, 248, 298, 348, 398])  # Polynomial Regression (degree=3)
y_pred_knn3 = np.array([108, 147, 203, 240, 305, 335, 380])  # KNN Regression (k=3)
y_pred_knn5 = np.array([107, 149, 202, 255, 290, 338, 385])  # KNN Regression (k=5)
y_pred_knn7 = np.array([106, 151, 200, 252, 295, 342, 395])  # KNN Regression (k=7)

models = {
    "Linear Regression": y_pred_lr,
    "Polynomial Regression (Degree 2)": y_pred_poly2,
    "Polynomial Regression (Degree 3)": y_pred_poly3,
    "KNN Regression (k=3)": y_pred_knn3,
    "KNN Regression (k=5)": y_pred_knn5,
    "KNN Regression (k=7)": y_pred_knn7,
}

plt.figure(figsize=(15, 10))
for i, (name, y_pred) in enumerate(models.items(), 1):
    plt.subplot(2, 3, i)
    plt.scatter(y_test, y_pred, alpha=0.5, label="Prediksi vs Aktual")
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label="Ideal Line")
    plt.xlabel("Nilai Aktual (y_test)")
    plt.ylabel("Nilai Prediksi (y_pred)")
    plt.title(f"{name}")
    plt.legend()

plt.tight_layout()
plt.savefig("scatter_plot_models.png", dpi=300)  # Simpan gambar sebagai PNG
plt.show()
