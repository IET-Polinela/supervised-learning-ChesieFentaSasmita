
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("train.csv")  # Sesuaikan dengan lokasi file dataset

# Pisahkan fitur numerik dan kategorikal
categorical_cols = df.select_dtypes(include=["object"]).columns

# Lakukan encoding pada fitur kategorikal
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Simpan encoder untuk nanti

# Simpan dataset hasil encoding sebelum splitting
df.to_csv("processed_data.csv", index=False)

# Pisahkan X (independent features) dan Y (target variable)
X = df.drop(columns=["Id", "SalePrice"])  # Menghapus ID dan SalePrice dari fitur input
Y = df["SalePrice"]  # Target variable

# Membagi dataset menjadi training dan testing (80:20)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Simpan dataset hasil split
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
Y_train.to_csv("Y_train.csv", index=False)
Y_test.to_csv("Y_test.csv", index=False)

# Menampilkan ukuran dataset hasil split
print(f"Training data: {X_train.shape}, Testing data: {X_test.shape}")
print("Preprocessed data saved as processed_data.csv")
