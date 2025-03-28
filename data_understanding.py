
import pandas as pd

# Konfigurasi tampilan output Pandas
pd.set_option("display.max_columns", None)  # Menampilkan semua kolom
pd.set_option("display.expand_frame_repr", False)  # Mencegah pemotongan frame
pd.set_option("display.float_format", "{:.2f}".format)  # Agar angka lebih rapi

# Membaca dataset
file_path = 'train.csv'  
house_data = pd.read_csv(file_path)

# Menampilkan statistik deskriptif untuk numerik dan kategorikal secara terpisah
stats_numerik = house_data.describe().transpose()  # Statistik numerik
stats_kategorikal = house_data.describe(include=["O"]).transpose()  # Statistik kategorikal

print("Statistik Deskriptif (Numerik):\n", stats_numerik)
print("\nStatistik Deskriptif (Kategorikal):\n", stats_kategorikal)

# Menampilkan jumlah data per kolom
print("\nJumlah Data per Kolom:")
print(house_data.count())

# Mengecek jumlah nilai yang hilang per kolom
missing_values = house_data.isnull().sum()
print("\nJumlah Nilai yang Hilang:")
print(missing_values[missing_values > 0])  # Hanya menampilkan kolom dengan missing values

# Menangani nilai yang hilang dengan assignment langsung
house_data_filled = house_data.copy()

for column in house_data_filled.columns:
    if house_data_filled[column].dtype == 'O':  # Jika kategorikal
        house_data_filled[column] = house_data_filled[column].fillna(house_data_filled[column].mode()[0])
    else:  # Jika numerik
        house_data_filled[column] = house_data_filled[column].fillna(house_data_filled[column].median())

# Memeriksa ulang apakah masih ada nilai yang hilang
missing_values_after = house_data_filled.isnull().sum()
print("\nJumlah Nilai yang Hilang Setelah Penanganan:")
print(missing_values_after[missing_values_after > 0])  # Hanya menampilkan jika masih ada nilai kosong
