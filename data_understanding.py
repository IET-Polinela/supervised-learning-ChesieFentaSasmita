import pandas as pd

file_path = "train.csv"  

df = pd.read_csv(file_path)

# Pilih hanya kolom numerik
df_numeric = df.select_dtypes(include=["number"])

# Hitung statistik deskriptif hanya untuk kolom numerik
stats = df_numeric.describe().T  

# Tambahkan kolom median, Q1, Q2, Q3, dan count
stats["median"] = df_numeric.median()
stats["Q1"] = df_numeric.quantile(0.25)
stats["Q2"] = df_numeric.quantile(0.50)
stats["Q3"] = df_numeric.quantile(0.75)
stats["count"] = df_numeric.count()

stats.to_csv("statistics_summary.csv")

print("Perhitungan statistik selesai. Hasil disimpan di statistics_summary.csv")
