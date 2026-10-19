# %% [markdown]
# # Prediksi Jarak Tempuh Mobil Listrik Berdasarkan Kapasitas Baterai
# X = battery_capacity_kWh | y = range_km
# Jalankan per sel (# %%) di VS Code / Jupyter, atau langsung sebagai script.

# %% 1. Import & muat data  (ini yang di-screenshot untuk TP-02)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

df = pd.read_csv("electric_vehicles_spec_2025_csv.csv")

print("--- 5 Baris Pertama ---")
print(df.head())
print("\n--- Nama Kolom ---")
print(df.columns)

# %% 2. Ambil dua kolom yang dipakai & cek kondisi data
data = df[["battery_capacity_kWh", "range_km"]].copy()
print("Jumlah baris:", len(data))
print("\nMissing value:\n", data.isna().sum())
print("\nStatistik deskriptif:\n", data.describe())
print("\nKorelasi:", round(data.corr().iloc[0, 1], 3))

# %% 3. EDA: scatterplot untuk validasi tren linear
plt.figure(figsize=(8, 6))
sns.scatterplot(x="battery_capacity_kWh", y="range_km", data=data, alpha=0.6)
plt.title("Kapasitas Baterai vs Jarak Tempuh")
plt.xlabel("Kapasitas Baterai (kWh)")
plt.ylabel("Jarak Tempuh (km)")
plt.show()

# %% 4. Cek outlier (metode IQR pada masing-masing variabel)
for col in data.columns:
    q1, q3 = data[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = ((data[col] < low) | (data[col] > high)).sum()
    print(f"{col}: batas [{low:.1f}, {high:.1f}] -> {n_out} outlier")
# Catatan: metode IQR tidak menemukan outlier pada kedua variabel,
# jadi tidak ada data yang dibuang. Mobil dengan error besar dibahas di sel 9.

# %% 5. Split data & latih model
X = data[["battery_capacity_kWh"]]
y = data["range_km"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)

# %% 6. Evaluasi
pred = model.predict(X_test)
mae = metrics.mean_absolute_error(y_test, pred)
rmse = np.sqrt(metrics.mean_squared_error(y_test, pred))
r2 = metrics.r2_score(y_test, pred)
print(f"MAE  : {mae:.2f} km")
print(f"RMSE : {rmse:.2f} km")
print(f"R2   : {r2:.3f}")

# %% 7. Interpretasi intercept & koefisien
b0 = model.intercept_
b1 = model.coef_[0]
print(f"Intercept (b0): {b0:.2f}")
print(f"Koefisien (b1): {b1:.2f}")
print(f"Persamaan: range_km = {b0:.2f} + {b1:.2f} x battery_capacity_kWh")
print(f"-> Tiap +1 kWh baterai, jarak tempuh naik rata-rata ~{b1:.2f} km.")

# %% 8. Visualisasi garis regresi & aktual vs prediksi
plt.figure(figsize=(8, 6))
plt.scatter(X_test, y_test, alpha=0.6, label="Data uji")
xs = np.linspace(X["battery_capacity_kWh"].min(), X["battery_capacity_kWh"].max(), 100)
plt.plot(xs, b0 + b1 * xs, color="red", lw=2, label="Garis regresi")
plt.xlabel("Kapasitas Baterai (kWh)")
plt.ylabel("Jarak Tempuh (km)")
plt.title("Garis Regresi Linear")
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(y_test, pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", lw=2)
plt.xlabel("Jarak Tempuh Aktual (km)")
plt.ylabel("Jarak Tempuh Prediksi (km)")
plt.title("Aktual vs Prediksi")
plt.show()

# %% 9. Mobil dengan error terbesar (bahan analisis kritis)
tmp = df.loc[X_test.index, ["brand", "model", "battery_capacity_kWh", "range_km"]].copy()
tmp["prediksi"] = pred
tmp["selisih"] = (tmp["range_km"] - tmp["prediksi"]).abs()
print(tmp.sort_values("selisih", ascending=False).head(5))

# %% 10. Simpan model (Joblib)
joblib.dump(model, "model_ev_range.pkl")
print("Model tersimpan: model_ev_range.pkl")

# Info tambahan untuk aplikasi Flask (batas input & metrik)
import json
info = {
    "min_kwh": float(X["battery_capacity_kWh"].min()),
    "max_kwh": float(X["battery_capacity_kWh"].max()),
    "intercept": float(b0),
    "coef": float(b1),
    "r2": float(r2),
    "rmse": float(rmse),
    "mae": float(mae),
    "n_data": int(len(data)),
}
with open("model_info.json", "w") as f:
    json.dump(info, f, indent=2)
print("Info model tersimpan: model_info.json")

# Titik data untuk grafik di aplikasi web
points = data.round(1).values.tolist()
with open("data_points.json", "w") as f:
    json.dump(points, f)
print("Titik data tersimpan: data_points.json")

# %% 11. Tes load model & prediksi contoh
loaded = joblib.load("model_ev_range.pkl")
contoh = pd.DataFrame({"battery_capacity_kWh": [75.0]})
print(f"Prediksi untuk 75 kWh: {loaded.predict(contoh)[0]:.0f} km")
