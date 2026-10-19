# EV Range: Prediksi Jarak Tempuh Mobil Listrik

Regresi Linear Sederhana: `battery_capacity_kWh` (X) -> `range_km` (y).
Aplikasi web dibuat dengan Flask, Scikit-learn, dan Bootstrap.

## Jalankan di komputer

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    |  Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Buka http://127.0.0.1:5000

## Latih ulang model (opsional)

```bash
python ta02_regresi_ev.py
```

Skrip ini menghasilkan ulang `model_ev_range.pkl`, `model_info.json`, dan `data_points.json`.
Pakai versi scikit-learn yang sama dengan `requirements.txt`.

## Deploy ke Railway

1. Upload folder ini ke repository GitHub.
2. Di Railway: New Project -> Deploy from GitHub repo -> pilih repo ini.
3. Setelah build selesai: Settings -> Networking -> Generate Domain.

Start command sudah ada di `Procfile` (`gunicorn app:app --bind 0.0.0.0:$PORT`).

## Struktur

```
app.py                 aplikasi Flask (halaman + endpoint /api/predict)
templates/index.html   tampilan web
model_ev_range.pkl     model hasil pelatihan (Joblib)
model_info.json        koefisien dan metrik model
data_points.json       titik data untuk grafik
ta02_regresi_ev.py     EDA, pelatihan, evaluasi
requirements.txt       dependensi
Procfile               perintah start untuk Railway
```
