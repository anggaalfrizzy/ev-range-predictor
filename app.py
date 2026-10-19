import json
import math
import os

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "model_ev_range.pkl"))

with open(os.path.join(BASE_DIR, "model_info.json")) as f:
    info = json.load(f)
with open(os.path.join(BASE_DIR, "data_points.json")) as f:
    points = json.load(f)

# Batas input dibulatkan ke dalam rentang data latih
LOW = math.ceil(info["min_kwh"])
HIGH = math.floor(info["max_kwh"])
DEFAULT_KWH = 75


@app.template_filter("idn")
def idn(value, digits=2):
    """Format angka gaya Indonesia: 1.234,56"""
    text = f"{value:,.{digits}f}"
    return text.replace(",", "§").replace(".", ",").replace("§", ".")


def predict_range(raw):
    """Validasi input lalu prediksi. Mengembalikan (hasil, pesan_error)."""
    try:
        value = float(str(raw).strip().replace(",", "."))
    except ValueError:
        return None, "Masukkan angka, misalnya 75 atau 62,5."
    if not math.isfinite(value) or value < LOW or value > HIGH:
        return None, f"Masukkan angka antara {LOW} dan {HIGH} kWh."

    X_new = pd.DataFrame({"battery_capacity_kWh": [value]})
    pred = float(model.predict(X_new)[0])
    return {
        "battery": value,
        "range": round(pred),
        "low": round(pred - info["rmse"]),
        "high": round(pred + info["rmse"]),
    }, None


@app.route("/", methods=["GET", "POST"])
def index():
    result, error = None, None
    raw = DEFAULT_KWH
    # Hasil hanya dihitung setelah tombol "Hitung estimasi" ditekan (POST)
    if request.method == "POST":
        raw = request.form.get("battery", DEFAULT_KWH)
        result, error = predict_range(raw)

    if error:
        battery_value = str(raw)
    elif result:
        battery_value = f"{result['battery']:g}".replace(".", ",")
    else:
        battery_value = str(DEFAULT_KWH)

    js_info = {
        "low": LOW,
        "high": HIGH,
        "min_kwh": info["min_kwh"],
        "max_kwh": info["max_kwh"],
        "intercept": info["intercept"],
        "coef": info["coef"],
        "rmse": info["rmse"],
    }
    return render_template(
        "index.html",
        result=result,
        error=error,
        battery_value=battery_value,
        info=info,
        low=LOW,
        high=HIGH,
        js_info=js_info,
        points=points,
    )


@app.route("/api/predict")
def api_predict():
    result, error = predict_range(request.args.get("battery", ""))
    if error:
        return jsonify(ok=False, error=error), 400
    return jsonify(ok=True, **result)


if __name__ == "__main__":
    app.run(debug=True)
