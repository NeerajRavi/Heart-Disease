from flask import Flask, render_template, request
import joblib
import os
import numpy as np
import requests
from datetime import datetime

app = Flask(__name__)
model = None
scaler = None

# ✅ Use ENV variable (fallback added for safety)
N8N_WEBHOOK_URL = os.environ.get(
    "N8N_WEBHOOK_URL",
    "https://neerajravi.app.n8n.cloud/webhook/heart-disease-prediction"
)

def load_assets():
    global model, scaler
    if model is None:
        model = joblib.load("heart_disease_model2.joblib")
    if scaler is None:
        scaler = joblib.load("heart_scaler.joblib")

@app.route("/health")
def health():
    return "OK", 200

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error = None
    missing = []

    values = {
        "age": "", "gender": "", "height": "", "weight": "",
        "ap_hi": "", "ap_lo": "", "cholesterol": "",
        "gluc": "", "smoke": "", "alco": "", "active": ""
    }

    if request.method == "POST":
        try:
            load_assets()

            for key in values:
                values[key] = request.form.get(key, "").strip()
                if values[key] == "":
                    missing.append(key)

            if missing:
                return render_template(
                    "home.html",
                    prediction=None,
                    values=values,
                    missing=missing,
                    error="Please fill all fields before predicting."
                )

            features = np.array([[float(values[k]) for k in values]])
            scaled = scaler.transform(features)
            pred = model.predict(scaled)[0]

            prediction = (
                "High chance of heart disease"
                if pred == 1
                else "Low chance of heart disease"
            )

            payload = {
                **values,
                "prediction": prediction,
                "timestamp": datetime.now().isoformat()
            }

            # 🔍 DEBUG LOGS (CRITICAL)
            print("N8N WEBHOOK URL:", N8N_WEBHOOK_URL)

            try:
                r = requests.post(N8N_WEBHOOK_URL, json=payload, timeout=10)
                print("n8n response status:", r.status_code)
                print("n8n response body:", r.text)
            except Exception as e:
                print("❌ n8n POST failed:", e)

        except Exception as e:
            error = f"Prediction failed: {e}"

    return render_template(
        "home.html",
        prediction=prediction,
        values=values,
        missing=missing,
        error=error
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
