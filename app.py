# from flask import Flask, render_template, request
# import joblib
# import os
# import numpy as np

# app = Flask(__name__)

# model = None
# scaler = None


# def load_assets():
#     global model, scaler

#     if model is None:
#         MODEL_PATH = "heart_disease_model2.joblib"

#         if not os.path.exists(MODEL_PATH):
#             raise FileNotFoundError("heart_disease_model2.joblib not found")

#         model = joblib.load(MODEL_PATH)

#     if scaler is None:
#         SCALER_PATH = "heart_scaler.joblib"

#         if not os.path.exists(SCALER_PATH):
#             raise FileNotFoundError("heart_scaler.joblib not found")

#         scaler = joblib.load(SCALER_PATH)


# # 🔹 health check for Render
# @app.route("/health")
# def health():
#     return "OK", 200


# @app.route("/", methods=["GET", "POST"])
# def home():
#     prediction = None
#     error = None
#     missing = []

#     values = {
#         "age": "",
#         "gender": "",
#         "height": "",
#         "weight": "",
#         "ap_hi": "",
#         "ap_lo": "",
#         "cholesterol": "",
#         "gluc": "",
#         "smoke": "",
#         "alco": "",
#         "active": ""
#     }

#     if request.method == "POST":
#         try:
#             load_assets()   # load only once

#             for key in values:
#                 values[key] = request.form.get(key, "").strip()
#                 if values[key] == "":
#                     missing.append(key)

#             if missing:
#                 error = "Please fill all fields before predicting."
#                 return render_template(
#                     "home1.html",
#                     prediction=None,
#                     values=values,
#                     missing=missing,
#                     error=error
#                 )

#             # ⚠️ explicit feature order (VERY IMPORTANT)
#             features = np.array([[
#                 float(values["age"]),
#                 float(values["gender"]),
#                 float(values["height"]),
#                 float(values["weight"]),
#                 float(values["ap_hi"]),
#                 float(values["ap_lo"]),
#                 float(values["cholesterol"]),
#                 float(values["gluc"]),
#                 float(values["smoke"]),
#                 float(values["alco"]),
#                 float(values["active"])
#             ]])

#             # safety check
#             if features.shape[1] != scaler.n_features_in_:
#                 raise ValueError(
#                     f"Feature mismatch: expected {scaler.n_features_in_}, got {features.shape[1]}"
#                 )

#             scaled = scaler.transform(features)
#             pred = model.predict(scaled)[0]

#             prediction = (
#                 "High chance of heart disease"
#                 if pred == 1
#                 else "Low chance of heart disease"
#             )

#         except Exception as e:
#             error = f"Prediction failed: {e}"

#     return render_template(
#         "home1.html",
#         prediction=prediction,
#         values=values,
#         missing=missing,
#         error=error
#     )
from flask import Flask, render_template, request
import joblib
import os
import numpy as np
import requests
from datetime import datetime

app = Flask(__name__)

model = None
scaler = None

N8N_WEBHOOK_URL = "https://neerajacadeno.app.n8n.cloud/webhook/heart-prediction"


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
                error = "Please fill all fields before predicting."
                return render_template("home1.html",
                                       prediction=None,
                                       values=values,
                                       missing=missing,
                                       error=error)

            features = np.array([[float(values[k]) for k in values]])
            scaled = scaler.transform(features)
            pred = model.predict(scaled)[0]

            prediction = (
                "High chance of heart disease"
                if pred == 1
                else "Low chance of heart disease"
            )

            # 🔹 SEND TO n8n
            payload = {
                **values,
                "prediction": prediction,
                "timestamp": datetime.now().isoformat()
            }

            try:
                requests.post(N8N_WEBHOOK_URL, json=payload, timeout=5)
            except Exception:
                pass

        except Exception as e:
            error = f"Prediction failed: {e}"

    return render_template("home1.html",
                           prediction=prediction,
                           values=values,
                           missing=missing,
                           error=error)
