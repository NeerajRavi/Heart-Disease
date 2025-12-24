from flask import Flask,render_template,request
import joblib
import os
import requests
import numpy as np

app=Flask(__name__)

model_url=joblib.load("https://huggingface.co/NeerajRavi/heart-disease-prediction-model/resolve/main/heart_disease_model.joblib")
model_path="heart_disease_model.joblib"
if not os.path.exists(model_path):
    r=requests.get(model_url)
    r.raise_for_status()
    with open(model_path,"wb") as f:
        f.write(r.content)
model = joblib.load(model_path)
scaler=joblib.load("heart_scaler.joblib")

@app.route("/",methods=["GET","POST"])
def home():
    prediction=None
    error=None
    missing=[]
    values = {
        "age":"",
        "gender":"",
        "height":"",
        "weight":"",
        "ap_hi":"",
        "ap_lo":"",
        "cholesterol":"",
        "gluc":"",
        "smoke":"",
        "alco":"",
        "active":""
    }
    if request.method=="POST":
        try:
            for key in values:
                values[key]=request.form.get(key,"")
                if values[key]=="":
                    missing.append(key)
            if missing:
                error = "Please fill all fields before predicting."
                return render_template("home.html",prediction=None,values=values,missing=missing,error=error)
            features = np.array([[float(values[k]) for k in values]])
            scaled=scaler.transform(features)
            pred = model.predict(scaled)[0]
            prediction = "High chance of heart disease" if pred == 1 else "Low chance of heart disease"
        except Exception as e:
            prediction = f"Error: {e}"
    return render_template("home1.html",prediction=prediction,values=values,missing=missing,error=error)
if __name__=="__main__":
    app.run(debug=True)