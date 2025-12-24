from flask import Flask,render_template,request
import joblib
import numpy as np
app=Flask(__name__)
model=joblib.load("heart_disease_model.joblib")
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