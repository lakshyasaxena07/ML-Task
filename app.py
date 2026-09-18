"""
Flask Web Application for Heart Attack Risk Prediction
Dataset: CDC 2022 Heart Attack Risk Predictors (JoseMBM Kaggle)
Model: Decision Tree Regressor with Cross-Validation & Hyperparameter Tuning
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__, static_folder="static", template_folder="templates")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "heart_model.pkl")
META_PATH = os.path.join(BASE_DIR, "model", "model_meta.json")

# Load model and metadata at startup
print("Loading trained Decision Tree Regressor pipeline...")
model = joblib.load(MODEL_PATH)

with open(META_PATH, "r", encoding="utf-8") as f:
    model_meta = json.load(f)

print(f"Loaded model successfully. Dataset: {model_meta['dataset']}")


def determine_risk_tier(risk_percent: float):
    if risk_percent < 3.0:
        return {
            "tier": "Minimal Risk",
            "badge_class": "badge-minimal",
            "color": "#10b981", # emerald green
            "accent": "#059669",
            "summary": "Your predicted risk is significantly below the general population baseline (5.46%). Keep up your healthy habits!"
        }
    elif risk_percent < 7.0:
        return {
            "tier": "Low Risk",
            "badge_class": "badge-low",
            "color": "#0ea5e9", # ocean blue
            "accent": "#0284c7",
            "summary": "Your predicted risk is close to or slightly below the average population baseline (5.46%). Maintain active lifestyle and regular health checkups."
        }
    elif risk_percent < 18.0:
        return {
            "tier": "Moderate Risk",
            "badge_class": "badge-moderate",
            "color": "#f59e0b", # amber
            "accent": "#d97706",
            "summary": "Moderate cardiovascular risk detected. Certain lifestyle or clinical factors (such as age, smoking, or blood sugar) elevate your risk above normal."
        }
    elif risk_percent < 35.0:
        return {
            "tier": "High Risk",
            "badge_class": "badge-high",
            "color": "#e11d48", # rose crimson
            "accent": "#be123c",
            "summary": "Elevated heart attack risk. Significant clinical warning signs are present. Proactive cardiac screening and physician consultation are strongly advised."
        }
    else:
        return {
            "tier": "Critical Risk",
            "badge_class": "badge-critical",
            "color": "#991b1b", # deep red
            "accent": "#7f1d1d",
            "summary": "Very high risk of myocardial infarction. History of angina, vascular complications, or severe co-morbidities require prompt medical attention."
        }


def generate_risk_factors_and_advice(data: dict):
    factors = []
    advice = []

    # Check Angina
    if data.get("HadAngina") == "Yes":
        factors.append({"name": "History of Angina", "impact": "High Risk", "desc": "Angina is the single strongest indicator of ischemic coronary disease."})
        advice.append("Consult your cardiologist regularly to monitor coronary artery circulation.")

    # Check Stroke
    if data.get("HadStroke") == "Yes":
        factors.append({"name": "Prior Stroke", "impact": "High Risk", "desc": "History of cerebrovascular events increases systemic cardiovascular risk."})
        advice.append("Adhere strictly to prescribed antiplatelet, anticoagulant, or blood pressure therapies.")

    # Check Diabetes
    if "Yes" in str(data.get("HadDiabetes", "")):
        factors.append({"name": "Diabetes Condition", "impact": "Moderate Risk", "desc": "Elevated blood glucose damages cardiac vessels and accelerates plaque buildup."})
        advice.append("Maintain tight glycemic control (HbA1c target) and adhere to diabetic care plans.")

    # Check Smoker
    smoker = data.get("SmokerStatus", "")
    if "every day" in smoker or "some days" in smoker:
        factors.append({"name": "Active Smoker", "impact": "Moderate Risk", "desc": "Smoking damages blood vessel lining and increases arterial plaque rupture."})
        advice.append("Participate in a smoking cessation program; heart attack risk drops rapidly after quitting.")
    elif "Former" in smoker:
        factors.append({"name": "Former Smoker", "impact": "Low-Moderate Risk", "desc": "Past smoking history carries residual vascular risk compared to never-smokers."})

    # Check Physical Activity
    if data.get("PhysicalActivities") == "No":
        factors.append({"name": "Sedentary Lifestyle", "impact": "Moderate Risk", "desc": "Lack of regular exercise increases vascular stiffness and cardiac strain."})
        advice.append("Aim for at least 150 minutes of moderate aerobic activity (brisk walking, cycling) per week.")

    # Check BMI
    bmi = float(data.get("BMI", 25.0))
    if bmi >= 30.0:
        factors.append({"name": f"Elevated BMI ({bmi:.1f})", "impact": "Moderate Risk", "desc": "Obesity places increased mechanical and metabolic burden on the heart."})
        advice.append("Target a balanced cardioprotective diet (Mediterranean or DASH) to gradually lower BMI.")
    elif bmi < 18.5:
        factors.append({"name": f"Low BMI ({bmi:.1f})", "impact": "Mild Impact", "desc": "Underweight status may reflect underlying frailty or systemic illness."})

    # Check General Health
    if data.get("GeneralHealth") in ["Poor", "Fair"]:
        factors.append({"name": f"General Health ({data.get('GeneralHealth')})", "impact": "Elevated Risk", "desc": "Sub-optimal self-rated health strongly correlates with multi-organ burden."})

    # Check Walking Difficulty
    if data.get("DifficultyWalking") == "Yes":
        factors.append({"name": "Mobility Limitations", "impact": "Elevated Risk", "desc": "Difficulty walking is often an indicator of peripheral arterial disease or frailty."})

    # Default advice if few are triggered
    if len(advice) == 0:
        advice.append("Maintain an active routine, heart-healthy nutrition, and regular annual medical checkups.")
        advice.append("Keep your blood pressure, cholesterol levels, and resting heart rate in optimal ranges.")

    return factors, advice


@app.route("/")
def index():
    return render_template("index.html", meta=model_meta)


@app.route("/api/meta", methods=["GET"])
def get_metadata():
    return jsonify(model_meta)


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No input JSON provided"}), 400

        # Construct single-row DataFrame matching trained features
        row_dict = {}
        for col in model_meta["categorical_features"]:
            row_dict[col] = [str(data.get(col, "No")).strip()]
        for col in model_meta["numerical_features"]:
            row_dict[col] = [float(data.get(col, 0.0))]

        df_row = pd.DataFrame(row_dict)

        # Predict continuous risk score
        raw_prediction = float(model.predict(df_row)[0])
        # Bound between 0.0% and 100.0%
        risk_percent = max(0.0, min(100.0, round(raw_prediction * 100, 2)))

        tier_info = determine_risk_tier(risk_percent)
        factors, advice = generate_risk_factors_and_advice(data)

        response = {
            "risk_percent": risk_percent,
            "raw_score": raw_prediction,
            "tier": tier_info["tier"],
            "badge_class": tier_info["badge_class"],
            "color": tier_info["color"],
            "accent": tier_info["accent"],
            "summary": tier_info["summary"],
            "factors": factors,
            "advice": advice,
            "population_baseline": model_meta["positive_rate_percent"],
            "relative_to_baseline": round(risk_percent / model_meta["positive_rate_percent"], 2) if model_meta["positive_rate_percent"] > 0 else 1.0,
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
