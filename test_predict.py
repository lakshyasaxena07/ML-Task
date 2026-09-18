"""
Verification test script for Heart Attack Risk prediction model and API
"""

import json
from app import app

def test_predictions():
    client = app.test_client()

    # 1. Test Metadata API
    meta_res = client.get("/api/meta")
    assert meta_res.status_code == 200, f"Expected 200, got {meta_res.status_code}"
    meta_json = meta_res.get_json()
    assert meta_json["model_type"] == "Decision Tree Regressor"
    assert "cv_results" in meta_json
    print("[PASS] /api/meta returned valid metadata.")

    # 2. Test Low Risk Profile (Healthy young non-smoker)
    low_risk_payload = {
        "AgeCategory": "Age 25 to 29",
        "Sex": "Female",
        "GeneralHealth": "Excellent",
        "LastCheckupTime": "Within past year (anytime less than 12 months ago)",
        "PhysicalActivities": "Yes",
        "SmokerStatus": "Never smoked",
        "HadAngina": "No",
        "HadStroke": "No",
        "HadAsthma": "No",
        "HadCOPD": "No",
        "HadDepressiveDisorder": "No",
        "HadKidneyDisease": "No",
        "HadArthritis": "No",
        "HadDiabetes": "No",
        "DifficultyWalking": "No",
        "ChestScan": "No",
        "AlcoholDrinkers": "No",
        "BMI": 22.0,
        "PhysicalHealthDays": 0.0,
        "MentalHealthDays": 0.0,
        "SleepHours": 8.0,
    }

    res_low = client.post("/api/predict", json=low_risk_payload)
    assert res_low.status_code == 200
    low_data = res_low.get_json()
    print(f"[PASS] Low Risk Patient: {low_data['risk_percent']}% ({low_data['tier']})")
    assert low_data["risk_percent"] < 5.0, f"Expected < 5%, got {low_data['risk_percent']}%"

    # 3. Test High Risk Profile (Older adult with Angina, Stroke, Diabetes, Smoker)
    high_risk_payload = {
        "AgeCategory": "Age 75 to 79",
        "Sex": "Male",
        "GeneralHealth": "Poor",
        "LastCheckupTime": "Within past year (anytime less than 12 months ago)",
        "PhysicalActivities": "No",
        "SmokerStatus": "Current smoker - now smokes every day",
        "HadAngina": "Yes",
        "HadStroke": "Yes",
        "HadAsthma": "No",
        "HadCOPD": "Yes",
        "HadDepressiveDisorder": "No",
        "HadKidneyDisease": "Yes",
        "HadArthritis": "Yes",
        "HadDiabetes": "Yes",
        "DifficultyWalking": "Yes",
        "ChestScan": "Yes",
        "AlcoholDrinkers": "No",
        "BMI": 33.5,
        "PhysicalHealthDays": 20.0,
        "MentalHealthDays": 10.0,
        "SleepHours": 5.0,
    }

    res_high = client.post("/api/predict", json=high_risk_payload)
    assert res_high.status_code == 200
    high_data = res_high.get_json()
    print(f"[PASS] High Risk Patient: {high_data['risk_percent']}% ({high_data['tier']})")
    assert high_data["risk_percent"] > 30.0, f"Expected > 30%, got {high_data['risk_percent']}%"

    print("\nAll model prediction tests passed successfully!")

if __name__ == "__main__":
    test_predictions()
