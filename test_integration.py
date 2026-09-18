from app import app

client = app.test_client()

# 1. Test GET /
res = client.get('/')
assert res.status_code == 200, f'GET / failed: {res.status_code}'
assert b'CardioRisk' in res.data
assert b'Decision Tree Regression' in res.data
print('[PASS] GET / returned 200 OK with proper HTML content.')

# 2. Test static files
res_css = client.get('/static/css/style.css')
assert res_css.status_code == 200, 'CSS failed'
print('[PASS] /static/css/style.css returned 200 OK.')

res_js = client.get('/static/js/app.js')
assert res_js.status_code == 200, 'JS failed'
print('[PASS] /static/js/app.js returned 200 OK.')

# 3. Test presets via /api/predict
presets = {
    'Healthy Adult': {
        'AgeCategory': 'Age 25 to 29', 'Sex': 'Female', 'GeneralHealth': 'Excellent',
        'BMI': 21.5, 'PhysicalActivities': 'Yes', 'SmokerStatus': 'Never smoked',
        'LastCheckupTime': 'Within past year (anytime less than 12 months ago)',
        'HadAngina': 'No', 'HadStroke': 'No', 'HadCOPD': 'No', 'HadKidneyDisease': 'No',
        'ChestScan': 'No', 'DifficultyWalking': 'No', 'HadArthritis': 'No',
        'HadDiabetes': 'No', 'HadDepressiveDisorder': 'No', 'AlcoholDrinkers': 'No',
        'HadAsthma': 'No', 'SleepHours': 8, 'PhysicalHealthDays': 0, 'MentalHealthDays': 0
    },
    'Moderate Risk Adult': {
        'AgeCategory': 'Age 65 to 69', 'Sex': 'Male', 'GeneralHealth': 'Fair',
        'BMI': 29.5, 'PhysicalActivities': 'No', 'SmokerStatus': 'Current smoker - now smokes every day',
        'LastCheckupTime': 'Within past year (anytime less than 12 months ago)',
        'HadAngina': 'No', 'HadStroke': 'No', 'HadCOPD': 'No', 'HadKidneyDisease': 'No',
        'ChestScan': 'Yes', 'DifficultyWalking': 'Yes', 'HadArthritis': 'Yes',
        'HadDiabetes': 'Yes', 'HadDepressiveDisorder': 'No',
        'AlcoholDrinkers': 'No', 'HadAsthma': 'No', 'SleepHours': 6.0, 'PhysicalHealthDays': 10, 'MentalHealthDays': 5
    },
    'High Risk Cardiac Profile': {
        'AgeCategory': 'Age 70 to 74', 'Sex': 'Male', 'GeneralHealth': 'Poor',
        'BMI': 33.2, 'PhysicalActivities': 'No', 'SmokerStatus': 'Current smoker - now smokes every day',
        'LastCheckupTime': 'Within past year (anytime less than 12 months ago)',
        'HadAngina': 'Yes', 'HadStroke': 'Yes', 'HadCOPD': 'Yes', 'HadKidneyDisease': 'Yes',
        'ChestScan': 'Yes', 'DifficultyWalking': 'Yes', 'HadArthritis': 'Yes',
        'HadDiabetes': 'Yes', 'HadDepressiveDisorder': 'Yes', 'AlcoholDrinkers': 'No',
        'HadAsthma': 'Yes', 'SleepHours': 5, 'PhysicalHealthDays': 15, 'MentalHealthDays': 10
    }
}

for name, payload in presets.items():
    res = client.post('/api/predict', json=payload)
    assert res.status_code == 200
    data = res.get_json()
    risk = data['risk_percent']
    tier = data['tier']
    color = data['color']
    print(f'[PASS] {name}: Risk={risk}%, Tier={tier}, Color={color}')

print('\nAll integration tests succeeded!')
