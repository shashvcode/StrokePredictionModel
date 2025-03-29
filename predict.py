import joblib
import pandas as pd

loaded_model = joblib.load('best_stroke_prediction_model.joblib')

data = {
    'age': [int(input("Enter your age: "))],
    'chest_pain': [int(input("Chest pain (0 or 1): "))],
    'high_blood_pressure': [int(input("High blood pressure (0 or 1): "))],
    'irregular_heartbeat': [int(input("Irregular heartbeat (0 or 1): "))],
    'shortness_of_breath': [int(input("Shortness of breath (0 or 1): "))],
    'fatigue_weakness': [int(input("Fatigue or weakness (0 or 1): "))],
    'dizziness': [int(input("Dizziness (0 or 1): "))],
    'swelling_edema': [int(input("Swelling or edema (0 or 1): "))],
    'neck_jaw_pain': [int(input("Neck or jaw pain (0 or 1): "))],
    'excessive_sweating': [int(input("Excessive sweating (0 or 1): "))],
    'persistent_cough': [int(input("Persistent cough (0 or 1): "))],
    'nausea_vomiting': [int(input("Nausea or vomiting (0 or 1): "))],
    'chest_discomfort': [int(input("Chest discomfort (0 or 1): "))],
    'cold_hands_feet': [int(input("Cold hands or feet (0 or 1): "))],
    'snoring_sleep_apnea': [int(input("Snoring or sleep apnea (0 or 1): "))],
    'anxiety_doom': [int(input("Feeling of doom or anxiety (0 or 1): "))],
    'gender': [input("Gender (Male or Female): ")]
}

X_new = pd.DataFrame(data)
prediction = loaded_model.predict(X_new)[0]
print(f"Estimated Stroke Risk: {round(prediction, 2)}%")