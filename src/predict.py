import pandas as pd
import joblib

def predict_patient():
    try:
        model = joblib.load('../models/lung_cancer_rf_model.pkl')
        scaler = joblib.load('../models/scaler.pkl')
    except FileNotFoundError:
        print("Error: Model or scaler not found. Please run preprocessing and training first.")
        return

    # Define the exact features the model was trained on
    feature_columns = ['GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY',
                       'PEER_PRESSURE', 'CHRONIC DISEASE', 'FATIGUE ', 'ALLERGY ',
                       'WHEEZING', 'ALCOHOL CONSUMING', 'COUGHING', 'SHORTNESS OF BREATH',
                       'SWALLOWING DIFFICULTY', 'CHEST PAIN']
    
    # Example Patient Data (1 = No/Female/Male depending on mapping, 2 = Yes)
    patient_data = {
        'GENDER': 1,               # 1=Male, 0=Female
        'AGE': 65,                 
        'SMOKING': 2,              # 2=Yes, 1=No
        'YELLOW_FINGERS': 1,       
        'ANXIETY': 1,
        'PEER_PRESSURE': 1,
        'CHRONIC DISEASE': 1,
        'FATIGUE ': 2,             
        'ALLERGY ': 1,             
        'WHEEZING': 2,
        'ALCOHOL CONSUMING': 1,
        'COUGHING': 2,
        'SHORTNESS OF BREATH': 2,
        'SWALLOWING DIFFICULTY': 1,
        'CHEST PAIN': 2
    }
    
    patient_df = pd.DataFrame([patient_data], columns=feature_columns)

    # Scale the patient's data
    patient_scaled = scaler.transform(patient_df)

    # Make the prediction
    prediction = model.predict(patient_scaled)
    probability = model.predict_proba(patient_scaled)[0][1]

    # Output results
    print("\n--- Lung Cancer Prediction ---")
    if prediction[0] == 1:
        print("Result: HIGH RISK (Positive)")
    else:
        print("Result: LOW RISK (Negative)")
    
    print(f"Model Confidence: {probability * 100:.2f}%")

if __name__ == "__main__":
    predict_patient()