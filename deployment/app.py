from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import uvicorn
import os

# Load model and artifacts
MODEL_PATH = "/content/drive/MyDrive/Project/HB1_VISA_PREDICTION/src/models/xgboost.pkl"
SCALER_PATH = "/content/drive/MyDrive/Project/HB1_VISA_PREDICTION/src/models/scaler.pkl"
ENCODERS_PATH = "/content/drive/MyDrive/Project/HB1_VISA_PREDICTION/src/models/label_encoders.pkl"
TARGET_ENCODER_PATH = "/content/drive/MyDrive/Project/HB1_VISA_PREDICTION/src/models/target_encoder.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
label_encoders = joblib.load(ENCODERS_PATH)
target_encoder = joblib.load(TARGET_ENCODER_PATH)

# Define input schema using Pydantic
class InputData(BaseModel):
    continent: str
    education_of_employee: str
    has_job_experience: str
    requires_job_training: str
    region_of_employment: str
    unit_of_wage: str
    full_time_position: str
    no_of_employees: int
    yr_of_estab: int
    prevailing_wage: float

# Define feature order
categorical_cols = [
    "continent",
    "education_of_employee",
    "has_job_experience",
    "requires_job_training",
    "region_of_employment",
    "unit_of_wage",
    "full_time_position"
]
numerical_cols = ["no_of_employees", "yr_of_estab", "prevailing_wage"]

# Create FastAPI app
app = FastAPI(title="Easy Labor Prediction Application")

@app.post("/predict", tags=["Prediction"])
def predict(input_data: InputData):
    try:
        data = input_data.dict()

        # 1. Encode categorical values
        for col in categorical_cols:
            le = label_encoders.get(col)
            value = data[col]
            if value not in le.classes_:
                return {
                    "error": f"Invalid value '{value}' for '{col}'. Expected one of: {list(le.classes_)}"
                }
            data[col] = le.transform([value])[0]

        # 2. Separate and scale numeric values
        X_num = np.array([[data[col] for col in numerical_cols]])
        X_num_scaled = scaler.transform(X_num)

        # 3. Combine all features: encoded categorical + scaled numeric
        X_final = np.array([
            data['continent'],
            data['education_of_employee'],
            data['has_job_experience'],
            data['requires_job_training'],
            data['region_of_employment'],
            data['unit_of_wage'],
            data['full_time_position'],
            *X_num_scaled[0]
        ]).reshape(1, -1)

        # 4. Predict
        pred_num = model.predict(X_final)[0]
        pred_label = target_encoder.inverse_transform([pred_num])[0]

        result = "Certified" if str(pred_label).lower() == "certified" or str(pred_label) == "1" else "Denied"

        return {
            "prediction_raw": int(pred_num),
            "prediction_label": result
        }

    except Exception as e:
        print(f"❌ INTERNAL ERROR: {e}")
        return {"error": str(e)}
