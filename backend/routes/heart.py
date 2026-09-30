from fastapi import APIRouter
from pydantic import BaseModel
import joblib

router = APIRouter()

heart_model = joblib.load("models/heart_model.pkl")


class HeartInput(BaseModel):
    age: float
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int


@router.post("/predict/heart")
def predict(data: HeartInput):

    features = [[
        data.age,
        data.sex,
        data.cp,
        data.trestbps,
        data.chol,
        data.fbs,
        data.restecg,
        data.thalach,
        data.exang,
        data.oldpeak,
        data.slope,
        data.ca,
        data.thal
    ]]

    # Prediction
    prediction = heart_model.predict(features)[0]

    # Probability of class 1
    probabilities = heart_model.predict_proba(features)[0]
    risk_probability = float(probabilities[1])

    # Risk category
    if risk_probability >= 0.70:
        risk_level = "High"
    elif risk_probability >= 0.40:
        risk_level = "Moderate"
    else:
        risk_level = "Low"

    return {
        "prediction": int(prediction),
        "probability": round(risk_probability, 4),
        "risk_percentage": round(risk_probability * 100, 2),
        "risk_level": risk_level
    }