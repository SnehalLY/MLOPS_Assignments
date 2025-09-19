# main.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load trained model (ensure iris_model.pkl exists or replace with MLflow loading)
model = joblib.load("iris_model.pkl")

app = FastAPI(title="Iris Model API")

# Request body format
class IrisRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def home():
    return {"message": "Iris Prediction API is running 🚀"}

@app.post("/predict")
def predict(data: IrisRequest):
    features = np.array([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
    prediction = model.predict(features)
    return {"prediction": int(prediction[0])}
