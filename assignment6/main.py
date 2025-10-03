# main.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# load model at startup (ensure iris_model.pkl present)
model = joblib.load("iris_model.pkl")

app = FastAPI(title="Iris Classifier API")

@app.get("/")
def home():
    return {"message": "Iris Classifier API is up"}

@app.post("/predict")
def predict(data: IrisInput):
    features = np.array([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
    pred = model.predict(features)[0]
    return {"prediction": int(pred)}
