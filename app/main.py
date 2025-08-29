# FastAPI Deployment

from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

model = joblib.load('src/model.pkl')
scaler = joblib.load('src/scaler.pkl')

class InputData(BaseModel):
    features: list

@app.post('/predict')
def predict(data: InputData):
    input_scaled = scaler.transform([data.features])
    prediction = model.predict(input_scaled)
    return {'prediction': int(prediction[0])}