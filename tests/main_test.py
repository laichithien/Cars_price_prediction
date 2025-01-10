from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from scripts.inference import ML_Model

# Initialize the FastAPI app
app = FastAPI()

# model = ML_Model()

# Define the schema for input features
class CarFeatures(BaseModel):
    manufacture_date: int
    brand: str
    model: str
    origin: str
    type: str
    seats: float
    gearbox: str
    fuel: str
    color: str
    mileage_v2: int
    condition: str

@app.post("/test")
async def test(data: CarFeatures):
    print(data.dict())
    input_data = pd.DataFrame([data.dict()])
    print(input_data)
    return data

@app.post("/predict", summary="Predict Car Price")
async def predict_price(
    features: CarFeatures,
    version: int = 1,
    loss: str = 'mse',
    method: str = 'xgb'
    ):
    """
    Predict the price of a car given its features.
    """
    try:
        features = features.dict()
        price = model.predict_price(features=features)
        price = float(round(price, 2))
        return {"predicted_price": price}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    