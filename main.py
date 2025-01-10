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

@app.post("/predict", summary="Predict Car Price with Query Parameters")
async def predict_price(
    features: CarFeatures,
    version: int = 1,  # Query parameter for version
    loss: str = "mse",  # Query parameter for loss type
    method: str = "xgb"  # Query parameter for method
):
    """
    Predict the price of a car given its features.
    Parameters:
        - features: JSON body containing car features.
        - version: Model and preprocessor version (integer).
        - loss: Loss type (e.g., 'mse', 'mae').
        - method: Model type (e.g., 'xgb').
    """
    # try:
    # Dynamically construct paths for the model and preprocessor
    model_path = f"models/models_v2/{method}_{loss}_model{version}.pkl"
    transformer_path = f"models/models_v2/preprocessor{version}.pkl"

    print(model_path)
    print(transformer_path)

    # Load the appropriate model and transformer
    ml_model = ML_Model(MODEL_PATH=model_path, TRANSFORMER_PATH=transformer_path)

    # Convert features to dictionary and predict price
    features_dict = features.dict()
    price = ml_model.predict_price(features=features_dict)

    # Format and return prediction
    price = float(round(price, 2))
    return {
        "predicted_price": price,
        "version": version,
        "loss": loss,
        "method": method,
    }

    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
