from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd


app = FastAPI(title="House Price Predictor API")

model = joblib.load("models/house_price_model.pkl")
scaler = joblib.load("models/scaler.pkl")


class HouseFeatures(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float


@app.get("/")
def health_check():
    return {"status": "House Price Predictor API is running!"}


@app.post("/predict")
def predict_price(features: HouseFeatures):
    input_data = pd.DataFrame([{
        "MedInc": features.MedInc,
        "HouseAge": features.HouseAge,
        "AveRooms": features.AveRooms,
        "AveBedrms": features.AveBedrms,
        "Population": features.Population,
        "AveOccup": features.AveOccup,
        "Latitude": features.Latitude,
        "Longitude": features.Longitude
    }])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    predicted_price = round(prediction[0] * 100000, 2)
    return {
        "predicted_price_usd": predicted_price,
        "predicted_price_formatted": f"${predicted_price:,.0f}"
    }
