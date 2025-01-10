from scripts.inference import ML_Model

model = ML_Model()

example_features = {
        "manufacture_date": 2021,
        "brand": "Toyota",
        "model": "Camry",
        "origin": "Việt Nam",
        "type": "Sedan",
        "seats": 5.0,
        "gearbox": "AT",
        "fuel": "petrol",
        "color": "black",
        "mileage_v2": 100000,
        "condition": "used"
    }

price_pred = model.predict_price(features=example_features)
print(f"Predicted Price: {price_pred:,.2f} VND")