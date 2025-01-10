import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

class ML_Model:
    def __init__(self, MODEL_PATH: str, TRANSFORMER_PATH: str):
        self.model = joblib.load(MODEL_PATH)
        self.transformer = joblib.load(TRANSFORMER_PATH)
    
    def predict_price(self, features: dict) -> float:
        """
        Predicts the price of a car given its features.
        """
        input_data = pd.DataFrame([features])
        transformed_data = self.transform_data(input_data=input_data)
        # Predict the log price
        log_price_pred = self.model.predict(transformed_data)


        # Reverse the log1p transformation to get the original price
        price_pred = np.expm1(log_price_pred)
        return price_pred[0]
    
    def transform_data(self, input_data: pd.DataFrame):
        """
        Transforms input data using the preprocessor.
        """
        categorical_features = ['brand', 'model', 'origin', 'type', 'gearbox', 'fuel', 'color']
        numerical_features = ['seats', 'mileage_v2', 'car_age']
        current_year = 2024
        input_data['car_age'] = current_year - input_data['manufacture_date']
        transformed_data = self.transformer.transform(input_data)
        one_hot_columns = self.transformer.transformers_[1][1].get_feature_names_out(categorical_features)
        all_columns = numerical_features + list(one_hot_columns)
        transformed_data = transformed_data.toarray()
        transformed_data = pd.DataFrame(transformed_data, columns=all_columns)
        return transformed_data
