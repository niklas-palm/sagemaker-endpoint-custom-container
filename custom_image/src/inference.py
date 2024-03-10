import os

import joblib
import pandas as pd
from xgboost import XGBClassifier


def load_model(model_dir: str) -> XGBClassifier:
    """
    Load the model from the specified directory.
    """
    return joblib.load(os.path.join(model_dir, "model.pkl"))


def predict(body: dict, model: XGBClassifier) -> dict:
    """
    Generate predictions for the incoming request using the model.
    """
    features = pd.DataFrame.from_records(body["features"])
    predictions = model.predict(features).tolist()
    return {"predictions": predictions}
