import os
import json
import pickle
import numpy as np

# Base path of this file (server/util.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths to model files (relative to server/)
MODEL_PATH = os.path.join(BASE_DIR, 'bangalore_home_prediction_model.pickle')
FEATURES_PATH = os.path.join(BASE_DIR, 'model_features.json')

# Load model
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

# Load feature names
with open(FEATURES_PATH, 'r') as f:
    feature_columns = json.load(f)

def predict_price(total_sqft, bhk, bath, location):
    input_array = np.zeros(len(feature_columns))
    input_array[feature_columns.index('total_squareft')] = total_sqft
    input_array[feature_columns.index('bhk')] = bhk
    input_array[feature_columns.index('bath')] = bath

    location_key = f"location_{location.lower()}"
    if location_key in feature_columns:
        input_array[feature_columns.index(location_key)] = 1

    pred_log_price = model.predict([input_array])[0]
    return round(np.expm1(pred_log_price), 2)


def get_all_locations():
    return sorted([
        col.replace("location_", "").title()
        for col in feature_columns
        if col.startswith("location_")
    ])
