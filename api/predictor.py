import joblib
import shap
import numpy as np
import pandas as pd
import os 
import sys

# Allow imports from src/
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.data import CONTINUOUS_FEATURES, CATEGORICAL_FEATURES

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")

# Load once at startup
pipeline = joblib.load(MODEL_PATH)
preprocessor = pipeline.named_steps['preprocessor']
classifier = pipeline.named_steps['classifier']

# Load training data as SHAP background
X_train_background = np.load(os.path.join(BASE_DIR, "models", "X_train_transformed.npy"))
explainer = shap.LinearExplainer(classifier, X_train_background)

ALL_FEATURES = CONTINUOUS_FEATURES + CATEGORICAL_FEATURES

def predict(patient_dict: dict) -> dict:
    input_df = pd.DataFrame([patient_dict], columns=ALL_FEATURES)

    X_transformed = preprocessor.transform(input_df)

    prediction = int(classifier.predict(X_transformed)[0])
    probability = float(classifier.predict_proba(X_transformed)[0][1])

    shap_vals = explainer(X_transformed)
    
    cat_feature_names = (preprocessor
                        .named_transformers_['categorical']
                        .named_steps['encoder']
                        .get_feature_names_out(CATEGORICAL_FEATURES)
                        .tolist())
    feature_names = CONTINUOUS_FEATURES + cat_feature_names

    # Map SHAP values to feature names
    shap_dict = {
        name: round(float(val), 4)
        for name, val in zip(feature_names, shap_vals.values[0])
    }

    return {
        "prediction": prediction,
        "prediction_label": "Disease" if prediction == 1 else "No Disease",
        "probability": round(probability, 4),
        "shap_values": shap_dict
    }