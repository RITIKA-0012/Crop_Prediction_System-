import joblib
import pandas as pd

MODEL_PATH = "models/crop_prediction.pkl"

model_data = joblib.load(MODEL_PATH)

model = model_data["model"]
label_encoder = model_data["label_encoder"]
features = model_data["features"]


def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
    input_data = pd.DataFrame([[
        N, P, K, temperature, humidity, ph, rainfall
    ]], columns=features)

    prediction = model.predict(input_data)
    crop = label_encoder.inverse_transform(prediction)[0]
    return crop


def predict_crop_details(N, P, K, temperature, humidity, ph, rainfall):
    input_data = pd.DataFrame([[
        N, P, K, temperature, humidity, ph, rainfall
    ]], columns=features)

    # Get probabilities for all crops
    probs = model.predict_proba(input_data)[0]
    top_indices = probs.argsort()[::-1][:3]

    top_predictions = [
        {
            "crop": label_encoder.inverse_transform([idx])[0],
            "confidence": round(float(probs[idx]) * 100, 2)
        }
        for idx in top_indices
    ]

    # Get feature importances from Random Forest
    feature_imp = dict(zip(features, model.feature_importances_))

    return top_predictions, feature_imp