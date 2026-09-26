import joblib
import pandas as pd

MODEL_PATH = "models/crop_prediction.pkl"

model_data = joblib.load(MODEL_PATH)

model = model_data["model"]
label_encoder = model_data["label_encoder"]
features = model_data["features"]


def predict_crop(N, P, K, temperature, humidity, ph, rainfall):

    input_data = pd.DataFrame([[
        N,
        P,
        K,
        temperature,
        humidity,
        ph,
        rainfall
    ]], columns=features)

    prediction = model.predict(input_data)

    crop = label_encoder.inverse_transform(prediction)[0]

    return crop