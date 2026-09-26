import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report


# Load dataset
DATA_PATH = "data/Crop_recommendation.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())


# Features and target
X = df.drop("label", axis=1)
y = df["label"]


# Encode crop names
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)


# Train Random Forest
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)


# Evaluate
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Training Completed!")
print("Accuracy:", accuracy)


# Save model + encoder + feature names
model_data = {
    "model": model,
    "label_encoder": label_encoder,
    "features": X.columns.tolist()
}

joblib.dump(model_data, "models/crop_prediction.pkl")

print("\nModel saved successfully!")
print("Location: models/crop_prediction.pkl")