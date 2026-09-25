import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load Dataset
df = pd.read_csv("dataset/Crop_recommendation.csv")

# Display Dataset Information
print("First 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

# Separate Features and Target
X = df.drop("label", axis=1)
y = df["label"]

# Encode Crop Names
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

print("\nCrop Classes:")
print(label_encoder.classes_)

print("\nPreprocessing Completed Successfully!")