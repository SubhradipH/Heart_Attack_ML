import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Load model and scaler
model = joblib.load("svm_heart_attack_model.pkl")
scaler = joblib.load("svm_scaler.pkl")

# Load test dataset
df_test = pd.read_csv(r"D:\H.A\new ha1\test.csv")

# Drop unnecessary columns
if "Patient_ID" in df_test.columns:
    df_test.drop(columns=["Patient_ID"], inplace=True)

# Handle missing values
numeric_cols = df_test.select_dtypes(include=["number"]).columns
categorical_cols = df_test.select_dtypes(include=["object"]).columns
df_test[numeric_cols] = df_test[numeric_cols].fillna(df_test[numeric_cols].median())
df_test[categorical_cols] = df_test[categorical_cols].fillna(df_test[categorical_cols].mode().iloc[0])

# Encode binary categorical variables
for col in categorical_cols:
    if df_test[col].nunique() == 2:
        df_test[col] = LabelEncoder().fit_transform(df_test[col])

# One-hot encode remaining categorical columns
df_test = pd.get_dummies(df_test, drop_first=True)

# Feature Engineering
df_test["age_cholestoral"] = df_test["age"] * df_test["cholestoral"]
df_test["oldpeak_squared"] = df_test["oldpeak"] ** 2
df_test["heart_rate_variability"] = df_test["Max_heart_rate"] - df_test["resting_blood_pressure"]
df_test["hypertension"] = (df_test["resting_blood_pressure"] > 140).astype(int)
df_test["exercise_intensity"] = df_test["Max_heart_rate"] / df_test["age"]

# Ensure same feature order as training
model_features = scaler.feature_names_in_
for col in model_features:
    if col not in df_test.columns:
        df_test[col] = 0  # Add missing column with 0

df_test = df_test[model_features]  # Reorder columns to match training

# Scale test data
X_test_scaled = scaler.transform(df_test)

# Predict
predictions = model.predict(X_test_scaled)

# Show result
print("\n🩺 Prediction Results:")
for i, pred in enumerate(predictions):
    result = "Yes" if pred == 1 else "No"
    print(f"Person {i + 1}: Heart Attack: {result}")
