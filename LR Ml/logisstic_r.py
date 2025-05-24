import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
file_path = r"D:\H.A\new ha1\HeartDiseaseTrain-Test.csv" # Update the path if needed
df = pd.read_csv(file_path)

# Feature Engineering: Creating new features
df["age_cholestoral"] = df["age"] * df["cholestoral"]  # Interaction feature
df["oldpeak_squared"] = df["oldpeak"] ** 2  # Non-linear relationship
df["heart_rate_variability"] = df["Max_heart_rate"] - df["resting_blood_pressure"]  # HR Variability
df["hypertension"] = (df["resting_blood_pressure"] > 140).astype(int)  # Binary hypertension feature
df["exercise_intensity"] = df["Max_heart_rate"] / df["age"]  # Normalized exercise intensity

# Creating age categories
df["age_category"] = pd.cut(df["age"], bins=[0, 40, 60, 100], labels=["Young", "Middle-aged", "Senior"])

# Define categorical and numerical features
categorical_features = ["sex", "chest_pain_type", "fasting_blood_sugar", "rest_ecg",
                        "exercise_induced_angina", "slope", "vessels_colored_by_flourosopy",
                        "thalassemia", "age_category"]

numerical_features = ["age", "resting_blood_pressure", "cholestoral", "Max_heart_rate", 
                      "oldpeak", "age_cholestoral", "oldpeak_squared", "heart_rate_variability", 
                      "hypertension", "exercise_intensity"]

# Convert categorical variables to numeric using OneHotEncoder
encoder = OneHotEncoder(handle_unknown="ignore", drop="first")
encoded_categorical = pd.DataFrame(encoder.fit_transform(df[categorical_features]).toarray())

# Standardize numerical features
scaler = StandardScaler()
scaled_numerical = pd.DataFrame(scaler.fit_transform(df[numerical_features]), columns=numerical_features)

# Combine numerical and encoded categorical features
X = pd.concat([scaled_numerical, encoded_categorical], axis=1)
y = df["target"]  # Target variable

# ✅ Fix: Convert feature names to string
X.columns = X.columns.astype(str)

# Splitting dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Apply SMOTE for class balancing
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Apply polynomial features to capture non-linearity
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_train_poly = poly.fit_transform(X_train_resampled)
X_test_poly = poly.transform(X_test)

# Define Logistic Regression model pipeline
model = LogisticRegression(C=15, max_iter=1000)

# Train the model
model.fit(X_train_poly, y_train_resampled)

# Evaluate model performance
y_pred = model.predict(X_test_poly)
accuracy = accuracy_score(y_test, y_pred)
print(f"Improved Logistic Regression Model Accuracy: {accuracy:.2%}")

# Save heart attack patient data
heart_attack_cases = df[df["target"] == 1]
output_file = "Heart_Attack_Cases_Logistic.xlsx"
heart_attack_cases.to_excel(output_file, index=False)
print(f"Heart attack patient data saved to {output_file}")

# Plot the bar chart
total_patients = len(df)
heart_attack_patients = len(heart_attack_cases)
healthy_patients = total_patients - heart_attack_patients

labels = ["Total Patients", "Heart Attack Patients", "Healthy Patients"]
counts = [total_patients, heart_attack_patients, healthy_patients]

plt.figure(figsize=(8, 5))
plt.bar(labels, counts, color=["blue", "red", "green"])
plt.ylabel("Number of Patients")
plt.title("Patient Distribution")

# Display values on bars
for i, count in enumerate(counts):
    plt.text(i, count + 10, str(count), ha="center", fontsize=12)

plt.show()
