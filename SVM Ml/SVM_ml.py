import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

# Load the dataset
file_path = r"D:\H.A\new ha1\HeartDiseaseTrain-Test.csv"

# Handle potential file format issues
try:
    if file_path.endswith(".xlsx"):
        df = pd.read_excel(file_path, engine="openpyxl")
    elif file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    else:
        raise ValueError("Unsupported file format. Please upload a .xlsx or .csv file.")
except Exception as e:
    print(f"Error loading file: {e}")
    exit()

# Drop unnecessary columns
if "Patient_ID" in df.columns:
    df.drop(columns=["Patient_ID"], inplace=True)

# Separate numeric and categorical columns
numeric_cols = df.select_dtypes(include=["number"]).columns
categorical_cols = df.select_dtypes(include=["object"]).columns

# Fill missing values separately
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])

# Convert categorical data to numerical
for col in categorical_cols:
    if df[col].nunique() == 2:  # Binary categorical data (e.g., Male/Female)
        df[col] = LabelEncoder().fit_transform(df[col])
    else:
        df = pd.get_dummies(df, columns=[col], drop_first=True)  # One-hot encoding

# Define features and target variable
X = df.drop(columns=["target"])
y = df["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE for class balancing
smote = SMOTE(sampling_strategy="auto", random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Feature Scaling
scaler = StandardScaler()
X_train_resampled = scaler.fit_transform(X_train_resampled)
X_test = scaler.transform(X_test)

# Optimize SVM model using GridSearchCV
param_grid = {
    "C": [0.1, 1, 10],
    "kernel": ["linear", "rbf"],
    "gamma": ["scale", "auto"]
}

svm_model = GridSearchCV(SVC(), param_grid, cv=5, scoring="accuracy")
svm_model.fit(X_train_resampled, y_train_resampled)

# Best model evaluation
best_svm = svm_model.best_estimator_
y_pred = best_svm.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Improved SVM Model Accuracy: {accuracy * 100:.2f}%")
print(report)

# Save results to Excel
output_file = "Heart_Attack_Cases_Results.xlsx"
df["Predicted_Heart_Attack"] = best_svm.predict(scaler.transform(X))

# Ensure the directory exists
output_dir = os.path.dirname(output_file)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)

df.to_excel(output_file, index=False)
print(f"Results saved to {output_file}")
