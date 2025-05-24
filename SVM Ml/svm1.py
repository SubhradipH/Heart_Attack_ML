import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Load dataset
file_path = r"D:\H.A\new ha1\HeartDiseaseTrain-Test.csv"  # Updated file path
df = pd.read_csv(file_path)

# Rename columns to match dataset
df.rename(columns={
    "Resting_BP": "resting_blood_pressure",
    "Max_Heart_Rate": "Max_heart_rate"
}, inplace=True)

# Feature Engineering
df["heart_rate_variability"] = df["Max_heart_rate"] - df["resting_blood_pressure"]
df["blood_pressure_ratio"] = df["resting_blood_pressure"] / df["Max_heart_rate"]
df["hypertension"] = (df["resting_blood_pressure"] > 140).astype(int)
df["exercise_intensity"] = df["Max_heart_rate"] / df["age"]
df["age_category"] = pd.cut(df["age"], bins=[0, 40, 60, 100], labels=["Young", "Middle-aged", "Senior"])
df["cholestoral_age_interaction"] = df["cholestoral"] * df["age"]
df["st_depression_squared"] = df["oldpeak"] ** 2


# Define categorical and numerical features
categorical_features = ["Sex", "Chest_Pain_Type", "Fasting_Blood_Sugar", "Resting_ECG",
                        "Exercise_Induced_Angina", "Slope_ST_Segment", "Major_Vessels",
                        "Thalassemia", "age_category", "hypertension"]

numerical_features = ["Age", "Resting_Blood_Pressure", "Cholesterol", "Max_heart_rate", "ST_Depression",
                      "heart_rate_variability", "blood_pressure_ratio", "exercise_intensity",
                      "cholesterol_age_interaction", "st_depression_squared"]

# Preprocessing Pipeline
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])

# Prepare features and target
X = df.drop(columns=["Heart_Attack", "Patient_ID"])
y = df["Heart_Attack"]

# Apply preprocessing before SMOTE
X_processed = preprocessor.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42, stratify=y)

# Apply improved SMOTE
smote = SMOTE(sampling_strategy=1.0, random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Train optimized SVM model
model = SVC(kernel="rbf", C=20, gamma="auto", probability=True, class_weight="balanced")
model.fit(X_train_resampled, y_train_resampled)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Improved SVM Model Accuracy: {accuracy:.2%}")
print(f"Precision: {precision:.2%}")
print(f"Recall: {recall:.2%}")
print(f"F1-Score: {f1:.2%}")

# Print classification report
print("\nDetailed Classification Report:")
report = classification_report(y_test, y_pred, target_names=["No Heart Attack", "Heart Attack"])
print(report)

# Save heart attack patient data
heart_attack_cases = df[df["Heart_Attack"] == 1]
output_file = "Heart_Attack_Cases_SVM_Improved.xlsx"
heart_attack_cases.to_excel(output_file, index=False)
print(f"Heart attack patient data saved to {output_file}")

# Plot patient distribution
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

# Plot Precision, Recall, and F1-score over iterations
metrics = ["Accuracy", "Precision", "Recall", "F1-score"]
values = [accuracy, precision, recall, f1]

plt.figure(figsize=(8, 5))
plt.plot(metrics, values, marker="o", linestyle="-", color="purple")
plt.xlabel("Metrics")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.title("Model Performance Metrics")
plt.grid(True)

for i, v in enumerate(values):
    plt.text(i, v, f"{v:.2%}", ha="center", fontsize=12)

plt.show()
