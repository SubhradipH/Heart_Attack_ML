import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

# Load the dataset
file_path = r"D:\H.A\new ha1\HeartDiseaseTrain-Test.csv"   # Update the path if needed
df = pd.read_csv(file_path)

# Define categorical and numerical features
categorical_features = ["sex", "chest_pain_type", "fasting_blood_sugar", "rest_ecg", 
                        "exercise_induced_angina", "slope", "vessels_colored_by_flourosopy", "thalassemia"]
numerical_features = ["age", "resting_blood_pressure", "cholestoral", "Max_heart_rate", "oldpeak"]

# Define preprocessing steps
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])

# Define XGBoost model pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss'))
])

# Split dataset into train and test sets
X = df.drop(columns=["target"])
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train the model
model.fit(X_train, y_train)

# Evaluate model performance
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]  # For ROC curve

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print(f"XGBoost Model Accuracy: {accuracy:.2%}")
print(f"Precision: {precision:.2%}")
print(f"Recall: {recall:.2%}")
print(f"F1 Score: {f1:.2%}")
print(f"ROC AUC: {roc_auc:.2%}")



# Filter heart attack patients
heart_attack_cases = df[df["target"] == 1]

# Save to an Excel file
output_file = "Heart_Attack_Cases_XGBoost.xlsx"
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
