import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE

# Load data
df = pd.read_csv(r"D:\H.A\new ha1\HeartDiseaseTrain-Test.csv")

# Drop Patient_ID if exists
if "Patient_ID" in df.columns:
    df.drop(columns=["Patient_ID"], inplace=True)

# Handle missing values
numeric_cols = df.select_dtypes(include=["number"]).columns
categorical_cols = df.select_dtypes(include=["object"]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])

# Encode categorical
for col in categorical_cols:
    if df[col].nunique() == 2:
        df[col] = LabelEncoder().fit_transform(df[col])
    else:
        df = pd.get_dummies(df, columns=[col], drop_first=True)

# Feature Engineering
df["age_cholestoral"] = df["age"] * df["cholestoral"]
df["oldpeak_squared"] = df["oldpeak"] ** 2
df["heart_rate_variability"] = df["Max_heart_rate"] - df["resting_blood_pressure"]
df["hypertension"] = (df["resting_blood_pressure"] > 140).astype(int)
df["exercise_intensity"] = df["Max_heart_rate"] / df["age"]

# Split features and target
X = df.drop(columns=["target"])
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Balance classes
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# Tune KNN
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

grid_knn = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_knn.fit(X_train_scaled, y_train_resampled)

best_knn = grid_knn.best_estimator_
print(f"\n Best KNN Parameters: {grid_knn.best_params_}")

# Evaluate
y_pred = best_knn.predict(X_test_scaled)
y_prob = best_knn.predict_proba(X_test_scaled)[:, 1]

acc = accuracy_score(y_test, y_pred)
print(f"\n Tuned KNN Accuracy: {acc * 100:.2f}%")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=["No Attack", "Attack"], yticklabels=["No Attack", "Attack"])
plt.title("Confusion Matrix - Tuned KNN")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f'Tuned KNN (AUC = {roc_auc:.2f})', color='green')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Tuned KNN')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# Save model and scaler
joblib.dump(best_knn, "knn_best_model.pkl")
joblib.dump(scaler, "knn_best_scaler.pkl")
print("\n Tuned KNN model and scaler saved successfully.")
