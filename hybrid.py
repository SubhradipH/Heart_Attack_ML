import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Load dataset
df = pd.read_csv(r"D:\H.A\new ha1\HeartDiseaseTrain-Test.csv")

# Drop ID column if exists
if "Patient_ID" in df.columns:
    df.drop(columns=["Patient_ID"], inplace=True)

# Handle missing values
numeric_cols = df.select_dtypes(include=["number"]).columns
categorical_cols = df.select_dtypes(include=["object"]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])

# Encode categorical features
for col in categorical_cols:
    if df[col].nunique() == 2:
        df[col] = LabelEncoder().fit_transform(df[col])
    else:
        df = pd.get_dummies(df, columns=[col], drop_first=True)

# Feature engineering
df["age_cholestoral"] = df["age"] * df["cholestoral"]
df["oldpeak_squared"] = df["oldpeak"] ** 2
df["heart_rate_variability"] = df["Max_heart_rate"] - df["resting_blood_pressure"]
df["hypertension"] = (df["resting_blood_pressure"] > 140).astype(int)
df["exercise_intensity"] = df["Max_heart_rate"] / df["age"]

# Split dataset
X = df.drop(columns=["target"])
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔥 Standard Scaler (same for all models)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ✅ Hyperparameter tuning for KNN
knn_param_grid = {
    'n_neighbors': [3, 5, 7],
    'weights': ['uniform', 'distance']
}
knn_grid = GridSearchCV(KNeighborsClassifier(), knn_param_grid, cv=5, scoring='accuracy')
knn_grid.fit(X_train_scaled, y_train)
best_knn = knn_grid.best_estimator_
print("Best KNN params:", knn_grid.best_params_)

# ✅ Hyperparameter tuning for SVM (probability=True for predict_proba)
svm_param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['rbf', 'linear'],
    'gamma': ['scale', 'auto']
}
svm_grid = GridSearchCV(SVC(probability=True), svm_param_grid, cv=5, scoring='accuracy')
svm_grid.fit(X_train_scaled, y_train)
best_svm = svm_grid.best_estimator_
print("Best SVM params:", svm_grid.best_params_)

# ✅ Optional: Add Logistic Regression for triple hybrid
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_scaled, y_train)

# Prediction probabilities
prob_knn = best_knn.predict_proba(X_test_scaled)[:, 1]
prob_svm = best_svm.predict_proba(X_test_scaled)[:, 1]
prob_log = log_model.predict_proba(X_test_scaled)[:, 1]

# ✅ Weighted soft voting
# Example weights: 0.5 for SVM, 0.3 for KNN, 0.2 for Logistic Regression
avg_prob = (0.5 * prob_svm + 0.3 * prob_knn + 0.2 * prob_log)
final_pred = (avg_prob >= 0.5).astype(int)

# Accuracy and report
accuracy = accuracy_score(y_test, final_pred)
print(f"\n✅ Hybrid Model Accuracy: {accuracy * 100:.2f}%\n")
print(classification_report(y_test, final_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, final_pred)
print("Confusion Matrix:\n", cm)

# Plot Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Attack", "Attack"], yticklabels=["No Attack", "Attack"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Hybrid Model Confusion Matrix")
plt.tight_layout()
plt.show()

# Plot ROC Curve
fpr, tpr, _ = roc_curve(y_test, avg_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='purple', lw=2, label=f'Hybrid ROC (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Hybrid Model ROC Curve')
plt.legend(loc='lower right')
plt.grid()
plt.tight_layout()
plt.show()
