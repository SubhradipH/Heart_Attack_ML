import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Load dataset
df = pd.read_csv(r"D:\H.A\new ha1\HeartDiseaseTrain-Test.csv")

# Drop ID column
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

# Scale data for Neural Network
scaler_nn = StandardScaler()
X_train_nn = scaler_nn.fit_transform(X_train)
X_test_nn = scaler_nn.transform(X_test)

# Build Neural Network model
model_nn = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_nn.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile model
model_nn.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# ✅ Train Neural Network
history = model_nn.fit(X_train_nn, y_train, epochs=20, batch_size=32, validation_split=0.1, verbose=1)

# ✅ Plot Training vs Validation Accuracy Curve
plt.figure(figsize=(6, 4))
plt.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='green')
plt.title('Epoch vs Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ✅ Plot Training vs Validation Loss Curve
plt.figure(figsize=(6, 4))
plt.plot(history.history['loss'], label='Training Loss', color='red')
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
plt.title('Epoch vs Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Load SVM and KNN models
svm_model = joblib.load(r"D:\H.A\svm_heart_attack_model.pkl")
svm_scaler = joblib.load(r"D:\H.A\svm_scaler.pkl")

knn_model = joblib.load(r"D:\H.A\knn_best_model.pkl")
knn_scaler = joblib.load(r"D:\H.A\knn_best_scaler.pkl")

# Transform test data
X_test_svm = svm_scaler.transform(X_test)
X_test_knn = knn_scaler.transform(X_test)

# Get prediction probabilities
prob_nn = model_nn.predict(X_test_nn).flatten()
prob_svm = svm_model.predict_proba(X_test_svm)[:, 1]
prob_knn = knn_model.predict_proba(X_test_knn)[:, 1]

# ✅ Triple Soft Voting
avg_prob = (prob_nn + prob_svm + prob_knn) / 3
final_pred = (avg_prob >= 0.5).astype(int)

# Accuracy and report
accuracy = accuracy_score(y_test, final_pred)
print(f"\n✅ Hybrid (NN + SVM + KNN) Accuracy: {accuracy * 100:.2f}%\n")
print(classification_report(y_test, final_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, final_pred)
print("Confusion Matrix:\n", cm)

# Plot Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Attack", "Attack"], yticklabels=["No Attack", "Attack"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Hybrid NN + SVM + KNN Confusion Matrix")
plt.tight_layout()
plt.show()

# Plot ROC Curve
fpr, tpr, _ = roc_curve(y_test, avg_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Hybrid ROC (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Hybrid NN + SVM + KNN ROC Curve')
plt.legend(loc='lower right')
plt.grid()
plt.tight_layout()
plt.show()
