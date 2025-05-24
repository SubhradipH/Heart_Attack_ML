import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, roc_curve
)
from imblearn.over_sampling import SMOTE


def main():
    # Load the dataset
    file_path = r"D:\H.A\new ha1\HeartDiseaseTrain-Test.csv"

    try:
        if file_path.endswith(".xlsx"):
            df = pd.read_excel(file_path, engine="openpyxl")
        elif file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        else:
            raise ValueError("Unsupported file format. Please upload a .xlsx or .csv file.")
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # Drop unnecessary columns
    if "Patient_ID" in df.columns:
        df.drop(columns=["Patient_ID"], inplace=True)

    # Handle missing values
    numeric_cols = df.select_dtypes(include=["number"]).columns
    categorical_cols = df.select_dtypes(include=["object"]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])

    # Encode categorical variables
    for col in categorical_cols:
        if df[col].nunique() == 2:
            df[col] = LabelEncoder().fit_transform(df[col])
        else:
            df = pd.get_dummies(df, columns=[col], drop_first=True)

    # ====================================
    # Feature Engineering
    # ====================================
    df["age_cholestoral"] = df["age"] * df["cholestoral"]
    df["oldpeak_squared"] = df["oldpeak"] ** 2
    df["heart_rate_variability"] = df["Max_heart_rate"] - df["resting_blood_pressure"]
    df["hypertension"] = (df["resting_blood_pressure"] > 140).astype(int)
    df["exercise_intensity"] = df["Max_heart_rate"] / df["age"]

    # Split features and target
    X = df.drop(columns=["target"])
    y = df["target"]

    # =============================
    # ✅ Patient Distribution Plot
    # =============================
    total_patients = len(df)
    heart_attack_patients = df["target"].sum()
    healthy_patients = total_patients - heart_attack_patients

    plt.figure(figsize=(8, 6))
    bars = plt.bar(
        ["Total Patients", "Heart Attack Patients", "Healthy Patients"],
        [total_patients, heart_attack_patients, healthy_patients],
        color=["blue", "red", "green"]
    )

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 10, int(yval), ha='center', fontsize=10)

    plt.title("Patient Distribution")
    plt.ylabel("Number of Patients")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # SMOTE for imbalance
    smote = SMOTE(sampling_strategy="auto", random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Feature Scaling
    scaler = StandardScaler()
    X_train_resampled = scaler.fit_transform(X_train_resampled)
    X_test = scaler.transform(X_test)

    # GridSearchCV for best SVM
    param_grid = {
        "C": [0.1, 1, 10],
        "kernel": ["linear", "rbf"],
        "gamma": ["scale", "auto"]
    }
    svm_model = GridSearchCV(SVC(), param_grid, cv=5, scoring="accuracy")
    svm_model.fit(X_train_resampled, y_train_resampled)

    best_svm = svm_model.best_estimator_
    y_pred = best_svm.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)

    print(f"Improved SVM Model Accuracy: {accuracy * 100:.2f}%")
    print(report)
    print(f"ROC AUC Score: {roc_auc:.2f}")

    # =============================
    # ✅ Confusion Matrix Plot
    # =============================
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_svm.classes_)
    disp.plot(cmap="Blues", values_format='d')
    plt.title("Confusion Matrix - SVM Model")
    plt.grid(False)
    plt.tight_layout()
    plt.show()

    # =============================
    # ✅ ROC Curve Plot Added
    # =============================
    y_scores = best_svm.decision_function(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})", color="darkorange")
    plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - SVM Model")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # =============================
    # ✅ Classification Report Bar Plot
    # =============================
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()

    metrics_to_plot = report_df.loc[["0", "1"], ["precision", "recall", "f1-score"]]
    metrics_to_plot.plot(kind="bar", figsize=(8, 5))
    plt.title("Classification Report - Precision, Recall, F1-Score")
    plt.ylabel("Score")
    plt.ylim(0, 1.1)
    plt.grid(axis="y")
    plt.xticks(rotation=0)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

    # =============================
    # ✅ Export Report to Excel
    # =============================
    output_file = "Heart_Attack_Cases_Results.xlsx"
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with pd.ExcelWriter(output_file, engine='openpyxl', mode='w') as writer:
        df.to_excel(writer, sheet_name="Predictions", index=False)
        report_df.to_excel(writer, sheet_name="Classification_Report")

    print(f"Classification report and results saved to {output_file}")

    # =============================
    # ✅ Save Model & Scaler
    # =============================
    joblib.dump(best_svm, "svm_heart_attack_model.pkl")
    joblib.dump(scaler, "svm_scaler.pkl")
    print("Trained SVM model and scaler saved successfully.")


if __name__ == "__main__":
    main()
