import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
from imblearn.over_sampling import SMOTE

def main():
    # Load the dataset
    file_path = r"D:/H.A/new ha1/HeartDiseaseTrain-Test.csv"
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

    # Ensure target variable is binary
    if df["target"].dtype == "object":
        df["target"] = LabelEncoder().fit_transform(df["target"])

    # Feature Engineering
    df["age_cholestoral"] = df["age"] * df["cholestoral"]
    df["oldpeak_squared"] = df["oldpeak"] ** 2
    df["heart_rate_variability"] = df["Max_heart_rate"] - df["resting_blood_pressure"]
    df["hypertension"] = (df["resting_blood_pressure"] > 140).astype(int)
    df["exercise_intensity"] = df["Max_heart_rate"] / df["age"]

    # Split features and target
    X = df.drop(columns=["target"])
    y = df["target"]
    
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
    param_grid = {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"], "gamma": ["scale", "auto"]}
    svm_model = GridSearchCV(SVC(probability=True), param_grid, cv=5, scoring="accuracy")
    svm_model.fit(X_train_resampled, y_train_resampled)
    
    best_svm = svm_model.best_estimator_
    y_pred = best_svm.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"Improved SVM Model Accuracy: {accuracy * 100:.2f}%")
    print(report)
    
    # ROC Curve
    y_prob_svm = best_svm.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob_svm)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve - SVM')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()
    
    # Classification Report Bar Plot
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
    
    # Export Report to Excel
    output_file = "Heart_Attack_Cases_Results.xlsx"
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with pd.ExcelWriter(output_file, engine='openpyxl', mode='w') as writer:
        df.to_excel(writer, sheet_name="Predictions", index=False)
        report_df.to_excel(writer, sheet_name="Classification_Report")
    
    print(f"Classification report and results saved to {output_file}")
    
    # Save Model & Scaler
    joblib.dump(best_svm, "svm_heart_attack_model.pkl")
    joblib.dump(scaler, "svm_scaler.pkl")
    print("Trained SVM model and scaler saved successfully.")

if __name__ == "__main__":
    main()
