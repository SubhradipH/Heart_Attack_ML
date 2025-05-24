import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def load_data(file_path):
    df = pd.read_csv(file_path)

    # Drop ID column if exists
    if 'Patient_ID' in df.columns:
        df.drop(columns=['Patient_ID'], inplace=True)

    # Handle missing values
    numeric_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])

    # Encode
    for col in categorical_cols:
        if df[col].nunique() == 2:
            df[col] = LabelEncoder().fit_transform(df[col])
        else:
            df = pd.get_dummies(df, columns=[col], drop_first=True)

    # Feature engineering
    df['age_cholestoral'] = df['age'] * df['cholestoral']
    df['oldpeak_squared'] = df['oldpeak'] ** 2
    df['heart_rate_variability'] = df['Max_heart_rate'] - df['resting_blood_pressure']
    df['hypertension'] = (df['resting_blood_pressure'] > 140).astype(int)
    df['exercise_intensity'] = df['Max_heart_rate'] / df['age']

    X = df.drop(columns=['target'])
    y = df['target']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def evaluate_models(model_paths, model_names, X_test, y_test):
    for path, name in zip(model_paths, model_names):
        model = joblib.load(path)
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title(f'{name} - Confusion Matrix')
        plt.show()

    plt.plot([0, 1], [0, 1], 'k--')
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()

def main():
    # Load and split data
    file_path = r"D:\H.A\new ha1\HeartDiseaseTrain-Test.csv"  # Provide the correct path here
    X_train, X_test, y_train, y_test = load_data(file_path)

    # SMOTE
    sm = SMOTE(random_state=42)
    X_train, y_train = sm.fit_resample(X_train, y_train)

    # Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Paths to models
    model_paths = ["D:\H.A\knn_heart_attack_model.pkl", "D:\H.A\svm_heart_attack_model.pkl"]  # Adjust filenames
    model_names = ["KNN", "SVM"]

    evaluate_models(model_paths, model_names, X_test, y_test)

if __name__ == "__main__":
    main()
