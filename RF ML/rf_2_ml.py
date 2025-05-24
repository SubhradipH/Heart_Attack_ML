import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import pickle

class HeartDiseaseModel:
    def __init__(self):  # Correct constructor
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.label_encoders = {}  # Store encoders for future decoding if needed

    def load_data(self, file_path):
        df = pd.read_csv(file_path)
        df = df.iloc[1:].reset_index(drop=True)  # Optional: remove first row if it's a duplicate header

        # Debugging: Show available columns
        print("📄 Columns in the dataset:", df.columns.tolist())

        # Encode all categorical columns
        for col in df.columns:
            if df[col].dtype == 'object':
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                self.label_encoders[col] = le

        # Use all columns except the last as features, last one as label
        self.X = df.iloc[:, :-1]
        self.y = df.iloc[:, -1]

        print("✅ Data loaded and encoded successfully.")
        print("📊 Sample features:\n", self.X.head())

    def train_model(self, test_size=0.2):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=42, stratify=self.y
        )
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        print(f"\n✅ Model Accuracy: {accuracy:.2f}")
        print("📄 Classification Report:\n", report)

    def save_model(self, filename="random_forest_model.pkl"):
        with open(filename, "wb") as file:
            pickle.dump(self.model, file)
        print(f"💾 Model saved as {filename}")

    def load_model(self, filename="random_forest_model.pkl"):
        with open(filename, "rb") as file:
            self.model = pickle.load(file)
        print("✅ Model loaded successfully.")

    def predict(self, input_data):
        return self.model.predict(input_data)

# ✅ Run if file is executed directly
if __name__ == "__main__":
    model = HeartDiseaseModel()
    model.load_data(r"D:\H.A\new ha1\HeartDiseaseTrain-Test.csv")  # Update path if needed
    model.train_model()
    model.save_model()
