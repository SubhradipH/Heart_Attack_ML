import pandas as pd
import pickle

class HeartDiseaseModelTester:
    def __init__(self, model_path):
        with open(model_path, "rb") as file:
            self.model = pickle.load(file)
        print("✅ Model loaded successfully.")
        
        # Set expected column names from the training data
        self.columns = [
            'age',
            'sex',
            'chest_pain_type',
            'resting_blood_pressure',
            'cholestoral',
            'fasting_blood_sugar',
            'rest_ecg',
            'Max_heart_rate',
            'exercise_induced_angina',
            'oldpeak',
            'slope',
            'vessels_colored_by_flourosopy',
            'thalassemia'
        ]
        print("📊 Model expects features:", self.columns)

    def predict_single(self, input_dict):
        try:
            df = pd.DataFrame([input_dict])
            df = df[self.columns]  # Ensure order matches training
            prediction = self.model.predict(df)
            print("🔮 Prediction:", "Heart Disease" if prediction[0] == 1 else "No Heart Disease")
        except KeyError as e:
            print("❌ Missing columns in input:", e)

# Example test
if __name__ == "__main__":
    tester = HeartDiseaseModelTester("random_forest_model.pkl")

    # ✅ Sample input (make sure the keys exactly match the expected columns)
    sample_input = {
        'age': 55,
        'sex': 1,
        'chest_pain_type': 0,
        'resting_blood_pressure': 140,
        'cholestoral': 240,
        'fasting_blood_sugar': 0,
        'rest_ecg': 1,
        'Max_heart_rate': 160,
        'exercise_induced_angina': 0,
        'oldpeak': 1.5,
        'slope': 2,
        'vessels_colored_by_flourosopy': 0,
        'thalassemia': 2
    }

    tester.predict_single(sample_input)
