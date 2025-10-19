import os
import unittest
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

DATA_PATH = "data/raw/iris.csv"
MODEL_PATH = "models/model.pkl"
METRICS_FILE = "metrics.txt"


class TestDataValidation(unittest.TestCase):
    """Test data quality and schema."""

    def test_data_file_exists(self):
        print("\n🔎 Checking if data file exists...")
        self.assertTrue(os.path.exists(DATA_PATH), f"Data file not found at {DATA_PATH}")

    def test_data_schema(self):
        print("🔎 Validating data schema...")
        df = pd.read_csv(DATA_PATH)
        expected_cols = 5  # 4 features + 1 label
        self.assertEqual(df.shape[1], expected_cols, "Data must have 5 columns (4 features + 1 label).")

    def test_no_missing_values(self):
        print("🔎 Checking for missing values...")
        df = pd.read_csv(DATA_PATH)
        self.assertFalse(df.isnull().values.any(), "Data contains missing values.")

    def test_class_distribution(self):
        print("🔎 Checking class distribution...")
        df = pd.read_csv(DATA_PATH)
        label_counts = df.iloc[:, -1].value_counts()
        for label, count in label_counts.items():
            self.assertGreater(count, 0, f"Label {label} has zero samples.")


class TestModelEvaluation(unittest.TestCase):
    """Test model quality and metrics."""

    def test_model_file_exists(self):
        print("\n🔎 Checking if model file exists...")
        self.assertTrue(os.path.exists(MODEL_PATH), f"Model file not found at {MODEL_PATH}")

    def test_model_performance(self):
        print("🔎 Loading data and model for evaluation...")
        df = pd.read_csv(DATA_PATH)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

        model = joblib.load(MODEL_PATH)
        y_pred = model.predict(X)

        print("\n📊 Evaluating model performance...")
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, average='macro', zero_division=0)
        recall = recall_score(y, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y, y_pred, average='macro', zero_division=0)

        # Print metrics to terminal
        print(f"Accuracy  : {accuracy:.4f}")
        print(f"Precision : {precision:.4f}")
        print(f"Recall    : {recall:.4f}")
        print(f"F1-Score  : {f1:.4f}")

        # Write metrics to metrics.txt
        with open(METRICS_FILE, "w") as f:
            f.write("=== Model Evaluation Metrics ===\n")
            f.write(f"Accuracy  : {accuracy:.4f}\n")
            f.write(f"Precision : {precision:.4f}\n")
            f.write(f"Recall    : {recall:.4f}\n")
            f.write(f"F1-Score  : {f1:.4f}\n")

        # Assertions to ensure quality thresholds
        self.assertGreaterEqual(accuracy, 0.80, "Accuracy below acceptable threshold!")
        self.assertGreaterEqual(f1, 0.80, "F1-score below acceptable threshold!")


if __name__ == "__main__":
    print("🚀 Starting unit tests for data validation and model evaluation...")
    unittest.main(verbosity=2)

