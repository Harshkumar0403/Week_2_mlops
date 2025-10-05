import os
import argparse
import pandas as pd
import joblib
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# ------------------------------
# Argument parsing
# ------------------------------
parser = argparse.ArgumentParser(description="Run inference on a dataset using a trained model.")
parser.add_argument("--data", required=True, help="Path to input CSV file for inference.")
parser.add_argument("--out", required=True, help="Output file to save inference results.")
parser.add_argument("--model", default="models/model.pkl", help="Path to trained model file.")
args = parser.parse_args()

# ------------------------------
# Load model
# ------------------------------
if not os.path.exists(args.model):
    raise FileNotFoundError(f"Model not found at {args.model}. Please run train.py first.")

model = joblib.load(args.model)

# ------------------------------
# Load dataset
# ------------------------------
if not os.path.exists(args.data):
    raise FileNotFoundError(f"Data file not found at {args.data}")

df = pd.read_csv(args.data)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# ------------------------------
# Inference
# ------------------------------
y_pred = model.predict(X)
acc = accuracy_score(y, y_pred)
report = classification_report(y, y_pred)
cm = confusion_matrix(y, y_pred)

# ------------------------------
# Print and save results
# ------------------------------
print(f"\n=== Inference Results for {args.data} ===")
print(report)
print(f"Accuracy: {acc:.4f}")
print("Confusion Matrix:\n", cm)

with open(args.out, "w") as f:
    f.write(f"Inference Results for {args.data}\n")
    f.write("=" * 50 + "\n")
    f.write(report + "\n")
    f.write(f"Accuracy: {acc:.4f}\n\n")
    f.write("Confusion Matrix:\n")
    f.write(str(cm) + "\n")

print(f"\n✅ Results saved to: {args.out}")

