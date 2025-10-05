import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

# Define paths
DATA_PATH = "data/raw/iris.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
RESULT_FILE = "result.txt"

# Create model directory if not present
os.makedirs(MODEL_DIR, exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)

# Assume last column is the label
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Initialize and train model
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# Compute metrics
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# Print metrics on screen
print("=== Classification Report ===")
print(report)
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", cm)

# Save model
joblib.dump(clf, MODEL_PATH)

# Save results to result.txt
with open(RESULT_FILE, "w") as f:
    f.write("=== Classification Report ===\n")
    f.write(report + "\n")
    f.write(f"Accuracy: {accuracy:.4f}\n\n")
    f.write("Confusion Matrix:\n")
    f.write(str(cm) + "\n")

print(f"\nModel saved to: {MODEL_PATH}")
print(f"Results saved to: {RESULT_FILE}")

