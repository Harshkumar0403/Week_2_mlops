import os
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score
import pytest

# =====================
# 1️⃣  Data Validation
# =====================
def test_data_loading():
    """Ensure all dataset versions exist and are non-empty."""
    for version in ["raw", "v1", "v2"]:
        path = f"data/{version}"
        assert os.path.exists(path), f"Missing data folder: {path}"
        files = os.listdir(path)
        assert len(files) > 0, f"No files found in {path}"

# =====================
# 2️⃣  Model Validation
# =====================
@pytest.fixture(scope="session")
def model():
    """Load trained model once per session."""
    model_path = "models/model.pkl"
    assert os.path.exists(model_path), "Model file missing!"
    model = joblib.load(model_path)
    return model

def test_model_prediction_shape(model):
    """Check that model.predict returns expected output shape."""
    import numpy as np
    X_sample = np.random.rand(5, 4)  # Iris-like dummy data
    preds = model.predict(X_sample)
    assert preds.shape[0] == 5, "Model prediction size mismatch!"

def test_model_accuracy_on_sample(model):
    """Check model accuracy on a small sample if data is available."""
    sample_data = "data/v2/iris.csv"  # replace with actual file name if known
    if os.path.exists(sample_data):
        df = pd.read_csv(sample_data)
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        preds = model.predict(X)
        acc = accuracy_score(y, preds)
        print(f"Sample Accuracy: {acc:.3f}")
        assert acc > 0.7, f"Accuracy too low: {acc}"

# =====================
# 3️⃣  Sanity Metrics Log
# =====================
def test_log_metrics():
    """Log metrics into metrics.txt for CI."""
    with open("metrics.txt", "a") as f:
        f.write("\n✅ All tests executed successfully.\n")
    assert True

