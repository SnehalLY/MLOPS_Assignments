# load_model.py (fix applied)
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from mlflow.exceptions import MlflowException

mlflow.set_tracking_uri("http://127.0.0.1:5000")
client = MlflowClient(tracking_uri=mlflow.get_tracking_uri())
model_name = "IrisClassifier"

# Try to load Production; fallback to latest
try:
    print("Trying to load Production model...")
    model = mlflow.pyfunc.load_model(f"models:/{model_name}/Production")
    print("Loaded Production model.")
except MlflowException:
    print("Production not found. Falling back...")
    versions = client.search_model_versions(f"name='{model_name}'")
    latest = sorted(versions, key=lambda v: int(v.version))[-1].version
    model = mlflow.pyfunc.load_model(f"models:/{model_name}/{latest}")
    print(f"Loaded version {latest}.")

# Prepare test data with column names
iris = load_iris(as_frame=True)   # use as_frame=True to keep names
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Pass DataFrame with correct column names
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)

print(f"Accuracy: {acc:.4f}")
print("Sample predictions:")
for i in range(5):
    print(f"Features: {X_test.iloc[i].to_dict()} -> Pred: {preds[i]} | Actual: {y_test.iloc[i]}")
