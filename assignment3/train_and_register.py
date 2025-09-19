# train_and_register.py
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

# Point scripts to the running tracking server:
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("iris-registry-demo")

# Load data
iris = load_iris(as_frame=True)
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training + logging + registering
registered_model_name = "IrisClassifier"

with mlflow.start_run() as run:
    params = {"n_estimators": 100, "max_depth": 5}
    mlflow.log_params(params)

    model = RandomForestClassifier(**params, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    mlflow.log_metric("accuracy", float(acc))

    # add signature + input example (helps avoid warnings)
    input_example = X_train.head(3)
    signature = infer_signature(X_train, model.predict(X_train))

    # log & register the model in the MLflow Model Registry
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name=registered_model_name,
        signature=signature,
        input_example=input_example
    )

    print(f"Run {run.info.run_id} logged. Accuracy: {acc:.4f}")

# Query registry to show versions
client = MlflowClient(tracking_uri=mlflow.get_tracking_uri())
versions = client.get_latest_versions(registered_model_name)
print("Registered model versions found:", [v.version for v in versions])
