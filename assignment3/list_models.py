# list_models.py (fixed)
from mlflow.tracking import MlflowClient
import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")
client = MlflowClient(tracking_uri=mlflow.get_tracking_uri())

for rm in client.search_registered_models():
    print("Model:", rm.name)
    for v in rm.latest_versions:
        print("  version:", v.version, "stage:", v.current_stage)
