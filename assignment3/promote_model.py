# promote_model.py
from mlflow.tracking import MlflowClient
import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")
client = MlflowClient(tracking_uri=mlflow.get_tracking_uri())

model_name = "IrisClassifier"

# Get latest non-archived version (if any)
versions = client.get_latest_versions(model_name, stages=None)
if not versions:
    raise SystemExit(f"No versions found for model {model_name}")

# pick latest numeric version
latest = sorted(versions, key=lambda v: int(v.version))[-1]
print(f"Promoting version {latest.version} (current stage={latest.current_stage}) to Production...")

client.transition_model_version_stage(
    name=model_name,
    version=latest.version,
    stage="Production",
    archive_existing_versions=True
)

print("Done. Promoted to Production.")
