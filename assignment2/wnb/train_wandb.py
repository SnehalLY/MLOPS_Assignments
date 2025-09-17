import wandb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Initialize W&B run
wandb.init(project="iris-demo", name="rf-classifier")

# Load dataset
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# Set hyperparameters in wandb.config
wandb.config.n_estimators = 100
wandb.config.max_depth = 5

# Train model
model = RandomForestClassifier(
    n_estimators=wandb.config.n_estimators,
    max_depth=wandb.config.max_depth,
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# Log metrics
wandb.log({"accuracy": acc})

# Save model locally
joblib.dump(model, "rf_model.pkl")

# Log model as artifact
artifact = wandb.Artifact("rf_model", type="model")
artifact.add_file("rf_model.pkl")
wandb.log_artifact(artifact)

wandb.finish()
