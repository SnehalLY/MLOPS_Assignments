from prefect import flow, task
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


@task
def load_data():
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target
    return X, y


@task
def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)


@task
def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model


@task
def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"✅ Model accuracy: {acc:.4f}")
    return acc


@flow
def iris_ml_pipeline():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    iris_ml_pipeline()
