import mlflow
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Load DVC-tracked dataset
data = pd.read_csv("data/train.csv")

X = data[["feature"]]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

# Create MLflow experiment
mlflow.set_experiment("LunarLander_MLOps")

n_estimators = 100

with mlflow.start_run():

    # Create and train model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Predictions
    predictions = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)

    # Log parameters
    mlflow.log_param("model", "RandomForestClassifier")
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("dataset", "data/train.csv")
    mlflow.log_param("dataset_version", "DVC tracked")

    # Log metric
    mlflow.log_metric("accuracy", accuracy)

    # Save model locally
    model_file = "random_forest_model.pkl"
    joblib.dump(model, model_file)

    # Log model file as artifact
    mlflow.log_artifact(model_file)

    print("MLflow experiment completed!")
    print("Model: RandomForestClassifier")
    print("Dataset: DVC tracked")
    print("Accuracy:", accuracy)
    print("Model saved:", model_file)