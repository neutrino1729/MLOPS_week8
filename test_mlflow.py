#!/usr/bin/env python3
"""Quick test of MLflow setup"""

import mlflow
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print("Testing MLflow setup...")

# Set tracking URI
mlflow.set_tracking_uri("./mlruns")
mlflow.set_experiment("mlflow_test")

# Load data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# Start run
with mlflow.start_run(run_name="test_run") as run:
    # Train simple model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Log parameters
    mlflow.log_param("n_estimators", 10)
    mlflow.log_param("test_size", 0.2)
    
    # Log metrics
    accuracy = accuracy_score(y_test, model.predict(X_test))
    mlflow.log_metric("accuracy", accuracy)
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
    
    print(f"✓ Run ID: {run.info.run_id}")
    print(f"✓ Accuracy: {accuracy:.4f}")
    print(f"✓ Artifacts URI: {run.info.artifact_uri}")

print("\n✓ MLflow test completed successfully!")
print("\nTo view results, run:")
print("  mlflow ui --host 0.0.0.0 --port 5000")
