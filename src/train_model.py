#!/usr/bin/env python3
"""
Model Training Script with MLflow Integration
Trains models on clean and poisoned datasets
"""

import pandas as pd
import numpy as np
import argparse
import yaml
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import joblib
import os
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration"""
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)

def get_model(model_name: str, config: dict):
    """Get model instance based on configuration"""
    models = {
        'random_forest': RandomForestClassifier(**config['training']['models']['random_forest']),
        'svm': SVC(**config['training']['models']['svm'], probability=True),
        'logistic_regression': LogisticRegression(**config['training']['models']['logistic_regression'])
    }
    return models.get(model_name)

def train_and_evaluate(data_path: str, model_name: str = 'random_forest', 
                      poison_level: float = 0.0, run_name: str = None):
    """
    Train model and log to MLflow
    
    Args:
        data_path: Path to dataset
        model_name: Type of model to train
        poison_level: Poison level of dataset (for logging)
        run_name: Custom run name
    """
    # Load config
    config = load_config()
    
    # Load data
    logger.info(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    
    # Prepare data
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split data
    test_size = config['training']['test_size']
    random_state = config['training']['random_state']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )
    
    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Get model
    model = get_model(model_name, config)
    if model is None:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Set MLflow experiment
    mlflow.set_experiment("iris_poisoning_training")
    
    # Start MLflow run
    if run_name is None:
        run_name = f"{model_name}_poison{int(poison_level*100)}pct"
    
    with mlflow.start_run(run_name=run_name) as run:
        logger.info(f"MLflow Run ID: {run.info.run_id}")
        
        # Log parameters
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("poison_level", poison_level)
        mlflow.log_param("dataset_path", data_path)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("n_test", len(X_test))
        
        # Log model parameters
        model_params = model.get_params()
        for param, value in model_params.items():
            mlflow.log_param(f"model_{param}", value)
        
        # Train model
        logger.info(f"Training {model_name}...")
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
        
        # Calculate metrics
        metrics = {
            'train_accuracy': accuracy_score(y_train, y_pred_train),
            'test_accuracy': accuracy_score(y_test, y_pred_test),
            'train_f1': f1_score(y_train, y_pred_train, average='macro'),
            'test_f1': f1_score(y_test, y_pred_test, average='macro'),
            'train_precision': precision_score(y_train, y_pred_train, average='macro'),
            'test_precision': precision_score(y_test, y_pred_test, average='macro'),
            'train_recall': recall_score(y_train, y_pred_train, average='macro'),
            'test_recall': recall_score(y_test, y_pred_test, average='macro')
        }
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, 
                                   cv=config['training']['cv_folds'], 
                                   scoring='accuracy')
        metrics['cv_accuracy_mean'] = cv_scores.mean()
        metrics['cv_accuracy_std'] = cv_scores.std()
        
        # Log all metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
            logger.info(f"{metric_name}: {metric_value:.4f}")
        
        # Save model artifacts
        artifacts_dir = f"artifacts/models/{run_name}"
        os.makedirs(artifacts_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(artifacts_dir, "model.pkl")
        joblib.dump(model, model_path)
        
        # Save scaler
        scaler_path = os.path.join(artifacts_dir, "scaler.pkl")
        joblib.dump(scaler, scaler_path)
        
        # Save label encoder
        le_path = os.path.join(artifacts_dir, "label_encoder.pkl")
        joblib.dump(le, le_path)
        
        # Save metrics
        metrics_path = os.path.join(artifacts_dir, "metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Log artifacts to MLflow
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_artifact(scaler_path)
        mlflow.log_artifact(le_path)
        mlflow.log_artifact(metrics_path)
        
        logger.info(f"Artifacts saved to: {artifacts_dir}")
        logger.info(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
        
        return model, scaler, le, metrics, run.info.run_id

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train models with MLflow tracking")
    parser.add_argument("--data-path", required=True, help="Path to dataset")
    parser.add_argument("--model", default="random_forest", 
                       choices=['random_forest', 'svm', 'logistic_regression'],
                       help="Model type")
    parser.add_argument("--poison-level", type=float, default=0.0,
                       help="Poison level for logging")
    parser.add_argument("--run-name", help="Custom run name")
    
    args = parser.parse_args()
    
    # Set MLflow tracking
    mlflow.set_tracking_uri("./mlruns")
    
    # Train model
    train_and_evaluate(args.data_path, args.model, args.poison_level, args.run_name)
