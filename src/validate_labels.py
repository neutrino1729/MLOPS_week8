#!/usr/bin/env python3
"""
Label Validation Script with MLflow Integration
Detects potentially poisoned labels using KNN and confidence methods
"""

import pandas as pd
import numpy as np
import argparse
import yaml
import mlflow
import json
import os
import logging
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import List, Tuple, Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration"""
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)

def convert_to_native_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_native_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(item) for item in obj]
    return obj

class LabelValidator:
    """Validates labels using KNN and confidence-based methods"""
    
    def __init__(self, k: int = 5, threshold: float = 0.5, normalize: bool = True):
        self.k = k
        self.threshold = threshold
        self.normalize = normalize
        self.scaler = StandardScaler() if normalize else None
        self.label_encoder = LabelEncoder()
    
    def knn_validation(self, data_path: str) -> Tuple[List[int], Dict]:
        """KNN-based label validation"""
        logger.info(f"KNN validation on: {data_path}")
        logger.info(f"Parameters: k={self.k}, threshold={self.threshold}")
        
        # Load data
        df = pd.read_csv(data_path)
        target_col = df.columns[-1]
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Scale features
        if self.normalize:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X.values
        
        # Fit KNN
        knn = KNeighborsClassifier(n_neighbors=self.k + 1)
        knn.fit(X_scaled, y_encoded)
        
        # Find neighbors
        distances, indices = knn.kneighbors(X_scaled)
        
        suspicious_indices = []
        detailed_report = []
        
        # Check each point
        for i in range(len(df)):
            original_label = y_encoded[i]
            neighbor_indices = indices[i][1:]  # Exclude self
            neighbor_labels = y_encoded[neighbor_indices]
            neighbor_distances = distances[i][1:]
            
            # Count disagreements
            disagreements = np.sum(neighbor_labels != original_label)
            disagreement_ratio = disagreements / self.k
            
            if disagreement_ratio >= self.threshold:
                suspicious_indices.append(int(i))  # Convert to native int
                
                neighbor_label_names = [
                    self.label_encoder.inverse_transform([label])[0]
                    for label in neighbor_labels
                ]
                
                detailed_report.append({
                    "index": int(i),
                    "original_label": str(y.iloc[i]),
                    "disagreements": int(disagreements),
                    "disagreement_ratio": float(disagreement_ratio),
                    "neighbor_labels": neighbor_label_names,
                    "avg_neighbor_distance": float(neighbor_distances.mean())
                })
        
        # Summary report
        summary = {
            "dataset_path": data_path,
            "total_rows": int(len(df)),
            "suspicious_rows": int(len(suspicious_indices)),
            "suspicion_rate": float(len(suspicious_indices) / len(df)),
            "validation_params": {
                "method": "knn",
                "k": int(self.k),
                "threshold": float(self.threshold),
                "normalize": bool(self.normalize)
            },
            "class_distribution": {str(k): int(v) for k, v in y.value_counts().items()},
            "suspicious_indices": suspicious_indices,
            "detailed_analysis": detailed_report
        }
        
        logger.info(f"Found {len(suspicious_indices)} suspicious labels ({summary['suspicion_rate']:.2%})")
        
        return suspicious_indices, summary
    
    def confidence_validation(self, data_path: str) -> Dict:
        """Confidence-based validation using Random Forest"""
        logger.info("Performing confidence-based validation...")
        
        df = pd.read_csv(data_path)
        target_col = df.columns[-1]
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Scale features
        if self.normalize:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X.values
        
        # Train confidence model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_scaled, y_encoded)
        
        # Get predictions and probabilities
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)
        
        # Find suspicious points
        suspicious = []
        
        for i in range(len(df)):
            true_label = y_encoded[i]
            pred_label = predictions[i]
            max_prob = np.max(probabilities[i])
            true_label_prob = probabilities[i][true_label]
            
            # Flag if model predicts different with high confidence
            if pred_label != true_label and max_prob > 0.8:
                suspicious.append({
                    "index": int(i),
                    "true_label": self.label_encoder.inverse_transform([true_label])[0],
                    "predicted_label": self.label_encoder.inverse_transform([pred_label])[0],
                    "prediction_confidence": float(max_prob),
                    "true_label_confidence": float(true_label_prob)
                })
        
        accuracy = (predictions == y_encoded).mean()
        
        logger.info(f"Confidence validation: {len(suspicious)} suspicious points")
        logger.info(f"Model accuracy: {accuracy:.4f}")
        
        return {
            "suspicious_points": suspicious,
            "model_accuracy": float(accuracy),
            "num_suspicious": int(len(suspicious))
        }

def validate_dataset(data_path: str, poison_level: float = 0.0, 
                    use_mlflow: bool = True) -> Dict:
    """
    Complete validation with both methods and MLflow logging
    """
    config = load_config()
    
    # Initialize validator
    val_params = config['validation']['knn']
    validator = LabelValidator(
        k=val_params['k'],
        threshold=val_params['threshold'],
        normalize=val_params['normalize_features']
    )
    
    # Set MLflow experiment
    if use_mlflow:
        mlflow.set_tracking_uri("./mlruns")
        mlflow.set_experiment("label_validation")
    
    # Run validations
    logger.info(f"\n{'='*60}")
    logger.info(f"Validating: {data_path}")
    logger.info(f"{'='*60}\n")
    
    # KNN validation
    suspicious_knn, knn_report = validator.knn_validation(data_path)
    
    # Confidence validation
    conf_report = validator.confidence_validation(data_path)
    
    # Combine reports
    full_report = {
        "dataset": data_path,
        "poison_level": float(poison_level),
        "knn_validation": knn_report,
        "confidence_validation": conf_report,
        "summary": {
            "knn_suspicious_count": int(len(suspicious_knn)),
            "knn_suspicion_rate": float(knn_report['suspicion_rate']),
            "conf_suspicious_count": int(conf_report['num_suspicious']),
            "model_accuracy": float(conf_report['model_accuracy'])
        }
    }
    
    # Convert all numpy types to native Python types
    full_report = convert_to_native_types(full_report)
    
    # Save report
    output_dir = "reports/validation"
    os.makedirs(output_dir, exist_ok=True)
    
    dataset_name = os.path.basename(data_path).replace('.csv', '')
    report_path = os.path.join(output_dir, f"{dataset_name}_validation.json")
    
    with open(report_path, 'w') as f:
        json.dump(full_report, f, indent=2)
    
    logger.info(f"\nValidation report saved to: {report_path}")
    
    # Log to MLflow
    if use_mlflow:
        run_name = f"validation_{dataset_name}"
        
        with mlflow.start_run(run_name=run_name):
            mlflow.log_param("dataset", data_path)
            mlflow.log_param("poison_level", poison_level)
            mlflow.log_param("knn_k", val_params['k'])
            mlflow.log_param("knn_threshold", val_params['threshold'])
            
            # Log metrics
            mlflow.log_metric("knn_suspicious_count", len(suspicious_knn))
            mlflow.log_metric("knn_suspicion_rate", knn_report['suspicion_rate'])
            mlflow.log_metric("conf_suspicious_count", conf_report['num_suspicious'])
            mlflow.log_metric("conf_model_accuracy", conf_report['model_accuracy'])
            
            # Log report
            mlflow.log_artifact(report_path)
    
    return full_report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate dataset labels")
    parser.add_argument("--data-path", required=True, help="Path to dataset")
    parser.add_argument("--poison-level", type=float, default=0.0, help="Known poison level")
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow")
    
    args = parser.parse_args()
    
    validate_dataset(args.data_path, args.poison_level, not args.no_mlflow)
