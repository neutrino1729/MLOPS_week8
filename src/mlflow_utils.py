"""
MLflow Utilities for IRIS Poisoning Experiments
Handles experiment tracking, model logging, and artifact management
"""

import mlflow
import mlflow.sklearn
import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
from datetime import datetime

class MLflowExperimentTracker:
    """Manages MLflow experiment tracking for poisoning experiments"""
    
    def __init__(self, experiment_name: str = "iris_poisoning_mlops"):
        """
        Initialize MLflow experiment tracker
        
        Args:
            experiment_name: Name of the MLflow experiment
        """
        self.experiment_name = experiment_name
        
        # Load configuration
        self.config = self._load_config()
        
        # Set tracking URI (local with GCS artifacts)
        mlflow.set_tracking_uri("./mlruns")
        
        # Set experiment
        mlflow.set_experiment(self.experiment_name)
        
        print(f"✓ MLflow experiment: {self.experiment_name}")
        print(f"✓ Tracking URI: ./mlruns")
        print(f"✓ Artifacts: {self.config['mlflow']['artifact_location']}")
    
    def _load_config(self) -> Dict:
        """Load configuration from params.yaml"""
        if os.path.exists("params.yaml"):
            with open("params.yaml", "r") as f:
                return yaml.safe_load(f)
        return {}
    
    def start_run(self, run_name: str, tags: Optional[Dict] = None) -> mlflow.ActiveRun:
        """
        Start a new MLflow run
        
        Args:
            run_name: Name for the run
            tags: Optional tags for the run
            
        Returns:
            MLflow active run object
        """
        if tags is None:
            tags = {}
        
        # Add default tags
        tags.update({
            "project": "iris_poisoning",
            "timestamp": datetime.now().isoformat()
        })
        
        return mlflow.start_run(run_name=run_name, tags=tags)
    
    def log_dataset_info(self, dataset_path: str, poison_level: float = 0.0):
        """Log dataset information"""
        df = pd.read_csv(dataset_path)
        
        mlflow.log_param("dataset_path", dataset_path)
        mlflow.log_param("dataset_rows", len(df))
        mlflow.log_param("dataset_features", len(df.columns) - 1)
        mlflow.log_param("poison_level", poison_level)
        
        # Log class distribution
        target_col = df.columns[-1]
        class_dist = df[target_col].value_counts().to_dict()
        mlflow.log_params({f"class_{k}": v for k, v in class_dist.items()})
    
    def log_model_params(self, model_name: str, model_params: Dict):
        """Log model parameters"""
        mlflow.log_param("model_type", model_name)
        
        for param_name, param_value in model_params.items():
            mlflow.log_param(f"{model_name}_{param_name}", param_value)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to MLflow"""
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value, step=step)
    
    def log_model(self, model, model_name: str):
        """Log trained model"""
        mlflow.sklearn.log_model(
            model, 
            model_name,
            registered_model_name=f"{self.experiment_name}_{model_name}"
        )
    
    def log_artifacts(self, artifact_dir: str):
        """Log directory of artifacts"""
        if os.path.exists(artifact_dir):
            mlflow.log_artifacts(artifact_dir)
    
    def log_figure(self, figure, filename: str):
        """Log matplotlib figure"""
        temp_path = f"temp_{filename}"
        figure.savefig(temp_path, dpi=300, bbox_inches='tight')
        mlflow.log_artifact(temp_path)
        os.remove(temp_path)
    
    def end_run(self):
        """End current MLflow run"""
        mlflow.end_run()

# Global tracker instance
_tracker = None

def get_tracker(experiment_name: str = "iris_poisoning_mlops") -> MLflowExperimentTracker:
    """Get or create global tracker instance"""
    global _tracker
    if _tracker is None:
        _tracker = MLflowExperimentTracker(experiment_name)
    return _tracker
