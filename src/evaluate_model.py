#!/usr/bin/env python3
"""
Model Evaluation Script
Comprehensive evaluation of models across poison levels
"""

import pandas as pd
import numpy as np
import yaml
import json
import os
import joblib
import mlflow
import logging
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)

def evaluate_all_models():
    """Evaluate all trained models and generate comparison report"""
    
    mlflow.set_tracking_uri("./mlruns")
    
    # Get all training runs
    client = mlflow.tracking.MlflowClient()
    experiment = mlflow.get_experiment_by_name("iris_poisoning_training")
    
    if experiment is None:
        logger.error("No training experiment found!")
        return
    
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"]
    )
    
    results = []
    
    for run in runs:
        run_data = {
            "run_id": run.info.run_id,
            "run_name": run.data.tags.get("mlflow.runName", "unknown"),
            "model_type": run.data.params.get("model_type", "unknown"),
            "poison_level": float(run.data.params.get("poison_level", 0.0)),
            "test_accuracy": run.data.metrics.get("test_accuracy", 0.0),
            "test_f1": run.data.metrics.get("test_f1", 0.0),
            "test_precision": run.data.metrics.get("test_precision", 0.0),
            "test_recall": run.data.metrics.get("test_recall", 0.0),
            "cv_accuracy_mean": run.data.metrics.get("cv_accuracy_mean", 0.0),
            "cv_accuracy_std": run.data.metrics.get("cv_accuracy_std", 0.0)
        }
        results.append(run_data)
    
    # Create DataFrame for analysis
    df_results = pd.DataFrame(results)
    
    # Save results
    output_dir = "reports/performance"
    os.makedirs(output_dir, exist_ok=True)
    
    results_path = os.path.join(output_dir, "model_comparison.csv")
    df_results.to_csv(results_path, index=False)
    
    logger.info(f"Results saved to: {results_path}")
    logger.info(f"\nModel Performance Summary:")
    logger.info(df_results.to_string())
    
    # Generate comparison report
    report = {
        "total_runs": len(results),
        "models_evaluated": df_results['model_type'].unique().tolist(),
        "poison_levels": sorted(df_results['poison_level'].unique().tolist()),
        "best_overall": {
            "run_id": df_results.loc[df_results['test_accuracy'].idxmax(), 'run_id'],
            "accuracy": df_results['test_accuracy'].max(),
            "model": df_results.loc[df_results['test_accuracy'].idxmax(), 'model_type']
        },
        "performance_by_poison_level": {}
    }
    
    # Analyze by poison level
    for poison_lvl in report['poison_levels']:
        subset = df_results[df_results['poison_level'] == poison_lvl]
        report['performance_by_poison_level'][f"{int(poison_lvl*100)}pct"] = {
            "mean_accuracy": float(subset['test_accuracy'].mean()),
            "std_accuracy": float(subset['test_accuracy'].std()),
            "best_model": subset.loc[subset['test_accuracy'].idxmax(), 'model_type']
        }
    
    report_path = os.path.join(output_dir, "evaluation_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"\nEvaluation report saved to: {report_path}")
    
    return df_results, report

if __name__ == "__main__":
    evaluate_all_models()
