#!/usr/bin/env python3
"""
Complete IRIS Data Poisoning Pipeline
Orchestrates the entire experiment workflow with MLflow tracking
"""

import os
import sys
import subprocess
import yaml
import mlflow
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)

def run_command(cmd, description):
    """Run shell command and log output"""
    logger.info(f"\n{'='*70}")
    logger.info(f"STEP: {description}")
    logger.info(f"{'='*70}")
    logger.info(f"Command: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
    
    if result.returncode != 0:
        logger.error(f"Failed: {description}")
        return False
    
    logger.info(f"✓ Completed: {description}\n")
    return True

def main():
    """Execute complete pipeline"""
    
    start_time = datetime.now()
    
    logger.info("""
    ╔══════════════════════════════════════════════════════════╗
    ║                                                          ║
    ║     IRIS DATA POISONING MLOPS PIPELINE                  ║
    ║     Full Experiment Workflow with MLflow                ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    config = load_config()
    
    # Set MLflow tracking
    mlflow.set_tracking_uri("./mlruns")
    
    # Pipeline steps
    steps = [
        # Step 1: Create poisoned datasets
        {
            "cmd": "python src/poison_data.py --create-all",
            "desc": "Creating poisoned datasets (5%, 10%, 50%)"
        },
        
        # Step 2: Validate all datasets
        {
            "cmd": "python src/validate_labels.py --data-path data/iris.csv --poison-level 0.0",
            "desc": "Validating clean dataset"
        },
        {
            "cmd": "python src/validate_labels.py --data-path data/poisoned/iris_poisoned_5pct.csv --poison-level 0.05",
            "desc": "Validating 5% poisoned dataset"
        },
        {
            "cmd": "python src/validate_labels.py --data-path data/poisoned/iris_poisoned_10pct.csv --poison-level 0.10",
            "desc": "Validating 10% poisoned dataset"
        },
        {
            "cmd": "python src/validate_labels.py --data-path data/poisoned/iris_poisoned_50pct.csv --poison-level 0.50",
            "desc": "Validating 50% poisoned dataset"
        },
        
        # Step 3: Train models on all datasets
        {
            "cmd": "python src/train_model.py --data-path data/iris.csv --model random_forest --poison-level 0.0 --run-name clean_rf",
            "desc": "Training Random Forest on clean data"
        },
        {
            "cmd": "python src/train_model.py --data-path data/poisoned/iris_poisoned_5pct.csv --model random_forest --poison-level 0.05 --run-name poison5_rf",
            "desc": "Training Random Forest on 5% poisoned data"
        },
        {
            "cmd": "python src/train_model.py --data-path data/poisoned/iris_poisoned_10pct.csv --model random_forest --poison-level 0.10 --run-name poison10_rf",
            "desc": "Training Random Forest on 10% poisoned data"
        },
        {
            "cmd": "python src/train_model.py --data-path data/poisoned/iris_poisoned_50pct.csv --model random_forest --poison-level 0.50 --run-name poison50_rf",
            "desc": "Training Random Forest on 50% poisoned data"
        },
        
        # Step 4: Evaluate all models
        {
            "cmd": "python src/evaluate_model.py",
            "desc": "Evaluating all models and generating comparison report"
        },
        
        # Step 5: Create visualizations
        {
            "cmd": "python src/visualize_results.py",
            "desc": "Creating comprehensive visualizations"
        }
    ]
    
    # Execute pipeline
    success_count = 0
    total_steps = len(steps)
    
    for i, step in enumerate(steps, 1):
        logger.info(f"\nProgress: {i}/{total_steps}")
        
        if run_command(step["cmd"], step["desc"]):
            success_count += 1
        else:
            logger.error(f"Pipeline failed at step {i}")
            break
    
    # Final summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    logger.info(f"\n{'='*70}")
    logger.info("PIPELINE EXECUTION SUMMARY")
    logger.info(f"{'='*70}")
    logger.info(f"Total steps: {total_steps}")
    logger.info(f"Successful: {success_count}")
    logger.info(f"Failed: {total_steps - success_count}")
    logger.info(f"Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
    logger.info(f"{'='*70}")
    
    if success_count == total_steps:
        logger.info("\n✓ PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("\nResults available in:")
        logger.info("  • MLflow UI: mlflow ui --host 0.0.0.0 --port 5000")
        logger.info("  • Reports: reports/")
        logger.info("  • Visualizations: results/plots/")
        logger.info("  • Models: artifacts/models/")
        return 0
    else:
        logger.error("\n✗ PIPELINE FAILED!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
