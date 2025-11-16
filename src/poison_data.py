#!/usr/bin/env python3
"""
Data Poisoning Script with MLflow Integration
Enhanced version with experiment tracking
"""

import pandas as pd
import numpy as np
import argparse
import os
import json
import logging
from pathlib import Path
import mlflow
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from params.yaml"""
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)

def poison_labels(input_path: str, output_path: str, poison_level: float, 
                 random_seed: int = 42, log_to_mlflow: bool = True) -> dict:
    """
    Create poisoned dataset with label flipping
    
    Args:
        input_path: Path to clean dataset
        output_path: Path to save poisoned dataset
        poison_level: Fraction of labels to flip (0.0 to 1.0)
        random_seed: Random seed for reproducibility
        log_to_mlflow: Whether to log to MLflow
    
    Returns:
        Dictionary with poisoning metadata
    """
    np.random.seed(random_seed)
    
    # Load data
    logger.info(f"Loading dataset from: {input_path}")
    df = pd.read_csv(input_path)
    
    # Get target column (last column)
    target_column = df.columns[-1]
    unique_labels = df[target_column].unique().tolist()
    
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Unique labels: {unique_labels}")
    
    # Calculate poisoning
    num_rows = len(df)
    num_to_poison = int(num_rows * poison_level)
    
    logger.info(f"Poisoning {num_to_poison} of {num_rows} rows ({poison_level*100:.1f}%)")
    
    # Create poisoned copy
    df_poisoned = df.copy()
    
    # Randomly select indices to poison
    poison_indices = np.random.choice(df.index, size=num_to_poison, replace=False)
    
    # Track label changes
    label_changes = {label: {new: 0 for new in unique_labels if new != label} 
                    for label in unique_labels}
    
    # Flip labels
    for idx in poison_indices:
        original_label = df_poisoned.loc[idx, target_column]
        possible_new = [l for l in unique_labels if l != original_label]
        new_label = np.random.choice(possible_new)
        df_poisoned.loc[idx, target_column] = new_label
        label_changes[original_label][new_label] += 1
    
    # Save poisoned dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_poisoned.to_csv(output_path, index=False)
    logger.info(f"Saved poisoned dataset to: {output_path}")
    
    # Create metadata
    metadata = {
        "original_file": input_path,
        "poisoned_file": output_path,
        "poison_level": poison_level,
        "total_rows": num_rows,
        "poisoned_rows": num_to_poison,
        "random_seed": random_seed,
        "label_changes": label_changes,
        "unique_labels": unique_labels
    }
    
    # Save metadata
    metadata_path = output_path.replace('.csv', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Log to MLflow
    if log_to_mlflow:
        mlflow.set_experiment("data_poisoning")
        with mlflow.start_run(run_name=f"poison_{int(poison_level*100)}pct"):
            mlflow.log_param("poison_level", poison_level)
            mlflow.log_param("total_rows", num_rows)
            mlflow.log_param("poisoned_rows", num_to_poison)
            mlflow.log_param("random_seed", random_seed)
            
            # Log label changes as metrics
            for orig, changes in label_changes.items():
                for new, count in changes.items():
                    if count > 0:
                        mlflow.log_metric(f"flipped_{orig}_to_{new}", count)
            
            # Log metadata as artifact
            mlflow.log_artifact(metadata_path)
    
    logger.info("Label flip summary:")
    for orig, changes in label_changes.items():
        for new, count in changes.items():
            if count > 0:
                logger.info(f"  {orig} -> {new}: {count} instances")
    
    return metadata

def create_all_poison_levels(input_path: str, output_dir: str = "data/poisoned"):
    """Create datasets for all poison levels defined in config"""
    config = load_config()
    poison_levels = config['poisoning']['levels']
    random_seed = config['poisoning']['random_seed']
    
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    for level in poison_levels:
        level_pct = int(level * 100)
        output_path = os.path.join(output_dir, f"iris_poisoned_{level_pct}pct.csv")
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Creating {level_pct}% poisoned dataset")
        logger.info(f"{'='*60}")
        
        metadata = poison_labels(input_path, output_path, level, random_seed)
        results[f"{level_pct}pct"] = metadata
    
    # Save summary
    summary_path = os.path.join(output_dir, "poisoning_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nSummary saved to: {summary_path}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create poisoned datasets")
    parser.add_argument("--input-path", default="data/iris.csv", help="Input CSV file")
    parser.add_argument("--output-path", help="Output CSV file")
    parser.add_argument("--poison-level", type=float, help="Poison level (0.0-1.0)")
    parser.add_argument("--create-all", action="store_true", help="Create all poison levels")
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow logging")
    
    args = parser.parse_args()
    
    # Set MLflow tracking
    mlflow.set_tracking_uri("./mlruns")
    
    if args.create_all:
        create_all_poison_levels(args.input_path)
    elif args.poison_level is not None and args.output_path:
        poison_labels(args.input_path, args.output_path, args.poison_level, 
                     log_to_mlflow=not args.no_mlflow)
    else:
        parser.error("Either --create-all or both --poison-level and --output-path required")
