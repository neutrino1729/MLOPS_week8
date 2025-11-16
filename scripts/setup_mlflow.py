#!/usr/bin/env python3
"""
MLflow GCS Backend Setup Script
Configures MLflow to use Google Cloud Storage for tracking and artifacts
"""

import os
import mlflow
from google.cloud import storage
import sys

def setup_mlflow_gcs():
    """Configure MLflow with GCS backend"""
    
    # GCS Configuration
    GCS_BUCKET = "mlops-course-dulcet-bastion-452612-v4-unique"
    PROJECT_ID = "theta-index-472515-d8"
    
    # MLflow paths in GCS
    TRACKING_URI = f"gs://{GCS_BUCKET}/mlruns"
    ARTIFACT_LOCATION = f"gs://{GCS_BUCKET}/mlartifacts"
    
    print("=" * 60)
    print("MLflow GCS Backend Setup")
    print("=" * 60)
    
    # Test GCS connection
    print("\n1. Testing GCS connection...")
    try:
        client = storage.Client(project=PROJECT_ID)
        bucket = client.bucket(GCS_BUCKET)
        
        if bucket.exists():
            print(f"   ✓ Successfully connected to bucket: {GCS_BUCKET}")
        else:
            print(f"   ✗ Bucket {GCS_BUCKET} not found")
            return False
            
    except Exception as e:
        print(f"   ✗ GCS connection failed: {e}")
        return False
    
    # Create MLflow directories in GCS
    print("\n2. Creating MLflow directories in GCS...")
    try:
        # Create placeholder files to ensure directories exist
        blob_tracking = bucket.blob("mlruns/.gitkeep")
        blob_artifacts = bucket.blob("mlartifacts/.gitkeep")
        
        if not blob_tracking.exists():
            blob_tracking.upload_from_string("")
            print(f"   ✓ Created mlruns/ directory")
        else:
            print(f"   ✓ mlruns/ directory already exists")
            
        if not blob_artifacts.exists():
            blob_artifacts.upload_from_string("")
            print(f"   ✓ Created mlartifacts/ directory")
        else:
            print(f"   ✓ mlartifacts/ directory already exists")
            
    except Exception as e:
        print(f"   ✗ Failed to create directories: {e}")
        return False
    
    # Set MLflow tracking URI
    print("\n3. Configuring MLflow tracking URI...")
    try:
        # For GCS backend, we need to use local tracking with GCS artifacts
        # This is the recommended approach for GCS
        os.environ['MLFLOW_TRACKING_URI'] = './mlruns'  # Local tracking
        os.environ['MLFLOW_ARTIFACT_LOCATION'] = ARTIFACT_LOCATION  # GCS artifacts
        
        mlflow.set_tracking_uri('./mlruns')
        
        print(f"   ✓ Tracking URI: ./mlruns (local)")
        print(f"   ✓ Artifact Location: {ARTIFACT_LOCATION}")
        
    except Exception as e:
        print(f"   ✗ MLflow configuration failed: {e}")
        return False
    
    # Test MLflow setup
    print("\n4. Testing MLflow setup...")
    try:
        mlflow.set_experiment("test_experiment")
        
        with mlflow.start_run(run_name="test_run") as run:
            mlflow.log_param("test_param", "test_value")
            mlflow.log_metric("test_metric", 1.0)
            
            run_id = run.info.run_id
            
        print(f"   ✓ MLflow test successful")
        print(f"   ✓ Test run ID: {run_id}")
        
        # Clean up test experiment
        mlflow.delete_experiment(mlflow.get_experiment_by_name("test_experiment").experiment_id)
        
    except Exception as e:
        print(f"   ✗ MLflow test failed: {e}")
        return False
    
    # Save configuration
    print("\n5. Saving configuration...")
    config_file = ".mlflow_config"
    with open(config_file, "w") as f:
        f.write(f"MLFLOW_TRACKING_URI=./mlruns\n")
        f.write(f"MLFLOW_ARTIFACT_LOCATION={ARTIFACT_LOCATION}\n")
        f.write(f"GCS_BUCKET={GCS_BUCKET}\n")
        f.write(f"PROJECT_ID={PROJECT_ID}\n")
    
    print(f"   ✓ Configuration saved to {config_file}")
    
    print("\n" + "=" * 60)
    print("✓ MLflow GCS Backend Setup Complete!")
    print("=" * 60)
    
    print("\nConfiguration Details:")
    print(f"  • Tracking: Local SQLite (./mlruns)")
    print(f"  • Artifacts: GCS ({ARTIFACT_LOCATION})")
    print(f"  • Bucket: {GCS_BUCKET}")
    print(f"  • Project: {PROJECT_ID}")
    
    print("\nNext Steps:")
    print("  1. Source the configuration: source .mlflow_config")
    print("  2. View MLflow UI: mlflow ui --host 0.0.0.0 --port 5000")
    print("  3. Start logging experiments!")
    
    return True

if __name__ == "__main__":
    success = setup_mlflow_gcs()
    sys.exit(0 if success else 1)
