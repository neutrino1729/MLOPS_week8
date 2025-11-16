# IRIS Data Poisoning MLOps Pipeline - Execution Guide

## 🚀 Quick Start

### Option 1: Run Complete Pipeline (Recommended)
```bash
# Run everything in one command
python run_complete_pipeline.py
```

### Option 2: Run with DVC
```bash
# Execute DVC pipeline
dvc repro

# View results
mlflow ui --host 0.0.0.0 --port 5000
```

### Option 3: Step-by-Step Manual Execution

#### Step 1: Create Poisoned Datasets
```bash
python src/poison_data.py --create-all
```

#### Step 2: Validate Datasets
```bash
# Clean data
python src/validate_labels.py --data-path data/iris.csv --poison-level 0.0

# Poisoned data
python src/validate_labels.py --data-path data/poisoned/iris_poisoned_5pct.csv --poison-level 0.05
python src/validate_labels.py --data-path data/poisoned/iris_poisoned_10pct.csv --poison-level 0.10
python src/validate_labels.py --data-path data/poisoned/iris_poisoned_50pct.csv --poison-level 0.50
```

#### Step 3: Train Models
```bash
# Clean data
python src/train_model.py --data-path data/iris.csv --model random_forest --poison-level 0.0

# Poisoned data
python src/train_model.py --data-path data/poisoned/iris_poisoned_5pct.csv --model random_forest --poison-level 0.05
python src/train_model.py --data-path data/poisoned/iris_poisoned_10pct.csv --model random_forest --poison-level 0.10
python src/train_model.py --data-path data/poisoned/iris_poisoned_50pct.csv --model random_forest --poison-level 0.50
```

#### Step 4: Evaluate
```bash
python src/evaluate_model.py
```

#### Step 5: Visualize
```bash
python src/visualize_results.py
```

## 📊 View Results

### MLflow UI
```bash
mlflow ui --host 0.0.0.0 --port 5000
```
Then open: `http://localhost:5000`

### Generated Files
- **Models**: `artifacts/models/`
- **Reports**: `reports/`
- **Visualizations**: `results/plots/`
- **MLflow Runs**: `mlruns/`

## 🧪 Testing

### Quick Test
```bash
python quick_test.py
```

### Full Test Suite
```bash
pytest tests/ -v
```

## 📦 Upload to GCS
```bash
# Upload results to GCS
gsutil -m cp -r results/ gs://mlops-course-dulcet-bastion-452612-v4-unique/results/
gsutil -m cp -r reports/ gs://mlops-course-dulcet-bastion-452612-v4-unique/reports/
gsutil -m cp -r artifacts/ gs://mlops-course-dulcet-bastion-452612-v4-unique/artifacts/
```

## 🔧 Troubleshooting

### Issue: MLflow not tracking
```bash
# Check MLflow config
source .mlflow_config
echo $MLFLOW_TRACKING_URI
```

### Issue: DVC errors
```bash
# Reinitialize DVC
dvc init -f
dvc remote add -d myremote gs://mlops-course-dulcet-bastion-452612-v4-unique/dvc-storage
```

## ✅ Success Criteria

After running the pipeline, you should have:
- ✓ 4 poisoned datasets (5%, 10%, 50% + clean)
- ✓ Validation reports for all datasets
- ✓ Trained models on all poison levels
- ✓ Performance comparison reports
- ✓ Comprehensive visualizations
- ✓ MLflow experiments tracked

