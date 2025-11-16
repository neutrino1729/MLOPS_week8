# 🔬 IRIS Data Poisoning MLOps Pipeline

> **Complete MLOps pipeline demonstrating data poisoning attacks and detection mechanisms using IRIS dataset**

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![MLflow](https://img.shields.io/badge/MLflow-2.9-orange.svg)](https://mlflow.org/)
[![DVC](https://img.shields.io/badge/DVC-3.30-purple.svg)](https://dvc.org/)
[![GCP](https://img.shields.io/badge/GCP-Vertex_AI-red.svg)](https://cloud.google.com/vertex-ai)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Pipeline Components](#pipeline-components)
- [Results](#results)
- [Project Structure](#project-structure)
- [MLflow Tracking](#mlflow-tracking)
- [Contributing](#contributing)

---

## 🎯 Overview

This project implements a complete MLOps pipeline for studying **data poisoning attacks** on machine learning systems. It demonstrates:

- **Label flipping attacks** at 5%, 10%, and 50% poisoning levels
- **KNN-based detection** mechanisms for identifying poisoned data
- **Model performance degradation** analysis across poison levels
- **MLflow experiment tracking** with GCS backend
- **DVC data versioning** for reproducibility
- **Comprehensive visualizations** and reporting

### Key Objectives

1. ✅ Demonstrate realistic data poisoning scenarios
2. ✅ Implement effective detection mechanisms
3. ✅ Quantify model robustness to poisoning
4. ✅ Provide mitigation strategies
5. ✅ Establish reproducible MLOps workflow

---

## ⚡ Features

### Data Poisoning
- **Multiple poison levels**: 5%, 10%, 50%
- **Label flipping attacks**: Random mislabeling
- **Reproducible**: Configurable random seeds
- **Metadata tracking**: Complete audit trail

### Detection & Validation
- **KNN-based validation**: Neighbor disagreement analysis
- **Confidence-based detection**: Model prediction analysis
- **Statistical monitoring**: Class distribution checks
- **Comprehensive reporting**: Detailed JSON reports

### ML Pipeline
- **Multiple algorithms**: Random Forest, SVM, Logistic Regression
- **Cross-validation**: Robust performance estimation
- **MLflow integration**: Complete experiment tracking
- **Artifact management**: Models, scalers, encoders

### Visualization & Analysis
- **Performance plots**: Accuracy vs poison level
- **Detection effectiveness**: KNN detection rates
- **Confusion matrices**: Error pattern analysis
- **Heatmaps**: Model comparison across poison levels

---

## 🏗️ Architecture
```
┌─────────────────────────────────────────────────────────┐
│                   Data Layer (GCS)                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐             │
│  │ Clean    │  │ Poisoned │  │ Poisoned │             │
│  │ IRIS     │  │ 5%       │  │ 10%, 50% │             │
│  └──────────┘  └──────────┘  └──────────┘             │
└─────────────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────┐
│              Poisoning & Validation Layer               │
│  ┌──────────────┐         ┌──────────────┐             │
│  │ Label        │────────→│ KNN          │             │
│  │ Flipping     │         │ Validation   │             │
│  └──────────────┘         └──────────────┘             │
└─────────────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────┐
│                  Training Layer                         │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐                │
│  │ Random  │  │   SVM   │  │Logistic │                │
│  │ Forest  │  │         │  │Regress. │                │
│  └─────────┘  └─────────┘  └─────────┘                │
└─────────────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────┐
│            MLflow Tracking & Artifacts (GCS)            │
│  • Experiments  • Metrics  • Models  • Reports          │
└─────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites
```bash
# Python 3.10+
python --version

# GCP credentials configured
gcloud auth list
```

### Installation
```bash
# Clone repository
git clone https://github.com/neutrino1729/MLOPS_week8.git
cd MLOPS_week8

# Install dependencies
pip install -r requirements-gcp.txt

# Configure DVC
dvc remote modify myremote projectname theta-index-472515-d8
```

### Run Complete Pipeline
```bash
# Option 1: One-command execution
python run_complete_pipeline.py

# Option 2: Using DVC
dvc repro

# Option 3: Step-by-step (see RUN_INSTRUCTIONS.md)
```

### View Results
```bash
# Start MLflow UI
mlflow ui --host 0.0.0.0 --port 5000

# Generate analysis report
python generate_analysis_report.py

# View visualizations
ls results/plots/
```

---

## 🔧 Pipeline Components

### 1. Data Poisoning (`src/poison_data.py`)
Creates poisoned datasets with configurable poison levels.
```bash
python src/poison_data.py --create-all
```

### 2. Label Validation (`src/validate_labels.py`)
Detects suspicious labels using KNN and confidence methods.
```bash
python src/validate_labels.py --data-path data/iris.csv
```

### 3. Model Training (`src/train_model.py`)
Trains models with MLflow tracking.
```bash
python src/train_model.py --data-path data/iris.csv --model random_forest
```

### 4. Evaluation (`src/evaluate_model.py`)
Compares model performance across poison levels.
```bash
python src/evaluate_model.py
```

### 5. Visualization (`src/visualize_results.py`)
Generates comprehensive plots and charts.
```bash
python src/visualize_results.py
```

---

## 📊 Results

### Model Performance

| Poison Level | Accuracy | F1 Score | Degradation |
|--------------|----------|----------|-------------|
| 0% (Clean)   | 0.9667   | 0.9665   | Baseline    |
| 5%           | 0.9333   | 0.9324   | 3.5%        |
| 10%          | 0.9000   | 0.8989   | 6.9%        |
| 50%          | 0.6667   | 0.6542   | 31.0%       |

### Detection Effectiveness

| Poison Level | KNN Detection Rate | Suspicious Samples |
|--------------|-------------------|-------------------|
| 0% (Clean)   | 5.3%              | 8 / 150           |
| 5%           | 8.7%              | 13 / 150          |
| 10%          | 14.0%             | 21 / 150          |
| 50%          | 52.7%             | 79 / 150          |

### Key Findings

1. **Linear Degradation**: Performance drops roughly linearly with poison level up to 10%
2. **Severe Impact**: 50% poisoning causes ~31% performance degradation
3. **Detection Works**: KNN method effectively identifies poisoned samples
4. **Model Resilience**: Random Forest shows better robustness than linear models

---

## 📁 Project Structure
```
MLOPS_week8/
├── README.md                      # This file
├── RUN_INSTRUCTIONS.md            # Detailed execution guide
├── params.yaml                    # Pipeline configuration
├── dvc.yaml                       # DVC pipeline definition
├── requirements-gcp.txt           # Python dependencies
│
├── src/                           # Source code
│   ├── poison_data.py            # Data poisoning implementation
│   ├── validate_labels.py        # Label validation (KNN, confidence)
│   ├── train_model.py            # Model training with MLflow
│   ├── evaluate_model.py         # Model evaluation & comparison
│   └── visualize_results.py      # Visualization generation
│
├── data/                          # Datasets (DVC tracked)
│   ├── iris.csv                  # Clean IRIS dataset
│   └── poisoned/                 # Poisoned datasets (5%, 10%, 50%)
│
├── artifacts/                     # Model artifacts
│   └── models/                   # Trained models, scalers, encoders
│
├── reports/                       # Analysis reports
│   ├── validation/               # Label validation reports
│   └── performance/              # Model performance reports
│
├── results/                       # Visualizations
│   └── plots/                    # Generated charts & graphs
│
├── mlruns/                        # MLflow experiments (local)
│
├── scripts/                       # Utility scripts
│   └── setup_mlflow.py          # MLflow configuration
│
└── tests/                         # Test suite
    ├── test_model.py
    └── test_data_validation.py
```

---

## 📈 MLflow Tracking

### Experiments Tracked

1. **data_poisoning**: Dataset creation experiments
2. **label_validation**: Validation effectiveness
3. **iris_poisoning_training**: Model training runs

### Logged Information

- **Parameters**: poison_level, model_type, hyperparameters
- **Metrics**: accuracy, F1, precision, recall, CV scores
- **Artifacts**: models, scalers, validation reports, plots
- **Tags**: experiment_type, dataset_name, timestamp

### Viewing MLflow UI
```bash
# Local tracking
mlflow ui --host 0.0.0.0 --port 5000

# GCS backend (configured)
export MLFLOW_ARTIFACT_LOCATION=gs://mlops-course-dulcet-bastion-452612-v4-unique/mlartifacts
```

---

## 🛡️ Mitigation Strategies

Based on experimental results, we recommend:

1. **Proactive Detection**
   - Implement KNN validation (k=5, threshold=0.5)
   - Monitor class distribution statistics
   - Track prediction confidence distributions

2. **Robust Training**
   - Use ensemble methods (Random Forest performed best)
   - Implement cross-validation for stability
   - Maintain clean baseline datasets

3. **Continuous Monitoring**
   - Regular model performance audits
   - Anomaly detection on incoming data
   - Data provenance tracking with DVC

4. **Defense in Depth**
   - Multi-source data verification
   - Model ensemble voting
   - Periodic retraining on validated data

---

## 🧪 Testing
```bash
# Quick component tests
python quick_test.py

# Full test suite
pytest tests/ -v

# Coverage report
pytest tests/ --cov=src --cov-report=html
```

---

## 📦 Deployment

### Upload to GCS
```bash
# Upload results
gsutil -m cp -r results/ gs://mlops-course-dulcet-bastion-452612-v4-unique/results/
gsutil -m cp -r artifacts/ gs://mlops-course-dulcet-bastion-452612-v4-unique/artifacts/
```

### DVC Push
```bash
# Push data to DVC remote
dvc push
```

---

## 👥 Contributing

Contributions welcome! Please follow these steps:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

---

## 📄 License

This project is part of MLOps coursework - educational purposes only.

---

## 🙏 Acknowledgments

- **IRIS Dataset**: Fisher, R.A. (1936)
- **MLflow**: Databricks
- **DVC**: Iterative.ai
- **GCP Vertex AI**: Google Cloud Platform

---

## 📧 Contact

**Rakesh Bhattacharjee**
- Email: rakesh.bhattacharjee24@gmail.com
- GitHub: [@neutrino1729](https://github.com/neutrino1729)

---

**Project Status**: ✅ Complete | **Last Updated**: November 2025
