#!/usr/bin/env python3
"""
Visualization Script
Creates comprehensive visualizations of poisoning impact
"""

import pandas as pd
import numpy as np
import yaml
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)

def create_performance_plots():
    """Create performance comparison visualizations"""
    
    # Load model comparison data
    results_path = "reports/performance/model_comparison.csv"
    
    if not os.path.exists(results_path):
        logger.error("Run evaluation first: python src/evaluate_model.py")
        return
    
    df = pd.read_csv(results_path)
    
    # Create output directory
    output_dir = "results/plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    config = load_config()
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # 1. Accuracy vs Poison Level
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('IRIS Data Poisoning Impact Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Accuracy by poison level
    ax1 = axes[0, 0]
    for model in df['model_type'].unique():
        model_data = df[df['model_type'] == model].sort_values('poison_level')
        ax1.plot(model_data['poison_level'] * 100, model_data['test_accuracy'], 
                marker='o', label=model, linewidth=2)
    
    ax1.set_xlabel('Poison Level (%)', fontsize=12)
    ax1.set_ylabel('Test Accuracy', fontsize=12)
    ax1.set_title('Model Accuracy vs Poison Level', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: F1 Score by poison level
    ax2 = axes[0, 1]
    for model in df['model_type'].unique():
        model_data = df[df['model_type'] == model].sort_values('poison_level')
        ax2.plot(model_data['poison_level'] * 100, model_data['test_f1'],
                marker='s', label=model, linewidth=2)
    
    ax2.set_xlabel('Poison Level (%)', fontsize=12)
    ax2.set_ylabel('Test F1 Score', fontsize=12)
    ax2.set_title('F1 Score vs Poison Level', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Performance degradation
    ax3 = axes[1, 0]
    clean_perf = df[df['poison_level'] == 0.0].groupby('model_type')['test_accuracy'].mean()
    
    degradation_data = []
    for model in df['model_type'].unique():
        model_df = df[df['model_type'] == model].sort_values('poison_level')
        for _, row in model_df.iterrows():
            if row['poison_level'] > 0:
                deg = (clean_perf[model] - row['test_accuracy']) / clean_perf[model] * 100
                degradation_data.append({
                    'model': model,
                    'poison_level': row['poison_level'] * 100,
                    'degradation': deg
                })
    
    deg_df = pd.DataFrame(degradation_data)
    
    for model in deg_df['model'].unique():
        model_deg = deg_df[deg_df['model'] == model]
        ax3.bar(model_deg['poison_level'] + (list(deg_df['model'].unique()).index(model) - 1) * 5,
               model_deg['degradation'], width=4, label=model, alpha=0.7)
    
    ax3.set_xlabel('Poison Level (%)', fontsize=12)
    ax3.set_ylabel('Performance Degradation (%)', fontsize=12)
    ax3.set_title('Relative Performance Degradation', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Model comparison heatmap
    ax4 = axes[1, 1]
    pivot_data = df.pivot_table(
        values='test_accuracy',
        index='model_type',
        columns='poison_level',
        aggfunc='mean'
    )
    pivot_data.columns = [f"{int(c*100)}%" for c in pivot_data.columns]
    
    sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax4,
               cbar_kws={'label': 'Accuracy'}, vmin=0.5, vmax=1.0)
    ax4.set_title('Accuracy Heatmap: Model × Poison Level', fontsize=14)
    ax4.set_xlabel('Poison Level', fontsize=12)
    ax4.set_ylabel('Model Type', fontsize=12)
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'performance_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Performance plots saved to: {plot_path}")
    
    # 2. Detection effectiveness plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Load validation reports
    validation_dir = "reports/validation"
    if os.path.exists(validation_dir):
        detection_data = []
        
        for filename in os.listdir(validation_dir):
            if filename.endswith('_validation.json'):
                filepath = os.path.join(validation_dir, filename)
                with open(filepath, 'r') as f:
                    report = json.load(f)
                    
                detection_data.append({
                    'poison_level': report['poison_level'] * 100,
                    'knn_detection_rate': report['summary']['knn_suspicion_rate'] * 100,
                    'dataset': os.path.basename(report['dataset'])
                })
        
        if detection_data:
            det_df = pd.DataFrame(detection_data).sort_values('poison_level')
            
            ax.plot(det_df['poison_level'], det_df['knn_detection_rate'],
                   marker='o', linewidth=2, markersize=8, color='#e74c3c')
            ax.fill_between(det_df['poison_level'], 0, det_df['knn_detection_rate'],
                           alpha=0.3, color='#e74c3c')
            
            ax.set_xlabel('Actual Poison Level (%)', fontsize=12)
            ax.set_ylabel('Detection Rate (%)', fontsize=12)
            ax.set_title('Poison Detection Effectiveness (KNN Method)', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            detection_plot = os.path.join(output_dir, 'detection_effectiveness.png')
            plt.savefig(detection_plot, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Detection plot saved to: {detection_plot}")
    
    logger.info("All visualizations created successfully!")
    
    return True

if __name__ == "__main__":
    create_performance_plots()
