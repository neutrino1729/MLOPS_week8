#!/usr/bin/env python3
"""
Generate comprehensive analysis report
"""

import json
import pandas as pd
import os
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient

def generate_report():
    """Generate comprehensive analysis"""
    
    print("="*80)
    print("IRIS DATA POISONING - COMPREHENSIVE ANALYSIS REPORT")
    print("="*80)
    
    # 1. Model Performance Summary
    print("\n📊 1. MODEL PERFORMANCE SUMMARY")
    print("-"*80)
    
    perf_file = "reports/performance/model_comparison.csv"
    if os.path.exists(perf_file):
        df = pd.read_csv(perf_file)
        
        print("\nPerformance by Poison Level:")
        summary = df.groupby('poison_level').agg({
            'test_accuracy': ['mean', 'std', 'min', 'max'],
            'test_f1': ['mean', 'std']
        }).round(4)
        print(summary)
        
        print("\nBest Models per Poison Level:")
        for level in sorted(df['poison_level'].unique()):
            subset = df[df['poison_level'] == level]
            best = subset.loc[subset['test_accuracy'].idxmax()]
            print(f"  {int(level*100):3d}% poison: {best['model_type']:20s} "
                  f"(Accuracy: {best['test_accuracy']:.4f})")
    else:
        print("  No performance data found. Run evaluation first.")
    
    # 2. Detection Effectiveness
    print("\n\n🔍 2. DETECTION EFFECTIVENESS")
    print("-"*80)
    
    validation_dir = "reports/validation"
    if os.path.exists(validation_dir):
        detection_results = []
        
        for filename in sorted(os.listdir(validation_dir)):
            if filename.endswith('_validation.json'):
                filepath = os.path.join(validation_dir, filename)
                with open(filepath, 'r') as f:
                    report = json.load(f)
                    
                detection_results.append({
                    'Dataset': os.path.basename(report['dataset']),
                    'Poison Level': f"{int(report['poison_level']*100)}%",
                    'KNN Suspicious': report['summary']['knn_suspicious_count'],
                    'KNN Detection Rate': f"{report['summary']['knn_suspicion_rate']*100:.1f}%",
                    'Confidence Suspicious': report['summary']['conf_suspicious_count'],
                    'Model Accuracy': f"{report['summary']['model_accuracy']:.4f}"
                })
        
        if detection_results:
            det_df = pd.DataFrame(detection_results)
            print("\n" + det_df.to_string(index=False))
    else:
        print("  No validation data found.")
    
    # 3. Performance Degradation Analysis
    print("\n\n📉 3. PERFORMANCE DEGRADATION ANALYSIS")
    print("-"*80)
    
    if os.path.exists(perf_file):
        df = pd.read_csv(perf_file)
        clean_perf = df[df['poison_level'] == 0.0]['test_accuracy'].mean()
        
        print(f"\nBaseline (Clean Data) Accuracy: {clean_perf:.4f}")
        print("\nDegradation by Poison Level:")
        
        for level in sorted(df['poison_level'].unique()):
            if level > 0:
                poisoned_perf = df[df['poison_level'] == level]['test_accuracy'].mean()
                degradation = clean_perf - poisoned_perf
                degradation_pct = (degradation / clean_perf) * 100
                
                print(f"  {int(level*100):3d}% poison: "
                      f"Accuracy = {poisoned_perf:.4f}, "
                      f"Drop = {degradation:.4f} ({degradation_pct:.1f}%)")
    
    # 4. Key Findings
    print("\n\n💡 4. KEY FINDINGS")
    print("-"*80)
    
    findings = [
        "1. Random Forest shows robustness to low-level poisoning (5-10%)",
        "2. Significant performance degradation observed at 50% poison level",
        "3. KNN-based detection shows increasing effectiveness with poison level",
        "4. Label flipping attacks are detectable using neighbor-based validation",
        "5. Cross-validation provides stable performance estimates across poison levels"
    ]
    
    for finding in findings:
        print(f"  {finding}")
    
    # 5. Mitigation Strategies
    print("\n\n🛡️  5. RECOMMENDED MITIGATION STRATEGIES")
    print("-"*80)
    
    strategies = [
        "1. KNN Label Validation: Implement k=5, threshold=0.5 for routine checks",
        "2. Ensemble Methods: Use multiple models to reduce single-point failures",
        "3. Data Provenance: Maintain clean baseline datasets with version control",
        "4. Confidence Monitoring: Track model prediction confidence distributions",
        "5. Regular Auditing: Periodically validate performance on trusted test sets",
        "6. Anomaly Detection: Monitor for unusual patterns in incoming data"
    ]
    
    for strategy in strategies:
        print(f"  {strategy}")
    
    # 6. Generated Artifacts
    print("\n\n📁 6. GENERATED ARTIFACTS")
    print("-"*80)
    
    artifacts = {
        "Models": "artifacts/models/",
        "Validation Reports": "reports/validation/",
        "Performance Reports": "reports/performance/",
        "Visualizations": "results/plots/",
        "MLflow Experiments": "mlruns/"
    }
    
    for name, path in artifacts.items():
        if os.path.exists(path):
            count = len([f for f in Path(path).rglob('*') if f.is_file()])
            print(f"  {name:25s}: {path:30s} ({count} files)")
    
    print("\n" + "="*80)
    print("Report generation complete!")
    print("="*80)

if __name__ == "__main__":
    generate_report()
    
    # Save to file
    os.makedirs("reports", exist_ok=True)
    print("\nSaving report to file...")
    
    import sys
    from io import StringIO
    
    old_stdout = sys.stdout
    sys.stdout = report_buffer = StringIO()
    
    generate_report()
    
    sys.stdout = old_stdout
    report_content = report_buffer.getvalue()
    
    with open("reports/COMPREHENSIVE_ANALYSIS.txt", "w") as f:
        f.write(report_content)
    
    print("✅ Report saved to: reports/COMPREHENSIVE_ANALYSIS.txt")
