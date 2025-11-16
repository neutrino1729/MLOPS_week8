#!/usr/bin/env python3
"""
Quick test of pipeline components
"""

import subprocess
import sys

def test_component(cmd, name):
    """Test a single component"""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode == 0:
        print(f"✓ {name} - PASSED")
        return True
    else:
        print(f"✗ {name} - FAILED")
        return False

def main():
    tests = [
        ("python src/poison_data.py --input-path data/iris.csv --output-path data/test_poison.csv --poison-level 0.05 --no-mlflow",
         "Data Poisoning"),
        ("python src/validate_labels.py --data-path data/iris.csv --poison-level 0.0 --no-mlflow",
         "Label Validation"),
        ("python src/train_model.py --data-path data/iris.csv --model random_forest --poison-level 0.0 --run-name test_run",
         "Model Training")
    ]
    
    results = []
    for cmd, name in tests:
        results.append(test_component(cmd, name))
    
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Passed: {sum(results)}/{len(results)}")
    print(f"Failed: {len(results) - sum(results)}/{len(results)}")
    
    if all(results):
        print("\n✓ ALL TESTS PASSED!")
        return 0
    else:
        print("\n✗ SOME TESTS FAILED!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
