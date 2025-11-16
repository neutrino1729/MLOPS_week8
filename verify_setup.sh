#!/bin/bash
echo "======================================"
echo "  MLOPS WEEK 8 - Environment Check"
echo "======================================"
echo ""

echo "1. Python Environment"
echo "   Python version: $(python --version)"
echo ""

echo "2. Critical Packages"
python << 'PYEOF'
import sys
packages = ['mlflow', 'sklearn', 'pandas', 'numpy', 'google.cloud.storage', 'dvc']
for pkg in packages:
    try:
        mod = __import__(pkg)
        version = getattr(mod, '__version__', 'OK')
        print(f"   ✓ {pkg}: {version}")
    except ImportError:
        print(f"   ✗ {pkg}: NOT INSTALLED")
PYEOF
echo ""

echo "3. Project Structure"
echo "   Working directory: $(pwd)"
ls -d */ 2>/dev/null | head -5
echo ""

echo "4. Datasets"
echo "   Available datasets:"
ls -lh data/*.csv 2>/dev/null | awk '{print "   - "$9" ("$5")"}'
echo ""

echo "5. GCP Configuration"
echo "   Project: $(gcloud config get-value project 2>/dev/null)"
echo "   Account: $(gcloud config get-value account 2>/dev/null)"
echo ""

echo "6. Disk Space"
df -h / | tail -1 | awk '{print "   Root: "$3" used / "$4" available ("$5" used)"}'
df -h /home/jupyter 2>/dev/null | tail -1 | awk '{print "   Home: "$3" used / "$4" available ("$5" used)"}'
echo ""

echo "======================================"
echo "  Setup Verification Complete!"
echo "======================================"
