"""
Local Setup & Verification Script
Run this once to verify everything is working before using Docker.

Usage:
    python setup_and_run.py
"""

import subprocess
import sys
import os

def check_python():
    v = sys.version_info
    print(f"Python {v.major}.{v.minor}.{v.micro}")
    if v.major != 3 or v.minor != 10:
        print("⚠️  Warning: Recommended Python is 3.10.x (you have {}.{})".format(v.major, v.minor))
    else:
        print("✅ Python version OK")

def check_imports():
    packages = [
        "streamlit", "sentence_transformers", "chromadb",
        "pandas", "numpy", "faker", "mlflow"
    ]
    for pkg in packages:
        try:
            __import__(pkg)
            print(f"  ✅ {pkg}")
        except ImportError:
            print(f"  ❌ {pkg} - NOT INSTALLED")

def run_quick_test():
    print("\nRunning quick pipeline test...")
    from app.data_generator import generate_synthetic_transactions
    from app.filters import extract_filters

    df = generate_synthetic_transactions(num_rows=10)
    print(f"  ✅ Generated {len(df)} rows")

    filters = extract_filters("high value SWIFT transactions to India above 20000")
    print(f"  ✅ Filters extracted: {filters}")

    print("\n✅ All checks passed! Run with:\n")
    print("    streamlit run streamlit_app.py\n")

if __name__ == "__main__":
    print("=" * 50)
    print("NLP Transaction Search - Setup Check")
    print("=" * 50)
    check_python()
    print("\nChecking installed packages:")
    check_imports()
    run_quick_test()
