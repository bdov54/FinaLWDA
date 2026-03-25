#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("Testing import of generate_model_comparison_commentary...")

try:
    from review_analytics.pipeline.infer_runtime_binary import generate_model_comparison_commentary
    print("✓ Successfully imported generate_model_comparison_commentary")
    print(f"✓ Function type: {type(generate_model_comparison_commentary)}")
    print(f"✓ Function name: {generate_model_comparison_commentary.__name__}")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test function exists and is callable
if callable(generate_model_comparison_commentary):
    print("✓ Function is callable")
else:
    print("✗ Function is not callable")
    sys.exit(1)

print("\nAll tests passed!")
