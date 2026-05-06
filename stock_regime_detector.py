"""
Launcher script -- equivalent to: python -m stock_regime_detector [options]

Usage:
    python stock_regime_detector.py [options]

Run with --help for all options.
"""

import runpy
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
runpy.run_module("stock_regime_detector", run_name="__main__", alter_sys=True)
