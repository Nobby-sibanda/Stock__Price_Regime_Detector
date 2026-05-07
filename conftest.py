"""
Root conftest: ensure the installed stock_regime_detector *package* is found
rather than the stock_regime_detector.py launcher script, which would shadow it
and cause "not a package" errors during test collection.
"""
import sys
import os

_project_dir = os.path.dirname(os.path.abspath(__file__))

# Remove the project directory itself from sys.path so that
# stock_regime_detector.py (the launcher) doesn't shadow the installed package.
while _project_dir in sys.path:
    sys.path.remove(_project_dir)
