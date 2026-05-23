#!/usr/bin/env python3
"""Backward-compatible alias: scripts/grid_search.py is the canonical entry point."""
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).with_name("grid_search.py")), run_name="__main__")
