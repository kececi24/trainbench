#!/usr/bin/env python3
"""Convenience alias for src/plot_curves.py."""
import sys
import os
import runpy

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("src"))

if __name__ == "__main__":
    runpy.run_module("src.plot_curves", run_name="__main__")
