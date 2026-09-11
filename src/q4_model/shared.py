"""Reuse frozen geometry only; do not import the Q3 solver or HTTP client."""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


core = load_file('_b_q4_frozen_geometry', ROOT / 'q3_model_v2/geometry.py')

