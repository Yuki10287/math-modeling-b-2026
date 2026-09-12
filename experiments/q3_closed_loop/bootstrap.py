"""Paths to the immutable Q3 model and the previous independent experiment."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
Q3 = ROOT / 'src' / 'q3_model_v2'
PREVIOUS = ROOT / 'experiments' / 'q3_exploration'
for directory in (Q3, PREVIOUS):
    if str(directory) not in sys.path:
        sys.path.append(str(directory))

import geometry as core
import solver as baseline
from belief_model import FeedbackModel
