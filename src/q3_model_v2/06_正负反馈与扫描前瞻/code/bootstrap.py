"""Import frozen Q3 modules without moving or modifying the release."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
Q3 = ROOT/'src/q3_model_v2'
if str(Q3) not in sys.path:
    sys.path.append(str(Q3))

import geometry as core
import solver as baseline
from belief_model import FeedbackModel, exclude_disk_hull

