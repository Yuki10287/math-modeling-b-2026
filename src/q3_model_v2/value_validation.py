"""Independent geometry checks including roundoff-sized nonconvex vertices."""
import numpy as np
from matplotlib.path import Path as PlotPath
from audit_forecast_results import contains_polygon, boundary_distance, TOLERANCE_M
from validate_model import independent_disk_cover


def trace_truth_audit(arena, trace):
    checked = optical = 0
    assert not any(r.get('phase') == 'belief_conflict' for r in trace)
    for index, row in enumerate(trace):
        if 'polygon' not in row or 'channel' not in row:
            continue
        P = np.asarray(row['polygon'], float)
        truth = np.asarray(arena._sources[row['channel']]['position'], float)
        member = contains_polygon(P, truth)
        independent = bool((len(P) >= 3 and PlotPath(P).contains_point(truth)) or
                           boundary_distance(P, truth) <= TOLERANCE_M)
        assert member == independent, (index, 'polygon membership algorithms disagree')
        assert member, (index, row.get('phase'), row['channel'], 'source excluded')
        checked += 1
        if row.get('optical_path') is not None:
            assert independent_disk_cover(P, row['optical_path'])['complete'], (index, 'optical cover failed')
            optical += 1
    return dict(polygons_checked=checked, full_optical_paths_checked=optical,
        truth_exclusion_violations=0, belief_conflicts=0, independent_membership_crosscheck=True)
