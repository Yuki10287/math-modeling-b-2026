"""Independent canonical-geometry audit plus the established trajectory audit."""
from copy import deepcopy
from functools import lru_cache
import hashlib
import importlib.util
from pathlib import Path

import numpy as np

from cell_cover import CellCover, CERTIFICATE_PATH, CERTIFICATE_SHA256
from validation import validate_run


@lru_cache(maxsize=4)
def _independent_geometry_audit(certificate_sha256, script_sha256):
    path = CERTIFICATE_PATH.with_name('independent_exact_audit.py')
    spec = importlib.util.spec_from_file_location('_q4_independent_cell_geometry', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module.audit(CERTIFICATE_PATH)
    assert report['passed'] and report['input_sha256'] == certificate_sha256
    assert report['target_disk_radius_1800_contained_exactly']
    assert report['exact_bisection_partition_verified']
    return report


def audit_canonical_geometry(cover=None):
    """Re-audit the exact saved construction; detect mutation of geometry views."""
    actual_hash = hashlib.sha256(CERTIFICATE_PATH.read_bytes()).hexdigest()
    assert actual_hash == CERTIFICATE_SHA256, 'canonical certificate changed'
    script = CERTIFICATE_PATH.with_name('independent_exact_audit.py')
    script_hash = hashlib.sha256(script.read_bytes()).hexdigest()
    report = _independent_geometry_audit(actual_hash, script_hash)
    if cover is not None:
        canonical = CellCover()
        for name in ('stations', 'vertices', 'indices', 'triangles'):
            assert np.array_equal(getattr(cover, name), getattr(canonical, name)), name
        assert cover.exact_cells == canonical.exact_cells
        assert cover.pre_witnesses == canonical.pre_witnesses
    return dict(report)


def validate_cell_run(sources, events, trace, result, cover=None):
    """Validate a CellCover run without confusing proof vertices with scan stops.

The old validator's ``stations`` parameter names geometric mesh vertices. Only
the private validation copy uses that legacy name; the actual result preserves
the 22 measurement stations separately from the 108-cell vertex/index view.
"""
    cover = CellCover() if cover is None else cover
    geometry_report = audit_canonical_geometry(cover)
    certificate = result['certificate']
    assert certificate['basis'] in ('directional_cell_cover', 'count_upper_bound')
    expected = cover.geometry_certificate()
    for key in ('geometry_sha256', 'station_count', 'cell_count', 'cell_geometry'):
        assert certificate[key] == expected[key], key
    for key in ('stations', 'vertices', 'triangles'):
        assert np.array_equal(np.asarray(certificate[key]), np.asarray(expected[key])), key
    adapted = deepcopy(result)
    adapted['certificate']['stations'] = certificate['vertices']
    report = validate_run(sources, events, trace, adapted, cover.vertices, cover.indices)
    report.update(actual_measurement_stations=len(cover.stations),
                  proof_cells=len(cover.triangles),
                  proof_vertices=len(cover.vertices),
                  canonical_geometry_passed=geometry_report['passed'],
                  canonical_geometry_sha256=geometry_report['input_sha256'])
    return report
