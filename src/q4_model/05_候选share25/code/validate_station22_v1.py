"""Independent actual-evidence and exact geometry validation for the new 22 cover."""
from copy import deepcopy
from fractions import Fraction as F
from functools import lru_cache
import json

import numpy as np
from audit_station_certificates_v1 import audit, cross, hull, inside, norm2


@lru_cache(maxsize=1)
def static_geometry(path):
    report = audit(path)
    data = json.loads(path.read_text())
    points = tuple(tuple(F(x) for x in p) for p in data['stations'])
    parents = [tuple(points[k] for k in ids) for ids in data['initial_triangles']]
    cells, vertices, indices = [], {}, []
    for row in data['cells']:
        triangle = parents[row['parent']]
        for digit in row['branch']:
            code = int(digit)
            a, b, c = code % 3, (code+1) % 3, (code+2) % 3
            middle = tuple((x+y)/2 for x, y in zip(triangle[a], triangle[b]))
            triangle = ((triangle[a], middle, triangle[c]) if code < 3
                        else (middle, triangle[b], triangle[c]))
        cells.append(triangle)
        index = []
        for p in triangle:
            if p not in vertices:
                vertices[p] = len(vertices)
            index.append(vertices[p])
        indices.append(index)
    return report, points, tuple(cells), np.array(list(vertices), float), np.array(indices)


@lru_cache(maxsize=16384)
def exact_actual_witness(triangle, keys):
    points = tuple(tuple(F(x) for x in p) for p in keys)
    poly = hull(points)
    return len(poly) >= 3 and all(inside(v, poly) and all(norm2(v, p) <= 1000**2 for p in points) for v in triangle)


def validate(result, events, path):
    static, stations, cells, vertices, indices = static_geometry(path)
    cert = result['certificate']
    assert result['complete'] and cert['basis'] in ('directional_cell_cover', 'count_upper_bound')
    assert cert['geometry_sha256'] == static['input_sha256']
    assert cert['station_count'] == len(stations) and cert['cell_count'] == len(cells)
    assert np.array_equal(np.asarray(cert['stations']), np.array(stations, float))
    assert np.array_equal(np.asarray(cert['vertices']), vertices)
    assert np.array_equal(np.asarray(cert['triangles']), indices)
    removed, positive = set(), set()
    accepted_negative = {c: set() for c in range(1, 21)}
    for event in events:
        c = event['channel']
        if event['action'] == 'measure':
            if event['measure_result'] == 'no_signal':
                accepted_negative[c].add(tuple(event['position']))
            else:
                positive.add(c)
        elif event['clear_result'] == 'success':
            removed.add(c)
    assert len(result['cleared']) == len(removed) and set(result['cleared']) == removed
    assert set(cert['cleared']) == removed
    absent = cert['absent']
    assert len(set(absent)) == len(absent)
    assert all(isinstance(c, int) and not isinstance(c, bool) and 1 <= c <= 20 for c in absent)
    assert not set(absent) & (removed | positive)
    checks = 0
    if cert['basis'] == 'count_upper_bound':
        assert len(removed) == 16
    else:
        assert len(removed)+len(absent) == 20
        assert set(cert['channels']) == {str(c) for c in absent}
        for c in absent:
            row = cert['channels'][str(c)]
            keys = tuple(tuple(p) for p in row['negative_points'])
            assert row['complete'] and row['triangles'] == len(cells) and row['covered'] == len(cells)
            assert all(key in accepted_negative[c] for key in keys)
            assert {int(t) for t in row['triangle_witnesses']} == set(range(len(cells)))
            for t, ids in row['triangle_witnesses'].items():
                assert all(isinstance(k, int) and not isinstance(k, bool) and 0 <= k < len(keys) for k in ids)
                assert exact_actual_witness(cells[int(t)], tuple(sorted(keys[k] for k in ids)))
                checks += 1
    adapted = deepcopy(result)
    adapted['certificate']['stations'] = cert['vertices']
    return dict(passed=True, actual_negative_witness_cells=checks, static=static), adapted, vertices, indices
