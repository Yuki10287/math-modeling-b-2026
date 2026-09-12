"""Local-only convex-region optical cover, after frozen fallback selection.

The old optical score still chooses measurement versus clearance and drives
sharing. Only an already selected optical execution plan is replaced. Cells
intersect the actual rational convex region; empty cells are skipped only by
exact clipping. An independent auditor reconstructs all intersections.
"""
import ast
from fractions import Fraction as F
import math
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
Q4 = HERE.parents[1]
sys.path.insert(0, str(Q4))
import localization
import task_sharing_solver
from shared import core


def exact(points):
    return [tuple(F(float(x)) for x in p) for p in points]


def hull(points):
    points = sorted(set(points))
    if len(points) < 3:
        return points
    def cross(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    chains = []
    for sequence in (points, points[::-1]):
        chain = []
        for p in sequence:
            while len(chain) >= 2 and cross(chain[-2], chain[-1], p) <= 0:
                chain.pop()
            chain.append(p)
        chains.append(chain)
    return chains[0][:-1]+chains[1][:-1]


def clip(poly, axis, boundary, lower):
    if not poly:
        return []
    output = []
    for a, b in zip(poly, poly[1:]+poly[:1]):
        da, db = a[axis]-boundary, b[axis]-boundary
        ia, ib = (da >= 0, db >= 0) if lower else (da <= 0, db <= 0)
        if ia:
            output.append(a)
        if ia != ib:
            t = da/(da-db)
            output.append(tuple(x+t*(y-x) for x, y in zip(a, b)))
    return hull(output)


def geometry(P):
    values = np.asarray(P, float)
    if values.ndim != 2 or values.shape[1] != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Expected nonempty finite polygon')
    points = exact(values)
    _, (i, j) = core.diameter(values)
    origin = points[i]
    u = tuple(b-a for a, b in zip(points[i], points[j]))
    if u == (0, 0):
        u = (F(1), F(0))
    v = (-u[1], u[0])
    scale = sum(x*x for x in u)
    def local(p):
        d = tuple(x-y for x, y in zip(p, origin))
        return sum(x*y for x, y in zip(d, u))/scale, sum(x*y for x, y in zip(d, v))/scale
    def world(p):
        return tuple(origin[k]+p[0]*u[k]+p[1]*v[k] for k in range(2))
    polygon = hull(list(map(local, points)))
    lo = tuple(min(p[k] for p in polygon) for k in range(2))
    hi = tuple(max(p[k] for p in polygon) for k in range(2))
    extent = tuple(b-a for a, b in zip(lo, hi))
    radius = F(20)-F(1, 10000)
    n = [max(1, math.ceil(float(width)*math.sqrt(float(scale))/(math.sqrt(2)*float(radius))))
         for width in extent]
    while scale*sum((width/count)**2 for width, count in zip(extent, n))/4 > radius**2:
        k = max(range(2), key=lambda k: extent[k]/n[k])
        n[k] += 1
    pieces = {}
    for x in range(n[0]):
        for y in range(n[1]):
            bounds = [(lo[k]+extent[k]*index/n[k], lo[k]+extent[k]*(index+1)/n[k])
                      for k, index in enumerate((x, y))]
            part = polygon
            for k in range(2):
                part = clip(clip(part, k, bounds[k][0], True), k, bounds[k][1], False)
            if not part:
                continue
            center = tuple((min(p[k] for p in part)+max(p[k] for p in part))/2 for k in range(2))
            actual = tuple(float(z) for z in world(center))
            actual_exact = tuple(F(z) for z in actual)
            assert all(sum((a-b)**2 for a, b in zip(world(p), actual_exact)) <= 400 for p in part)
            pieces[(x, y)] = actual
    if not pieces:
        raise RuntimeError('Empty optical cover for nonempty region')
    return pieces, dict(version='convex-grid-optical-v1', original_polygon=values.tolist(),
        basis_indices=[int(i), int(j)], grid_counts=n, radius_m=20,
        partition='closed_uniform_grid_in_exact_orthogonal_basis',
        omitted_only_when_exact_intersection_empty=True)


def optical_plan(P, start, quadrature=None):
    pieces, certificate = geometry(P)
    nx, ny = certificate['grid_counts']
    points = localization.samples(P) if quadrature is None or not len(quadrature) else quadrature
    best = None
    for ys in (range(ny), range(ny-1, -1, -1)):
        for reverse in (False, True):
            cells = []
            for k, y in enumerate(ys):
                xs = range(nx-1, -1, -1) if bool(k % 2)^reverse else range(nx)
                cells.extend((x, y) for x in xs if (x, y) in pieces)
            path = np.asarray([pieces[cell] for cell in cells])
            cumulative = np.cumsum(np.linalg.norm(np.diff(np.vstack((start, path)), axis=0), axis=1))/5
            hits = np.linalg.norm(np.asarray(points)[:, None, :]-path, axis=2) <= 20
            if not hits.any(axis=1).all():
                raise RuntimeError('Independent scoring samples outside exact optical cover')
            first = hits.argmax(axis=1)
            mean = float(np.mean(cumulative[first]+3*first+5))
            worst = float(cumulative[-1]+3*(len(path)-1)+5)
            score = .8*mean+.2*worst
            if best is None or score < best['score']:
                proof = dict(certificate, occupied_cells=[list(c) for c in cells], actual_points=path.tolist())
                best = dict(path=path, mean_s=mean, worst_s=worst, score=score,
                    occupied_count=len(path), rectangle_count=nx*ny,
                    omitted_cells=nx*ny-len(path), optical_certificate=proof)
    return best


def make_solver():
    parsed = ast.parse(Path(task_sharing_solver.__file__).read_text(encoding='utf-8-sig'))
    function = next(n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name == 'solve_multi')
    service = next(n for n in function.body if isinstance(n, ast.FunctionDef) and n.name == 'service')
    matches = []
    for index, statement in enumerate(service.body):
        if isinstance(statement, ast.Expr) and "phase='optical_plan'" in ast.unparse(statement):
            matches.append(index)
    assert len(matches) == 1
    replacement = ast.parse('''
old_rectangle_plan = plan
plan = _polygon_optical_plan(belief.P, np.asarray(api.position))
trace.append(dict(phase='polygon_optical_plan', channel=c, event=calls,
    polygon=belief.P.tolist(), start=np.asarray(api.position).tolist(),
    old_rectangle_count=len(old_rectangle_plan['path']),
    old_rectangle_score=old_rectangle_plan['score'],
    old_rectangle_worst_s=old_rectangle_plan['worst_s'],
    **{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in plan.items()}))
''').body
    index = matches[0]
    service.body[index:index+1] = replacement
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = task_sharing_solver.solve_multi.__globals__.copy()
    namespace['_polygon_optical_plan'] = optical_plan
    exec(compile(module, str(Path(__file__)), 'exec'), namespace)
    return namespace['solve_multi']


def solve_multi(api, trace=None):
    return make_solver()(api, variant='share25', trace=trace)
