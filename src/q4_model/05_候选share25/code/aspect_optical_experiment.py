"""Local-only ablation of rectangular optical cover aspect ratios.

The frozen runtime modules are unchanged. Only the rectangular cell counts
vary; every cell still has half diagonal strictly below the 20 m clear radius.
All four snake endpoints are evaluated with the existing 0.8/0.2 cost score.
"""
import math
import types

import numpy as np

import guarded_policy
import localization
import task_sharing_solver
from shared import core


def optical_plan(P, start, quadrature=None):
    _, (i, j) = core.diameter(P)
    u = P[j] - P[i]
    u = u / np.linalg.norm(u) if np.linalg.norm(u) > 1e-10 else np.array([1., 0.])
    rotation = np.column_stack((u, [-u[1], u[0]]))
    origin = P.mean(axis=0)
    local = (P-origin) @ rotation
    lo, hi = local.min(axis=0), local.max(axis=0)
    extent = hi-lo
    radius = 20-1e-4
    base = np.maximum(1, np.ceil(extent/(math.sqrt(2)*radius)).astype(int))
    grids = {tuple(map(int, base))}
    # Search every row count up to the square-grid row count, plus one.
    # For each row count choose the fewest columns giving a valid diagonal.
    # Including the old grid means the scored fallback can never get worse
    # at an identical belief/start, although closed-loop time still can.
    for ny in range(1, int(base[1])+2):
        dy = extent[1]/ny
        if dy >= 2*radius:
            continue
        dx_limit = math.sqrt((2*radius)**2-dy**2)
        nx = max(1, int(math.ceil(extent[0]/dx_limit)))
        if nx*ny <= int(base.prod()):
            grids.add((nx, ny))
    points = localization.samples(P) if quadrature is None or not len(quadrature) else quadrature
    best = None
    for nx, ny in sorted(grids):
        cell_radius = float(np.linalg.norm(extent/[nx, ny])/2)
        if cell_radius > radius+1e-10:
            raise RuntimeError('invalid rectangular optical partition')
        x = lo[0]+(np.arange(nx)+.5)*extent[0]/nx
        y = lo[1]+(np.arange(ny)+.5)*extent[1]/ny
        for ys in (y, y[::-1]):
            for reverse in (False, True):
                line = []
                for k, yy in enumerate(ys):
                    xs = x[::-1] if bool(k % 2)^reverse else x
                    line.extend((xx, yy) for xx in xs)
                path = origin+np.array(line) @ rotation.T
                cumulative = np.cumsum(np.linalg.norm(np.diff(np.vstack((start, path)), axis=0), axis=1))/5
                hits = np.linalg.norm(points[:, None, :]-path, axis=2) <= 20
                if not hits.any(axis=1).all():
                    raise RuntimeError('optical rectangle failed sample sanity check')
                first = hits.argmax(axis=1)
                costs = cumulative[first]+3*first+5
                worst = float(cumulative[-1]+3*(len(path)-1)+5)
                mean = float(costs.mean())
                score = .8*mean+.2*worst
                if best is None or score < best['score']:
                    best = dict(path=path, score=score, worst_s=worst, mean_s=mean,
                                origin=origin, rotation=rotation, lower=lo, upper=hi,
                                cells=[nx, ny], cell_radius=cell_radius)
    return best


def _bind(function, **overrides):
    namespace = dict(function.__globals__)
    namespace.update(overrides)
    bound = types.FunctionType(function.__code__, namespace, function.__name__,
                               function.__defaults__, function.__closure__)
    bound.__kwdefaults__ = function.__kwdefaults__
    return bound


# Private function namespaces prevent accidental mutation of the frozen model.
_choose_measure = _bind(localization.choose_measure, optical_plan=optical_plan)
_score_at = _bind(guarded_policy.score_at, optical_plan=optical_plan)
_solve = _bind(task_sharing_solver.solve_multi, optical_plan=optical_plan,
               choose_measure=_choose_measure, score_at=_score_at)


def solve_multi(api, trace=None):
    return _solve(api, variant='share25', trace=trace)
