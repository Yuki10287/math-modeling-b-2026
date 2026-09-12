"""Local development experiment: integrate feasible heading/radius prior volume.

The frozen 25-station policy is injected with a Belief subclass in a private
function globals dictionary. No frozen module globals, conservative position
polygon, optical fallback, termination certificate or official entry change.

Scoring assumptions (not distributions stated by the problem): equal initial
source-type mass, uniform heading on [0, 2*pi), uniform radius on [1000,1500],
and the existing 16-point position quadrature. Observed visibility changes only
at observation-vector angles +/- pi/2. Feasible interval prior volume is exact
apart from float tolerances. Prediction at a NEW point is still approximate:
three-point Gauss rules integrate each heading interval and radius interval.
The unseen point's own visibility discontinuity is not an observation boundary.

Zero-volume or missed sampled hypotheses never remove the continuous position
set. Empty scoring scenarios cause the original certified optical fallback.
Failed clearances filter scoring samples only, exactly as in the old scorer.

Run this file from any directory with --out pointing to a NEW results folder.
Five fixed development populations, one layout each, are paired at seeds
32000,32100,32200,32300,32400; there is no search or automatic policy promotion.
"""
import argparse
import hashlib
import json
import math
import platform
from pathlib import Path
import sys
import time
import traceback
import types

import numpy as np
import scipy

HERE = Path(__file__).resolve().parent
Q4 = HERE.parents[1]
SRC = Q4.parent
sys.path.insert(0, str(Q4))

import localization
import task_sharing_solver
import q4_share25_client
from polar_cover import PolarCover
from shared import load_file

ENVIRONMENT_PATH = SRC/'q3_model_v2/00_主方案_lean/code/environment.py'
VALIDATION_PATH = Q4/'00_公共几何与定位/code/validation.py'
LocalArena = load_file('_continuous_hypothesis_local_environment', ENVIRONMENT_PATH).LocalArena
validate_run = load_file('_continuous_hypothesis_independent_validation', VALIDATION_PATH).validate_run

POPULATIONS = ('uniform', 'boundary', 'cluster', 'omni_heavy', 'dir_heavy')
FIELD = 'smooth'
GAUSS_NODES = (-math.sqrt(3/5), 0., math.sqrt(3/5))
GAUSS_WEIGHTS = (5/18, 4/9, 5/18)  # Unit-interval weights sum to one.


class VolumeBelief(localization.Belief):
    """Change finite action scoring only; inherit the full conservative update."""

    def scenarios(self):
        gs = localization.samples(self.P)
        rows = []
        positive = np.asarray(self.positives, float).reshape(-1, 2)
        negative = np.asarray(self.negatives, float).reshape(-1, 2)
        position_weight = 1 / len(gs)
        for index, g in enumerate(gs):
            if any(np.linalg.norm(g-q) <= 20 for q in self.failed_clears):
                continue
            pv, nv = positive-g, negative-g
            pd = np.linalg.norm(pv, axis=1)
            nd = np.linalg.norm(nv, axis=1)
            lower = max(1000., float(pd.max()))
            if lower >= 1500.:
                continue

            def add(n, upper, angle_mass):
                width = upper-lower
                if width <= 0:
                    return
                # Uniform R has density 1/500; initial type probability is 1/2.
                prior_mass = position_weight * .5 * angle_mass * width/500
                middle, half = (lower+upper)/2, width/2
                for node, weight in zip(GAUSS_NODES, GAUSS_WEIGHTS):
                    rows.append((index, g, n, middle+half*node, prior_mass*weight))

            omni_upper = min(1500., float(nd.min())-1e-7) if len(nd) else 1500.
            add(None, omni_upper, 1.)
            vectors = np.vstack((pv, nv))
            vectors = vectors[np.linalg.norm(vectors, axis=1) > 1e-10]
            angles = np.arctan2(vectors[:, 1], vectors[:, 0])
            cuts = np.unique(np.concatenate(([0., 2*math.pi],
                (angles-math.pi/2) % (2*math.pi),
                (angles+math.pi/2) % (2*math.pi))))
            for left, right in zip(cuts[:-1], cuts[1:]):
                width = right-left
                if width <= 1e-12:
                    continue
                middle, half = (left+right)/2, width/2
                heading = np.array([math.cos(middle), math.sin(middle)])
                if np.any(pv @ heading < -1e-10):
                    continue
                visible_negative = nd[nv @ heading >= 0]
                upper = (min(1500., float(visible_negative.min())-1e-7)
                         if len(visible_negative) else 1500.)
                if upper <= lower:
                    continue
                for node, weight in zip(GAUSS_NODES, GAUSS_WEIGHTS):
                    angle = middle+half*node
                    n = np.array([math.cos(angle), math.sin(angle)])
                    add(n, upper, width/(2*math.pi)*weight)
        return gs, rows


def injected_solver():
    """A fresh function, with only its Belief binding changed in private globals."""
    original = task_sharing_solver.solve_multi
    namespace = original.__globals__.copy()
    namespace['Belief'] = VolumeBelief
    result = types.FunctionType(original.__code__, namespace, original.__name__,
                                original.__defaults__, original.__closure__)
    result.__kwdefaults__ = original.__kwdefaults__
    assert original.__globals__['Belief'] is localization.Belief
    return result


def quadrature_checks():
    """Analytic prior-volume identities, including a zero-volume orientation set."""
    results = []
    for name, positives, negatives, failed, expected in (
        ('one_positive_half_heading', [[100., 0.]], [], [], .75),
        ('opposite_negative_directional_only', [[100., 0.]], [[-100., 0.]], [], .25),
        ('visible_negative_radius_volume', [[100., 0.]], [[1200., 0.]], [], .75*(200-1e-7)/500),
        ('opposite_positives_zero_directional_volume', [[100., 0.], [-100., 0.]], [], [], .5),
        ('failed_clear_filters_samples_only', [[100., 0.]], [], [[0., 0.]], 0.)):
        belief = VolumeBelief.__new__(VolumeBelief)
        belief.P = np.array([[0., 0.]])
        belief.positives = list(map(np.array, positives))
        belief.negatives = list(map(np.array, negatives))
        belief.failed_clears = list(map(np.array, failed))
        before = belief.P.copy()
        gs, scenarios = belief.scenarios()
        mass = sum(row[4] for row in scenarios)
        assert abs(mass-expected) < 1e-10, (name, mass, expected)
        assert np.array_equal(before, belief.P) and np.array_equal(gs, before)
        for _, g, n, radius, weight in scenarios:
            assert weight > 0 and 1000 <= radius <= 1500
            assert all(np.linalg.norm(q-g) <= radius and (n is None or (q-g)@n >= -1e-7)
                       for q in belief.positives)
            assert all(np.linalg.norm(q-g) > radius or (n is not None and (q-g)@n < 1e-7)
                       for q in belief.negatives)
        results.append(dict(name=name, passed=True, expected=expected, actual=mass, scenarios=len(scenarios)))
    return results


def sources_for(seed, population):
    """Same generator rules as benchmark.make_case + benchmark_cells.sources_for.

    Kept here to avoid importing their obsolete flat runtime dependencies.
    These truth values are consumed only by the arena and independent auditor.
    """
    rng = np.random.default_rng(seed)
    count = int(rng.integers(10, 17))
    channels = rng.choice(np.arange(1, 21), count, replace=False)
    sources = []
    for k, channel in enumerate(channels):
        angle = rng.uniform(0, 2*math.pi)
        if population == 'boundary':
            distance = 1800. if k % 2 == 0 else rng.uniform(1740, 1800)
            orientation = angle if k % 3 else angle+math.pi/2
            radius = 1000.
        elif population == 'cluster':
            angle = .25+rng.uniform(-.2, .2)
            distance = rng.uniform(800, 1300)
            orientation = rng.uniform(0, 2*math.pi)
            radius = rng.uniform(1000, 1500)
        else:
            distance = 1800*math.sqrt(rng.uniform())
            orientation = rng.uniform(0, 2*math.pi)
            radius = rng.uniform(1000, 1500)
        directional = k != 0 and (population == 'boundary' or k % 2 == 1)
        sources.append(dict(channel=int(channel),
            position=[distance*math.cos(angle), distance*math.sin(angle)],
            radius=radius, orientation=orientation if directional else None))
    if population in ('omni_heavy', 'dir_heavy'):
        rng = np.random.default_rng(seed+1000000)
        minority = int(rng.integers(len(sources)))
        for k, source in enumerate(sources):
            directional = k == minority if population == 'omni_heavy' else k != minority
            source['orientation'] = float(rng.uniform(0, 2*np.pi)) if directional else None
    return sources


def public_api(arena):
    class Public:
        __slots__ = ()
        @property
        def position(self): return arena.position.copy()
        @property
        def channel(self): return arena.channel
        def measure(self, q, c): return arena.measure(q, c)
        def clear(self, q, c): return arena.clear(q, c)
    return Public()


def hashes():
    result = {f'src/{k}': v for k, v in q4_share25_client.source_hashes().items()}
    for path in (Path(__file__), ENVIRONMENT_PATH, VALIDATION_PATH):
        result[path.relative_to(SRC.parent).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def write(path, data, exclusive=False):
    with path.open('x' if exclusive else 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def run_case(seed, population, variant):
    sources = sources_for(seed, population)
    arena, trace = LocalArena(sources, seed, FIELD), []
    result, error = None, None
    started = time.perf_counter()
    try:
        solver = task_sharing_solver.solve_multi if variant == 'share25' else injected_solver()
        result = solver(public_api(arena), variant='share25', trace=trace)
    except Exception:
        error = traceback.format_exc()
    runtime = time.perf_counter()-started
    try:
        if error:
            raise RuntimeError(error)
        cover = PolarCover()
        audit = validate_run(sources, arena.events, trace, result, cover.stations, cover.indices)
    except Exception:
        audit = dict(passed=False, error=traceback.format_exc())
    row = dict(seed=seed, population=population, field=FIELD, variant=variant,
               runtime_s=runtime, audit=audit, **arena.evaluation())
    return dict(summary=row, sources=sources, result=result, events=arena.events, trace=trace)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    checks = quadrature_checks()
    snapshot = hashes()
    frozen = json.loads((Q4/'05_候选share25/results/share25_client_release.json').read_text(encoding='utf-8'))
    current = q4_share25_client.source_hashes()
    assert all(frozen['source_sha256'][f'src/{k}'] == v for k, v in current.items())
    write(out/'manifest.json', dict(code_hashes=snapshot, local_only=True,
        official_simulator_used=False, frozen_runtime_unchanged=True,
        python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
        positions=16, angular_quadrature_points_per_interval=3, radius_quadrature_points=3,
        quadrature='Gauss-Legendre', initial_type_prior=[.5, .5], radius_prior='uniform [1000,1500]',
        heading_prior='uniform [0,2*pi)', populations=POPULATIONS, seeds=[32000+100*k for k in range(5)],
        layouts=5, field=FIELD, variants=['share25', 'volume_integral25'],
        purpose='Five predeclared development layouts; no holdout or official performance claim.',
        quadrature_checks=checks), exclusive=True)
    rows = []
    for index, population in enumerate(POPULATIONS):
        seed = 32000+100*index
        for variant in ('share25', 'volume_integral25'):
            case = run_case(seed, population, variant)
            row = case['summary']
            rows.append(row)
            write(out/f'{seed}-{population}-{variant}.json', case, exclusive=True)
            write(out/'rows.json', rows)
            print(f'{seed} {population} {variant}: {row["total_s"]:.2f}s; '
                  f'{row["cleared"]}/{row["source_count"]}; runtime {row["runtime_s"]:.2f}s; '
                  f'audit={row["audit"]["passed"]}', flush=True)
    pairs = []
    for index, population in enumerate(POPULATIONS):
        baseline, candidate = rows[2*index:2*index+2]
        valid = baseline['audit']['passed'] and candidate['audit']['passed']
        pairs.append(dict(population=population, seed=32000+100*index, passed=valid,
            baseline_s=baseline['total_s'], candidate_s=candidate['total_s'],
            saved_s=baseline['total_s']-candidate['total_s'] if valid else None,
            reduction_percent=100*(baseline['total_s']-candidate['total_s'])/baseline['total_s'] if valid else None))
    groups = {}
    for variant in ('share25', 'volume_integral25'):
        group = [r for r in rows if r['variant'] == variant]
        valid = all(r['audit']['passed'] for r in group)
        groups[variant] = dict(runs=len(group), audit_passes=sum(r['audit']['passed'] for r in group),
            source_instances=sum(r['source_count'] for r in group), cleared=sum(r['cleared'] for r in group),
            mean_total_s=float(np.mean([r['total_s'] for r in group])) if valid else None,
            pooled_s_per_source=sum(r['total_s'] for r in group)/sum(r['source_count'] for r in group) if valid else None,
            mean_solver_wall_s=float(np.mean([r['runtime_s'] for r in group])))
    stable = hashes() == snapshot
    assert task_sharing_solver.Belief is localization.Belief
    summary = dict(passed=stable and all(r['audit']['passed'] for r in rows),
                   source_snapshot_stable=stable, baseline_global_unchanged=True,
                   local_only=True, independent_development_layouts=5, groups=groups, pairs=pairs)
    write(out/'summary.json', summary, exclusive=True)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if summary['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
