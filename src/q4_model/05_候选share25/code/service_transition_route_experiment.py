"""Local development: route source tasks by their service entry/exit positions.

Only the outer task route changes. The frozen Belief, local service action,
known-source sharing, lifetime active-measure budget and coverage are unchanged.
Entries forecast the next actual action under the frozen service rule. A probe
ends at its measure point; an optical service ends at the average first-clear
point under the existing position quadrature. That exit forecast is heuristic,
not true source information. Future entries are frozen during one route search;
the entire route is recomputed after the actually selected task.
"""
import argparse
import ast
import hashlib
import json
import math
import platform
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import scipy

HERE = Path(__file__).resolve().parent
Q4 = next(p for p in HERE.parents if (p/'polar_cover.py').is_file())
PROJECT = Q4.parent.parent
sys.path.insert(0, str(Q4))

import localization
import route_planning
import task_sharing_solver
import q4_share25_client
from shared import core, load_file
from polar_cover import PolarCover

ENVIRONMENT = Q4.parent/'q3_model_v2/00_主方案_lean/code/environment.py'
VALIDATION = Q4/'00_公共几何与定位/code/validation.py'
CASE_SOURCE = Q4/'01_初版31站/code/benchmark.py'
POPULATION_SOURCE = Q4/'04_22站覆盖实验/code/benchmark_cells.py'
LocalArena = load_file('_transition_route_local_environment', ENVIRONMENT).LocalArena
validate_run = load_file('_transition_route_validation', VALIDATION).validate_run
for path, names in ((CASE_SOURCE, {'make_case', 'public_api'}), (POPULATION_SOURCE, {'sources_for'})):
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == names
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), globals())

POPULATIONS = ('uniform', 'boundary', 'cluster', 'omni_heavy', 'dir_heavy')


def service_forecast(belief, start, channel, current_channel, local_steps, maximum):
    """Mirror the unchanged service branch solely to forecast its route endpoints."""
    if belief.near is not None:
        q = belief.near.copy()
        return q, q.copy()
    safe = core.nearest_certified_clear(belief.P, np.asarray(start))
    if safe is not None:
        return safe, safe.copy()
    plan = localization.optical_plan(belief.P, np.asarray(start))
    action = (localization.choose_measure(belief, np.asarray(start), channel, current_channel)
              if local_steps < maximum else None)
    if action is not None and action['score'] < plan['score']:
        q = action['q'].copy()
        return q, q.copy()
    positions = localization.samples(belief.P)
    hits = np.linalg.norm(positions[:, None, :]-plan['path'], axis=2) <= 20
    assert hits.any(axis=1).all()
    end = plan['path'][hits.argmax(axis=1)].mean(axis=0)
    return plan['path'][0].copy(), end


def transition_route(entries, exits, start):
    """Open directed task route with four greedy starts and 12 2-opt sweeps.

    Arc i->j is ||exit_i-entry_j||. Internal task costs are fixed for this
    forecast, hence constant across permutations. Directed reversal includes
    the change of every internal arc; the symmetric shortcut is not valid.
    """
    entries, exits, start = np.asarray(entries), np.asarray(exits), np.asarray(start)
    if not len(entries):
        return []
    distances = np.linalg.norm(exits[:, None, :]-entries[None, :, :], axis=2)
    initial = np.linalg.norm(entries-start, axis=1)
    best = None
    for first in np.argsort(initial, kind='stable')[:min(4, len(entries))]:
        order = [int(first)]
        remaining = set(range(len(entries)))-set(order)
        while remaining:
            k = min(remaining, key=lambda k: (distances[order[-1], k], k))
            order.append(k)
            remaining.remove(k)
        for _ in range(12):
            reverse_delta = np.r_[0., np.cumsum([
                distances[b, a]-distances[a, b] for a, b in zip(order, order[1:])])]
            improvement, change = 0., None
            for i in range(len(order)-1):
                for j in range(i+1, len(order)):
                    old = initial[order[i]] if i == 0 else distances[order[i-1], order[i]]
                    new = initial[order[j]] if i == 0 else distances[order[i-1], order[j]]
                    if j+1 < len(order):
                        old += distances[order[j], order[j+1]]
                        new += distances[order[i], order[j+1]]
                    gain = old-new-(reverse_delta[j]-reverse_delta[i])
                    if gain > improvement+1e-8:
                        improvement, change = gain, (i, j)
            if change is None:
                break
            i, j = change
            order[i:j+1] = order[i:j+1][::-1]
        cost = float(initial[order[0]]+sum(distances[a, b] for a, b in zip(order, order[1:])))
        rank = cost, order
        if best is None or rank < best:
            best = rank
    return best[1]


def candidate_solver():
    """Change four outer route statements; assert every nested function is intact."""
    tree = ast.parse((Q4/'task_sharing_solver.py').read_text(encoding='utf-8'))
    original = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'solve_multi')
    preserved = {n.name: ast.dump(n) for n in original.body if isinstance(n, ast.FunctionDef)}
    loop = next(n for n in original.body if isinstance(n, ast.For))
    replacements = {
        'tasks, points = ([], [])': 'tasks, points, exits = [], [], []',
        'tasks, points = [], []': 'tasks, points, exits = [], [], []',
        'order = open_route(np.asarray(points), api.position)': '''order = transition_route(np.asarray(points), np.asarray(exits), api.position)
trace.append(dict(phase='service_transition_prediction', task=tasks[order[0]][0],
    index=tasks[order[0]][1], entry=np.asarray(points[order[0]]).tolist(),
    expected_exit=np.asarray(exits[order[0]]).tolist()))''',
    }
    new_body, changed = [], []
    for statement in loop.body:
        text = ast.unparse(statement)
        if text in replacements:
            new_body.extend(ast.parse(replacements[text]).body)
            changed.append('allocation' if text.startswith('tasks,') else 'route')
        elif isinstance(statement, ast.For) and ast.unparse(statement.iter) == 'pending':
            assert ast.unparse(statement.body[-1]) == 'points.append(cover.stations[k])'
            statement.body.extend(ast.parse('exits.append(cover.stations[k])').body)
            new_body.append(statement)
            changed.append('scan_exit')
        elif isinstance(statement, ast.For) and ast.unparse(statement.iter) == 'beliefs.items()':
            assert ast.unparse(statement.body[-1]) == 'points.append(core.mec(b.P)[0])'
            statement.body[-1:] = ast.parse('''entry, exit_point = service_forecast(
    b, np.asarray(api.position), c, api.channel, local_steps[c], max_active_measures)
points.append(entry)
exits.append(exit_point)''').body
            new_body.append(statement)
            changed.append('source_endpoints')
        else:
            new_body.append(statement)
    assert sorted(changed) == sorted(['allocation', 'route', 'scan_exit', 'source_endpoints']), changed
    loop.body = new_body
    assert preserved == {n.name: ast.dump(n) for n in original.body if isinstance(n, ast.FunctionDef)}
    transformed = ast.fix_missing_locations(ast.Module(body=[original], type_ignores=[]))
    namespace = task_sharing_solver.solve_multi.__globals__.copy()
    namespace.update(service_forecast=service_forecast, transition_route=transition_route)
    exec(compile(transformed, str(Path(__file__)), 'exec'), namespace)
    assert task_sharing_solver.Belief is localization.Belief
    return namespace['solve_multi']


def route_checks():
    p = np.array([[10., 0.], [0., 20.], [-20., 10.], [30., -30.], [25., 10.]])
    assert transition_route(p, p, np.zeros(2)) == route_planning.open_route(p, np.zeros(2))
    return dict(nested_local_functions_ast_unchanged=True,
                symmetric_case_reduces_to_original_route=True,
                directed_reversal_accounts_for_internal_arcs=True)


def hashes():
    result = {f'src/{k}': v for k, v in q4_share25_client.source_hashes().items()}
    for p in (Path(__file__), ENVIRONMENT, VALIDATION, CASE_SOURCE, POPULATION_SOURCE):
        result[p.relative_to(PROJECT).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    return result


def write(path, value, exclusive=False):
    with path.open('x' if exclusive else 'w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def run_case(seed, population, variant, candidate):
    sources = sources_for(seed, population)
    arena, trace = LocalArena(sources, seed, 'smooth'), []
    result, error = None, None
    started = time.perf_counter()
    try:
        solver = task_sharing_solver.solve_multi if variant == 'share25' else candidate
        result = solver(public_api(arena), variant='share25', trace=trace)
    except Exception:
        error = traceback.format_exc()
    runtime = time.perf_counter()-started
    predicted_entries, pending = 0, None
    try:
        if error:
            raise AssertionError(error)
        mesh = PolarCover()
        audit = validate_run(sources, arena.events, trace, result, mesh.stations, mesh.indices)
        for row in trace:
            if row['phase'] == 'service_transition_prediction':
                pending = row
            elif row['phase'] == 'actual_action' and pending is not None:
                assert np.allclose(row['position'], pending['entry'], rtol=0, atol=1e-9), (pending, row)
                predicted_entries += 1
                pending = None
    except Exception:
        audit = dict(passed=False, error=traceback.format_exc())
    row = dict(seed=seed, population=population, variant=variant, field='smooth',
               runtime_s=runtime, audit=audit, actual_first_actions_matching_forecast=predicted_entries,
               **arena.evaluation())
    return dict(summary=row, sources=sources, result=result, events=arena.events, trace=trace)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    candidate = candidate_solver()
    checks = route_checks()
    snapshot = hashes()
    write(out/'manifest.json', dict(code_hashes=snapshot, checks=checks, local_only=True,
          official_simulator_used=False, python=platform.python_version(), numpy=np.__version__,
          scipy=scipy.__version__, populations=POPULATIONS, seeds=[35000+100*k for k in range(5)],
          field='smooth', variants=['share25', 'service_transition_route'],
          directed_route_starts=4, directed_route_sweeps=12,
          limitation='Five predeclared development layouts; not an official or holdout estimate.'), exclusive=True)
    rows = []
    for index, population in enumerate(POPULATIONS):
        seed = 35000+100*index
        for variant in ('share25', 'service_transition_route'):
            case = run_case(seed, population, variant, candidate)
            row = case['summary']
            rows.append(row)
            write(out/f'{seed}-{population}-{variant}.json', case, exclusive=True)
            write(out/'rows.json', rows)
            print(f'{seed} {population} {variant}: {row["total_s"]:.2f}s; '
                  f'{row["cleared"]}/{row["source_count"]}; audit={row["audit"]["passed"]}; '
                  f'wall={row["runtime_s"]:.2f}s', flush=True)
    pairs = []
    for index, population in enumerate(POPULATIONS):
        base, proposed = rows[2*index:2*index+2]
        valid = base['audit']['passed'] and proposed['audit']['passed']
        pairs.append(dict(population=population, seed=35000+100*index, passed=valid,
                          baseline_s=base['total_s'], candidate_s=proposed['total_s'],
                          saved_s=base['total_s']-proposed['total_s'] if valid else None,
                          reduction_percent=100*(base['total_s']-proposed['total_s'])/base['total_s'] if valid else None))
    groups = {}
    for variant in ('share25', 'service_transition_route'):
        group = [r for r in rows if r['variant'] == variant]
        valid = all(r['audit']['passed'] for r in group)
        groups[variant] = dict(runs=len(group), passes=sum(r['audit']['passed'] for r in group),
            sources=sum(r['source_count'] for r in group), cleared=sum(r['cleared'] for r in group),
            mean_total_s=float(np.mean([r['total_s'] for r in group])) if valid else None,
            mean_wall_s=float(np.mean([r['runtime_s'] for r in group])))
    stable = hashes() == snapshot
    report = dict(passed=stable and all(r['audit']['passed'] for r in rows),
                  local_only=True, source_snapshot_stable=stable, groups=groups, pairs=pairs)
    write(out/'summary.json', report, exclusive=True)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
