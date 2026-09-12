"""Preserve optical order; skip only disks proved disjoint from the source set.

No point moves, belief update, sensing decision, route, or formal CLI changes.
The original rectangle plan is kept as a theoretical continuous-cover witness.
Execution uses its ordered subsequence, whose removed radius-20 closed disks
are strictly disjoint from the certified region. No feedback labels or truth
are passed to the selector. New output folders are exclusive.
"""
import argparse
import ast
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import traceback

import numpy as np
import scipy

HERE = Path(__file__).resolve().parent
Q4 = HERE.parents[1]
PROJECT = Q4.parent.parent
sys.path.insert(0, str(Q4))
import task_sharing_solver
import q4_share25_client
from polar_cover import PolarCover
from continuous_hypothesis_experiment import (
    LocalArena, public_api, sources_for, validate_run, ENVIRONMENT_PATH,
    VALIDATION_PATH, POPULATIONS,
)
from negative_region_boxes_v1 import contract_position
from audit_negative_region_boxes_v1 import verify_certificate, demand
from late_negative_optical_experiment import validation


def rational_point(p):
    return tuple(F(float(x)) for x in p)


def cross(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])


def exact_hull(points):
    points = sorted(set(map(rational_point, points)))
    demand(bool(points), 'empty polygon')
    if len(points) <= 1:
        return points
    halves = []
    for sequence in (points, points[::-1]):
        stack = []
        for p in sequence:
            while len(stack) >= 2 and cross(stack[-2], stack[-1], p) <= 0:
                stack.pop()
            stack.append(p)
        halves.append(stack)
    return halves[0][:-1]+halves[1][:-1]


def distance_squared(q, hull):
    """Exact distance to the convex hull of the ACTUAL binary-float vertices."""
    q = rational_point(q)
    if len(hull) >= 3 and all(cross(a, b, q) >= 0 for a, b in zip(hull, hull[1:]+hull[:1])):
        return F(0)
    values = []
    for a, b in zip(hull, hull[1:]+hull[:1]):
        v = (b[0]-a[0], b[1]-a[1])
        length2 = v[0]*v[0]+v[1]*v[1]
        t = ((q[0]-a[0])*v[0]+(q[1]-a[1])*v[1])/length2 if length2 else F(0)
        t = min(F(1), max(F(0), t))
        values.append(sum((q[i]-a[i]-t*v[i])**2 for i in (0, 1)))
    return min(values)


def prune_path(P, positives, negatives, path):
    """Selector receives geometry and prior measurements, never outcomes/truth."""
    original = np.asarray(P, float)
    contracted, info = contract_position(original, positives, negatives)
    region = np.asarray(contracted, float) if info['applied'] else original
    h0, h1 = exact_hull(original), exact_hull(region)
    d0 = [distance_squared(q, h0) for q in path]
    d1 = [distance_squared(q, h1) for q in path]
    # The output float hull may slightly expand original P. Keep both sound
    # source sets by intersecting their logical constraints: either disjoint
    # disk proof is sufficient, without constructing another float polygon.
    removed0 = [i for i, d in enumerate(d0) if d > 400]
    removed1 = [i for i, d in enumerate(d1) if d > 400]
    removed = sorted(set(removed0) | set(removed1))
    kept = [i for i in range(len(path)) if i not in set(removed)]
    demand(bool(kept), 'all optical points pruned; fail closed')
    return dict(original_polygon=original.tolist(), polygon=region.tolist(),
                positives=[np.asarray(p).tolist() for p in positives],
                negatives=[np.asarray(p).tolist() for p in negatives],
                original_path=np.asarray(path).tolist(), original_removed_indices=removed0,
                contracted_removed_indices=removed1, removed_indices=removed,
                kept_indices=kept, original_distances_squared=[str(d) for d in d0],
                contracted_distances_squared=[str(d) for d in d1], info=info)


def make_solver():
    tree = ast.parse(Path(task_sharing_solver.__file__).read_text(encoding='utf-8-sig'))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'solve_multi')
    original_other = {n.name: ast.dump(n) for n in function.body if isinstance(n, ast.FunctionDef) and n.name != 'service'}
    outer = ast.dump(next(n for n in function.body if isinstance(n, ast.For)))
    service = next(n for n in function.body if isinstance(n, ast.FunctionDef) and n.name == 'service')
    changed = 0
    for index, statement in enumerate(service.body):
        if isinstance(statement, ast.For) and ast.unparse(statement.iter) == "plan['path']":
            statement.iter = ast.parse('execution_path', mode='eval').body
            patch = ast.parse('''
pruning = _prune_path(belief.P, belief.positives, belief.negatives, plan['path'])
trace.append(dict(phase='ordered_optical_prune', channel=c, event=calls, **pruning))
execution_path = [plan['path'][i] for i in pruning['kept_indices']]
''').body
            # Insert immediately before the original optical_plan trace so
            # the original witness is still present and unchanged.
            demand(index >= 1 and 'optical_plan' in ast.unparse(service.body[index-1]), 'unexpected optical trace')
            service.body[index-1:index-1] = patch
            changed += 1
            break
    demand(changed == 1, 'must patch one optical loop')
    demand(original_other == {n.name: ast.dump(n) for n in function.body if isinstance(n, ast.FunctionDef) and n.name != 'service'}, 'local helper changed')
    demand(outer == ast.dump(next(n for n in function.body if isinstance(n, ast.For))), 'route changed')
    demand(not any(isinstance(n, ast.Attribute) and n.attr == 'P' and isinstance(n.ctx, ast.Store)
                   for statement in patch for n in ast.walk(statement)), 'belief polygon assigned')
    namespace = task_sharing_solver.solve_multi.__globals__.copy()
    namespace['_prune_path'] = prune_path
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                 str(Path(__file__)), 'exec'), namespace)
    return namespace['solve_multi']


def audit_prune_case(case):
    """Bind exact disk and contraction proofs to actual preceding observations."""
    beliefs, latest, rows = {}, 0, []
    events, trace = case['events'], case['trace']
    truths = {s['channel']: s['position'] for s in case['sources']}
    for index, row in enumerate(trace):
        if row['phase'] == 'actual_action':
            latest = row['event']
        if row['phase'] == 'belief':
            beliefs[row['channel']] = row
        if row['phase'] != 'ordered_optical_prune':
            continue
        c = row['channel']
        demand(row['event'] == latest, 'prune event binding')
        old = beliefs[c]
        demand(row['original_polygon'] == old['polygon'] and row['positives'] == old['positives']
               and row['negatives'] == old['negatives'], 'prune differs from actual belief')
        observations = [e for e in events[:latest] if e['action'] == 'measure' and e['channel'] == c]
        pos = {tuple(e['position']) for e in observations if e['measure_result'] in ('direction', 'near')}
        neg = {tuple(e['position']) for e in observations if e['measure_result'] == 'no_signal'}
        demand(all(tuple(p) in pos for p in row['positives']), 'unobserved positive')
        demand(all(tuple(p) in neg for p in row['negatives']), 'unobserved negative')
        if row['info']['applied']:
            verify_certificate(row['info']['certificate'], row['original_polygon'], row['positives'],
                               row['negatives'], row['polygon'], require_reduction=True)
        else:
            demand(row['polygon'] == row['original_polygon'], 'unproved new polygon')
        demand(validation.polygon_contains(row['polygon'], truths[c]), 'true source lost')
        plan = trace[index+1]
        demand(plan['phase'] == 'optical_plan' and plan['channel'] == c and
               plan['polygon'] == row['original_polygon'] and plan['path'] == row['original_path'], 'original plan witness changed')
        d0 = [distance_squared(q, exact_hull(row['original_polygon'])) for q in row['original_path']]
        d1 = [distance_squared(q, exact_hull(row['polygon'])) for q in row['original_path']]
        r0, r1 = [i for i,d in enumerate(d0) if d>400], [i for i,d in enumerate(d1) if d>400]
        removed = sorted(set(r0)|set(r1))
        demand(row['original_removed_indices'] == r0 and row['contracted_removed_indices'] == r1
               and row['removed_indices'] == removed, 'incorrect deletion')
        demand(row['kept_indices'] == [i for i in range(len(d0)) if i not in removed], 'reordered kept path')
        actual = []
        for r in trace[index+2:]:
            if r['phase'] != 'actual_action':
                break
            demand(r['reason'] == 'optical_cover' and r['action'] == 'clear' and r['channel'] == c, 'optical action interleaved')
            actual.append(r)
            if r['result'] == 'success':
                break
        demand(actual and actual[-1]['result'] == 'success', 'optical service did not clear')
        demand([a['position'] for a in actual] == [row['original_path'][i] for i in row['kept_indices'][:len(actual)]], 'executed path differs')
        rows.append(dict(channel=c, removed=len(removed), original_only=len(r0),
                         additional_from_contraction=len(set(removed)-set(r0)),
                         contraction_applied=row['info']['applied']))
    return dict(passed=True, plans=len(rows), rows=rows,
                removed=sum(r['removed'] for r in rows),
                original_only=sum(r['original_only'] for r in rows),
                additional_from_contraction=sum(r['additional_from_contraction'] for r in rows))


def event_without_time(event):
    return {k:v for k,v in event.items() if k != 'time_s'}


def retime(events):
    out, position, channel, total = [], np.zeros(2), 1, 0.
    for event in events:
        q, c = np.asarray(event['position']), event['channel']
        total += float(np.linalg.norm(q-position))/5
        if event['action'] == 'measure':
            total += 5+int(c != channel)
            channel = c
        else:
            total += 5 if event['clear_result'] == 'success' else 3
        out.append(dict(event_without_time(event), time_s=total))
        position = q
    return out


def constrained_replay(case):
    """Delete only selector-certified doomed clears from an existing full path."""
    beliefs, latest, removed_events, proofs = {}, 0, set(), []
    for index, row in enumerate(case['trace']):
        if row['phase'] == 'actual_action':
            latest = row['event']
        if row['phase'] == 'belief':
            beliefs[row['channel']] = row
        if row['phase'] != 'optical_plan':
            continue
        c, old = row['channel'], beliefs[row['channel']]
        demand(row['polygon'] == old['polygon'], 'baseline optical plan differs from belief')
        proof = prune_path(old['polygon'], old['positives'], old['negatives'], row['path'])
        if proof['info']['applied']:
            verify_certificate(proof['info']['certificate'], proof['original_polygon'], proof['positives'],
                               proof['negatives'], proof['polygon'], require_reduction=True)
        actual = []
        for action in case['trace'][index+1:]:
            if action['phase'] != 'actual_action':
                break
            demand(action['reason'] == 'optical_cover' and action['channel'] == c, 'baseline optical service interleaved')
            i = len(actual)
            demand(action['position'] == row['path'][i], 'baseline actual point differs')
            actual.append(action)
            if i in proof['removed_indices']:
                demand(action['result'] == 'no_target_in_range', 'selector removed actual success')
                removed_events.add(action['event'])
            if action['result'] == 'success':
                break
        demand(actual and actual[-1]['result'] == 'success', 'baseline fallback no success')
        demand(len(actual)-1 in proof['kept_indices'], 'first successful point removed')
        proof.update(channel=c, event=latest, actual_original_indices=list(range(len(actual))),
                     first_success_index=len(actual)-1,
                     removed_actual_events=[a['event'] for i,a in enumerate(actual) if i in proof['removed_indices']],
                     removed_original_only=sum(i in proof['original_removed_indices'] for i in range(len(actual))),
                     removed_total=sum(i in proof['removed_indices'] for i in range(len(actual))))
        proofs.append(proof)
    new = retime([e for i,e in enumerate(case['events'], 1) if i not in removed_events])
    old_retime = retime(case['events'])
    demand(all(abs(a['time_s']-b['time_s']) < 1e-6 for a,b in zip(old_retime, case['events'])), 'baseline time reconstruction failed')
    old_time, new_time = old_retime[-1]['time_s'], new[-1]['time_s']
    demand(new_time <= old_time+1e-6, 'deletion increased total cost')
    return dict(passed=True, original_s=old_time, candidate_s=new_time, saved_s=old_time-new_time,
                removed_actual=len(removed_events),
                removed_original_only=sum(p['removed_original_only'] for p in proofs),
                additional_from_contraction=sum(p['removed_total']-p['removed_original_only'] for p in proofs),
                removed_event_numbers=sorted(removed_events), events=new, proofs=proofs)


def hashes():
    values = {f'src/{k}':v for k,v in q4_share25_client.source_hashes().items()}
    for p in [Path(__file__), HERE/'negative_region_boxes_v1.py', HERE/'audit_negative_region_boxes_v1.py',
              HERE/'continuous_hypothesis_experiment.py', HERE/'late_negative_optical_experiment.py',
              ENVIRONMENT_PATH, VALIDATION_PATH]:
        values[p.relative_to(PROJECT).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    return values


def write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def run_case(seed, population, variant, candidate):
    sources = sources_for(seed, population)
    arena, trace = LocalArena(sources, seed, 'smooth'), []
    t0 = time.perf_counter()
    result = (task_sharing_solver.solve_multi if variant == 'share25' else candidate)(public_api(arena), variant='share25', trace=trace)
    wall = time.perf_counter()-t0
    case = dict(sources=sources, result=result, events=arena.events, trace=trace)
    mesh = PolarCover()
    audit = validate_run(sources, arena.events, trace, result, mesh.stations, mesh.indices)
    audit['ordered_prune'] = audit_prune_case(case)
    case['summary'] = dict(seed=seed, population=population, variant=variant, runtime_s=wall,
                           audit=audit, **arena.evaluation())
    return case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--replay-batch', type=Path)
    parser.add_argument('--replay-only', action='store_true')
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    snapshot = hashes()
    write(out/'manifest.json', dict(code_hashes=snapshot, local_only=True, official_simulator_contacted=False,
        seeds=[48000+100*i for i in range(5)], populations=POPULATIONS, field='smooth',
        python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
        rule='Preserve original optical path order; remove only exact dist(q,P)^2 > 400, with original or independently certified contracted P.',
        contraction_depth=6, contraction_node_budget=255))
    replay_rows = []
    if args.replay_batch:
        for path in sorted(args.replay_batch.glob('*-share25.json')):
            case = json.loads(path.read_text(encoding='utf-8'))
            replay = constrained_replay(case)
            replay['input_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            write(out/(path.stem+'-replay.json'), replay)
            replay_rows.append(dict(file=path.name, **{k:v for k,v in replay.items() if k not in ('events','proofs')}))
            print(f'replay {path.name}: saved {replay["saved_s"]:.3f}s, removed {replay["removed_actual"]}', flush=True)
        write(out/'replay_summary.json', dict(passed=True, rows=replay_rows))
    if args.replay_only:
        demand(snapshot == hashes(), 'source changed during replay')
        return 0
    solver, rows, pairs = make_solver(), [], []
    for i, population in enumerate(POPULATIONS):
        seed, cases = 48000+100*i, []
        for variant in ('share25','ordered_prune'):
            case = run_case(seed, population, variant, solver)
            cases.append(case)
            rows.append(case['summary'])
            write(out/f'{seed}-{population}-{variant}.json', case)
            print(f'{seed} {population} {variant}: {case["summary"]["total_s"]:.2f}s; audit=True', flush=True)
        base, new = cases
        replay = constrained_replay(base)
        same_actions = [event_without_time(e) for e in replay['events']] == [event_without_time(e) for e in new['events']]
        time_error = max(abs(a['time_s']-b['time_s']) for a,b in zip(replay['events'],new['events'])) if same_actions else None
        demand(same_actions and time_error < 1e-6, 'actual run not equal to certified deletion replay')
        demand(base['result'] == new['result'], 'completion certificate changed')
        write(out/f'{seed}-{population}-homomorphism.json', replay)
        pairs.append(dict(seed=seed,population=population,baseline_s=replay['original_s'],candidate_s=replay['candidate_s'],
            saved_s=replay['saved_s'], reduction_percent=100*replay['saved_s']/replay['original_s'],
            removed_actual=replay['removed_actual'], removed_original_only=replay['removed_original_only'],
            additional_from_contraction=replay['additional_from_contraction'], same_actions=same_actions,
            time_error_s=time_error, identical_completion_certificate=True))
    write(out/'rows.json',rows)
    a = float(np.mean([p['baseline_s'] for p in pairs])); b = float(np.mean([p['candidate_s'] for p in pairs]))
    summary = dict(passed=all(r['audit']['passed'] for r in rows) and snapshot==hashes(),
        source_snapshot_stable=snapshot==hashes(), pairs=pairs, baseline_mean_s=a,candidate_mean_s=b,
        reduction_percent=100*(a-b)/a, faster=sum(p['saved_s']>1e-7 for p in pairs),
        sources=sum(r['source_count'] for r in rows if r['variant']=='ordered_prune'),
        candidate_cleared=sum(r['cleared'] for r in rows if r['variant']=='ordered_prune'),
        removed_actual=sum(p['removed_actual'] for p in pairs),
        removed_original_only=sum(p['removed_original_only'] for p in pairs),
        additional_from_contraction=sum(p['additional_from_contraction'] for p in pairs))
    write(out/'summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False),flush=True)
    return 0 if summary['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
