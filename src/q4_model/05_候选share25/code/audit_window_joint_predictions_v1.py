"""Audit saved window rollout records without sampling worlds or solving cases.

Actual and hypothetical public states are reconstructed ONLY from their saved
feedback events. No sources field is inspected. Original observation geometry
and open-route utilities are reused; no original macro/solver is executed.
"""
import argparse
from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
Q4 = HERE.parents[1]
PROJECT = Q4.parent.parent
sys.path.insert(0, str(Q4))
from localization import Belief, optical_plan
from polar_cover import PolarCover
from route_planning import open_route
from shared import core


def demand(value, message):
    if not value:
        raise ValueError(message)


def close(a, b, name):
    demand(math.isfinite(float(a)) and math.isfinite(float(b)) and abs(a-b) <= 1e-7,
           f'{name}: {a!r} != {b!r}')


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(type(value).__name__)


def public_digest(s):
    payload = dict(position=s.position, channel=s.channel,
        beliefs={str(c): dict(P=b.P, positives=b.positives, negatives=b.negatives,
            measured=b.measured, failed_clears=b.failed_clears, near=b.near)
            for c, b in sorted(s.beliefs.items())}, local_steps=s.local_steps,
        discovered=s.discovered, cleared=s.cleared, ledger=s.ledger,
        negative=s.cover.negative, covered=s.cover.covered, witnesses=s.cover.witnesses,
        max_active_measures=s.max_active_measures)
    raw = json.dumps(payload, sort_keys=True, allow_nan=False, separators=(',', ':'), default=json_value)
    return hashlib.sha256(raw.encode()).hexdigest()


def initial_state():
    return SimpleNamespace(position=np.zeros(2), channel=1, beliefs={}, local_steps={},
        cover=PolarCover(), discovered=set(), cleared=set(), ledger=[], max_active_measures=6)


def apply_event(state, event, active_view=False):
    """Exact old public update; explicit active_view distinguishes shared reads."""
    c, q, action = int(event['channel']), np.asarray(event['position'], float), event['action']
    demand(action in ('measure', 'clear') and q.shape == (2,) and np.isfinite(q).all(), 'invalid saved public event')
    key = 'measure_result' if action == 'measure' else 'clear_result'
    reply = {k: event[k] for k in (key, 'svd_deg') if k in event}
    parts = dict(move_s=math.dist(state.position, q)/5, measure_s=0., switch_s=0., clear_s=0.)
    if action == 'measure':
        parts.update(measure_s=5., switch_s=float(c != state.channel))
        state.channel = c
        if event[key] == 'no_signal':
            state.cover.observe_negative(c, q)
            if c in state.beliefs:
                state.beliefs[c].update(q, reply)
        elif event[key] in ('direction', 'near'):
            demand(c not in state.cleared, 'positive after successful source clear')
            if c not in state.beliefs:
                state.beliefs[c] = Belief(q, reply)
                state.beliefs[c].negatives = [p.copy() for p in state.cover.negative[c]]
                state.local_steps[c] = 0
                state.discovered.add(c)
            else:
                state.beliefs[c].update(q, reply)
        else:
            raise ValueError('invalid measure response')
        if active_view:
            demand(state.local_steps[c] < state.max_active_measures, 'active view exceeds lifetime budget')
            state.local_steps[c] += 1
    else:
        demand(not active_view, 'clear cannot consume active measurement allowance')
        if event[key] == 'success':
            demand(c not in state.cleared, 'duplicate successful clear')
            parts['clear_s'] = 5.
            state.cleared.add(c)
            state.beliefs.pop(c, None)
        elif event[key] == 'no_target_in_range':
            parts['clear_s'] = 3.
            if c in state.beliefs:
                state.beliefs[c].failed_clears.append(q.copy())
        else:
            raise ValueError('invalid clear response')
    state.position = q.copy()
    state.ledger.append(dict(action=action, position=q.tolist(), channel=c, **reply))
    return parts


def unknown(s):
    return [] if len(s.discovered) >= 16 else [c for c in range(1, 21)
        if c not in s.discovered and not s.cover.complete(c)]


def tasks_points(s, fixed=None):
    tasks = [('scan', int(k)) for k in s.cover.needed_stations(unknown(s))]
    points = [s.cover.stations[t[1]] for t in tasks]
    for c, belief in s.beliefs.items():
        tasks.append(('source', c)); points.append(core.mec(belief.P)[0])
    filtered = [(t, p) for t, p in zip(tasks, points) if fixed is None or t in fixed]
    return [t for t, _ in filtered], np.asarray([p for _, p in filtered]).reshape((-1, 2))


def baseline_task(s, fixed=None):
    tasks, points = tasks_points(s, fixed)
    return tasks[open_route(points, s.position)[0]] if tasks else None


def expected_roots(s):
    tasks, points = tasks_points(s)
    original = baseline_task(s)
    chosen = []
    for kind, count in (('source', 2), ('scan', 3)):
        # Match the frozen selector's float64 BLAS norm exactly; math.dist can
        # change a tie at the last bit and create a false audit failure.
        group = sorted([(float(np.linalg.norm(p-s.position)), t) for t, p in zip(tasks, points) if t[0] == kind],
                       key=lambda row: (row[0], row[1][1]))
        part = [t for _, t in group[:count]]
        if original is not None and original[0] == kind and original not in part:
            part[-1:] = [original]
        chosen.extend(part)
    return ([original] if original in chosen else [])+sorted(t for t in chosen if t != original)


def check_terminal(s, roots, recorded):
    tasks, points = tasks_points(s, roots)
    order = open_route(points, s.position)
    legs = [math.dist(s.position if i == 0 else points[order[i-1]], points[k])/5 for i, k in enumerate(order)]
    service, scan, radio = [], [], s.channel
    channels = unknown(s)
    for k in order:
        task, q = tasks[k], points[k]
        if task[0] == 'source':
            b = s.beliefs[task[1]]
            safe = b.near if b.near is not None else core.nearest_certified_clear(b.P, q)
            service.append(math.dist(safe, q)/5+5 if safe is not None else optical_plan(b.P, q)['score'])
        else:
            for c in sorted(channels, key=lambda c: (c != radio, c)):
                if any(math.dist(q, p) < 1e-7 for p in s.cover.negative[c]):
                    continue
                scan.append(5+int(c != radio)); radio = c
    expected = dict(travel_s=math.fsum(legs), service_s=math.fsum(service), scan_s=float(sum(scan)))
    expected['total_s'] = math.fsum(expected.values())
    for key, value in expected.items():
        close(value, recorded[key], 'public fixed-task terminal '+key)


def saved_macro(start, task, event_record, recorded_cost):
    """No feedback is generated. Infer source-active allowance from first event.

    Frozen source service either first measures its target once (one active
    view), or begins clearing it. Subsequent measures belong to share_at_stop.
    Scan never consumes a source's lifetime active-view count. The separate AST
    checks bind this inference to unchanged nested macro control flow.
    """
    end = copy.deepcopy(start)
    records = event_record['events']
    costs = []
    counts = Counter()
    for index, event in enumerate(records):
        active = index == 0 and task[0] == 'source' and event['action'] == 'measure'
        if index == 0 and task[0] == 'source':
            demand(event['channel'] == task[1], 'source macro first action does not target its source')
        if index == 0 and task[0] == 'scan':
            demand(event['action'] == 'measure' and event['position'] == start.cover.stations[task[1]].tolist(),
                   'scan macro does not enter its station')
        previous_radio = end.channel
        parts = apply_event(end, event, active_view=active)
        if event['action'] == 'clear':
            demand(end.channel == previous_radio, 'optical clear retuned the radio')
        costs.append(math.fsum(parts.values()))
        counts[event['action']] += 1
    cost = math.fsum(costs)
    close(cost, recorded_cost, 'saved macro fee from public entry state')
    return end, counts


def check_window(state, row, successor_map):
    demand(row['root_digest'] == public_digest(state), 'window root digest not bound to actual prefix')
    roots = list(map(tuple, row['root_tasks']))
    demand(roots == expected_roots(state), 'root action inventory/order differs from real state')
    original = baseline_task(state)
    demand(tuple(row['baseline_task']) == original and tuple(row['selected']) in roots, 'invalid baseline/selected root')
    depth, budget = row['depth'], row['budget']
    demand(depth in (1, 2) and budget['max_transitions'] == 60 and budget['max_api_calls'] == 6000, 'depth/budget changed')
    demand(0 <= budget['used_transitions'] <= 60 and 0 <= budget['predicted_measure']+budget['predicted_clear'] <= 6000,
           'planner exceeded recorded budget')
    demand([tuple(r['task']) for r in row['rows']] == roots[:len(row['rows'])], 'saved root rows are not the deterministic prefix')
    counted, terminal_count = Counter(), 0
    for root_row in row['rows']:
        root = tuple(root_row['task'])
        demand([s['world_index'] for s in root_row['samples']] == list(range(6)), 'completed root lacks six world records')
        for sample in root_row['samples']:
            first, counts = saved_macro(state, root, sample['first_events'], sample['first_cost_s'])
            counted.update(counts); counted['transitions'] += 1
            first_digest = public_digest(first)
            demand(first_digest == sample['first_public_digest'], 'first macro public state not bound to saved feedback')
            second = None if sample['second_task'] is None else tuple(sample['second_task'])
            if depth == 2:
                demand(second == baseline_task(first, roots), 'second task differs from public fixed-task policy')
                demand(second is None or second in roots, 'second task left fixed root episode')
                key = (first_digest, tuple(roots))
                demand(key not in successor_map or successor_map[key] == second, 'same public state chose different second tasks')
                successor_map[key] = second
            else:
                demand(second is None, 'depth one unexpectedly has a second task')
            if second is not None:
                end, counts = saved_macro(first, second, sample['second_events'], sample['second_cost_s'])
                counted.update(counts); counted['transitions'] += 1
            else:
                demand(sample['second_events'] is None, 'events exist without second task')
                close(sample['second_cost_s'], 0., 'missing second task cost')
                end = first
            check_terminal(end, roots, sample['terminal'])
            close(sample['total_s'], sample['first_cost_s']+sample['second_cost_s']+sample['terminal']['total_s'], 'sample total')
            terminal_count += 1
        close(root_row['mean_s'], math.fsum(s['total_s'] for s in root_row['samples'])/6, 'complete root mean')
    complete = row['reason'] == 'complete_root_comparison'
    if complete:
        demand(row['world_count'] == 6 and len(row['rows']) == len(roots), 'incomplete root comparison marked complete')
        chosen, value = original, math.inf
        for root_row in row['rows']:
            if root_row['mean_s'] < value-1e-8:
                chosen, value = tuple(root_row['task']), root_row['mean_s']
        demand(tuple(row['selected']) == chosen, 'wrong full root minimum or tie policy')
        for key, observed in (('used_transitions', counted['transitions']), ('predicted_measure', counted['measure']),
                              ('predicted_clear', counted['clear']), ('terminal_evaluations', terminal_count)):
            demand(budget[key] == observed, 'complete-round budget mismatch '+key)
    else:
        demand(tuple(row['selected']) == original and row['applied'] is False, 'partial/rejected round changed actual task')
        for key, observed in (('used_transitions', counted['transitions']), ('predicted_measure', counted['measure']),
                              ('predicted_clear', counted['clear']), ('terminal_evaluations', terminal_count)):
            demand(budget[key] >= observed, 'partial budget smaller than preserved complete rows '+key)
    demand(row['applied'] is (tuple(row['selected']) != original), 'applied flag differs from selected root')
    return dict(complete_comparison=complete, applied=row['applied'], reason=row['reason'],
                saved_complete_roots=len(row['rows']), preserved_macro_transitions=counted['transitions'],
                preserved_predicted_measure=counted['measure'], preserved_predicted_clear=counted['clear'],
                preserved_terminal_evaluations=terminal_count,
                interrupted_unrecorded_transitions=budget['used_transitions']-counted['transitions'],
                interrupted_unrecorded_api_calls=budget['predicted_measure']+budget['predicted_clear']-counted['measure']-counted['clear'])


def audit_case(path, successor_map):
    payload = json.loads(path.read_text(encoding='utf-8'))
    events, trace = payload['events'], payload['trace']
    del payload
    state, count, windows = initial_state(), 0, []
    pending_window, pending_dispatch, previous_window_prefix = None, None, -1
    actual_costs = []
    for index, row in enumerate(trace):
        phase = row.get('phase')
        if phase == 'window_rollout_prediction':
            demand(row['event_prefix'] == count and count > previous_window_prefix, 'window did not replan after new actual events')
            demand(row['window_call'] == len(windows)+1 <= 4, 'window call limit/order changed')
            demand(pending_window is None and pending_dispatch is None, 'unconsumed earlier task/window')
            report = check_window(state, row, successor_map)
            windows.append(dict(trace_index=index, actual_event_prefix=count, depth=row['depth'], **report))
            demand(index+1 < len(trace) and trace[index+1]['phase'] == 'task_choice', 'window not followed immediately by task choice')
            pending_window, previous_window_prefix = row, count
        elif phase == 'task_choice':
            demand(pending_dispatch is None, 'previous chosen task has no actual first action')
            task = (row['task'], row['index'])
            if pending_window is not None:
                demand(task == tuple(pending_window['selected']), 'actual task choice differs from selected first step')
                pending_window = None
            else:
                demand(not (state.beliefs and state.cover.needed_stations(unknown(state)) and len(windows) < 4), 'eligible window record missing')
                demand(task == baseline_task(state), 'non-window task differs from frozen baseline')
            demand(row['known'] == len(state.beliefs) and row['unknown'] == len(unknown(state)) and
                   row['pending_stations'] == len(state.cover.needed_stations(unknown(state))), 'actual task counters differ')
            pending_dispatch = task
        elif phase == 'actual_action':
            demand(row['event'] == count+1 and count < len(events), 'nonconsecutive actual event ledger')
            event = events[count]
            key = 'measure_result' if event['action'] == 'measure' else 'clear_result'
            demand((event['action'], event['channel'], event['position'], event[key]) ==
                   (row['action'], row['channel'], row['position'], row['result']), 'actual trace/ledger mismatch')
            if pending_dispatch is not None:
                task = pending_dispatch
                if task[0] == 'source':
                    demand(event['channel'] == task[1] and row['reason'] in ('source_measure', 'near_clear', 'certified_clear', 'optical_cover'),
                           'chosen source first action mismatch')
                else:
                    demand(row['reason'] == 'scan' and event['position'] == state.cover.stations[task[1]].tolist(), 'chosen station first action mismatch')
                pending_dispatch = None
            parts = apply_event(state, event, active_view=row['reason'] == 'source_measure')
            close(parts['move_s'], row['move_s'], 'actual move fee')
            actual_costs.append(math.fsum(parts.values())); count += 1
            close(math.fsum(actual_costs), event['time_s'], 'actual cumulative virtual clock')
        elif phase == 'belief':
            b = state.beliefs[row['channel']]
            demand(b.P.tolist() == row['polygon'] and [p.tolist() for p in b.positives] == row['positives']
                   and [p.tolist() for p in b.negatives] == row['negatives'], 'actual belief geometry/observations mismatch')
    demand(count == len(events) and pending_window is None and pending_dispatch is None, 'unfinished actual trace audit')
    return dict(input=binding(path), passed=True, actual_actions=count, windows=windows,
                window_count=len(windows), complete_comparisons=sum(w['complete_comparison'] for w in windows),
                applied_windows=sum(w['applied'] for w in windows),
                preserved_macro_transitions=sum(w['preserved_macro_transitions'] for w in windows),
                preserved_predicted_api_calls=sum(w['preserved_predicted_measure']+w['preserved_predicted_clear'] for w in windows),
                interrupted_unrecorded_transitions=sum(w['interrupted_unrecorded_transitions'] for w in windows),
                interrupted_unrecorded_api_calls=sum(w['interrupted_unrecorded_api_calls'] for w in windows))


def binding(path):
    path = path.resolve()
    return dict(path=path.relative_to(PROJECT).as_posix(), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True, help='NEW audit directory; no overwrite')
    args = parser.parse_args()
    demand(not args.out.exists(), 'refuse to overwrite old audit')
    manifest_path = args.input/'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    paths = sorted(args.input.glob('*-window_depth[12].json'))
    demand(len(paths) == 2*manifest['paired_conditions'] and len(list(args.input.glob('*-baseline_share25_prune.json'))) == manifest['paired_conditions'],
           'full planned triple batch not yet saved')
    dependencies = [Q4/'localization.py', Q4/'polar_cover.py', Q4/'directional_cover.py', Q4/'shared.py',
                    Q4/'route_planning.py', PROJECT/'src/q3_model_v2/geometry.py',
                    HERE/'window_joint_planner_v1.py', HERE/'window_worlds_v1.py']
    hashes = [binding(p) for p in dependencies]
    for record in hashes:
        demand(record['sha256'] == manifest['code_hashes'][record['path']], 'audited source differs from frozen batch')
    successor_map, rows = {}, []
    for path in paths:
        row = audit_case(path, successor_map)
        rows.append(row)
        print(json.dumps(dict(case=path.name, windows=row['window_count'], complete=row['complete_comparisons']), ensure_ascii=False), flush=True)
    demand(hashes == [binding(p) for p in dependencies], 'audited dependency changed during execution')
    report = dict(passed=True, audit_source=binding(Path(__file__)), manifest=binding(manifest_path), dependencies=hashes,
        scope='Only saved actual/hypothetical feedback records; no sources inspected, world sampling, macro execution or solver run.',
        rows=rows, same_public_state_successor_keys=len(successor_map),
        totals={key:sum(r[key] for r in rows) for key in ('actual_actions', 'window_count', 'complete_comparisons',
            'applied_windows', 'preserved_macro_transitions', 'preserved_predicted_api_calls',
            'interrupted_unrecorded_transitions', 'interrupted_unrecorded_api_calls')},
        reconstruction_basis='Public geometry/cover/ledger restored from saved events. A source macro first measuring its own target consumes one active view; later shared measurements do not. Clear keeps radio channel.',
        limitations=['Counter consistency and reconstructed public-state decisions do not prove general hidden-state noninterference; separate bounded structural/state-isolation tests supply mechanism evidence.',
                     'Rejected interrupted macros have no saved internal event sequence beyond complete root rows. Their extra work is only bounded against declared caps, never reconstructed or treated as fully verified.',
                     'World sample provenance, completeness, distribution calibration and physical latent-source compatibility are not proved here. No hidden sources were used.',
                     'Terminal is the fixed-root-task heuristic, not a complete global continuation or two-step optimality guarantee.',
                     'Actual continuous optical/absence geometry certificates belong to separate complete trajectory audits.'])
    args.out.mkdir(parents=True, exist_ok=False)
    with (args.out/'summary.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2); stream.write('\n')
    print(json.dumps(dict(output=str(args.out/'summary.json'), passed=True, **report['totals']), ensure_ascii=False))


if __name__ == '__main__':
    main()
