"""Reproducible local-only geometric checks and single-source development.

Reuses positive observations and service-start states from the old development
archive, so every tested policy has identical information and physical starts.
No test here connects to the official simulator or consumes seeds >= 4000.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import baseline_solver as baseline
import geometry as core
from environment import LocalArena
from local_policy import choose_action, best_optical_plan, cover_is_complete, OpticalCoverExhausted


def polygon_contains(P, q):
    if len(P) < 3:
        return True
    e = np.roll(P, -1, axis=0) - P
    d = q - P
    return bool(np.min(e[:, 0] * d[:, 1] - e[:, 1] * d[:, 0]) >= -1e-5)


def geometry_checks():
    cases = [np.array([[-80., -4], [80, -4], [80, 4], [-80, 4]]),
             np.array([[-28., -28], [28, -28], [28, 28], [-28, 28]]),
             np.array([[0., 0], [90., -10], [90., 10]]),
             np.array([[0., 0], [50., 0]]), np.array([[17., -6.]])]
    rows = []
    for i, P in enumerate(cases):
        plan = best_optical_plan(P, np.array([-100., -80.]))
        assert plan is not None and cover_is_complete(P, plan['path'])
        # Deliberately corrupted sets must fail: a disk on each square vertex
        # misses the middle even though all original vertices are covered.
        rows.append(dict(case=i, parts=plan['parts'], complete=True, worst_s=plan['worst_s']))
    P = np.array([[-35., -35], [35., -35], [35., 35], [-35., 35]])
    assert not cover_is_complete(P, P)
    try:
        choose_action(P, np.zeros(2), np.zeros(2), 1, state={'exhausted': True})
    except OpticalCoverExhausted:
        pass
    else:
        raise AssertionError('failed whole cover silently restarted')
    # Historical fixed-error measurements cannot be treated as fresh evidence.
    P = core.initial_belief(np.zeros(2), 0.)
    original = choose_action(P, np.zeros(2), np.zeros(2), 1, current_channel=1, mode='baseline')
    alternate = choose_action(P, np.zeros(2), np.zeros(2), 1, current_channel=1, mode='baseline',
                              observed_positions=[original['q']])
    assert np.linalg.norm(original['q'] - alternate['q']) > .1
    switched = choose_action(P, np.zeros(2), np.zeros(2), 1, current_channel=2, mode='baseline')
    assert np.isfinite(original['estimated_s']) and abs(switched['estimated_s'] - original['estimated_s'] - 1) < 1e-7
    return dict(covers=rows, vertex_only_false_certificate_rejected=True, exhausted_cover_rejected=True,
                repeated_measurement_rejected=True, switch_cost_in_score_verified=True)


def run_local(source, seed, field, anchor, first, start, mode):
    arena = LocalArena([source], seed, field, start=start)
    channel = source['channel']
    arena.channel = channel
    if first['measure_result'] == 'near':
        result = arena.clear(anchor, channel)
        assert result['clear_result'] == 'success'
        return {**arena.evaluation(), 'mode': mode, 'steps': 1, 'actions': ['near_clear']}
    active_core = baseline if mode == 'baseline' else core
    P = active_core.initial_belief(np.array(anchor), first['svd_deg'])
    witness, state, actions = np.array(anchor), None, []
    source_position = np.array(source['position'])  # only post-decision auditing
    for k in range(50):
        assert polygon_contains(P, source_position), 'belief excluded truth'
        if mode == 'baseline':
            q = baseline.nearest_certified_clear(P, arena.position)
            action = dict(kind='clear', q=q, state=None, rationale='baseline_certified_clear') if q is not None else dict(
                kind='measure', q=baseline.choose_measure(P, arena.position, 'time', witness),
                state=None, rationale='baseline_time_rollout')
        else:
            action = choose_action(P, arena.position, witness, channel,
                                   current_channel=channel, mode='hybrid' if mode == 'hybrid_short' else mode,
                                   candidate_mode='short' if mode == 'hybrid_short' else 'standard', state=state)
        actions.append(action['rationale'])
        if action['kind'] == 'clear':
            outcome = arena.clear(action['q'], channel)
            state = action['state']
            if outcome['clear_result'] == 'success':
                return {**arena.evaluation(), 'mode': mode, 'steps': k + 1, 'actions': actions}
        else:
            q = action['q']
            assert core.reception_certified(P, q, witness)
            result = arena.measure(q, channel)
            assert result['measure_result'] != 'no_signal'
            if result['measure_result'] == 'near':
                assert arena.clear(q, channel)['clear_result'] == 'success'
                return {**arena.evaluation(), 'mode': mode, 'steps': k + 2, 'actions': actions}
            P = active_core.update_belief(P, q, result['svd_deg'])
            witness, state = q.copy(), None
    raise AssertionError('local action limit')


def development(seeds, fields, modes):
    rows = []
    old = Path(__file__).resolve().parents[1] / 'q3_improved/results/development-v1/cases'
    for seed in seeds:
        if seed >= 4000:
            raise ValueError('holdout seeds forbidden in local development')
        for field in fields:
            case = json.loads((old / f'{seed}-{field}-dynamic.json').read_text(encoding='utf-8'))
            positives = {}
            for e in case['events']:
                if e['action'] == 'measure' and e['measure_result'] != 'no_signal':
                    positives.setdefault(e['channel'], (e['position'], e))
            starts = {}
            for item in case['trace']:
                if item.get('phase') == 'localize':
                    starts.setdefault(item['channel'], item['position'])
            for source in case['sources']:
                channel = source['channel']
                anchor, first = positives[channel]
                start = starts.get(channel, anchor)
                for mode in modes:
                    before = time.perf_counter()
                    metrics = run_local(source, seed, field, anchor, first, start, mode)
                    row = dict(seed=seed, field=field, channel=channel,
                               runtime_s=time.perf_counter() - before, **metrics)
                    rows.append(row)
                print(json.dumps(dict(seed=seed, field=field, channel=channel,
                                      times={x['mode']: round(x['total_s'], 2) for x in rows[-len(modes):]})), flush=True)
    summary = {}
    for mode in modes:
        items = [r for r in rows if r['mode'] == mode]
        summary[mode] = dict(sources=len(items), cleared=sum(r['all_cleared'] for r in items),
                             mean_s=float(np.mean([r['total_s'] for r in items])),
                             sum_s=float(sum(r['total_s'] for r in items)),
                             mean_measure=float(np.mean([r['counts']['measure'] for r in items])),
                             clear_failures=sum(r['counts']['clear_fail'] for r in items),
                             runtime_s=float(sum(r['runtime_s'] for r in items)))
    return dict(summary=summary, rows=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', default='1000')
    parser.add_argument('--fields', default='smooth')
    parser.add_argument('--modes', default='baseline,optical,hybrid')
    parser.add_argument('--output', default='local_experiments.json')
    args = parser.parse_args()
    hashes = {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
              for name in ['local_policy.py', 'local_checks.py', 'baseline_solver.py', 'geometry.py', 'environment.py']}
    results = dict(code_hashes_at_start=hashes,
                   bearing_half_angle_deg={'baseline': float(np.degrees(baseline.ANGLE_EPS)),
                                           'optical_and_hybrid': float(np.degrees(core.ANGLE_EPS))},
                   geometry_checks=geometry_checks(),
                   development=development([int(s) for s in args.seeds.split(',')],
                                           args.fields.split(','), args.modes.split(',')))
    (Path(__file__).parent / args.output).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(results['development']['summary'], ensure_ascii=False, indent=2))
