"""OFFLINE ONLY: compare predicted action differences with actual continuations.

Hidden source layouts are used only by this audit's synthetic feedback APIs,
after all online choices have been saved. They never enter the online selector.
"""
import argparse
import json
from pathlib import Path
import numpy as np

from bootstrap import FeedbackModel
from validate_model import BoundedArena, PublicAPI
from rollout_solver import continue_lean
from scenario_model import clone_for_world, scenario_worlds, _copy_coverage
from closed_preview import state_key
from scan_preview import _mask, _rollout
from v1_solver import rolling_route


class SnapshotAPI:
    def __init__(self, position, channel):
        self.position = np.asarray(position).copy()
        self.channel = channel

    def measure(self, *args):
        raise AssertionError('an offline evidence snapshot cannot execute actions')

    clear = measure


def snapshot(model):
    assert not model.beliefs and model.discovered == model.cleared
    clone = FeedbackModel(SnapshotAPI(model.position, model.channel), [],
                          use_negatives=model.use_negatives, range_cuts=model.range_cuts,
                          search_fraction=model.search_fraction)
    clone.coverage = _copy_coverage(model.coverage)
    clone.positions = {c: [np.asarray(p).copy() for p in ps] for c, ps in model.positions.items()}
    clone.discovered, clone.cleared = set(model.discovered), set(model.cleared)
    return clone


def old_costs(model, tasks, candidates, worlds):
    """Evaluate the old proxy on exactly the same source worlds, not a new draw."""
    channels = model.unknown()
    points = np.asarray([task['position'] for task in tasks])
    masks = np.asarray([_mask(model.coverage, q) for q in points])
    residual = ~model.coverage._excluded[[model.coverage._index(c) for c in channels]]
    values = []
    for candidate in candidates:
        first = next(i for i, task in enumerate(tasks) if task['key'] == candidate['key'])
        rest = [dict(task, preview_index=i) for i, task in enumerate(tasks) if i != first]
        tail, _ = rolling_route(points[first], rest) if rest else ([], 0.)
        order = [first] + [task['preview_index'] for task in tail]
        remaining, gains = residual.copy(), []
        for index in order:
            gains.append(np.any(remaining & masks[index], axis=1))
            remaining &= ~masks[index]
        values.append([_rollout(np.asarray(model.position), order, points, gains, channels,
                                world, len(model.discovered)) for world in worlds])
    return np.asarray(values)


def analyze(path):
    saved = json.loads(path.read_text(encoding='utf-8'))
    case = saved['result']
    records = [r for r in saved['trace'] if r.get('phase') == 'closed_scan_forecast']
    arena = BoundedArena(saved['sources'], case['seed'], case['field'], max_actions=3000)
    model = FeedbackModel(PublicAPI(arena), [], range_cuts=True, search_fraction=.30)
    checkpoints = []

    def replay(model, iteration, tasks, route):
        matches = [row for row in records if row['iteration'] == iteration]
        if not matches:
            return None
        record = matches[0]
        if record['prediction'] is None:
            return None
        candidates = [r['task'] for r in record['prediction']['candidates']]
        assert state_key(model, iteration, candidates, record['prediction']['scenario_count']) == record['state_key']
        checkpoints.append((snapshot(model), iteration, tasks, candidates, record))
        return candidates[record['chosen_index']] if record['changed'] else None

    result = continue_lean(model, selector=replay)
    assert result['complete']
    assert arena.events == saved['events'], 'recorded online decisions failed exact replay'
    output = []
    for model, iteration, tasks, candidates, record in checkpoints:
        truth = {s['channel']: (np.asarray(s['position']), s['radius'])
                 for s in saved['sources'] if s['channel'] not in model.cleared}
        actual = []
        for task in candidates:
            clone, sim = clone_for_world(model, truth, field=case['field'], scenario_index=case['seed'])
            outcome = continue_lean(clone, start_iteration=iteration, first_task=task)
            assert outcome['complete'] and sim.evaluation()['all_cleared']
            actual.append(float(sim.time_s))
        closed = np.asarray([row['costs_s'] for row in record['prediction']['candidates']], dtype=float)
        worlds = scenario_worlds(model, count=record['prediction']['scenario_count'])
        old = old_costs(model, tasks, candidates, worlds)
        alternatives = []
        for i in range(1, len(candidates)):
            actual_delta = actual[i]-actual[0]
            old_delta = float(np.mean(old[i]-old[0]))
            closed_delta = float(np.mean(closed[i]-closed[0]))
            alternatives.append(dict(index=i, actual_delta_s=actual_delta,
                old_predicted_delta_s=old_delta, closed_predicted_delta_s=closed_delta,
                old_absolute_delta_error_s=abs(old_delta-actual_delta),
                closed_absolute_delta_error_s=abs(closed_delta-actual_delta),
                old_correct_improvement_sign=bool((old_delta < -5) == (actual_delta < -5)),
                closed_correct_improvement_sign=bool((closed_delta < -5) == (actual_delta < -5))))
        output.append(dict(iteration=iteration, historical_cleared=len(model.cleared),
            candidates=candidates, actual_continuation_s=actual, old_scenario_costs_s=old.tolist(),
            closed_scenario_costs_s=closed.tolist(), alternatives=alternatives,
            online_chosen_index=record['chosen_index']))
    return dict(name=case['name'], online_variant=case['schedule'], exact_online_replay=True,
                checkpoints=output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', nargs='+', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    assert not args.out.exists(), 'preserve earlier diagnosis'
    rows = []
    for path in args.cases:
        rows.append(analyze(path))
        print(json.dumps(dict(diagnosed=str(path), checkpoints=len(rows[-1]['checkpoints']))), flush=True)
    alternatives = [a for row in rows for c in row['checkpoints'] for a in c['alternatives']]
    summary = dict(comparisons=len(alternatives))
    for method in ('old', 'closed'):
        summary[method] = dict(mean_absolute_delta_error_s=float(np.mean([
            a[f'{method}_absolute_delta_error_s'] for a in alternatives])),
            correct_improvement_sign=sum(a[f'{method}_correct_improvement_sign'] for a in alternatives))
    payload = dict(local_only=True, official_simulator_used=False, diagnostic_only=True,
        note='Truth is used only for post-hoc counterfactual feedback. Samples and checkpoints are development data; rankings cannot be retuned and called held-out.',
        cases=rows, summary=summary)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
