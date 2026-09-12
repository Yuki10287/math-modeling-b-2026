"""Paired full-policy scenario costs, never clairvoyant service visits."""
import copy
import hashlib
import json
import math
import time
import numpy as np

import bootstrap
from scan_preview import rank_scan_tasks
from scenario_model import scenario_worlds, clone_for_world
from rollout_solver import continue_lean


def applicable(model, tasks, route):
    return bool(route and route[0]['kind'] == 'scan' and not model.beliefs
                and model.discovered == model.cleared and len(tasks) >= 2
                and all(task['kind'] == 'scan' for task in tasks))


def shortlist(model, tasks, route, limit=3):
    """Include lean, the old preview's best alternative, and lean's next scan."""
    selected = [route[0]]
    old = rank_scan_tasks(model, tasks, route)
    for candidate in [*(task for _, task in old), *route[1:]]:
        if candidate['key'] not in {task['key'] for task in selected}:
            selected.append(candidate)
            break
    for candidate in route[1:]:
        if len(selected) >= limit:
            break
        if candidate['key'] not in {task['key'] for task in selected}:
            selected.append(candidate)
    return selected[:limit]


def state_key(model, iteration, candidates, count):
    public = dict(position=np.asarray(model.position).tolist(), channel=model.channel,
                  iteration=iteration, discovered=sorted(model.discovered), cleared=sorted(model.cleared),
                  positions={str(c): [np.asarray(p).tolist() for p in ps] for c, ps in model.positions.items()},
                  negatives=model.coverage._points, channels=model.coverage.channels,
                  coverage_config=[model.coverage.target_r, model.coverage.signal_r, model.coverage.cell_size],
                  use_negatives=model.use_negatives, range_cuts=model.range_cuts,
                  search_fraction=model.search_fraction, candidates=candidates, scenarios=count)
    digest = hashlib.sha256(json.dumps(public, sort_keys=True).encode())
    digest.update(model.coverage._excluded.tobytes())
    digest.update(model.coverage._counts.tobytes())
    return digest.hexdigest()


def forecast(model, iteration, candidates, count=6):
    worlds = scenario_worlds(model, count=count)
    if not worlds:
        return None
    started = time.perf_counter()
    rows = []
    for task in candidates:
        costs, errors, details = [], [], []
        for index, world in enumerate(worlds):
            field = ('smooth', 'extreme', 'hash')[index % 3]
            try:
                clone, arena = clone_for_world(model, world, field=field, scenario_index=index)
                outcome = continue_lean(clone, start_iteration=iteration, first_task=task)
                if not outcome['complete'] or not arena.evaluation()['all_cleared']:
                    raise RuntimeError(f'incomplete scenario: {outcome["status"]}')
                cost = float(arena.time_s)
                if not math.isfinite(cost):
                    raise ValueError('nonfinite scenario cost')
                costs.append(cost)
                errors.append(None)
                details.append(dict(counts=arena.counts.copy(), final_position=arena.position.tolist(),
                                    fallback_count=outcome['fallback_count'], field=field))
            except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
                costs.append(None)
                errors.append(f'{type(exc).__name__}: {exc}')
                details.append(None)
        rows.append(dict(task=copy.deepcopy(task), costs_s=costs, errors=errors, scenarios=details))
    return dict(candidates=rows, world_counts=[len(world) for world in worlds],
                scenario_count=len(worlds), forecast_runtime_s=time.perf_counter()-started)


def decision(prediction, risk=.2, margin_s=5.):
    """Finite-scenario relative regret; its maximum is not a global bound."""
    if prediction is None or any(v is None for v in prediction['candidates'][0]['costs_s']):
        return 0, []
    base = np.asarray(prediction['candidates'][0]['costs_s'])
    rows = []
    for index, row in enumerate(prediction['candidates']):
        if any(v is None for v in row['costs_s']):
            rows.append(dict(index=index, valid=False))
            continue
        delta = np.asarray(row['costs_s']) - base
        score = (1-risk)*float(delta.mean()) + risk*float(delta.max())
        rows.append(dict(index=index, valid=True, mean_delta_s=float(delta.mean()),
                         worst_delta_s=float(delta.max()), best_delta_s=float(delta.min()),
                         improved_scenarios=int(sum(delta < -1e-6)), regret_score_s=score))
    eligible = [row for row in rows if row['valid'] and row['regret_score_s'] < -margin_s]
    chosen = min(eligible, key=lambda row: (row['regret_score_s'], row['index']))['index'] if eligible else 0
    return chosen, rows


class ClosedLoopSelector:
    """At most two real decisions; cached only by complete relevant public state."""
    def __init__(self, *, scenario_count=6, max_candidates=3, risk=.2,
                 margin_s=5., max_decisions=2, cache=None):
        self.scenario_count = scenario_count
        self.max_candidates = max_candidates
        self.risk = risk
        self.margin_s = margin_s
        self.max_decisions = max_decisions
        self.calls = 0
        self.cache = {} if cache is None else cache

    def __call__(self, model, iteration, tasks, route):
        if self.calls >= self.max_decisions or not applicable(model, tasks, route):
            return None
        self.calls += 1
        candidates = shortlist(model, tasks, route, self.max_candidates)
        key = state_key(model, iteration, candidates, self.scenario_count)
        cached = key in self.cache
        if not cached:
            self.cache[key] = forecast(model, iteration, candidates, self.scenario_count)
        prediction = self.cache[key]
        chosen, scores = decision(prediction, self.risk, self.margin_s)
        model.trace.append(dict(phase='closed_scan_forecast', iteration=iteration, state_key=key,
            cached=cached, prediction=prediction, scores=scores, chosen_index=chosen,
            risk=self.risk, margin_s=self.margin_s, changed=chosen != 0))
        return candidates[chosen] if chosen else None
